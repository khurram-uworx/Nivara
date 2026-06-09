using Nivara.Exceptions;
using Nivara.Operations;
using Nivara.Query;
using System.Collections.Concurrent;

namespace Nivara.Execution;

sealed class ParallelExecutionStrategy : ExecutionStrategyBase
{
    static bool isParallelizable(string operationType)
    {
        return operationType switch
        {
            "Filter" => true,
            "Select" => false,
            "Sort" => true,
            "GroupBy" => true,
            "Join" => true,
            _ when operationType.StartsWith("Concatenate", StringComparison.Ordinal) => true,
            _ => false
        };
    }

    static bool shouldUseParallelism(IReadOnlyDictionary<string, IColumn> input, NivaraExecutionContext context)
    {
        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        return ParallelExecutionHelper.ShouldUseParallelProcessing(
            totalRows,
            context.MaxDegreeOfParallelism);
    }

    static void validateParallelismConfiguration(NivaraExecutionContext context)
    {
        ParallelExecutionHelper.ValidateParallelConfiguration(context.MaxDegreeOfParallelism);
    }

    // ── Shared chunk kernel for simple slice+execute operations (Filter) ──

    static Func<int, int, IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>
        createSliceExecuteKernel(IQueryOperation operation)

        => (start, length, columns) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(columns, start, length);
            return operation.Execute(subset);
        };

    // ── Sync dispatch ──

    IReadOnlyDictionary<string, IColumn> executeOperationParallelSync(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        NivaraExecutionContext context)
    {
        if (!isParallelizable(operation.OperationType) || !shouldUseParallelism(input, context))
            return operation.Execute(input);

        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);
        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(totalRows, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(totalRows, chunkSize);

        if (ranges.Count <= 1)
            return operation.Execute(input);

        var opType = operation.OperationType;
        if (opType == "Filter")
            return executeFilterParallelSync(operation, input, ranges, maxDop, context.CancellationToken);
        if (opType == "Sort")
            return executeSortParallelSync((SortOperation)operation, input, ranges, maxDop, context.CancellationToken);
        if (opType == "GroupBy")
            return executeGroupByParallelSync((GroupByOperation)operation, input, ranges, maxDop, context.CancellationToken);
        if (opType == "Join")
            return executeJoinParallelSync((JoinOperation)operation, maxDop, context.CancellationToken);
        if (opType.StartsWith("Concatenate", StringComparison.Ordinal))
            return executeConcatenationParallelSync((ConcatenationOperation)operation, input, maxDop, context.CancellationToken);

        return operation.Execute(input);
    }

    IReadOnlyDictionary<string, IColumn> executeSortParallelSync(
        SortOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges,
        int maxDop,
        CancellationToken cancellationToken)
    {
        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        var sortKeys = operation.SortKeys.ToList();
        var stable = operation.IsStable;
        var chunkStarts = ranges.Select(r => r.Start).ToArray();

        var sortedLocalIndices = new int[ranges.Count][];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(input, range.Start, range.Length);
            var localIndices = Enumerable.Range(0, range.Length).ToArray();
            var comparer = new MultiColumnComparer(subset, sortKeys);

            if (stable)
                localIndices = localIndices.OrderBy(i => i, comparer).ToArray();
            else
                Array.Sort(localIndices, comparer);

            sortedLocalIndices[index] = localIndices;
        });

        var globalIndices = new int[totalRows];
        var currentPtrs = new int[ranges.Count];
        var mergeComparer = new MultiColumnComparer(input, sortKeys);

        for (int outputPos = 0; outputPos < totalRows; outputPos++)
        {
            int bestChunk = -1;
            int bestOriginalRow = -1;

            for (int c = 0; c < ranges.Count; c++)
            {
                if (currentPtrs[c] >= ranges[c].Length)
                    continue;

                var originalRow = chunkStarts[c] + sortedLocalIndices[c][currentPtrs[c]];

                if (bestChunk == -1)
                {
                    bestChunk = c;
                    bestOriginalRow = originalRow;
                }
                else
                {
                    var cmp = mergeComparer.Compare(originalRow, bestOriginalRow);
                    if (cmp < 0 || (cmp == 0 && c < bestChunk))
                    {
                        bestChunk = c;
                        bestOriginalRow = originalRow;
                    }
                }
            }

            globalIndices[outputPos] = bestOriginalRow;
            currentPtrs[bestChunk]++;
        }

        var sortedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in input)
            sortedColumns[kvp.Key] = SortOperation.ReorderColumn(kvp.Value, globalIndices);

        return sortedColumns;
    }

    IReadOnlyDictionary<string, IColumn> executeFilterParallelSync(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var kernel = createSliceExecuteKernel(operation);
        var results = new IReadOnlyDictionary<string, IColumn>[ranges.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            results[(int)index] = kernel(range.Start, range.Length, input);
        });

        return ParallelExecutionHelper.ConcatenateColumnDictionaries(results);
    }

    IReadOnlyDictionary<string, IColumn> executeGroupByParallelSync(
        GroupByOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var keyColumnNames = operation.GroupByColumns
            .Select(expr => GetColumnName(expr, input))
            .ToArray();

        var partialMaps = new ConcurrentBag<Dictionary<GroupKey, List<int>>>();
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, range =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(input, range.Start, range.Length);
            var groupedData = GroupByOperation.CreateGroupsInternal(subset, keyColumnNames, range.Start);
            partialMaps.Add(groupedData.Groups);
        });

        var mergedMap = ParallelExecutionHelper.MergeGroupByDictionaries(partialMaps.ToArray());
        var mergedGroupedData = new GroupedData(mergedMap, keyColumnNames, input);

        var resultColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var keyColumnName in keyColumnNames)
        {
            var sourceColumn = input[keyColumnName];
            resultColumns[keyColumnName] = GroupByOperation.ExtractDistinctKeyValues(mergedGroupedData, keyColumnName, sourceColumn);
        }

        return resultColumns;
    }

    IReadOnlyDictionary<string, IColumn> executeJoinParallelSync(
        JoinOperation operation, int maxDop, CancellationToken cancellationToken)
    {
        var rightColumns = operation.RightColumns;
        var rightRowCount = rightColumns.Values.FirstOrDefault()?.Length ?? 0;
        var rightKeyColumns = operation.JoinKeys.Select(jk => jk.RightColumn).ToArray();

        if (rightRowCount <= 1)
            return operation.Execute(new Dictionary<string, IColumn>());

        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(rightRowCount, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(rightRowCount, chunkSize);

        if (ranges.Count <= 1)
            return operation.Execute(new Dictionary<string, IColumn>());

        var partialMaps = new ConcurrentBag<Dictionary<CompositeKey, List<int>>>();
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        var results = new Dictionary<CompositeKey, List<int>>[ranges.Count];
        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(rightColumns, range.Start, range.Length);
            var subRowCount = subset.Values.FirstOrDefault()?.Length ?? 0;
            var partialMap = JoinOperation.BuildHashMap(subset, rightKeyColumns, subRowCount, range.Start);
            results[(int)index] = partialMap;
        });

        var mergedMap = ParallelExecutionHelper.MergeJoinHashMaps(results);

        var joinIndices = operation.ComputeJoinIndicesWithHashMap(mergedMap);
        return operation.MaterializeResult(joinIndices);
    }

    IReadOnlyDictionary<string, IColumn> executeConcatenationParallelSync(
        ConcatenationOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDop, CancellationToken cancellationToken)
    {
        var allSources = new List<IReadOnlyDictionary<string, IColumn>> { input };
        allSources.AddRange(operation.Sources);

        if (operation.Direction == ConcatenationDirection.Vertical)
        {
            var allColumnNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var source in allSources)
                foreach (var key in source.Keys)
                    allColumnNames.Add(key);

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            var columnList = allColumnNames.ToList();

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDop,
                CancellationToken = cancellationToken
            };

            var columnResults = new ConcurrentBag<(string Name, IColumn Column)>();

            Parallel.ForEach(columnList, parallelOptions, columnName =>
            {
                var columnsToConcat = new List<IColumn>();
                foreach (var source in allSources)
                {
                    if (source.TryGetValue(columnName, out var column))
                        columnsToConcat.Add(column);
                    else if (operation.MismatchHandling == ConcatenationMismatchHandling.FillWithNulls)
                    {
                        var sourceLength = source.Values.FirstOrDefault()?.Length ?? 0;
                        if (sourceLength > 0)
                        {
                            var referenceColumn = GetReferenceColumn(allSources, columnName);
                            if (referenceColumn != null)
                            {
                                var nullCol = operation.CreateNullColumn(referenceColumn.ElementType, sourceLength);
                                columnsToConcat.Add(nullCol);
                            }
                        }
                    }
                }

                if (columnsToConcat.Count > 0)
                {
                    var concatenated = ConcatenationOperation.ConcatenateColumns(columnsToConcat);
                    columnResults.Add((columnName, concatenated));
                }
            });

            foreach (var (name, column) in columnResults)
                result[name] = column;

            return result;
        }
        else
            return operation.Execute(input);
    }

    static IColumn? GetReferenceColumn(IReadOnlyList<IReadOnlyDictionary<string, IColumn>> allSources, string columnName)
    {
        foreach (var source in allSources)
            if (source.TryGetValue(columnName, out var column))
                return column;

        return null;
    }

    static string GetColumnName(Nivara.Expressions.ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        if (expression is Nivara.Expressions.ColumnReference columnRef)
            return columnRef.ColumnName;

        return expression.Name;
    }

    // ── Async dispatch ──

    async Task<IReadOnlyDictionary<string, IColumn>> executeOperationParallelAsync(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        NivaraExecutionContext context)
    {
        if (!isParallelizable(operation.OperationType) || !shouldUseParallelism(input, context))
            return await Task.Run(() => operation.Execute(input), context.CancellationToken);

        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);
        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(totalRows, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(totalRows, chunkSize);

        if (ranges.Count <= 1)
            return await Task.Run(() => operation.Execute(input), context.CancellationToken);

        var opType = operation.OperationType;
        if (opType == "Filter")
            return await executeFilterParallelAsync(operation, input, ranges, maxDop, context.CancellationToken);
        if (opType == "Sort")
            return await Task.Run(() => executeSortParallelSync((SortOperation)operation, input, ranges, maxDop, context.CancellationToken), context.CancellationToken);
        if (opType == "GroupBy")
            return await executeGroupByParallelAsync((GroupByOperation)operation, input, ranges, maxDop, context.CancellationToken);
        if (opType == "Join")
            return await executeJoinParallelAsync((JoinOperation)operation, maxDop, context.CancellationToken);
        if (opType.StartsWith("Concatenate", StringComparison.Ordinal))
            return await executeConcatenationParallelAsync((ConcatenationOperation)operation, input, maxDop, context.CancellationToken);

        return await Task.Run(() => operation.Execute(input), context.CancellationToken);
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeFilterParallelAsync(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var kernel = createSliceExecuteKernel(operation);
        var results = new IReadOnlyDictionary<string, IColumn>[ranges.Count];
        var indexedRanges = ranges.Select((r, i) => (r.Start, r.Length, Index: i)).ToList();
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        await Parallel.ForEachAsync(indexedRanges, parallelOptions, (item, ct) =>
        {
            results[item.Index] = kernel(item.Start, item.Length, input);
            return ValueTask.CompletedTask;
        });

        return ParallelExecutionHelper.ConcatenateColumnDictionaries(results);
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeGroupByParallelAsync(
        GroupByOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var keyColumnNames = operation.GroupByColumns
            .Select(expr => GetColumnName(expr, input))
            .ToArray();

        var partialMaps = new ConcurrentBag<Dictionary<GroupKey, List<int>>>();
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        await Parallel.ForEachAsync(ranges, parallelOptions, (range, ct) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(input, range.Start, range.Length);
            var groupedData = GroupByOperation.CreateGroupsInternal(subset, keyColumnNames);
            partialMaps.Add(groupedData.Groups);
            return ValueTask.CompletedTask;
        });

        var mergedMap = ParallelExecutionHelper.MergeGroupByDictionaries(partialMaps.ToArray());
        var mergedGroupedData = new GroupedData(mergedMap, keyColumnNames, input);

        var resultColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var keyColumnName in keyColumnNames)
        {
            var sourceColumn = input[keyColumnName];
            resultColumns[keyColumnName] = GroupByOperation.ExtractDistinctKeyValues(mergedGroupedData, keyColumnName, sourceColumn);
        }

        return resultColumns;
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeJoinParallelAsync(
        JoinOperation operation, int maxDop, CancellationToken cancellationToken)
    {
        var rightColumns = operation.RightColumns;
        var rightRowCount = rightColumns.Values.FirstOrDefault()?.Length ?? 0;
        var rightKeyColumns = operation.JoinKeys.Select(jk => jk.RightColumn).ToArray();

        if (rightRowCount <= 1)
            return await Task.Run(() => operation.Execute(new Dictionary<string, IColumn>()), cancellationToken);

        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(rightRowCount, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(rightRowCount, chunkSize);

        if (ranges.Count <= 1)
            return await Task.Run(() => operation.Execute(new Dictionary<string, IColumn>()), cancellationToken);

        var partialMaps = new ConcurrentBag<Dictionary<CompositeKey, List<int>>>();
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        var indexedRanges = ranges.Select((r, i) => (r.Start, r.Length, Index: i)).ToList();
        await Parallel.ForEachAsync(indexedRanges, parallelOptions, (item, ct) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(rightColumns, item.Start, item.Length);
            var subRowCount = subset.Values.FirstOrDefault()?.Length ?? 0;
            var partialMap = JoinOperation.BuildHashMap(subset, rightKeyColumns, subRowCount, item.Start);
            partialMaps.Add(partialMap);
            return ValueTask.CompletedTask;
        });

        var mergedMap = ParallelExecutionHelper.MergeJoinHashMaps(partialMaps.ToArray());

        var joinIndices = operation.ComputeJoinIndicesWithHashMap(mergedMap);
        return operation.MaterializeResult(joinIndices);
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeConcatenationParallelAsync(
        ConcatenationOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDop, CancellationToken cancellationToken)
    {
        var allSources = new List<IReadOnlyDictionary<string, IColumn>> { input };
        allSources.AddRange(operation.Sources);

        if (operation.Direction == ConcatenationDirection.Vertical)
        {
            var allColumnNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var source in allSources)
                foreach (var key in source.Keys)
                    allColumnNames.Add(key);

            var columnList = allColumnNames.ToList();
            var columnResults = new ConcurrentBag<(string Name, IColumn Column)>();

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDop,
                CancellationToken = cancellationToken
            };

            await Parallel.ForEachAsync(columnList, parallelOptions, (columnName, ct) =>
            {
                var columnsToConcat = new List<IColumn>();
                foreach (var source in allSources)
                {
                    if (source.TryGetValue(columnName, out var col))
                        columnsToConcat.Add(col);
                    else if (operation.MismatchHandling == ConcatenationMismatchHandling.FillWithNulls)
                    {
                        var sourceLength = source.Values.FirstOrDefault()?.Length ?? 0;
                        if (sourceLength > 0)
                        {
                            var referenceColumn = GetReferenceColumn(allSources, columnName);
                            if (referenceColumn != null)
                            {
                                var nullCol = operation.CreateNullColumn(referenceColumn.ElementType, sourceLength);
                                columnsToConcat.Add(nullCol);
                            }
                        }
                    }
                }

                if (columnsToConcat.Count > 0)
                {
                    var concatenated = ConcatenationOperation.ConcatenateColumns(columnsToConcat);
                    columnResults.Add((columnName, concatenated));
                }
                return ValueTask.CompletedTask;
            });

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var (name, column) in columnResults)
                result[name] = column;
            return result;
        }

        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    // ── Chunked source reading ──

    static async Task<IReadOnlyDictionary<string, IColumn>> readSourceAsync(
        IQuerySource source, NivaraExecutionContext context)
    {
        if (!source.CanReadInChunks)
            return await source.ExecuteAsync(context.CancellationToken);

        var estimatedCount = source.EstimatedRowCount;
        if (estimatedCount == null || estimatedCount.Value <= 0)
            return await source.ExecuteAsync(context.CancellationToken);

        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);
        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(estimatedCount.Value, maxDop);
        var totalChunks = (estimatedCount.Value + chunkSize - 1) / chunkSize;

        if (totalChunks <= 1)
            return await source.ExecuteAsync(context.CancellationToken);

        var chunkResults = new ConcurrentBag<(int Index, IReadOnlyDictionary<string, IColumn> Data)>();

        await Parallel.ForEachAsync(
            Enumerable.Range(0, totalChunks),
            new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDop,
                CancellationToken = context.CancellationToken
            },
            async (chunkIndex, ct) =>
            {
                var chunkData = await source.ReadChunkAsync(chunkIndex, chunkSize, ct);
                chunkResults.Add((chunkIndex, chunkData));
            });

        var ordered = chunkResults.OrderBy(c => c.Index).Select(c => c.Data).ToList();
        return ParallelExecutionHelper.ConcatenateColumnDictionaries(ordered);
    }

    // ── Overrides ──

    protected override string StrategyName => "Parallel";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
        validateParallelismConfiguration(context);

        ReportProgress(context, "Starting sync parallel execution", 0, plan.Operations.Count + 1);

        var currentColumns = plan.Source.Execute();
        ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                currentColumns = executeOperationParallelSync(operation, currentColumns, context);
                ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Sync parallel execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
            throw new QueryExecutionException("Sync parallel execution resulted in no columns");

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
        validateParallelismConfiguration(context);

        ReportProgress(context, "Starting async parallel execution", 0, plan.Operations.Count + 1);

        var currentColumns = await readSourceAsync(plan.Source, context);
        ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                currentColumns = await executeOperationParallelAsync(operation, currentColumns, context);
                ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Async parallel execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
            throw new QueryExecutionException("Async parallel execution resulted in no columns");

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    public override bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            if (!executor.ValidatePlan(plan))
                return false;

            validateParallelismConfiguration(context);
            return true;
        }
        catch
        {
            return false;
        }
    }

    public override long EstimateExecutionCost(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return long.MaxValue;

        try
        {
            long cost = 180;
            cost += plan.Source.IsLazy ? 120 : 100;

            foreach (var operation in plan.Operations)
            {
                var baseCost = operation.OperationType switch
                {
                    "Filter" => 300,
                    "Select" => 150,
                    "Sort" => 800,
                    "GroupBy" => 1000,
                    "Join" => 1500,
                    _ when operation.OperationType.StartsWith("Concatenate", StringComparison.Ordinal) => 250,
                    _ => 500
                };

                var parallelismFactor = Math.Min(context.MaxDegreeOfParallelism, Environment.ProcessorCount);
                var parallelDiscount = isParallelizable(operation.OperationType)
                    ? baseCost * (1.0 - (0.7 / parallelismFactor))
                    : baseCost * 0.95;

                cost += (long)parallelDiscount;
            }

            cost += plan.Operations.Count * 50;
            return Math.Max(cost, 180);
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
