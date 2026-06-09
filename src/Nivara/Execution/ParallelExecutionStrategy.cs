using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Operations;
using Nivara.Query;

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

        if (operation is IParallelSortOperation sortOp)
            return executeSortParallelSync(sortOp, input, ranges, maxDop, context.CancellationToken);
        if (operation is IParallelGroupByOperation groupByOp)
            return executeGroupByParallelSync(groupByOp, input, ranges, maxDop, context.CancellationToken);
        if (operation is IParallelJoinOperation joinOp)
            return executeJoinParallelSync(joinOp, maxDop, context.CancellationToken);
        if (operation is IParallelConcatenationOperation concatOp)
            return executeConcatenationParallelSync(concatOp, input, maxDop, context.CancellationToken);
        if (isParallelizable(operation.OperationType))
            return executeFilterParallelSync(operation, input, ranges, maxDop, context.CancellationToken);

        return operation.Execute(input);
    }

    IReadOnlyDictionary<string, IColumn> executeSortParallelSync(
        IParallelSortOperation operation,
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

        var globalIndices = ParallelExecutionHelper.MergeSortedChunks(
            input, sortedLocalIndices, chunkStarts, ranges, sortKeys);

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
        IParallelGroupByOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var keyColumnNames = operation.GroupByColumns
            .Select(expr => GetColumnName(expr, input))
            .ToArray();

        var partialMaps = new Dictionary<GroupKey, List<int>>[ranges.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(input, range.Start, range.Length);
            var groupedData = GroupByOperation.CreateGroupsInternal(subset, keyColumnNames, range.Start);
            partialMaps[(int)index] = groupedData.Groups;
        });

        var mergedMap = ParallelExecutionHelper.MergeGroupByDictionaries(partialMaps);
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
        IParallelJoinOperation operation, int maxDop, CancellationToken cancellationToken)
    {
        var rightColumns = operation.RightColumns;
        var rightRowCount = rightColumns.Values.FirstOrDefault()?.Length ?? 0;
        var rightKeyColumns = operation.JoinKeys.Select(jk => jk.RightColumn).ToArray();

        if (rightRowCount <= 1)
            return ((IQueryOperation)operation).Execute(new Dictionary<string, IColumn>());

        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(rightRowCount, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(rightRowCount, chunkSize);

        if (ranges.Count <= 1)
            return ((IQueryOperation)operation).Execute(new Dictionary<string, IColumn>());

        var results = new Dictionary<CompositeKey, List<int>>[ranges.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

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
        IParallelConcatenationOperation operation, IReadOnlyDictionary<string, IColumn> input,
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

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDop,
                CancellationToken = cancellationToken
            };

            var columnResults = new (string Name, IColumn Column)[columnList.Count];

            Parallel.ForEach(columnList, parallelOptions, (columnName, _, index) =>
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
                    columnResults[(int)index] = (columnName, concatenated);
                }
            });

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var (name, column) in columnResults)
                if (column != null)
                    result[name] = column;

            return result;
        }
        else
            return ((IQueryOperation)operation).Execute(input);
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
            return await Task.Run(() => operation.Execute(input), context.CancellationToken).ConfigureAwait(false);

        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);
        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(totalRows, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(totalRows, chunkSize);

        if (ranges.Count <= 1)
            return await Task.Run(() => operation.Execute(input), context.CancellationToken).ConfigureAwait(false);

        if (operation is IParallelSortOperation sortOp)
            return await Task.Run(() => executeSortParallelSync(sortOp, input, ranges, maxDop, context.CancellationToken), context.CancellationToken).ConfigureAwait(false);
        if (operation is IParallelGroupByOperation groupByOp)
            return executeGroupByParallelAsync(groupByOp, input, ranges, maxDop, context.CancellationToken);
        if (operation is IParallelJoinOperation joinOp)
            return executeJoinParallelAsync(joinOp, maxDop, context.CancellationToken);
        if (operation is IParallelConcatenationOperation concatOp)
            return executeConcatenationParallelAsync(concatOp, input, maxDop, context.CancellationToken);
        if (isParallelizable(operation.OperationType))
            return executeFilterParallelAsync(operation, input, ranges, maxDop, context.CancellationToken);

        return await Task.Run(() => operation.Execute(input), context.CancellationToken).ConfigureAwait(false);
    }

    IReadOnlyDictionary<string, IColumn> executeFilterParallelAsync(
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

    IReadOnlyDictionary<string, IColumn> executeGroupByParallelAsync(
        IParallelGroupByOperation operation, IReadOnlyDictionary<string, IColumn> input,
        List<(int Start, int Length)> ranges, int maxDop, CancellationToken cancellationToken)
    {
        var keyColumnNames = operation.GroupByColumns
            .Select(expr => GetColumnName(expr, input))
            .ToArray();

        var results = new Dictionary<GroupKey, List<int>>[ranges.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(input, range.Start, range.Length);
            var groupedData = GroupByOperation.CreateGroupsInternal(subset, keyColumnNames, range.Start);
            results[(int)index] = groupedData.Groups;
        });

        var mergedMap = ParallelExecutionHelper.MergeGroupByDictionaries(results);
        var mergedGroupedData = new GroupedData(mergedMap, keyColumnNames, input);

        var resultColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var keyColumnName in keyColumnNames)
        {
            var sourceColumn = input[keyColumnName];
            resultColumns[keyColumnName] = GroupByOperation.ExtractDistinctKeyValues(mergedGroupedData, keyColumnName, sourceColumn);
        }

        return resultColumns;
    }

    IReadOnlyDictionary<string, IColumn> executeJoinParallelAsync(
        IParallelJoinOperation operation, int maxDop, CancellationToken cancellationToken)
    {
        var rightColumns = operation.RightColumns;
        var rightRowCount = rightColumns.Values.FirstOrDefault()?.Length ?? 0;
        var rightKeyColumns = operation.JoinKeys.Select(jk => jk.RightColumn).ToArray();

        if (rightRowCount <= 1)
            return ((IQueryOperation)operation).Execute(new Dictionary<string, IColumn>());

        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(rightRowCount, maxDop);
        var ranges = ParallelExecutionHelper.CreateChunkRanges(rightRowCount, chunkSize);

        if (ranges.Count <= 1)
            return ((IQueryOperation)operation).Execute(new Dictionary<string, IColumn>());

        var results = new Dictionary<CompositeKey, List<int>>[ranges.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(ranges, parallelOptions, (range, _, index) =>
        {
            var subset = ParallelExecutionHelper.CreateRowSubset(rightColumns, range.Start, range.Length);
            var subRowCount = subset.Values.FirstOrDefault()?.Length ?? 0;
            results[(int)index] = JoinOperation.BuildHashMap(subset, rightKeyColumns, subRowCount, range.Start);
        });

        var mergedMap = ParallelExecutionHelper.MergeJoinHashMaps(results);

        var joinIndices = operation.ComputeJoinIndicesWithHashMap(mergedMap);
        return operation.MaterializeResult(joinIndices);
    }

    IReadOnlyDictionary<string, IColumn> executeConcatenationParallelAsync(
        IParallelConcatenationOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDop, CancellationToken cancellationToken)
    {
        var allSources = new List<IReadOnlyDictionary<string, IColumn>> { input };
        allSources.AddRange(operation.Sources);

        if (operation.Direction != ConcatenationDirection.Vertical)
            return ((IQueryOperation)operation).Execute(input);

        var allColumnNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var source in allSources)
            foreach (var key in source.Keys)
                allColumnNames.Add(key);

        var columnList = allColumnNames.ToList();
        var columnResults = new (string Name, IColumn Column)[columnList.Count];

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDop,
            CancellationToken = cancellationToken
        };

        Parallel.ForEach(columnList, parallelOptions, (columnName, _, index) =>
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
                columnResults[(int)index] = (columnName, concatenated);
            }
        });

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, column) in columnResults)
            if (column != null)
                result[name] = column;
        return result;
    }

    // ── Chunked source reading ──

    static async Task<IReadOnlyDictionary<string, IColumn>> readSourceAsync(
        IQuerySource source, NivaraExecutionContext context)
    {
        if (!source.CanReadInChunks)
            return await source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);

        var estimatedCount = source.EstimatedRowCount;
        if (estimatedCount == null || estimatedCount.Value <= 0)
            return await source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);

        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);
        var chunkSize = ParallelExecutionHelper.CalculateOptimalChunkSize(estimatedCount.Value, maxDop);
        var totalChunks = (estimatedCount.Value + chunkSize - 1) / chunkSize;

        if (totalChunks <= 1)
            return await source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);

        var chunkResults = new IReadOnlyDictionary<string, IColumn>[totalChunks];

        await Parallel.ForEachAsync(
            Enumerable.Range(0, totalChunks),
            new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDop,
                CancellationToken = context.CancellationToken
            },
            async (chunkIndex, ct) =>
            {
                var chunkData = await source.ReadChunkAsync(chunkIndex, chunkSize, ct).ConfigureAwait(false);
                chunkResults[chunkIndex] = chunkData;
            }).ConfigureAwait(false);

        return ParallelExecutionHelper.ConcatenateColumnDictionaries(chunkResults);
    }

    // ── Overrides ──

    protected override string StrategyName => "Parallel";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
        validateParallelismConfiguration(context);

        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "ParallelExecution") : null;
        context.Progress?.Report(new ExecutionProgress("Starting sync parallel execution", 0, plan.Operations.Count + 1));

        var currentColumns = diag != null
            ? DiagnosticHelper.ExecuteWithDiagnostics(diag, "SourceExecute", () => plan.Source.Execute())
            : plan.Source.Execute();
        context.Progress?.Report(new ExecutionProgress("Data source executed", 1, plan.Operations.Count + 1));

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                var capturedOp = operation;
                currentColumns = diag != null
                    ? DiagnosticHelper.ExecuteWithDiagnostics(diag, operation.OperationType, () => executeOperationParallelSync(capturedOp, currentColumns, context))
                    : executeOperationParallelSync(capturedOp, currentColumns, context);
                context.Progress?.Report(new ExecutionProgress($"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1));
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Sync parallel execution at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
        => await executeCoreInternalAsync(plan, context).ConfigureAwait(false);

    async Task<NivaraFrame> executeCoreInternalAsync(
        QueryPlan plan, NivaraExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
        validateParallelismConfiguration(context);

        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "ParallelExecutionAsync") : null;
        context.Progress?.Report(new ExecutionProgress("Starting async parallel execution", 0, plan.Operations.Count + 1));

        var currentColumns = diag != null
            ? await DiagnosticHelper.ExecuteWithDiagnosticsAsync(diag, "SourceExecute", () => readSourceAsync(plan.Source, context)).ConfigureAwait(false)
            : await readSourceAsync(plan.Source, context).ConfigureAwait(false);
        context.Progress?.Report(new ExecutionProgress("Data source executed", 1, plan.Operations.Count + 1));

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                var capturedOp = operation;
                currentColumns = diag != null
                    ? await DiagnosticHelper.ExecuteWithDiagnosticsAsync(diag, operation.OperationType, () => executeOperationParallelAsync(capturedOp, currentColumns, context)).ConfigureAwait(false)
                    : await executeOperationParallelAsync(capturedOp, currentColumns, context).ConfigureAwait(false);
                context.Progress?.Report(new ExecutionProgress($"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1));
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Async parallel execution at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

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
