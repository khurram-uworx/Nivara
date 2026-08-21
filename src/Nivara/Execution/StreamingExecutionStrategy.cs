using Nivara.Diagnostics;
using Nivara.Expressions;
using Nivara.Query;
using System.Threading.Channels;

namespace Nivara.Execution;

sealed class StreamingExecutionStrategy : ExecutionStrategyBase
{
    static readonly HashSet<string> NonStreamableOperations = new() { Query.OperationType.Sort, Query.OperationType.SortByExpression, Query.OperationType.GroupBy, Query.OperationType.Join, Query.OperationType.Distinct, Query.OperationType.Rolling, Query.OperationType.Cumulative, Query.OperationType.Shift, Query.OperationType.Rank };

    static bool isSuitableForStreaming(QueryPlan plan)
    {
        foreach (var operation in plan.Operations)
            if (NonStreamableOperations.Contains(operation.OperationType))
                return false;
        return true;
    }

    /// <summary>
    /// Derives a default chunk size from the memory budget. This is the fallback used when
    /// the caller did not set an explicit <see cref="NivaraExecutionContext.ChunkSize"/>.
    /// </summary>
    static int calculateChunkSize(long memoryBudget)
    {
        const long estimatedBytesPerRow = 100;
        var chunkMemory = memoryBudget / 10;
        var calculatedChunkSize = (int)(chunkMemory / estimatedBytesPerRow);
        return Math.Max(1000, Math.Min(calculatedChunkSize, 100000));
    }

    /// <summary>
    /// Resolves the chunk size for a context: an explicit <see cref="NivaraExecutionContext.ChunkSize"/>
    /// wins; otherwise the value is derived from <see cref="NivaraExecutionContext.MemoryBudget"/>.
    /// </summary>
    static int resolveChunkSize(NivaraExecutionContext context)
        => context.ChunkSize ?? calculateChunkSize(context.MemoryBudget);

    static IReadOnlyDictionary<string, IColumn> executeOperationsOnData(
        IReadOnlyDictionary<string, IColumn> data,
        IReadOnlyList<IQueryOperation> operations)
    {
        var current = data;
        foreach (var op in operations)
            current = op.Execute(current);
        return current;
    }

    static void recordMaterialization(NivaraExecutionContext context, IQueryOperation boundaryOp, long rowCount)
    {
        var description = boundaryOp.ToString() ?? boundaryOp.GetType().Name;
        context.ExecutionDiagnostics?.AddBoundaryMaterialization(description, rowCount);
        context.Progress?.Report(new ExecutionProgress(
            $"Materializing boundary '{description}' over {rowCount:N0} rows", 0, 1));
    }

    static async ValueTask<IReadOnlyDictionary<string, IColumn>> executeOperationsOnDataAsync(
        IReadOnlyDictionary<string, IColumn> data,
        IReadOnlyList<IQueryOperation> operations,
        CancellationToken ct)
    {
        var current = data;
        foreach (var op in operations)
        {
            ct.ThrowIfCancellationRequested();
            current = await op.ExecuteAsync(current, ct).ConfigureAwait(false);
        }
        return current;
    }

    record OperationSegment(IReadOnlyList<IQueryOperation> StreamableOps, IQueryOperation? BoundaryOp);

    static List<OperationSegment> PartitionAtNonStreamableOps(IReadOnlyList<IQueryOperation> operations)
    {
        var segments = new List<OperationSegment>();
        var current = new List<IQueryOperation>();
        foreach (var op in operations)
        {
            if (NonStreamableOperations.Contains(op.OperationType)
                || WindowExpressionInspector.HasWindowExpression(op))
            {
                segments.Add(new(current, op));
                current = new();
            }
            else
            {
                current.Add(op);
            }
        }
        segments.Add(new(current, null));
        return segments;
    }

    internal static int CalculateChannelCapacity(long memoryBudget, int chunkSize)
    {
        const long estimatedBytesPerRow = 100;
        var bytesPerChunk = (long)chunkSize * estimatedBytesPerRow;
        if (bytesPerChunk <= 0) return 2;
        var capacity = (int)(memoryBudget / bytesPerChunk);
        return Math.Max(2, Math.Min(capacity, 16));
    }

    internal static Channel<NivaraFrame> CreateBoundChannel(long memoryBudget, int chunkSize)
    {
        var capacity = CalculateChannelCapacity(memoryBudget, chunkSize);
        return Channel.CreateBounded<NivaraFrame>(capacity);
    }

    protected override string StrategyName => "Streaming";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamingExecution") : null;
        context.Progress?.Report(new ExecutionProgress("Starting streaming execution", 0, 1));

        if (!plan.Source.CanReadInChunks)
        {
            var fullResult = executor.Execute(plan, diag);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return fullResult;
        }

        var chunkSize = resolveChunkSize(context);
        var estimatedRows = plan.Source.EstimatedRowCount;
        var totalChunks = estimatedRows.HasValue
            ? (int)((estimatedRows.Value + chunkSize - 1) / chunkSize)
            : -1;

        var segments = PartitionAtNonStreamableOps(plan.Operations);

        var windowProcessor = StreamingWindowProcessor.TryCreate(segments.Count > 0 ? segments[0].BoundaryOp : null);
        int overlapBoundarySegIdx = windowProcessor != null ? 0 : -1;

        using var budgetTracker = new StreamingBudgetTracker(context.MemoryBudget);
        var chunkFrames = new List<NivaraFrame>();
        int chunkIndex = 0;

        while (true)
        {
            context.CancellationToken.ThrowIfCancellationRequested();

            using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

            var chunkData = plan.Source.ReadChunk(chunkIndex, chunkSize);
            if (chunkData == null || chunkData.Count == 0 || chunkData.Values.All(c => c.Length == 0))
                break;
            diag?.AddRowsRead(QueryExecutor.GetRowCount(chunkData));

            var processedData = executeOperationsOnData(chunkData, segments[0].StreamableOps);

            var chunkFrame = windowProcessor != null
                ? NivaraFrame.Create(windowProcessor.ProcessChunk(processedData))
                : NivaraFrame.Create(processedData);

            if (chunkScope != null)
                chunkScope.SetRowCount(chunkFrame.RowCount);
            budgetTracker.RecordFrame(chunkFrame);
            chunkFrames.Add(chunkFrame);

            chunkIndex++;
            var completedWork = chunkIndex;
            var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
            context.Progress?.Report(new ExecutionProgress($"Processing chunk {chunkIndex}", completedWork, totalWork));
        }

        budgetTracker.RecordWarningIfExceeded(diag);

        NivaraFrame result;
        if (chunkFrames.Count == 0)
        {
            context.Progress?.Report(new ExecutionProgress("No data from chunks, falling back to full execution", 0, 1));
            var fallbackResult = executor.Execute(plan, diag);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return fallbackResult;
        }

        result = chunkFrames.Count == 1
            ? chunkFrames[0]
            : NivaraFrameExtensions.ConcatenateVertical(chunkFrames);

        if (chunkFrames.Count > 1)
        {
            foreach (var f in chunkFrames) f.Dispose();
        }

        for (int segIdx = 0; segIdx < segments.Count; segIdx++)
        {
            if (segIdx == overlapBoundarySegIdx) continue;

            var segment = segments[segIdx];

            if (segment.BoundaryOp != null)
            {
                recordMaterialization(context, segment.BoundaryOp, result.RowCount);
                var columns = result.ColumnNames.ToDictionary(
                    name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = segment.BoundaryOp.Execute(columns);
                var newResult = new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
                result = newResult;
            }

            if (segIdx == 0) continue;

            if (segment.StreamableOps.Count > 0)
            {
                var columns = result.ColumnNames.ToDictionary(
                    name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = executeOperationsOnData(columns, segment.StreamableOps);
                var newResult = new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
                result.Dispose();
                result = newResult;
            }
        }

        context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
        return result;
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
        => await executeCoreInternalAsync(plan, context).ConfigureAwait(false);

    async Task<NivaraFrame> executeCoreInternalAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamingExecutionAsync") : null;
        context.Progress?.Report(new ExecutionProgress("Starting streaming execution", 0, 1));

        if (!plan.Source.CanReadInChunks)
        {
            var columns = await plan.Source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);
            diag?.AddRowsRead(QueryExecutor.GetRowCount(columns));
            var processed = await executeOperationsOnDataAsync(columns, plan.Operations, context.CancellationToken).ConfigureAwait(false);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
        }

        var chunkSize = resolveChunkSize(context);
        var estimatedRows = plan.Source.EstimatedRowCount;
        var totalChunks = estimatedRows.HasValue
            ? (int)((estimatedRows.Value + chunkSize - 1) / chunkSize)
            : -1;

        var segments = PartitionAtNonStreamableOps(plan.Operations);

        var windowProcessor = StreamingWindowProcessor.TryCreate(segments.Count > 0 ? segments[0].BoundaryOp : null);
        int overlapBoundarySegIdx = windowProcessor != null ? 0 : -1;

        var channel = CreateBoundChannel(context.MemoryBudget, chunkSize);
        var chunkIndex = 0;

        var producer = Task.Run(async () =>
        {
            NivaraFrame? inFlight = null;
            try
            {
                await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, context.CancellationToken)
                    .ConfigureAwait(false))
                {
                    context.CancellationToken.ThrowIfCancellationRequested();
                    diag?.AddRowsRead(QueryExecutor.GetRowCount(chunkData));

                    using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

                    var processedData = await executeOperationsOnDataAsync(
                        chunkData, segments[0].StreamableOps, context.CancellationToken).ConfigureAwait(false);

                    var chunkFrame = windowProcessor != null
                        ? NivaraFrame.Create(windowProcessor.ProcessChunk(processedData))
                        : NivaraFrame.Create(processedData);

                    if (chunkScope != null)
                        chunkScope.SetRowCount(chunkFrame.RowCount);

                    inFlight = chunkFrame;
                    await channel.Writer.WriteAsync(chunkFrame, context.CancellationToken).ConfigureAwait(false);
                    inFlight = null;

                    chunkIndex++;
                    var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
                    context.Progress?.Report(new ExecutionProgress($"Processing chunk {chunkIndex}", chunkIndex, totalWork));
                }
            }
            finally
            {
                inFlight?.Dispose();
                channel.Writer.TryComplete();
            }
        }, context.CancellationToken);

        using var budgetTracker = new StreamingBudgetTracker(context.MemoryBudget);
        var chunkFrames = new List<NivaraFrame>();
        try
        {
            await foreach (var chunkFrame in channel.Reader.ReadAllAsync(context.CancellationToken)
                .ConfigureAwait(false))
            {
                budgetTracker.RecordFrame(chunkFrame);
                chunkFrames.Add(chunkFrame);
            }
            await producer.ConfigureAwait(false);
        }
        catch
        {
            foreach (var f in chunkFrames) f.Dispose();
            channel.Writer.TryComplete();
            while (channel.Reader.TryRead(out var buffered))
                buffered.Dispose();
            try { await producer.ConfigureAwait(false); } catch { }
            throw;
        }

        budgetTracker.RecordWarningIfExceeded(diag);

        NivaraFrame result;
        if (chunkFrames.Count == 0)
        {
            context.Progress?.Report(new ExecutionProgress("No data from chunks, falling back to full execution", 0, 1));
            var fallbackResult = executor.Execute(plan, diag);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return fallbackResult;
        }

        result = chunkFrames.Count == 1
            ? chunkFrames[0]
            : NivaraFrameExtensions.ConcatenateVertical(chunkFrames);

        if (chunkFrames.Count > 1)
        {
            foreach (var f in chunkFrames) f.Dispose();
        }

        for (int segIdx = 0; segIdx < segments.Count; segIdx++)
        {
            if (segIdx == overlapBoundarySegIdx) continue;

            var segment = segments[segIdx];

            if (segment.BoundaryOp != null)
            {
                recordMaterialization(context, segment.BoundaryOp, result.RowCount);
                var columns = result.ColumnNames.ToDictionary(
                    name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = segment.BoundaryOp.Execute(columns);
                var newResult = new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
                result = newResult;
            }

            if (segIdx == 0) continue;

            if (segment.StreamableOps.Count > 0)
            {
                var columns = result.ColumnNames.ToDictionary(
                    name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = await executeOperationsOnDataAsync(
                    columns, segment.StreamableOps, context.CancellationToken).ConfigureAwait(false);
                var newResult = new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
                result.Dispose();
                result = newResult;
            }
        }

        context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
        return result;
    }

    /// <summary>
    /// Streams processed chunks from the source as an async enumerable.
    /// Each yielded frame is a source chunk with the streamable operations applied.
    /// </summary>
    /// <remarks>
    /// A fully streamable plan (only Filter, Select, Slice, SelectRows; no window expressions)
    /// over a chunk-capable source yields one frame per source chunk. A mixed plan partitions
    /// at non-streamable boundary operations (Sort, GroupBy, Join, Distinct, Rolling, etc.):
    /// leading streamable ops run per chunk (yielded immediately), boundary ops run once over
    /// the concatenated chunks (yielded as one frame), and trailing streamable ops resume per
    /// chunk. A source that cannot read in chunks falls back to a single frame.
    /// The caller owns every yielded frame (the pipeline never disposes them).
    /// </remarks>
    public async IAsyncEnumerable<NivaraFrame> StreamChunksAsync(
        QueryPlan plan,
        NivaraExecutionContext context,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamChunksAsync") : null;

        if (!plan.Source.CanReadInChunks)
        {
            var columns = await plan.Source.ExecuteAsync(ct).ConfigureAwait(false);
            diag?.AddRowsRead(QueryExecutor.GetRowCount(columns));
            var processed = await executeOperationsOnDataAsync(columns, plan.Operations, ct).ConfigureAwait(false);
            var frame = NivaraFrame.Create(processed);
            if (diag != null)
            {
                diag.RowsReturned = frame.RowCount;
                diag.MaterializedColumns = frame.ColumnCount;
            }
            yield return frame;
            yield break;
        }

        var chunkSize = resolveChunkSize(context);
        var segments = PartitionAtNonStreamableOps(plan.Operations);
        var firstSegment = segments[0];
        var hasAnyBoundary = segments.Any(s => s.BoundaryOp != null);

        var windowProcessor = StreamingWindowProcessor.TryCreate(hasAnyBoundary ? firstSegment.BoundaryOp : null);

        using var budgetTracker = new StreamingBudgetTracker(context.MemoryBudget);

        if (!hasAnyBoundary && firstSegment.StreamableOps.Count == plan.Operations.Count)
        {
            await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, ct)
                .ConfigureAwait(false))
            {
                ct.ThrowIfCancellationRequested();
                diag?.AddRowsRead(QueryExecutor.GetRowCount(chunkData));

                var processedData = await executeOperationsOnDataAsync(
                    chunkData, firstSegment.StreamableOps, ct).ConfigureAwait(false);

                var chunkFrame = NivaraFrame.Create(processedData);
                if (diag != null)
                {
                    diag.RowsReturned += chunkFrame.RowCount;
                    diag.MaterializedColumns = chunkFrame.ColumnCount;
                }
                yield return chunkFrame;
            }
            yield break;
        }

        if (windowProcessor != null)
        {
            var chunkFrames = new List<NivaraFrame>();
            var hasTrailingBoundaries = segments.Count > 1 && segments.Skip(1).Any(s => s.BoundaryOp != null);

            await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, ct)
                .ConfigureAwait(false))
            {
                ct.ThrowIfCancellationRequested();
                diag?.AddRowsRead(QueryExecutor.GetRowCount(chunkData));

                var processedData = await executeOperationsOnDataAsync(
                    chunkData, firstSegment.StreamableOps, ct).ConfigureAwait(false);

                var finalData = windowProcessor.ProcessChunk(processedData);

                var chunkFrame = NivaraFrame.Create(finalData);
                budgetTracker.RecordFrame(chunkFrame);

                if (!hasTrailingBoundaries)
                {
                    if (diag != null)
                    {
                        diag.RowsReturned += chunkFrame.RowCount;
                        diag.MaterializedColumns = chunkFrame.ColumnCount;
                    }
                    yield return chunkFrame;
                }
                else
                {
                    chunkFrames.Add(chunkFrame);
                }
            }

            if (!hasTrailingBoundaries)
            {
                budgetTracker.RecordWarningIfExceeded(diag);
                yield break;
            }

            budgetTracker.RecordWarningIfExceeded(diag);

            if (chunkFrames.Count == 0)
            {
                var columns = await plan.Source.ExecuteAsync(ct).ConfigureAwait(false);
                var processed = await executeOperationsOnDataAsync(columns, plan.Operations, ct).ConfigureAwait(false);
                var frame = NivaraFrame.Create(processed);
                if (diag != null)
                {
                    diag.RowsReturned = frame.RowCount;
                    diag.MaterializedColumns = frame.ColumnCount;
                }
                yield return frame;
                yield break;
            }

            var result = chunkFrames.Count == 1
                ? chunkFrames[0]
                : NivaraFrameExtensions.ConcatenateVertical(chunkFrames);

            if (chunkFrames.Count > 1)
            {
                foreach (var f in chunkFrames) f.Dispose();
            }

            for (int segIdx = 0; segIdx < segments.Count; segIdx++)
            {
                if (segIdx == 0) continue;

                var segment = segments[segIdx];

                if (segment.BoundaryOp != null)
                {
                    recordMaterialization(context, segment.BoundaryOp, result.RowCount);
                    var columns = result.ColumnNames.ToDictionary(
                        name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                    var processed = segment.BoundaryOp.Execute(columns);
                    var newResult = NivaraFrame.Create(processed);
                    result = newResult;
                }

                if (segIdx > 0 && segment.StreamableOps.Count > 0)
                {
                    var columns = result.ColumnNames.ToDictionary(
                        name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                    var processed = await executeOperationsOnDataAsync(
                        columns, segment.StreamableOps, ct).ConfigureAwait(false);
                    var newResult = NivaraFrame.Create(processed);
                    result.Dispose();
                    result = newResult;
                }
            }

            if (diag != null)
            {
                diag.RowsReturned += result.RowCount;
                diag.MaterializedColumns = result.ColumnCount;
            }
            yield return result;
            yield break;
        }

        var chunkFramesLegacy = new List<NivaraFrame>();

        await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, ct)
            .ConfigureAwait(false))
        {
            ct.ThrowIfCancellationRequested();
            diag?.AddRowsRead(QueryExecutor.GetRowCount(chunkData));

            var processedData = await executeOperationsOnDataAsync(
                chunkData, firstSegment.StreamableOps, ct).ConfigureAwait(false);

            var chunkFrame = NivaraFrame.Create(processedData);
            budgetTracker.RecordFrame(chunkFrame);
            chunkFramesLegacy.Add(chunkFrame);
        }

        budgetTracker.RecordWarningIfExceeded(diag);

        if (chunkFramesLegacy.Count == 0)
        {
            var columns = await plan.Source.ExecuteAsync(ct).ConfigureAwait(false);
            var processed = await executeOperationsOnDataAsync(columns, plan.Operations, ct).ConfigureAwait(false);
            var frame = NivaraFrame.Create(processed);
            if (diag != null)
            {
                diag.RowsReturned = frame.RowCount;
                diag.MaterializedColumns = frame.ColumnCount;
            }
            yield return frame;
            yield break;
        }

        foreach (var chunk in chunkFramesLegacy)
        {
            if (firstSegment.StreamableOps.Count > 0)
            {
                if (diag != null)
                {
                    diag.RowsReturned += chunk.RowCount;
                    diag.MaterializedColumns = chunk.ColumnCount;
                }
                yield return chunk;
            }
        }

        var legacyResult = chunkFramesLegacy.Count == 1
            ? chunkFramesLegacy[0]
            : NivaraFrameExtensions.ConcatenateVertical(chunkFramesLegacy);

        for (int segIdx = 0; segIdx < segments.Count; segIdx++)
        {
            var segment = segments[segIdx];

            if (segment.BoundaryOp != null)
            {
                recordMaterialization(context, segment.BoundaryOp, legacyResult.RowCount);
                var columns = legacyResult.ColumnNames.ToDictionary(
                    name => name, name => legacyResult.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = segment.BoundaryOp.Execute(columns);
                var newResult = NivaraFrame.Create(processed);
                legacyResult = newResult;
            }

            if (segIdx > 0 && segment.StreamableOps.Count > 0)
            {
                var columns = legacyResult.ColumnNames.ToDictionary(
                    name => name, name => legacyResult.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = await executeOperationsOnDataAsync(
                    columns, segment.StreamableOps, ct).ConfigureAwait(false);
                var newResult = NivaraFrame.Create(processed);
                legacyResult.Dispose();
                legacyResult = newResult;
            }
        }

        if (diag != null)
        {
            diag.RowsReturned += legacyResult.RowCount;
            diag.MaterializedColumns = legacyResult.ColumnCount;
        }
        yield return legacyResult;
    }

    public override bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            if (!executor.ValidatePlan(plan))
                return false;

            if (context.MemoryBudget <= 0)
                return false;

            if (context.ChunkSize is <= 0)
                return false;

            return isSuitableForStreaming(plan);
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
            long cost = 150;
            cost += plan.Source.IsLazy ? 100 : 120;

            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    Query.OperationType.Filter => 250,
                    Query.OperationType.Select => 120,
                    Query.OperationType.Sort => 2000,
                    Query.OperationType.GroupBy => 2500,
                    Query.OperationType.Join => 3000,
                    _ when operation.OperationType.StartsWith(Query.OperationType.ConcatenationPrefix, StringComparison.Ordinal) => 200,
                    _ => 400
                };
            }

            if (isSuitableForStreaming(plan))
            {
                var streamingDiscount = Math.Min(cost * 0.15, 800);
                cost -= (long)streamingDiscount;
            }
            else
                cost += 1000;

            return Math.Max(cost, 150);
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
