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
            if (NonStreamableOperations.Contains(operation.OperationType)
                || WindowExpressionInspector.HasWindowExpression(operation))
                return false;
        return true;
    }

    static int calculateChunkSize(long memoryBudget)
    {
        const long estimatedBytesPerRow = 100;
        var chunkMemory = memoryBudget / 10;
        var calculatedChunkSize = (int)(chunkMemory / estimatedBytesPerRow);
        return Math.Max(1000, Math.Min(calculatedChunkSize, 100000));
    }

    static IReadOnlyDictionary<string, IColumn> executeOperationsOnData(
        IReadOnlyDictionary<string, IColumn> data,
        IReadOnlyList<IQueryOperation> operations)
    {
        var current = data;
        foreach (var op in operations)
            current = op.Execute(current);
        return current;
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
            if (NonStreamableOperations.Contains(op.OperationType))
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

    static int CalculateChannelCapacity(long memoryBudget, int chunkSize)
    {
        const long estimatedBytesPerRow = 100;
        var bytesPerChunk = (long)chunkSize * estimatedBytesPerRow;
        if (bytesPerChunk <= 0) return 2;
        var capacity = (int)(memoryBudget / bytesPerChunk);
        return Math.Max(2, Math.Min(capacity, 16));
    }

    protected override string StrategyName => "Streaming";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamingExecution") : null;
        context.Progress?.Report(new ExecutionProgress("Starting streaming execution", 0, 1));

        if (!isSuitableForStreaming(plan))
            return new LazyExecutionStrategy().Execute(plan, context);

        if (!plan.Source.CanReadInChunks)
        {
            var result = executor.Execute(plan);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return result;
        }

        var chunkSize = calculateChunkSize(context.MemoryBudget);
        var estimatedRows = plan.Source.EstimatedRowCount;
        var totalChunks = estimatedRows.HasValue
            ? (int)((estimatedRows.Value + chunkSize - 1) / chunkSize)
            : -1;

        var chunkFrames = new List<NivaraFrame>();
        int chunkIndex = 0;

        while (true)
        {
            context.CancellationToken.ThrowIfCancellationRequested();

            using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

            var chunkData = plan.Source.ReadChunk(chunkIndex, chunkSize);
            if (chunkData == null || chunkData.Count == 0 || chunkData.Values.All(c => c.Length == 0))
                break;

            var processedData = executeOperationsOnData(chunkData, plan.Operations);
            if (chunkScope != null)
                chunkScope.SetRowCount(processedData.Values.FirstOrDefault()?.Length ?? 0);
            var chunkFrame = NivaraFrame.Create(processedData);
            chunkFrames.Add(chunkFrame);

            chunkIndex++;
            var completedWork = chunkIndex;
            var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
            context.Progress?.Report(new ExecutionProgress($"Processing chunk {chunkIndex}", completedWork, totalWork));
        }

        if (chunkFrames.Count == 0)
        {
            context.Progress?.Report(new ExecutionProgress("No data from chunks, falling back to full execution", 0, 1));
            return executor.Execute(plan);
        }

        if (chunkFrames.Count == 1)
        {
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return chunkFrames[0];
        }

        var mergedResult = NivaraFrameExtensions.ConcatenateVertical(chunkFrames);
        context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
        return mergedResult;
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
        => await executeCoreInternalAsync(plan, context).ConfigureAwait(false);

    async Task<NivaraFrame> executeCoreInternalAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamingExecutionAsync") : null;
        context.Progress?.Report(new ExecutionProgress("Starting streaming execution", 0, 1));

        if (plan.Operations.Any(op => WindowExpressionInspector.HasWindowExpression(op)))
            return await new LazyExecutionStrategy().ExecuteAsync(plan, context).ConfigureAwait(false);

        if (!plan.Source.CanReadInChunks)
        {
            var columns = await plan.Source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);
            var processed = await executeOperationsOnDataAsync(columns, plan.Operations, context.CancellationToken).ConfigureAwait(false);
            context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
            return new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
        }

        var chunkSize = calculateChunkSize(context.MemoryBudget);
        var estimatedRows = plan.Source.EstimatedRowCount;
        var totalChunks = estimatedRows.HasValue
            ? (int)((estimatedRows.Value + chunkSize - 1) / chunkSize)
            : -1;

        var segments = PartitionAtNonStreamableOps(plan.Operations);
        var channelCapacity = CalculateChannelCapacity(context.MemoryBudget, chunkSize);
        var channel = Channel.CreateBounded<NivaraFrame>(channelCapacity);
        var chunkIndex = 0;

        var producer = Task.Run(async () =>
        {
            try
            {
                await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, context.CancellationToken)
                    .ConfigureAwait(false))
                {
                    context.CancellationToken.ThrowIfCancellationRequested();

                    using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

                    var processedData = await executeOperationsOnDataAsync(
                        chunkData, segments[0].StreamableOps, context.CancellationToken).ConfigureAwait(false);

                    if (chunkScope != null)
                        chunkScope.SetRowCount(processedData.Values.FirstOrDefault()?.Length ?? 0);

                    var chunkFrame = NivaraFrame.Create(processedData);
                    await channel.Writer.WriteAsync(chunkFrame, context.CancellationToken).ConfigureAwait(false);

                    chunkIndex++;
                    var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
                    context.Progress?.Report(new ExecutionProgress($"Processing chunk {chunkIndex}", chunkIndex, totalWork));
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, context.CancellationToken);

        var chunkFrames = new List<NivaraFrame>();
        try
        {
            await foreach (var chunkFrame in channel.Reader.ReadAllAsync(context.CancellationToken)
                .ConfigureAwait(false))
            {
                chunkFrames.Add(chunkFrame);
            }
            await producer.ConfigureAwait(false);
        }
        catch
        {
            foreach (var f in chunkFrames) f.Dispose();
            channel.Writer.Complete();
            throw;
        }

        NivaraFrame result;
        if (chunkFrames.Count == 0)
        {
            context.Progress?.Report(new ExecutionProgress("No data from chunks, falling back to full execution", 0, 1));
            result = executor.Execute(plan);
        }
        else
        {
            result = chunkFrames.Count == 1
                ? chunkFrames[0]
                : NivaraFrameExtensions.ConcatenateVertical(chunkFrames);

            if (chunkFrames.Count > 1)
            {
                foreach (var f in chunkFrames) f.Dispose();
            }
        }

        for (int segIdx = 0; segIdx < segments.Count; segIdx++)
        {
            var segment = segments[segIdx];

            if (segment.BoundaryOp != null)
            {
                var columns = result.ColumnNames.ToDictionary(
                    name => name, name => result.GetColumn(name), StringComparer.OrdinalIgnoreCase);
                var processed = segment.BoundaryOp.Execute(columns);
                var newResult = new NivaraFrame(processed.Select(kvp => (kvp.Key, kvp.Value)));
                result.Dispose();
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
    /// Plans that are not streamable (non-streamable operations) or whose source
    /// cannot read in chunks fall back to a single frame produced from the full source.
    /// </summary>
    public async IAsyncEnumerable<NivaraFrame> StreamChunksAsync(
        QueryPlan plan,
        NivaraExecutionContext context,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "StreamChunksAsync") : null;

        if (!isSuitableForStreaming(plan) || !plan.Source.CanReadInChunks)
        {
            var columns = await plan.Source.ExecuteAsync(ct).ConfigureAwait(false);
            var processed = await executeOperationsOnDataAsync(columns, plan.Operations, ct).ConfigureAwait(false);
            yield return NivaraFrame.Create(processed);
            yield break;
        }

        var chunkSize = calculateChunkSize(context.MemoryBudget);

        await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, ct)
            .ConfigureAwait(false))
        {
            ct.ThrowIfCancellationRequested();

            var processedData = await executeOperationsOnDataAsync(
                chunkData, plan.Operations, ct).ConfigureAwait(false);

            yield return NivaraFrame.Create(processedData);
        }
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
