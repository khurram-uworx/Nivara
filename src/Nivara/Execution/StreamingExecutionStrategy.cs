using Nivara.Diagnostics;
using Nivara.Query;

namespace Nivara.Execution;

sealed class StreamingExecutionStrategy : ExecutionStrategyBase
{
    static readonly HashSet<string> NonStreamableOperations = new() { Query.OperationType.Sort, Query.OperationType.GroupBy, Query.OperationType.Join };
    static readonly HashSet<string> StreamableOperationTypes = new() { Query.OperationType.Filter, Query.OperationType.Select, Query.OperationType.ConcatenationPrefix };

    static bool isSuitableForStreaming(QueryPlan plan)
    {
        foreach (var operation in plan.Operations)
            if (NonStreamableOperations.Contains(operation.OperationType))
                return false;
        return true;
    }

    static bool isOperationStreamable(IQueryOperation op)
        => !NonStreamableOperations.Contains(op.OperationType);

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

        if (!isSuitableForStreaming(plan))
            return await new LazyExecutionStrategy().ExecuteAsync(plan, context).ConfigureAwait(false);

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

        await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, context.CancellationToken).ConfigureAwait(false))
        {
            context.CancellationToken.ThrowIfCancellationRequested();

            using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

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
