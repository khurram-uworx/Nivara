using Nivara.Query;

namespace Nivara.Execution;

sealed class StreamingExecutionStrategy : ExecutionStrategyBase
{
    static readonly HashSet<string> NonStreamableOperations = new() { "Sort", "GroupBy", "Join" };
    static readonly HashSet<string> StreamableOperationTypes = new() { "Filter", "Select", "Concatenation" };

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
        => Task.Run(() => executeCoreInternalAsync(plan, context), context.CancellationToken).GetAwaiter().GetResult();

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
        => await executeCoreInternalAsync(plan, context).ConfigureAwait(false);

    async Task<NivaraFrame> executeCoreInternalAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        ReportProgress(context, "Starting streaming execution", 0, 1);

        if (!isSuitableForStreaming(plan))
        {
            var lazyStrategy = new LazyExecutionStrategy();
            return await lazyStrategy.ExecuteAsync(plan, context).ConfigureAwait(false);
        }

        if (!plan.Source.CanReadInChunks)
        {
            var fallbackResult = await Task.Run(() => executor.Execute(plan), context.CancellationToken).ConfigureAwait(false);
            ReportProgress(context, "Streaming execution completed", 1, 1);
            return fallbackResult;
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

            var processedData = executeOperationsOnData(chunkData, plan.Operations);
            var chunkFrame = NivaraFrame.Create(processedData);
            chunkFrames.Add(chunkFrame);

            chunkIndex++;
            var completedWork = chunkIndex;
            var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
            ReportProgress(context, $"Processing chunk {chunkIndex}", completedWork, totalWork);
        }

        if (chunkFrames.Count == 0)
        {
            ReportProgress(context, "No data from chunks, falling back to full execution", 0, 1);
            return await Task.Run(() => executor.Execute(plan), context.CancellationToken).ConfigureAwait(false);
        }

        if (chunkFrames.Count == 1)
        {
            ReportProgress(context, "Streaming execution completed", 1, 1);
            return chunkFrames[0];
        }

        var mergedResult = NivaraFrameExtensions.ConcatenateVertical(chunkFrames);
        ReportProgress(context, "Streaming execution completed", 1, 1);
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
                    "Filter" => 250,
                    "Select" => 120,
                    "Sort" => 2000,
                    "GroupBy" => 2500,
                    "Join" => 3000,
                    "Concatenation" => 200,
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
