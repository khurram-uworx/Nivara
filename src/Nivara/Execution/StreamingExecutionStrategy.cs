using Nivara.Query;

namespace Nivara.Execution;

sealed class StreamingExecutionStrategy : ExecutionStrategyBase
{
    static bool isSuitableForStreaming(QueryPlan plan)
    {
        var nonStreamableOperations = new HashSet<string> { "Sort", "GroupBy", "Join" };
        foreach (var operation in plan.Operations)
            if (nonStreamableOperations.Contains(operation.OperationType))
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

    protected override string StrategyName => "Streaming";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        calculateChunkSize(context.MemoryBudget);
        ReportProgress(context, "Starting streaming execution", 0, 1);

        if (!isSuitableForStreaming(plan))
        {
            var lazyStrategy = new LazyExecutionStrategy();
            return lazyStrategy.Execute(plan, context);
        }

        var result = executor.Execute(plan);
        ReportProgress(context, "Streaming execution completed", 1, 1);
        return result;
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        calculateChunkSize(context.MemoryBudget);
        ReportProgress(context, "Starting async streaming execution", 0, 1);

        if (!isSuitableForStreaming(plan))
        {
            var lazyStrategy = new LazyExecutionStrategy();
            return await lazyStrategy.ExecuteAsync(plan, context);
        }

        var result = await Task.Run(() => executor.Execute(plan), context.CancellationToken);
        ReportProgress(context, "Async streaming execution completed", 1, 1);
        return result;
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
