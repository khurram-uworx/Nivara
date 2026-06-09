using Nivara.Query;

namespace Nivara.Execution;

sealed class LazyExecutionStrategy : ExecutionStrategyBase
{
    protected override string StrategyName => "Lazy";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        ReportProgress(context, "Starting lazy execution", 0, 1);
        var result = executor.Execute(plan);
        ReportProgress(context, "Lazy execution completed", 1, 1);
        return result;
    }

    public override bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            return executor.ValidatePlan(plan);
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
            long cost = 100;
            cost += plan.Source.IsLazy ? 50 : 100;

            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 200,
                    "Select" => 100,
                    "Sort" => 1000,
                    "GroupBy" => 1500,
                    "Join" => 2000,
                    "Concatenation" => 300,
                    _ => 500
                };
            }

            var optimizationDiscount = Math.Min(cost * 0.2, 1000);
            cost -= (long)optimizationDiscount;
            return Math.Max(cost, 100);
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
