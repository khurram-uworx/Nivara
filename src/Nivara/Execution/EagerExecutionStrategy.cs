using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Execution;

sealed class EagerExecutionStrategy : ExecutionStrategyBase
{
    protected override string StrategyName => "Eager";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        ReportProgress(context, "Starting eager execution", 0, plan.Operations.Count + 1);

        var currentColumns = plan.Source.Execute();
        ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                currentColumns = operation.Execute(currentColumns);
                ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Eager execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
            throw new QueryExecutionException("Eager execution resulted in no columns");

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        ReportProgress(context, "Starting async eager execution", 0, plan.Operations.Count + 1);

        var currentColumns = await plan.Source.ExecuteAsync(context.CancellationToken);
        ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                currentColumns = await Task.Run(() => operation.Execute(currentColumns), context.CancellationToken);
                ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Async eager execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
            throw new QueryExecutionException("Async eager execution resulted in no columns");

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
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
            long cost = 200;
            cost += plan.Source.IsLazy ? 200 : 150;

            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 300,
                    "Select" => 150,
                    "Sort" => 1200,
                    "GroupBy" => 1800,
                    "Join" => 2500,
                    "Concatenation" => 400,
                    _ => 600
                };
            }

            cost += plan.Operations.Count * 100;
            return Math.Max(cost, 200);
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
