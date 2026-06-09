using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Execution;

sealed class EagerExecutionStrategy : ExecutionStrategyBase
{
    protected override string StrategyName => "Eager";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "EagerExecution") : null;
        context.Progress?.Report(new ExecutionProgress("Starting eager execution", 0, plan.Operations.Count + 1));

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
                    ? DiagnosticHelper.ExecuteWithDiagnostics(diag, operation.OperationType, () => capturedOp.Execute(currentColumns))
                    : capturedOp.Execute(currentColumns);
                context.Progress?.Report(new ExecutionProgress($"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1));
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
        var diag = context.ExecutionDiagnostics;
        using var overallScope = diag != null ? DiagnosticHelper.CreateScope(diag, "EagerExecutionAsync") : null;
        context.Progress?.Report(new ExecutionProgress("Starting async eager execution", 0, plan.Operations.Count + 1));

        var currentColumns = diag != null
            ? await DiagnosticHelper.ExecuteWithDiagnosticsAsync(diag, "SourceExecute", () => plan.Source.ExecuteAsync(context.CancellationToken)).ConfigureAwait(false)
            : await plan.Source.ExecuteAsync(context.CancellationToken).ConfigureAwait(false);
        context.Progress?.Report(new ExecutionProgress("Data source executed", 1, plan.Operations.Count + 1));

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                var capturedOp = operation;
                currentColumns = diag != null
                    ? await DiagnosticHelper.ExecuteWithDiagnosticsAsync(diag, operation.OperationType,
                        () => Task.Run(() => capturedOp.Execute(currentColumns), context.CancellationToken)).ConfigureAwait(false)
                    : await Task.Run(() => capturedOp.Execute(currentColumns), context.CancellationToken).ConfigureAwait(false);
                context.Progress?.Report(new ExecutionProgress($"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1));
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
                    Query.OperationType.Filter => 300,
                    Query.OperationType.Select => 150,
                    Query.OperationType.Sort => 1200,
                    Query.OperationType.GroupBy => 1800,
                    Query.OperationType.Join => 2500,
                    _ when operation.OperationType.StartsWith(Query.OperationType.ConcatenationPrefix, StringComparison.Ordinal) => 400,
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
