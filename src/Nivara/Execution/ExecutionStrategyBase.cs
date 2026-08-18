using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Execution;

abstract class ExecutionStrategyBase : IExecutionStrategy
{
    protected readonly QueryExecutor executor;

    protected ExecutionStrategyBase()
    {
        executor = new QueryExecutor();
    }

    public NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context)
    {
        ValidateArgs(plan, context);
        context.CancellationToken.ThrowIfCancellationRequested();
        TryExtractPredicatePushdown(plan, context);
        try { return ExecuteCore(plan, context); }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"{StrategyName} execution failed: {ex.Message}", ex);
        }
    }

    public async Task<NivaraFrame> ExecuteAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        ValidateArgs(plan, context);
        context.CancellationToken.ThrowIfCancellationRequested();
        TryExtractPredicatePushdown(plan, context);
        try
        {
            return await ExecuteCoreAsync(plan, context).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Async {StrategyName} execution failed: {ex.Message}", ex);
        }
    }

    protected abstract NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context);

    protected virtual Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
        => Task.Run(() => ExecuteCore(plan, context), context.CancellationToken);

    protected abstract string StrategyName { get; }

    public abstract bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context);

    public abstract long EstimateExecutionCost(QueryPlan plan, NivaraExecutionContext context);

    protected static void ValidateArgs(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null) throw new ArgumentNullException(nameof(plan));
        if (context == null) throw new ArgumentNullException(nameof(context));
    }

    static void TryExtractPredicatePushdown(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan.Operations.Count == 0)
            return;

        if (plan.Operations[0] is not FilterOperation filter)
            return;

        if (plan.Source is not IPredicatePushdownSource pushdownSource)
            return;

        if (filter.Condition is not ComparisonExpression comparison)
            return;

        if (!pushdownSource.CanPushdownFilter(comparison, plan.ResultSchema))
            return;

        context.ExecutionDiagnostics?.RecordOptimization(
            new OptimizationApplied(
                "RowGroupPredicatePushdown",
                $"Pushed filter '{comparison}' to source for row-group statistics pruning"));
        pushdownSource.ApplyFilterPredicate(comparison, plan.ResultSchema);
    }
}
