using Nivara.Exceptions;
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
        try { return await ExecuteCoreAsync(plan, context); }
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

    protected static void ReportProgress(NivaraExecutionContext context, string operationName, long completedWork, long totalWork)
    {
        if (context.Progress != null)
        {
            var progress = new ExecutionProgress(operationName, completedWork, totalWork);
            context.Progress.Report(progress);
        }
    }
}
