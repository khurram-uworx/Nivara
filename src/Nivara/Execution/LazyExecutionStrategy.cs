using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Execution;

sealed class LazyExecutionStrategy : ExecutionStrategyBase
{
    protected override string StrategyName => "Lazy";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        context.Progress?.Report(new ExecutionProgress("Starting lazy execution", 0, 1));
        var result = diag != null
            ? DiagnosticHelper.ExecuteWithDiagnostics(diag, "LazyExecution", () => executor.Execute(plan))
            : executor.Execute(plan);
        context.Progress?.Report(new ExecutionProgress("Lazy execution completed", 1, 1));
        return result;
    }

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        var diag = context.ExecutionDiagnostics;
        context.Progress?.Report(new ExecutionProgress("Starting lazy execution", 0, 1));
        var result = diag != null
            ? await DiagnosticHelper.ExecuteWithDiagnosticsAsync(diag, "LazyExecutionAsync", () => executeAsyncCore(plan, context)).ConfigureAwait(false)
            : await executeAsyncCore(plan, context).ConfigureAwait(false);
        context.Progress?.Report(new ExecutionProgress("Lazy execution completed", 1, 1));
        return result;
    }

    /// <summary>
    /// Mirrors <see cref="QueryExecutor.Execute"/> semantics over the async seams: the source is
    /// read via <see cref="IQuerySource.ExecuteAsync"/> and each operation is applied via
    /// <see cref="IQueryOperation.ExecuteAsync"/>, so IO-bound sources participate in a genuine
    /// async pipeline instead of the whole plan being pushed onto a thread-pool thread.
    /// </summary>
    async Task<NivaraFrame> executeAsyncCore(QueryPlan plan, NivaraExecutionContext context)
    {
        var ct = context.CancellationToken;

        if (!executor.ValidatePlan(plan))
        {
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
            throw new QueryExecutionException($"Query plan validation failed. {diagnosticInfo}");
        }

        IReadOnlyDictionary<string, IColumn> currentColumns;
        try
        {
            currentColumns = await plan.Source.ExecuteAsync(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
            throw new QueryExecutionException($"Data source execution failed: {ex.Message}. {diagnosticInfo}", ex);
        }

        if (currentColumns == null)
        {
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
            throw new QueryExecutionException($"Data source returned null columns. {diagnosticInfo}");
        }

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            try
            {
                currentColumns = await operation.ExecuteAsync(currentColumns, ct).ConfigureAwait(false);

                if (currentColumns == null)
                {
                    var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
                    throw new QueryExecutionException($"Operation '{operation.OperationType}' at position {i + 1} returned null columns. {diagnosticInfo}");
                }
            }
            catch (Exception ex) when (ex is not QueryExecutionException && ex is not OperationCanceledException)
            {
                var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
                throw new QueryExecutionException(
                    $"Operation '{operation.OperationType}' at position {i + 1} failed: {ex.Message}. {diagnosticInfo}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
        {
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
            throw new QueryExecutionException($"Query execution resulted in no columns. {diagnosticInfo}");
        }

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
            long cost = 100;
            cost += plan.Source.IsLazy ? 50 : 100;

            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    Query.OperationType.Filter => 200,
                    Query.OperationType.Select => 100,
                    Query.OperationType.Sort => 1000,
                    Query.OperationType.GroupBy => 1500,
                    Query.OperationType.Join => 2000,
                    _ when operation.OperationType.StartsWith(Query.OperationType.ConcatenationPrefix, StringComparison.Ordinal) => 300,
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
