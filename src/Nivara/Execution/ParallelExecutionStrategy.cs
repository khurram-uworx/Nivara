using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Execution;

sealed class ParallelExecutionStrategy : ExecutionStrategyBase
{
    static bool isParallelizable(string operationType)
    {
        return operationType switch
        {
            "Filter" => true,
            "Select" => false,
            "Sort" => true,
            "GroupBy" => true,
            "Join" => true,
            "Concatenation" => true,
            _ => false
        };
    }

    static bool shouldUseParallelism(IReadOnlyDictionary<string, IColumn> input, NivaraExecutionContext context)
    {
        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        return ParallelExecutionHelper.ShouldUseParallelProcessing(
            totalRows,
            context.MaxDegreeOfParallelism);
    }

    static void validateParallelismConfiguration(NivaraExecutionContext context)
    {
        ParallelExecutionHelper.ValidateParallelConfiguration(context.MaxDegreeOfParallelism);
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeOperationParallelAsync(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        NivaraExecutionContext context)
    {
        ParallelExecutionHelper.ValidateParallelConfiguration(context.MaxDegreeOfParallelism);

        if (isParallelizable(operation.OperationType) && shouldUseParallelism(input, context))
        {
            var recommendedParallelism = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);

            return operation.OperationType switch
            {
                "Filter" => await executeFilterParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Sort" => await executeSortParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "GroupBy" => await executeGroupByParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Join" => await executeJoinParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Concatenation" => await executeConcatenationParallel(operation, input, recommendedParallelism, context.CancellationToken),
                _ => await Task.Run(() => operation.Execute(input), context.CancellationToken)
            };
        }
        else
            return await Task.Run(() => operation.Execute(input), context.CancellationToken);
    }

    async Task<IReadOnlyDictionary<string, IColumn>> executeFilterParallel(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism, CancellationToken cancellationToken)
        => await Task.Run(() => operation.Execute(input), cancellationToken);

    async Task<IReadOnlyDictionary<string, IColumn>> executeSortParallel(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism, CancellationToken cancellationToken)
        => await Task.Run(() => operation.Execute(input), cancellationToken);

    async Task<IReadOnlyDictionary<string, IColumn>> executeGroupByParallel(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism, CancellationToken cancellationToken)
        => await Task.Run(() => operation.Execute(input), cancellationToken);

    async Task<IReadOnlyDictionary<string, IColumn>> executeJoinParallel(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism, CancellationToken cancellationToken)
        => await Task.Run(() => operation.Execute(input), cancellationToken);

    async Task<IReadOnlyDictionary<string, IColumn>> executeConcatenationParallel(
        IQueryOperation operation, IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism, CancellationToken cancellationToken)
        => await Task.Run(() => operation.Execute(input), cancellationToken);

    protected override string StrategyName => "Parallel";

    protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
        => ExecuteCoreAsync(plan, context).GetAwaiter().GetResult();

    protected override async Task<NivaraFrame> ExecuteCoreAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
        validateParallelismConfiguration(context);

        ReportProgress(context, "Starting async parallel execution", 0, plan.Operations.Count + 1);

        var currentColumns = await plan.Source.ExecuteAsync(context.CancellationToken);
        ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

        for (int i = 0; i < plan.Operations.Count; i++)
        {
            var operation = plan.Operations[i];
            context.CancellationToken.ThrowIfCancellationRequested();
            try
            {
                currentColumns = await executeOperationParallelAsync(operation, currentColumns, context);
                ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
            }
            catch (Exception ex) when (ex is not QueryExecutionException)
            {
                throw new QueryExecutionException(
                    $"Async parallel execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                    operation.OperationType,
                    ex);
            }
        }

        if (currentColumns.Count == 0)
            throw new QueryExecutionException("Async parallel execution resulted in no columns");

        var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    public override bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            if (!executor.ValidatePlan(plan))
                return false;

            validateParallelismConfiguration(context);
            return true;
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
            long cost = 180;
            cost += plan.Source.IsLazy ? 120 : 100;

            foreach (var operation in plan.Operations)
            {
                var baseCost = operation.OperationType switch
                {
                    "Filter" => 300,
                    "Select" => 150,
                    "Sort" => 800,
                    "GroupBy" => 1000,
                    "Join" => 1500,
                    "Concatenation" => 250,
                    _ => 500
                };

                var parallelismFactor = Math.Min(context.MaxDegreeOfParallelism, Environment.ProcessorCount);
                var parallelDiscount = isParallelizable(operation.OperationType)
                    ? baseCost * (1.0 - (0.7 / parallelismFactor))
                    : baseCost * 0.95;

                cost += (long)parallelDiscount;
            }

            cost += plan.Operations.Count * 50;
            return Math.Max(cost, 180);
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
