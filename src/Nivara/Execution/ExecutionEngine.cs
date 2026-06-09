using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Query;
using System.Collections.Concurrent;

namespace Nivara.Execution;

/// <summary>
/// Defines the execution strategy for query operations
/// </summary>
public enum ExecutionStrategy
{
    /// <summary>
    /// Build query plan, execute on Collect() - default strategy
    /// </summary>
    Lazy,

    /// <summary>
    /// Execute operations immediately without building query plans
    /// </summary>
    Eager,

    /// <summary>
    /// Process data in chunks for large datasets to manage memory usage
    /// </summary>
    Streaming,

    /// <summary>
    /// Use multiple threads for parallelizable operations
    /// </summary>
    Parallel
}

/// <summary>
/// Defines the interface for execution strategies
/// </summary>
public interface IExecutionStrategy
{
    /// <summary>
    /// Executes a query plan synchronously
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <param name="context">The execution context</param>
    /// <returns>The materialized result</returns>
    NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context);

    /// <summary>
    /// Executes a query plan asynchronously
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <param name="context">The execution context</param>
    /// <returns>A task representing the asynchronous execution</returns>
    Task<NivaraFrame> ExecuteAsync(QueryPlan plan, NivaraExecutionContext context);

    /// <summary>
    /// Validates that a query plan can be executed with this strategy
    /// </summary>
    /// <param name="plan">The query plan to validate</param>
    /// <param name="context">The execution context</param>
    /// <returns>True if the plan is valid, false otherwise</returns>
    bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context);

    /// <summary>
    /// Estimates the execution cost for this strategy
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <param name="context">The execution context</param>
    /// <returns>An estimated execution cost</returns>
    long EstimateExecutionCost(QueryPlan plan, NivaraExecutionContext context);
}

/// <summary>
/// Executes query plans using different execution strategies.
/// Provides strategy pattern for lazy, eager, streaming, and parallel execution modes.
/// </summary>
public sealed class ExecutionEngine
{
    QueryOptimizer? optimizer;
    ExecutionDiagnostics? lastDiagnostics;
    readonly ConcurrentDictionary<ExecutionStrategy, IExecutionStrategy> strategies;

    /// <summary>
    /// Initializes a new instance of ExecutionEngine
    /// </summary>
    public ExecutionEngine()
    {
        this.optimizer = null; // Will be set via internal method
        this.strategies = new ConcurrentDictionary<ExecutionStrategy, IExecutionStrategy>();

        // Register default execution strategies
        RegisterStrategy(ExecutionStrategy.Lazy, new LazyExecutionStrategy());
        RegisterStrategy(ExecutionStrategy.Eager, new EagerExecutionStrategy());
        RegisterStrategy(ExecutionStrategy.Streaming, new StreamingExecutionStrategy());
        RegisterStrategy(ExecutionStrategy.Parallel, new ParallelExecutionStrategy());
    }

    /// <summary>
    /// Sets the query optimizer for this execution engine (internal use only)
    /// </summary>
    /// <param name="queryOptimizer">The query optimizer to use</param>
    internal void SetOptimizer(QueryOptimizer? queryOptimizer)
    {
        optimizer = queryOptimizer;
    }

    /// <summary>
    /// Gets the diagnostics from the most recent execution
    /// </summary>
    public ExecutionDiagnostics? LastDiagnostics => lastDiagnostics;

    /// <summary>
    /// Executes a query plan using the specified execution context
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <param name="context">The execution context containing strategy and configuration</param>
    /// <returns>The materialized result of the query</returns>
    /// <exception cref="QueryExecutionException">Thrown when execution fails</exception>
    public NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        // Create or use existing diagnostics
        var diagnostics = context.ExecutionDiagnostics ?? new ExecutionDiagnostics();
        context.ExecutionDiagnostics = diagnostics;
        diagnostics.ExecutionStrategy = context.Strategy;
        diagnostics.ParallelismDegree = context.MaxDegreeOfParallelism;
        diagnostics.StartExecution();
        lastDiagnostics = diagnostics;

        try
        {
            // Apply optimization if optimizer is available
            var optimizedPlan = optimizer?.Optimize(plan) ?? plan;

            // Record optimization if plan was changed
            if (!ReferenceEquals(plan, optimizedPlan) && optimizer != null)
                diagnostics.RecordOptimization(new OptimizationApplied(
                    "QueryOptimization",
                    "Query optimizer modified the execution plan",
                    estimatedImprovement: null));

            // Get the execution strategy
            if (!strategies.TryGetValue(context.Strategy, out var strategy))
                throw new QueryExecutionException($"Unknown execution strategy: {context.Strategy}");

            // Execute using the selected strategy
            var result = strategy.Execute(optimizedPlan, context);

            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Critical,
                $"Query execution failed: {ex.Message}",
                "Check query plan and input data"));
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
            throw new QueryExecutionException($"Query execution failed: {ex.Message}. {diagnosticInfo}", ex);
        }
        finally
        {
            diagnostics.EndExecution();
            diagnostics.ImportFromDiagnosticsTracker();
        }
    }

    /// <summary>
    /// Executes a query plan using the default lazy execution strategy
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <returns>The materialized result of the query</returns>
    public NivaraFrame Execute(QueryPlan plan)
    {
        return Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Lazy));
    }

    /// <summary>
    /// Executes a query plan asynchronously using the specified execution context
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <param name="context">The execution context containing strategy and configuration</param>
    /// <returns>A task representing the asynchronous execution</returns>
    public async Task<NivaraFrame> ExecuteAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        // Create or use existing diagnostics
        var diagnostics = context.ExecutionDiagnostics ?? new ExecutionDiagnostics();
        context.ExecutionDiagnostics = diagnostics;
        diagnostics.ExecutionStrategy = context.Strategy;
        diagnostics.ParallelismDegree = context.MaxDegreeOfParallelism;
        diagnostics.StartExecution();
        lastDiagnostics = diagnostics;

        try
        {
            // Apply optimization if optimizer is available
            var optimizedPlan = optimizer?.Optimize(plan) ?? plan;

            // Record optimization if plan was changed
            if (!ReferenceEquals(plan, optimizedPlan) && optimizer != null)
                diagnostics.RecordOptimization(new OptimizationApplied(
                    "QueryOptimization",
                    "Query optimizer modified the execution plan",
                    estimatedImprovement: null));

            // Get the execution strategy
            if (!strategies.TryGetValue(context.Strategy, out var strategy))
                throw new QueryExecutionException($"Unknown execution strategy: {context.Strategy}");

            // Execute using the selected strategy
            return await strategy.ExecuteAsync(optimizedPlan, context);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Critical,
                $"Async query execution failed: {ex.Message}",
                "Check query plan and input data"));
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
            throw new QueryExecutionException($"Async query execution failed: {ex.Message}. {diagnosticInfo}", ex);
        }
        finally
        {
            diagnostics.EndExecution();
            diagnostics.ImportFromDiagnosticsTracker();
        }
    }

    /// <summary>
    /// Registers a custom execution strategy
    /// </summary>
    /// <param name="strategyType">The execution strategy type</param>
    /// <param name="strategy">The strategy implementation</param>
    public void RegisterStrategy(ExecutionStrategy strategyType, IExecutionStrategy strategy)
    {
        if (strategy == null)
            throw new ArgumentNullException(nameof(strategy));

        strategies.AddOrUpdate(strategyType, strategy, (_, _) => strategy);
    }

    /// <summary>
    /// Validates that a query plan can be executed without actually executing it
    /// </summary>
    /// <param name="plan">The query plan to validate</param>
    /// <param name="context">The execution context</param>
    /// <returns>True if the plan is valid for the given context, false otherwise</returns>
    public bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            // Apply optimization if optimizer is available
            var optimizedPlan = optimizer?.Optimize(plan) ?? plan;

            // Get the execution strategy
            if (!strategies.TryGetValue(context.Strategy, out var strategy))
                return false;

            // Validate using the selected strategy
            return strategy.ValidatePlan(optimizedPlan, context);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Estimates the execution cost of a query plan for the given context
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <param name="context">The execution context</param>
    /// <returns>An estimated execution cost (higher values indicate more expensive operations)</returns>
    public long EstimateExecutionCost(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return long.MaxValue;

        try
        {
            // Apply optimization if optimizer is available
            var optimizedPlan = optimizer?.Optimize(plan) ?? plan;

            // Get the execution strategy
            if (!strategies.TryGetValue(context.Strategy, out var strategy))
                return long.MaxValue;

            // Estimate cost using the selected strategy
            return strategy.EstimateExecutionCost(optimizedPlan, context);
        }
        catch
        {
            return long.MaxValue;
        }
    }

    /// <summary>
    /// Gets the available execution strategies
    /// </summary>
    /// <returns>A collection of available execution strategy types</returns>
    public IReadOnlyCollection<ExecutionStrategy> GetAvailableStrategies()
        => strategies.Keys.ToList();

    /// <summary>
    /// Returns a string representation of the execution engine
    /// </summary>
    /// <returns>A formatted string describing the execution engine</returns>
    public override string ToString()
    {
        var strategyCount = strategies.Count;
        var hasOptimizer = optimizer != null;
        return $"ExecutionEngine {{ Strategies: {strategyCount}, Optimizer: {hasOptimizer} }}";
    }
}
