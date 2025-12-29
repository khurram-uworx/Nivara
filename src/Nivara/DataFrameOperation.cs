using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Abstract base class for DataFrame operations that implement IQueryOperation&lt;NivaraFrame&gt;
/// </summary>
public abstract class DataFrameOperation : IQueryOperation<NivaraFrame>
{
    /// <summary>
    /// Initializes a new instance of DataFrameOperation
    /// </summary>
    /// <param name="plan">The query plan for this operation</param>
    /// <param name="strategy">The execution strategy</param>
    protected DataFrameOperation(QueryPlan plan, ExecutionStrategy strategy = ExecutionStrategy.Lazy)
    {
        Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        Strategy = strategy;
    }

    /// <inheritdoc />
    public QueryPlan Plan { get; }

    /// <inheritdoc />
    public ExecutionStrategy Strategy { get; }

    /// <summary>
    /// Gets the type of operation
    /// </summary>
    public abstract string OperationType { get; }

    /// <inheritdoc />
    public abstract IQueryOperation<TResult> Transform<TResult>(Func<NivaraFrame, TResult> transform);

    /// <inheritdoc />
    public abstract Task<NivaraFrame> ExecuteAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the operation synchronously
    /// </summary>
    /// <returns>The result of the operation</returns>
    public abstract NivaraFrame Execute();

    /// <summary>
    /// Executes the operation with the specified execution context
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>The result of the operation</returns>
    public virtual NivaraFrame Execute(ExecutionContext context)
    {
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        return context.Strategy switch
        {
            ExecutionStrategy.Lazy => ExecuteLazy(context),
            ExecutionStrategy.Eager => ExecuteEager(context),
            ExecutionStrategy.Streaming => ExecuteStreaming(context),
            ExecutionStrategy.Parallel => ExecuteParallel(context),
            _ => throw new ArgumentException($"Unknown execution strategy: {context.Strategy}")
        };
    }

    /// <summary>
    /// Executes the operation using lazy evaluation
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>The result of the operation</returns>
    protected virtual NivaraFrame ExecuteLazy(ExecutionContext context)
    {
        // Default implementation delegates to synchronous Execute
        return Execute();
    }

    /// <summary>
    /// Executes the operation using eager evaluation
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>The result of the operation</returns>
    protected virtual NivaraFrame ExecuteEager(ExecutionContext context)
    {
        // Default implementation delegates to synchronous Execute
        return Execute();
    }

    /// <summary>
    /// Executes the operation using streaming evaluation
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>The result of the operation</returns>
    protected virtual NivaraFrame ExecuteStreaming(ExecutionContext context)
    {
        // Default implementation delegates to synchronous Execute
        // Derived classes can override for true streaming behavior
        return Execute();
    }

    /// <summary>
    /// Executes the operation using parallel evaluation
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>The result of the operation</returns>
    protected virtual NivaraFrame ExecuteParallel(ExecutionContext context)
    {
        // Default implementation delegates to synchronous Execute
        // Derived classes can override for true parallel behavior
        return Execute();
    }

    /// <summary>
    /// Validates that the operation can be executed with the given context
    /// </summary>
    /// <param name="context">The execution context to validate</param>
    /// <exception cref="QueryExecutionException">Thrown when the operation cannot be executed with the given context</exception>
    protected virtual void ValidateExecutionContext(ExecutionContext context)
    {
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        if (context.MaxDegreeOfParallelism <= 0)
            throw new QueryExecutionException("MaxDegreeOfParallelism must be greater than zero");

        if (context.MemoryBudget <= 0)
            throw new QueryExecutionException("MemoryBudget must be greater than zero");
    }

    /// <summary>
    /// Reports progress for long-running operations
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <param name="operationName">The name of the current operation</param>
    /// <param name="completedWork">The amount of work completed</param>
    /// <param name="totalWork">The total amount of work</param>
    protected static void ReportProgress(ExecutionContext context, string operationName, long completedWork, long totalWork)
    {
        if (context.Progress != null)
        {
            var progress = new ExecutionProgress(operationName, completedWork, totalWork);
            context.Progress.Report(progress);
        }
    }

    /// <summary>
    /// Checks if the operation should be cancelled
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <exception cref="OperationCanceledException">Thrown when cancellation is requested</exception>
    protected static void ThrowIfCancellationRequested(ExecutionContext context)
    {
        context.CancellationToken.ThrowIfCancellationRequested();
    }

    /// <summary>
    /// Returns a string representation of the operation
    /// </summary>
    /// <returns>A formatted string describing the operation</returns>
    public override string ToString()
    {
        return $"{OperationType}Operation {{ Strategy: {Strategy} }}";
    }
}

/// <summary>
/// Generic transformation operation that wraps a transformation function
/// </summary>
/// <typeparam name="TResult">The result type of the transformation</typeparam>
internal sealed class TransformOperation<TResult> : IQueryOperation<TResult>
{
    readonly IQueryOperation<NivaraFrame> sourceOperation;
    readonly Func<NivaraFrame, TResult> transformFunction;

    /// <summary>
    /// Initializes a new instance of TransformOperation
    /// </summary>
    /// <param name="sourceOperation">The source operation</param>
    /// <param name="transformFunction">The transformation function</param>
    public TransformOperation(IQueryOperation<NivaraFrame> sourceOperation, Func<NivaraFrame, TResult> transformFunction)
    {
        this.sourceOperation = sourceOperation ?? throw new ArgumentNullException(nameof(sourceOperation));
        this.transformFunction = transformFunction ?? throw new ArgumentNullException(nameof(transformFunction));
    }

    /// <inheritdoc />
    public QueryPlan Plan => sourceOperation.Plan;

    /// <inheritdoc />
    public ExecutionStrategy Strategy => sourceOperation.Strategy;

    /// <inheritdoc />
    public IQueryOperation<TNewResult> Transform<TNewResult>(Func<TResult, TNewResult> transform)
    {
        if (transform == null)
            throw new ArgumentNullException(nameof(transform));

        // Chain transformations
        var chainedTransform = new Func<NivaraFrame, TNewResult>(frame => transform(transformFunction(frame)));
        return new TransformOperation<TNewResult>(sourceOperation, chainedTransform);
    }

    /// <inheritdoc />
    public async Task<TResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        var sourceResult = await sourceOperation.ExecuteAsync(cancellationToken);
        return transformFunction(sourceResult);
    }
}