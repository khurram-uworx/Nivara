using Nivara.Execution;

namespace Nivara.Query;

/// <summary>
/// Represents a generic query operation that can transform data in a query pipeline.
/// Operations are composable and can be optimized by the query engine.
/// </summary>
/// <typeparam name="T">The type of data being processed</typeparam>
public interface IQueryOperation<T>
{
    /// <summary>
    /// Gets the query plan for this operation
    /// </summary>
    QueryPlan Plan { get; }

    /// <summary>
    /// Gets the execution strategy for this operation
    /// </summary>
    ExecutionStrategy Strategy { get; }

    /// <summary>
    /// Transforms this operation to produce a new operation with a different result type
    /// </summary>
    /// <typeparam name="TResult">The result type of the transformation</typeparam>
    /// <param name="transform">The transformation function</param>
    /// <returns>A new operation that produces the transformed result</returns>
    IQueryOperation<TResult> Transform<TResult>(Func<T, TResult> transform);

    /// <summary>
    /// Executes the operation asynchronously
    /// </summary>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>A task that represents the asynchronous operation</returns>
    Task<T> ExecuteAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a query operation that can transform data in a query pipeline.
/// Operations are composable and can be optimized by the query engine.
/// </summary>
public interface IQueryOperation
{
    /// <summary>
    /// Gets the type of operation
    /// </summary>
    string OperationType { get; }

    /// <summary>
    /// Transforms the input schema to produce the output schema
    /// </summary>
    /// <param name="inputSchema">The schema of the input data</param>
    /// <returns>The schema of the output data after applying this operation</returns>
    /// <exception cref="SchemaValidationException">Thrown when the operation cannot be applied to the input schema</exception>
    Schema TransformSchema(Schema inputSchema);

    /// <summary>
    /// Executes the operation on the input columns
    /// </summary>
    /// <param name="input">The input columns to transform</param>
    /// <returns>The transformed columns</returns>
    /// <exception cref="QueryExecutionException">Thrown when the operation fails to execute</exception>
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);
}