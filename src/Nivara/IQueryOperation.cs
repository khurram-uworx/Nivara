namespace Nivara;

/// <summary>
/// Represents a query operation that can transform data in a query pipeline.
/// Operations are composable and can be optimized by the query engine.
/// </summary>
internal interface IQueryOperation
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