namespace Nivara.Exceptions;

/// <summary>
/// Exception thrown when a column is not found in a frame
/// </summary>
public sealed class ColumnNotFoundException : Exception
{
    /// <summary>
    /// Initializes a new instance of ColumnNotFoundException
    /// </summary>
    /// <param name="columnName">The name of the column that was not found</param>
    public ColumnNotFoundException(string columnName)
        : base($"Column '{columnName}' not found")
    {
        ColumnName = columnName;
    }

    /// <summary>
    /// Initializes a new instance of ColumnNotFoundException with available columns
    /// </summary>
    /// <param name="columnName">The name of the column that was not found</param>
    /// <param name="availableColumns">The names of available columns</param>
    public ColumnNotFoundException(string columnName, IEnumerable<string> availableColumns)
        : base($"Column '{columnName}' not found. Available columns: {string.Join(", ", availableColumns)}")
    {
        ColumnName = columnName;
        AvailableColumns = availableColumns.ToList();
    }

    /// <summary>
    /// Gets the name of the column that was not found
    /// </summary>
    public string ColumnName { get; }

    /// <summary>
    /// Gets the list of available column names, if provided
    /// </summary>
    public IReadOnlyList<string>? AvailableColumns { get; }
}

/// <summary>
/// Exception thrown when a column type doesn't match the expected type
/// </summary>
public sealed class ColumnTypeMismatchException : Exception
{
    /// <summary>
    /// Initializes a new instance of ColumnTypeMismatchException
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <param name="expectedType">The expected type</param>
    /// <param name="actualType">The actual type</param>
    public ColumnTypeMismatchException(string columnName, Type expectedType, Type actualType)
        : base($"Column '{columnName}' is of type {actualType.Name}, but expected {expectedType.Name}")
    {
        ColumnName = columnName;
        ExpectedType = expectedType;
        ActualType = actualType;
    }

    /// <summary>
    /// Gets the name of the column
    /// </summary>
    public string ColumnName { get; }

    /// <summary>
    /// Gets the expected type
    /// </summary>
    public Type ExpectedType { get; }

    /// <summary>
    /// Gets the actual type
    /// </summary>
    public Type ActualType { get; }
}

/// <summary>
/// Exception thrown when a data source cannot be executed
/// </summary>
public sealed class DataSourceException : Exception
{
    /// <summary>
    /// Initializes a new instance of DataSourceException
    /// </summary>
    /// <param name="message">The error message</param>
    public DataSourceException(string message) : base(message)
    { }

    /// <summary>
    /// Initializes a new instance of DataSourceException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception</param>
    public DataSourceException(string message, Exception innerException) : base(message, innerException)
    { }
}

/// <summary>
/// Exception thrown when schema validation fails
/// </summary>
public sealed class SchemaValidationException : Exception
{
    /// <summary>
    /// Initializes a new instance of SchemaValidationException
    /// </summary>
    /// <param name="message">The error message</param>
    public SchemaValidationException(string message) : base(message)
    { }

    /// <summary>
    /// Initializes a new instance of SchemaValidationException with schema details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="expectedSchema">The expected schema</param>
    /// <param name="actualSchema">The actual schema</param>
    public SchemaValidationException(string message, Schema expectedSchema, Schema actualSchema) : base(message)
    {
        ExpectedSchema = expectedSchema;
        ActualSchema = actualSchema;
    }

    /// <summary>
    /// Gets the expected schema, if provided
    /// </summary>
    public Schema? ExpectedSchema { get; }

    /// <summary>
    /// Gets the actual schema, if provided
    /// </summary>
    public Schema? ActualSchema { get; }
}

/// <summary>
/// Exception thrown when query execution fails
/// </summary>
public sealed class QueryExecutionException : Exception
{
    /// <summary>
    /// Initializes a new instance of QueryExecutionException
    /// </summary>
    /// <param name="message">The error message</param>
    public QueryExecutionException(string message) : base(message)
    { }

    /// <summary>
    /// Initializes a new instance of QueryExecutionException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception</param>
    public QueryExecutionException(string message, Exception innerException) : base(message, innerException)
    { }

    /// <summary>
    /// Initializes a new instance of QueryExecutionException with operation context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationType">The type of operation that failed</param>
    /// <param name="innerException">The inner exception</param>
    public QueryExecutionException(string message, string operationType, Exception innerException) : base(message, innerException)
    {
        OperationType = operationType;
    }

    /// <summary>
    /// Gets the type of operation that failed, if provided
    /// </summary>
    public string? OperationType { get; }
}
