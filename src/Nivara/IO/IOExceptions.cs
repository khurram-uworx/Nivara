namespace Nivara.IO;

/// <summary>
/// Base exception for all Nivara I/O operations
/// </summary>
public class NivaraIOException : Exception
{
    /// <summary>
    /// Initializes a new instance of NivaraIOException
    /// </summary>
    /// <param name="message">The error message</param>
    public NivaraIOException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of NivaraIOException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception</param>
    public NivaraIOException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of NivaraIOException with file path and operation context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="filePath">The file path where the error occurred</param>
    /// <param name="operationContext">The operation context</param>
    public NivaraIOException(string message, string? filePath, string? operationContext) : base(message)
    {
        FilePath = filePath;
        OperationContext = operationContext;
    }

    /// <summary>
    /// Initializes a new instance of NivaraIOException with file path, operation context, and inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="filePath">The file path where the error occurred</param>
    /// <param name="operationContext">The operation context</param>
    /// <param name="innerException">The inner exception</param>
    public NivaraIOException(string message, string? filePath, string? operationContext, Exception innerException) : base(message, innerException)
    {
        FilePath = filePath;
        OperationContext = operationContext;
    }

    /// <summary>
    /// Gets the file path where the error occurred, if provided
    /// </summary>
    public string? FilePath { get; init; }

    /// <summary>
    /// Gets the operation context, if provided
    /// </summary>
    public string? OperationContext { get; init; }
}

/// <summary>
/// Exception thrown when an unsupported type is encountered during I/O operations
/// </summary>
public sealed class UnsupportedTypeException : NivaraIOException
{
    /// <summary>
    /// Initializes a new instance of UnsupportedTypeException
    /// </summary>
    /// <param name="unsupportedType">The unsupported type</param>
    public UnsupportedTypeException(Type unsupportedType)
        : base($"Type '{unsupportedType.Name}' is not supported for I/O operations")
    {
        UnsupportedType = unsupportedType;
    }

    /// <summary>
    /// Initializes a new instance of UnsupportedTypeException with suggested alternatives
    /// </summary>
    /// <param name="unsupportedType">The unsupported type</param>
    /// <param name="suggestedAlternatives">Suggested alternative types</param>
    public UnsupportedTypeException(Type unsupportedType, IEnumerable<string> suggestedAlternatives)
        : base($"Type '{unsupportedType.Name}' is not supported for I/O operations. Suggested alternatives: {string.Join(", ", suggestedAlternatives)}")
    {
        UnsupportedType = unsupportedType;
        SuggestedAlternatives = suggestedAlternatives.ToList();
    }

    /// <summary>
    /// Gets the unsupported type
    /// </summary>
    public Type UnsupportedType { get; init; }

    /// <summary>
    /// Gets the list of suggested alternative types, if provided
    /// </summary>
    public IReadOnlyList<string> SuggestedAlternatives { get; init; } = new List<string>();
}

/// <summary>
/// Exception thrown when schema validation fails during I/O operations
/// </summary>
public sealed class SchemaValidationException : NivaraIOException
{
    /// <summary>
    /// Initializes a new instance of SchemaValidationException
    /// </summary>
    /// <param name="message">The error message</param>
    public SchemaValidationException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of SchemaValidationException with type mismatches and schema details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="typeMismatches">The type mismatches found</param>
    /// <param name="expectedSchema">The expected schema description</param>
    /// <param name="actualSchema">The actual schema description</param>
    public SchemaValidationException(string message, IEnumerable<string> typeMismatches, string expectedSchema, string actualSchema)
        : base(message)
    {
        TypeMismatches = typeMismatches.ToList();
        ExpectedSchema = expectedSchema;
        ActualSchema = actualSchema;
    }

    /// <summary>
    /// Gets the list of type mismatches found during validation
    /// </summary>
    public IReadOnlyList<string> TypeMismatches { get; init; } = new List<string>();

    /// <summary>
    /// Gets the expected schema description
    /// </summary>
    public string ExpectedSchema { get; init; } = string.Empty;

    /// <summary>
    /// Gets the actual schema description
    /// </summary>
    public string ActualSchema { get; init; } = string.Empty;
}

/// <summary>
/// Exception thrown when data corruption is detected during I/O operations
/// </summary>
public sealed class DataCorruptionException : NivaraIOException
{
    /// <summary>
    /// Initializes a new instance of DataCorruptionException
    /// </summary>
    /// <param name="message">The error message</param>
    public DataCorruptionException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of DataCorruptionException with inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception</param>
    public DataCorruptionException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of DataCorruptionException with affected columns and row range
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="affectedColumns">The affected column names</param>
    /// <param name="affectedRowRange">The affected row range</param>
    public DataCorruptionException(string message, IEnumerable<string> affectedColumns, Range affectedRowRange)
        : base(message)
    {
        AffectedColumns = affectedColumns.ToList();
        AffectedRowRange = affectedRowRange;
    }

    /// <summary>
    /// Initializes a new instance of DataCorruptionException with file path and operation context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="filePath">The file path where the error occurred</param>
    /// <param name="operationContext">The operation context</param>
    /// <param name="affectedColumns">The affected column names</param>
    /// <param name="affectedRowRange">The affected row range</param>
    public DataCorruptionException(string message, string? filePath, string? operationContext, IEnumerable<string> affectedColumns, Range affectedRowRange)
        : base(message, filePath, operationContext)
    {
        AffectedColumns = affectedColumns.ToList();
        AffectedRowRange = affectedRowRange;
    }

    /// <summary>
    /// Gets the list of affected column names
    /// </summary>
    public IReadOnlyList<string> AffectedColumns { get; init; } = new List<string>();

    /// <summary>
    /// Gets the affected row range
    /// </summary>
    public Range AffectedRowRange { get; init; }
}