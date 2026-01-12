using Nivara.Exceptions;

namespace Nivara.Helpers;

/// <summary>
/// Handles deferred error reporting for lazy operations.
/// Allows errors to be captured during query building and reported during execution.
/// </summary>
public sealed class DeferredErrorHandler
{
    private readonly List<DeferredError> deferredErrors;
    private readonly object lockObject;

    /// <summary>
    /// Initializes a new instance of DeferredErrorHandler
    /// </summary>
    public DeferredErrorHandler()
    {
        deferredErrors = new List<DeferredError>();
        lockObject = new object();
    }

    /// <summary>
    /// Gets a value indicating whether there are any deferred errors
    /// </summary>
    public bool HasDeferredErrors
    {
        get
        {
            lock (lockObject)
            {
                return deferredErrors.Count > 0;
            }
        }
    }

    /// <summary>
    /// Gets the count of deferred errors
    /// </summary>
    public int ErrorCount
    {
        get
        {
            lock (lockObject)
            {
                return deferredErrors.Count;
            }
        }
    }

    /// <summary>
    /// Adds a deferred error to be reported later
    /// </summary>
    /// <param name="error">The error to defer</param>
    /// <param name="context">The context where the error occurred</param>
    /// <param name="operationType">The type of operation that caused the error</param>
    public void AddDeferredError(Exception error, string context, string operationType)
    {
        if (error == null)
            throw new ArgumentNullException(nameof(error));

        if (string.IsNullOrWhiteSpace(context))
            throw new ArgumentException("Context cannot be null or whitespace", nameof(context));

        if (string.IsNullOrWhiteSpace(operationType))
            throw new ArgumentException("Operation type cannot be null or whitespace", nameof(operationType));

        lock (lockObject)
        {
            deferredErrors.Add(new DeferredError(error, context, operationType, DateTime.UtcNow));
        }
    }

    /// <summary>
    /// Adds a deferred error for file access issues
    /// </summary>
    /// <param name="filePath">The file path that caused the error</param>
    /// <param name="error">The underlying error</param>
    /// <param name="operationType">The type of operation (e.g., "ScanCsv", "ReadJson")</param>
    public void AddFileAccessError(string filePath, Exception error, string operationType)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filePath));

        var context = $"File access: {filePath}";
        var wrappedError = new DataSourceException($"Failed to access file '{filePath}': {error.Message}", error);

        AddDeferredError(wrappedError, context, operationType);
    }

    /// <summary>
    /// Adds a deferred error for schema validation issues
    /// </summary>
    /// <param name="schemaError">The schema validation error</param>
    /// <param name="context">The context where the error occurred</param>
    /// <param name="operationType">The type of operation</param>
    public void AddSchemaValidationError(SchemaValidationException schemaError, string context, string operationType)
    {
        if (schemaError == null)
            throw new ArgumentNullException(nameof(schemaError));

        AddDeferredError(schemaError, context, operationType);
    }

    /// <summary>
    /// Checks for deferred errors and throws an aggregate exception if any exist
    /// </summary>
    /// <param name="operationContext">The context of the operation that is checking for errors</param>
    /// <exception cref="QueryExecutionException">Thrown when deferred errors exist</exception>
    public void ThrowIfHasDeferredErrors(string operationContext)
    {
        if (string.IsNullOrWhiteSpace(operationContext))
            throw new ArgumentException("Operation context cannot be null or whitespace", nameof(operationContext));

        lock (lockObject)
        {
            if (deferredErrors.Count == 0)
                return;

            if (deferredErrors.Count == 1)
            {
                var singleError = deferredErrors[0];
                throw new QueryExecutionException(
                    $"Deferred error in {operationContext}: {singleError.Error.Message} (Context: {singleError.Context})",
                    singleError.OperationType,
                    singleError.Error);
            }

            // Multiple errors - create aggregate exception
            var errorMessages = deferredErrors.Select(e =>
                $"[{e.OperationType}] {e.Context}: {e.Error.Message}");

            var aggregateMessage = $"Multiple deferred errors in {operationContext}:\n" +
                                 string.Join("\n", errorMessages);

            var innerExceptions = deferredErrors.Select(e => e.Error).ToArray();
            var aggregateException = new AggregateException(aggregateMessage, innerExceptions);

            throw new QueryExecutionException(aggregateMessage, aggregateException);
        }
    }

    /// <summary>
    /// Gets all deferred errors for diagnostic purposes
    /// </summary>
    /// <returns>A read-only list of deferred errors</returns>
    public IReadOnlyList<DeferredError> GetDeferredErrors()
    {
        lock (lockObject)
        {
            return deferredErrors.ToList();
        }
    }

    /// <summary>
    /// Clears all deferred errors
    /// </summary>
    public void ClearDeferredErrors()
    {
        lock (lockObject)
        {
            deferredErrors.Clear();
        }
    }

    /// <summary>
    /// Creates a summary of all deferred errors
    /// </summary>
    /// <returns>A formatted string summarizing the deferred errors</returns>
    public string CreateErrorSummary()
    {
        lock (lockObject)
        {
            if (deferredErrors.Count == 0)
                return "No deferred errors";

            var summary = new System.Text.StringBuilder();
            summary.AppendLine($"Deferred Errors Summary ({deferredErrors.Count} errors):");
            summary.AppendLine("=".PadRight(50, '='));

            var groupedErrors = deferredErrors.GroupBy(e => e.OperationType);

            foreach (var group in groupedErrors)
            {
                summary.AppendLine($"\n{group.Key} Operations ({group.Count()} errors):");

                foreach (var error in group)
                {
                    summary.AppendLine($"  • {error.Context}");
                    summary.AppendLine($"    {error.Error.GetType().Name}: {error.Error.Message}");
                    summary.AppendLine($"    Occurred: {error.Timestamp:yyyy-MM-dd HH:mm:ss} UTC");
                }
            }

            return summary.ToString();
        }
    }
}

/// <summary>
/// Represents a deferred error with context information
/// </summary>
public sealed class DeferredError
{
    /// <summary>
    /// Initializes a new instance of DeferredError
    /// </summary>
    /// <param name="error">The error that was deferred</param>
    /// <param name="context">The context where the error occurred</param>
    /// <param name="operationType">The type of operation that caused the error</param>
    /// <param name="timestamp">The timestamp when the error was deferred</param>
    internal DeferredError(Exception error, string context, string operationType, DateTime timestamp)
    {
        Error = error ?? throw new ArgumentNullException(nameof(error));
        Context = context ?? throw new ArgumentNullException(nameof(context));
        OperationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
        Timestamp = timestamp;
    }

    /// <summary>
    /// Gets the deferred error
    /// </summary>
    public Exception Error { get; }

    /// <summary>
    /// Gets the context where the error occurred
    /// </summary>
    public string Context { get; }

    /// <summary>
    /// Gets the type of operation that caused the error
    /// </summary>
    public string OperationType { get; }

    /// <summary>
    /// Gets the timestamp when the error was deferred
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Returns a string representation of the deferred error
    /// </summary>
    /// <returns>A formatted string</returns>
    public override string ToString()
    {
        return $"[{OperationType}] {Context}: {Error.GetType().Name} - {Error.Message} (at {Timestamp:HH:mm:ss})";
    }
}