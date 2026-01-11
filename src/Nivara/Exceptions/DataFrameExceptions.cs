using Nivara.Operations;

namespace Nivara.Exceptions;

/// <summary>
/// Base exception for all DataFrame operations
/// </summary>
public abstract class DataFrameException : Exception
{
    /// <summary>
    /// Initializes a new instance of DataFrameException
    /// </summary>
    /// <param name="message">The error message</param>
    protected DataFrameException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of DataFrameException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception</param>
    protected DataFrameException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of DataFrameException with query context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="failedPlan">The query plan that failed</param>
    /// <param name="failedOperation">The operation that failed</param>
    protected DataFrameException(string message, QueryPlan? failedPlan, IQueryOperation? failedOperation) : base(message)
    {
        FailedPlan = failedPlan;
        FailedOperation = failedOperation;
    }

    /// <summary>
    /// Initializes a new instance of DataFrameException with query context and inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="failedPlan">The query plan that failed</param>
    /// <param name="failedOperation">The operation that failed</param>
    /// <param name="innerException">The inner exception</param>
    protected DataFrameException(string message, QueryPlan? failedPlan, IQueryOperation? failedOperation, Exception innerException) 
        : base(message, innerException)
    {
        FailedPlan = failedPlan;
        FailedOperation = failedOperation;
    }

    /// <summary>
    /// Gets the query plan that failed, if available
    /// </summary>
    public QueryPlan? FailedPlan { get; }

    /// <summary>
    /// Gets the operation that failed, if available
    /// </summary>
    public IQueryOperation? FailedOperation { get; }

    /// <summary>
    /// Gets detailed context information about the failure
    /// </summary>
    public virtual string GetDetailedContext()
    {
        var context = new System.Text.StringBuilder();
        context.AppendLine($"Exception Type: {GetType().Name}");
        context.AppendLine($"Message: {Message}");

        if (FailedOperation != null)
        {
            context.AppendLine($"Failed Operation: {FailedOperation.OperationType}");
        }

        if (FailedPlan != null)
        {
            context.AppendLine($"Query Plan Operations: {string.Join(" → ", FailedPlan.Operations.Select(op => op.OperationType))}");
            context.AppendLine($"Source Schema: {string.Join(", ", FailedPlan.Source.Schema.ColumnNames)}");
            context.AppendLine($"Expected Result Schema: {string.Join(", ", FailedPlan.ResultSchema.ColumnNames)}");
        }

        if (InnerException != null)
        {
            context.AppendLine($"Inner Exception: {InnerException.GetType().Name}: {InnerException.Message}");
        }

        return context.ToString();
    }
}

/// <summary>
/// Exception thrown when DataFrame schema validation fails
/// </summary>
public sealed class DataFrameSchemaValidationException : DataFrameException
{
    /// <summary>
    /// Initializes a new instance of DataFrameSchemaValidationException
    /// </summary>
    /// <param name="message">The error message</param>
    public DataFrameSchemaValidationException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of DataFrameSchemaValidationException with schema details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="expectedSchema">The expected schema</param>
    /// <param name="actualSchema">The actual schema</param>
    public DataFrameSchemaValidationException(string message, Schema expectedSchema, Schema actualSchema) : base(message)
    {
        ExpectedSchema = expectedSchema;
        ActualSchema = actualSchema;
    }

    /// <summary>
    /// Initializes a new instance of DataFrameSchemaValidationException with query context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="expectedSchema">The expected schema</param>
    /// <param name="actualSchema">The actual schema</param>
    /// <param name="failedPlan">The query plan that failed</param>
    /// <param name="failedOperation">The operation that failed</param>
    public DataFrameSchemaValidationException(string message, Schema expectedSchema, Schema actualSchema, 
        QueryPlan? failedPlan, IQueryOperation? failedOperation) 
        : base(message, failedPlan, failedOperation)
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

    /// <summary>
    /// Gets the list of schema mismatches
    /// </summary>
    public IReadOnlyList<SchemaMismatch> Mismatches
    {
        get
        {
            if (ExpectedSchema == null || ActualSchema == null)
                return Array.Empty<SchemaMismatch>();

            var mismatches = new List<SchemaMismatch>();

            // Check for missing columns
            foreach (var expectedColumn in ExpectedSchema.ColumnNames)
            {
                if (!ActualSchema.ColumnNames.Contains(expectedColumn))
                {
                    mismatches.Add(new SchemaMismatch(
                        SchemaMismatchType.MissingColumn,
                        expectedColumn,
                        null,
                        null,
                        $"Column '{expectedColumn}' is missing from actual schema"));
                }
            }

            // Check for extra columns
            foreach (var actualColumn in ActualSchema.ColumnNames)
            {
                if (!ExpectedSchema.ColumnNames.Contains(actualColumn))
                {
                    mismatches.Add(new SchemaMismatch(
                        SchemaMismatchType.ExtraColumn,
                        actualColumn,
                        null,
                        ActualSchema.GetColumnType(actualColumn),
                        $"Column '{actualColumn}' is not expected in schema"));
                }
            }

            // Check for type mismatches
            foreach (var columnName in ExpectedSchema.ColumnNames.Intersect(ActualSchema.ColumnNames))
            {
                var expectedType = ExpectedSchema.GetColumnType(columnName);
                var actualType = ActualSchema.GetColumnType(columnName);
                
                if (expectedType != actualType)
                {
                    mismatches.Add(new SchemaMismatch(
                        SchemaMismatchType.TypeMismatch,
                        columnName,
                        expectedType,
                        actualType,
                        $"Column '{columnName}' has type {actualType?.Name} but expected {expectedType?.Name}"));
                }
            }

            return mismatches;
        }
    }

    /// <summary>
    /// Gets detailed context information about the schema validation failure
    /// </summary>
    public override string GetDetailedContext()
    {
        var context = new System.Text.StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (ExpectedSchema != null)
        {
            context.AppendLine($"Expected Schema: {string.Join(", ", ExpectedSchema.ColumnNames.Select(name => $"{name}:{ExpectedSchema.GetColumnType(name)?.Name}"))}");
        }

        if (ActualSchema != null)
        {
            context.AppendLine($"Actual Schema: {string.Join(", ", ActualSchema.ColumnNames.Select(name => $"{name}:{ActualSchema.GetColumnType(name)?.Name}"))}");
        }

        var mismatches = Mismatches;
        if (mismatches.Count > 0)
        {
            context.AppendLine("Schema Mismatches:");
            foreach (var mismatch in mismatches)
            {
                context.AppendLine($"  • {mismatch}");
            }
        }

        return context.ToString();
    }
}

/// <summary>
/// Exception thrown when DataFrame join operations fail
/// </summary>
public sealed class JoinException : DataFrameException
{
    /// <summary>
    /// Initializes a new instance of JoinException
    /// </summary>
    /// <param name="message">The error message</param>
    public JoinException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of JoinException with join details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="joinType">The type of join that was attempted</param>
    /// <param name="leftKeys">The left join keys</param>
    /// <param name="rightKeys">The right join keys</param>
    /// <param name="conflictReason">The reason for the join failure</param>
    public JoinException(string message, JoinType joinType, string[] leftKeys, string[] rightKeys, string conflictReason) 
        : base(message)
    {
        AttemptedJoinType = joinType;
        LeftKeys = leftKeys ?? Array.Empty<string>();
        RightKeys = rightKeys ?? Array.Empty<string>();
        ConflictReason = conflictReason;
    }

    /// <summary>
    /// Initializes a new instance of JoinException with query context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="joinType">The type of join that was attempted</param>
    /// <param name="leftKeys">The left join keys</param>
    /// <param name="rightKeys">The right join keys</param>
    /// <param name="conflictReason">The reason for the join failure</param>
    /// <param name="failedPlan">The query plan that failed</param>
    /// <param name="failedOperation">The operation that failed</param>
    public JoinException(string message, JoinType joinType, string[] leftKeys, string[] rightKeys, string conflictReason,
        QueryPlan? failedPlan, IQueryOperation? failedOperation) 
        : base(message, failedPlan, failedOperation)
    {
        AttemptedJoinType = joinType;
        LeftKeys = leftKeys ?? Array.Empty<string>();
        RightKeys = rightKeys ?? Array.Empty<string>();
        ConflictReason = conflictReason;
    }

    /// <summary>
    /// Gets the type of join that was attempted
    /// </summary>
    public JoinType AttemptedJoinType { get; }

    /// <summary>
    /// Gets the left join keys
    /// </summary>
    public IReadOnlyList<string> LeftKeys { get; } = Array.Empty<string>();

    /// <summary>
    /// Gets the right join keys
    /// </summary>
    public IReadOnlyList<string> RightKeys { get; } = Array.Empty<string>();

    /// <summary>
    /// Gets the reason for the join failure
    /// </summary>
    public string? ConflictReason { get; }

    /// <summary>
    /// Gets detailed context information about the join failure
    /// </summary>
    public override string GetDetailedContext()
    {
        var context = new System.Text.StringBuilder();
        context.AppendLine(base.GetDetailedContext());
        context.AppendLine($"Join Type: {AttemptedJoinType}");
        context.AppendLine($"Left Keys: {string.Join(", ", LeftKeys)}");
        context.AppendLine($"Right Keys: {string.Join(", ", RightKeys)}");
        
        if (!string.IsNullOrEmpty(ConflictReason))
        {
            context.AppendLine($"Conflict Reason: {ConflictReason}");
        }

        return context.ToString();
    }
}

/// <summary>
/// Represents a schema mismatch between expected and actual schemas
/// </summary>
public sealed class SchemaMismatch
{
    /// <summary>
    /// Initializes a new instance of SchemaMismatch
    /// </summary>
    /// <param name="mismatchType">The type of mismatch</param>
    /// <param name="columnName">The name of the column involved</param>
    /// <param name="expectedType">The expected type, if applicable</param>
    /// <param name="actualType">The actual type, if applicable</param>
    /// <param name="description">A description of the mismatch</param>
    public SchemaMismatch(SchemaMismatchType mismatchType, string columnName, Type? expectedType, Type? actualType, string description)
    {
        MismatchType = mismatchType;
        ColumnName = columnName ?? throw new ArgumentNullException(nameof(columnName));
        ExpectedType = expectedType;
        ActualType = actualType;
        Description = description ?? throw new ArgumentNullException(nameof(description));
    }

    /// <summary>
    /// Gets the type of mismatch
    /// </summary>
    public SchemaMismatchType MismatchType { get; }

    /// <summary>
    /// Gets the name of the column involved in the mismatch
    /// </summary>
    public string ColumnName { get; }

    /// <summary>
    /// Gets the expected type, if applicable
    /// </summary>
    public Type? ExpectedType { get; }

    /// <summary>
    /// Gets the actual type, if applicable
    /// </summary>
    public Type? ActualType { get; }

    /// <summary>
    /// Gets a description of the mismatch
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Returns a string representation of the schema mismatch
    /// </summary>
    /// <returns>A formatted string describing the mismatch</returns>
    public override string ToString()
    {
        return Description;
    }
}

/// <summary>
/// Defines the types of schema mismatches that can occur
/// </summary>
public enum SchemaMismatchType
{
    /// <summary>
    /// A column is missing from the actual schema
    /// </summary>
    MissingColumn,

    /// <summary>
    /// An extra column exists in the actual schema
    /// </summary>
    ExtraColumn,

    /// <summary>
    /// A column has a different type than expected
    /// </summary>
    TypeMismatch
}