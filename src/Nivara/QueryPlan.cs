using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Represents a complete query execution plan with a data source and sequence of operations.
/// Provides the foundation for query optimization and execution.
/// </summary>
internal sealed class QueryPlan
{
    /// <summary>
    /// Initializes a new instance of QueryPlan
    /// </summary>
    /// <param name="source">The data source for the query</param>
    /// <param name="operations">The sequence of operations to apply</param>
    /// <exception cref="ArgumentNullException">Thrown when source or operations is null</exception>
    public QueryPlan(IQuerySource source, IEnumerable<IQueryOperation> operations)
    {
        Source = source ?? throw new ArgumentNullException(nameof(source));
        Operations = operations?.ToList() ?? throw new ArgumentNullException(nameof(operations));
        ResultSchema = ComputeResultSchema();
    }

    /// <summary>
    /// Gets the data source for this query
    /// </summary>
    public IQuerySource Source { get; }

    /// <summary>
    /// Gets the sequence of operations in this query
    /// </summary>
    public IReadOnlyList<IQueryOperation> Operations { get; }

    /// <summary>
    /// Gets the schema that will result from executing this query
    /// </summary>
    public Schema ResultSchema { get; }

    /// <summary>
    /// Creates a new query plan with additional operations
    /// </summary>
    /// <param name="additionalOperations">The operations to add</param>
    /// <returns>A new query plan with the additional operations</returns>
    public QueryPlan WithOperations(IEnumerable<IQueryOperation> additionalOperations)
    {
        if (additionalOperations == null)
            throw new ArgumentNullException(nameof(additionalOperations));

        var allOperations = Operations.Concat(additionalOperations);
        return new QueryPlan(Source, allOperations);
    }

    /// <summary>
    /// Creates a new query plan with a single additional operation
    /// </summary>
    /// <param name="operation">The operation to add</param>
    /// <returns>A new query plan with the additional operation</returns>
    public QueryPlan WithOperation(IQueryOperation operation)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        return WithOperations(new[] { operation });
    }

    /// <summary>
    /// Computes the result schema by applying all operations to the source schema
    /// </summary>
    /// <returns>The computed result schema</returns>
    private Schema ComputeResultSchema()
    {
        var schema = Source.Schema;
        
        foreach (var operation in Operations)
        {
            try
            {
                schema = operation.TransformSchema(schema);
            }
            catch (Exception ex)
            {
                throw new SchemaValidationException(
                    $"Operation '{operation.OperationType}' failed to transform schema: {ex.Message}");
            }
        }

        return schema;
    }

    /// <summary>
    /// Returns a string representation of the query plan
    /// </summary>
    /// <returns>A formatted string describing the query plan</returns>
    public override string ToString()
    {
        var operationNames = Operations.Select(op => op.OperationType);
        var pipeline = string.Join(" -> ", operationNames);
        
        return $"QueryPlan {{ Source: {Source.GetType().Name}, Pipeline: {pipeline}, ResultSchema: {ResultSchema} }}";
    }
}

/// <summary>
/// Provides methods for analyzing and explaining query plans
/// </summary>
internal static class QueryPlanAnalyzer
{
    /// <summary>
    /// Generates a detailed explanation of a query plan
    /// </summary>
    /// <param name="plan">The query plan to explain</param>
    /// <returns>A detailed explanation string</returns>
    public static string Explain(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        var explanation = new System.Text.StringBuilder();
        
        explanation.AppendLine("Query Execution Plan:");
        explanation.AppendLine($"├─ Source: {plan.Source.GetType().Name}");
        explanation.AppendLine($"│  └─ Schema: {plan.Source.Schema}");
        explanation.AppendLine($"│  └─ Lazy: {plan.Source.IsLazy}");
        
        if (plan.Operations.Count > 0)
        {
            explanation.AppendLine("├─ Operations:");
            
            var currentSchema = plan.Source.Schema;
            for (int i = 0; i < plan.Operations.Count; i++)
            {
                var operation = plan.Operations[i];
                var isLast = i == plan.Operations.Count - 1;
                var prefix = isLast ? "└─" : "├─";
                
                explanation.AppendLine($"│  {prefix} {i + 1}. {operation.OperationType}");
                
                try
                {
                    var newSchema = operation.TransformSchema(currentSchema);
                    if (!newSchema.Equals(currentSchema))
                    {
                        explanation.AppendLine($"│  {(isLast ? "   " : "│  ")}└─ Schema: {newSchema}");
                    }
                    currentSchema = newSchema;
                }
                catch (Exception ex)
                {
                    explanation.AppendLine($"│  {(isLast ? "   " : "│  ")}└─ Error: {ex.Message}");
                }
            }
        }
        
        explanation.AppendLine($"└─ Result Schema: {plan.ResultSchema}");
        
        return explanation.ToString();
    }

    /// <summary>
    /// Analyzes a query plan for potential optimization opportunities
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>A list of optimization suggestions</returns>
    public static IReadOnlyList<string> AnalyzeOptimizations(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        var suggestions = new List<string>();

        // Check for filter operations that could be pushed down
        var filterOperations = plan.Operations
            .Select((op, index) => new { Operation = op, Index = index })
            .Where(x => x.Operation.OperationType == "Filter")
            .ToList();

        if (filterOperations.Count > 1)
        {
            suggestions.Add("Multiple filter operations detected - consider combining them for better performance");
        }

        // Check for select operations
        var selectOperations = plan.Operations
            .Where(op => op.OperationType == "Select")
            .ToList();

        if (selectOperations.Count > 1)
        {
            suggestions.Add("Multiple select operations detected - consider combining projections");
        }

        // Check for operations that could benefit from predicate pushdown
        if (plan.Source.IsLazy && filterOperations.Any())
        {
            suggestions.Add("Filter operations on lazy source - predicate pushdown optimization available");
        }

        // Check for unused columns
        var sourceColumns = plan.Source.Schema.ColumnNames.Count;
        var resultColumns = plan.ResultSchema.ColumnNames.Count;
        
        if (resultColumns < sourceColumns && !selectOperations.Any())
        {
            suggestions.Add("Some columns are unused - consider adding explicit column selection for better performance");
        }

        return suggestions;
    }
}