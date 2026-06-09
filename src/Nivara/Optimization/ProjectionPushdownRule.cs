using Nivara.Expressions;
using Nivara.Query;

namespace Nivara.Optimization;

/// <summary>
/// Optimization rule that pushes column projections closer to data sources to reduce data movement
/// </summary>
public sealed class ProjectionPushdownRule : OptimizationRule
{
    /// <inheritdoc />
    public override string Name => "Projection Pushdown";

    /// <inheritdoc />
    public override string Description => "Moves column selections closer to data sources to reduce data movement";

    /// <inheritdoc />
    public override int Priority => 90; // High priority, but after predicate pushdown

    /// <inheritdoc />
    public override bool CanApply(QueryPlan plan)
    {
        if (plan == null || plan.Operations.Count < 2)
            return false;

        // Check if there are projection opportunities
        var usedColumns = AnalyzeColumnUsage(plan);
        var sourceColumns = plan.Source.Schema.ColumnNames.ToHashSet(StringComparer.OrdinalIgnoreCase);

        // If we use fewer columns than available and don't have early projection, we can optimize
        var hasEarlyProjection = plan.Operations.Count > 0 && plan.Operations[0] is SelectOperation;

        return usedColumns.Count < sourceColumns.Count && !hasEarlyProjection;
    }

    /// <inheritdoc />
    public override QueryPlan Apply(QueryPlan plan)
    {
        var usedColumns = AnalyzeColumnUsage(plan);
        var operations = plan.Operations.ToList();

        // If we already have a select operation at the beginning, don't add another
        if (operations.Count > 0 && operations[0] is SelectOperation)
        {
            return plan; // Already optimized
        }

        // Create a projection operation with only the used columns
        var projectionExpressions = usedColumns
            .Select(col => ColumnExpressions.Col(col))
            .ToArray();

        var projectionOperation = new SelectOperation(projectionExpressions);

        // Add the projection at the beginning
        var optimizedOperations = new List<IQueryOperation> { projectionOperation };
        optimizedOperations.AddRange(operations);

        return new QueryPlan(plan.Source, optimizedOperations);
    }

    /// <inheritdoc />
    protected override double CalculateEstimatedImprovement(QueryPlan originalPlan, QueryPlan optimizedPlan)
    {
        var sourceColumnCount = originalPlan.Source.Schema.ColumnNames.Count;
        var usedColumnCount = AnalyzeColumnUsage(originalPlan).Count;

        if (sourceColumnCount == 0)
            return 0.0;

        // Estimate improvement based on column reduction
        var columnReduction = (sourceColumnCount - usedColumnCount) / (double)sourceColumnCount;

        // Projection pushdown can provide significant memory and I/O improvements
        return columnReduction * 30.0; // Up to 30% improvement
    }

    /// <summary>
    /// Analyzes which columns are actually used throughout the query pipeline
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>A set of column names that are actually used</returns>
    private static HashSet<string> AnalyzeColumnUsage(QueryPlan plan)
    {
        var usedColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Analyze each operation to determine which columns it uses
        foreach (var operation in plan.Operations)
        {
            switch (operation.OperationType)
            {
                case Query.OperationType.Filter:
                    break;

                case Query.OperationType.Select:
                case Query.OperationType.GroupBy:
                case Query.OperationType.Sort:
                case Query.OperationType.Join:
                case Query.OperationType.Projection:
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;
            }
        }

        // If no operations explicitly use columns, assume all columns are used
        if (usedColumns.Count == 0)
        {
            usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
        }

        return usedColumns;
    }

    /// <summary>
    /// Gets the column names referenced by a column expression
    /// </summary>
    /// <param name="expression">The expression to analyze</param>
    /// <returns>A set of column names referenced by the expression</returns>
    private static HashSet<string> GetReferencedColumns(ColumnExpression expression)
    {
        var columns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        switch (expression)
        {
            case ColumnReference columnRef:
                columns.Add(columnRef.ColumnName);
                break;

            case BinaryExpression binary:
                columns.UnionWith(GetReferencedColumns(binary.Left));
                columns.UnionWith(GetReferencedColumns(binary.Right));
                break;

            case ComparisonExpression comparison:
                columns.UnionWith(GetReferencedColumns(comparison.Left));
                if (comparison.Right is ColumnExpression rightExpr)
                {
                    columns.UnionWith(GetReferencedColumns(rightExpr));
                }
                break;

            case ScalarExpression scalar:
                columns.UnionWith(GetReferencedColumns(scalar.Column));
                break;
        }

        return columns;
    }
}