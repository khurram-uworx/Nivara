using Nivara.Expressions;
using Nivara.Query;

namespace Nivara.Optimization;

/// <summary>
/// Optimization rule that eliminates unused columns early in the pipeline
/// </summary>
public sealed class ColumnEliminationRule : OptimizationRule
{
    /// <inheritdoc />
    public override string Name => "Column Elimination";

    /// <inheritdoc />
    public override string Description => "Removes unused columns early in the pipeline to reduce memory usage";

    /// <inheritdoc />
    public override int Priority => 70; // Medium priority

    /// <inheritdoc />
    public override bool CanApply(QueryPlan plan)
    {
        if (plan == null || plan.Operations.Count == 0)
            return false;

        // Check if there are unused columns that can be eliminated
        var usedColumns = AnalyzeColumnUsage(plan);
        var sourceColumns = plan.Source.Schema.ColumnNames.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var unusedColumns = sourceColumns.Except(usedColumns, StringComparer.OrdinalIgnoreCase).ToList();

        // Can apply if there are unused columns and no explicit select at the beginning
        var hasEarlySelect = plan.Operations.Count > 0 && plan.Operations[0] is SelectOperation;

        return unusedColumns.Count > 0 && !hasEarlySelect;
    }

    /// <inheritdoc />
    public override QueryPlan Apply(QueryPlan plan)
    {
        var usedColumns = AnalyzeColumnUsage(plan);
        var operations = plan.Operations.ToList();

        // If we already have a select operation at the beginning, don't modify
        if (operations.Count > 0 && operations[0] is SelectOperation)
        {
            return plan; // Already has column selection
        }

        // Create a select operation with only the used columns
        var usedColumnExpressions = usedColumns
            .OrderBy(col => col, StringComparer.OrdinalIgnoreCase) // Consistent ordering
            .Select(col => ColumnExpressions.Col(col))
            .ToArray();

        if (usedColumnExpressions.Length == 0)
        {
            // If no columns are explicitly used, keep all columns
            return plan;
        }

        var selectOperation = new SelectOperation(usedColumnExpressions);

        // Add the select operation at the beginning
        var optimizedOperations = new List<IQueryOperation> { selectOperation };
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

        // Calculate improvement based on column reduction
        var columnReduction = (sourceColumnCount - usedColumnCount) / (double)sourceColumnCount;

        // Column elimination can provide significant memory savings
        return columnReduction * 35.0; // Up to 35% improvement
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
                case "Filter":
                    // We can't access filter conditions, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "Select":
                    // We can't access select columns, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "GroupBy":
                    // We can't access group by details, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "Sort":
                    // We can't access sort keys, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "Join":
                    // We can't access join keys, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "Projection":
                    // We can't access projection mappings, so assume all source columns are used
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;

                case "Concatenation":
                    // Concatenation operations typically use all columns
                    usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
                    break;
            }
        }

        // Special case: if no operations explicitly reference columns,
        // we need to be conservative and assume all columns are used
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

            case LiteralExpression:
                // Literal expressions don't reference columns
                break;

            default:
                // For unknown expression types, be conservative and don't assume any columns
                break;
        }

        return columns;
    }
}