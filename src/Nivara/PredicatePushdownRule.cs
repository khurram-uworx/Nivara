using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Optimization rule that pushes filter predicates closer to data sources
/// </summary>
public sealed class PredicatePushdownRule : OptimizationRule
{
    /// <inheritdoc />
    public override string Name => "Predicate Pushdown";

    /// <inheritdoc />
    public override string Description => "Moves filter operations closer to data sources to reduce data movement";

    /// <inheritdoc />
    public override int Priority => 100; // High priority - apply early

    /// <inheritdoc />
    public override bool CanApply(QueryPlan plan)
    {
        if (plan == null || plan.Operations.Count < 2)
            return false;

        // Check if there are filter operations that can be moved earlier
        var operations = plan.Operations.ToList();

        for (int i = 1; i < operations.Count; i++)
        {
            if (operations[i] is FilterOperation filter)
            {
                // Check if this filter can be moved before any earlier operation
                for (int j = 0; j < i; j++)
                {
                    if (CanMoveFilterBefore(filter, operations[j], plan.Source.Schema))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /// <inheritdoc />
    public override QueryPlan Apply(QueryPlan plan)
    {
        var operations = plan.Operations.ToList();
        var optimizedOperations = new List<IQueryOperation>();
        var pendingFilters = new List<FilterOperation>();

        // Process operations and move filters as early as possible
        foreach (var operation in operations)
        {
            if (operation is FilterOperation filter)
            {
                pendingFilters.Add(filter);
            }
            else
            {
                // Add applicable filters before this operation
                var applicableFilters = new List<FilterOperation>();
                var remainingFilters = new List<FilterOperation>();

                foreach (var pendingFilter in pendingFilters)
                {
                    if (CanMoveFilterBefore(pendingFilter, operation, plan.Source.Schema))
                    {
                        applicableFilters.Add(pendingFilter);
                    }
                    else
                    {
                        remainingFilters.Add(pendingFilter);
                    }
                }

                // Add applicable filters first, then the operation
                optimizedOperations.AddRange(applicableFilters);
                optimizedOperations.Add(operation);

                // Keep remaining filters for later
                pendingFilters = remainingFilters;
            }
        }

        // Add any remaining filters at the end
        optimizedOperations.AddRange(pendingFilters);

        return new QueryPlan(plan.Source, optimizedOperations);
    }

    /// <inheritdoc />
    protected override double CalculateEstimatedImprovement(QueryPlan originalPlan, QueryPlan optimizedPlan)
    {
        // Count how many filter operations were moved earlier
        var originalFilters = originalPlan.Operations.OfType<FilterOperation>().ToList();
        var optimizedFilters = optimizedPlan.Operations.OfType<FilterOperation>().ToList();

        if (originalFilters.Count == 0)
            return 0.0;

        // Calculate improvement based on how much earlier filters are applied
        double improvement = 0.0;
        var originalPositions = GetFilterPositions(originalPlan.Operations);
        var optimizedPositions = GetFilterPositions(optimizedPlan.Operations);

        foreach (var filter in originalFilters)
        {
            var originalPos = originalPositions.GetValueOrDefault(filter, -1);
            var optimizedPos = optimizedPositions.GetValueOrDefault(filter, -1);

            if (originalPos > optimizedPos && optimizedPos >= 0)
            {
                // Filter moved earlier - estimate improvement based on position change
                var positionImprovement = (originalPos - optimizedPos) / (double)originalPlan.Operations.Count;
                improvement += positionImprovement * 20.0; // Up to 20% improvement per filter moved
            }
        }

        return Math.Min(improvement, 50.0); // Cap at 50% improvement
    }

    /// <summary>
    /// Gets the positions of filter operations in the operation list
    /// </summary>
    /// <param name="operations">The list of operations</param>
    /// <returns>A dictionary mapping filters to their positions</returns>
    private static Dictionary<FilterOperation, int> GetFilterPositions(IReadOnlyList<IQueryOperation> operations)
    {
        var positions = new Dictionary<FilterOperation, int>();

        for (int i = 0; i < operations.Count; i++)
        {
            if (operations[i] is FilterOperation filter)
            {
                positions[filter] = i;
            }
        }

        return positions;
    }

    /// <summary>
    /// Determines if a filter can be moved before a given operation
    /// </summary>
    /// <param name="filter">The filter operation</param>
    /// <param name="operation">The operation to check against</param>
    /// <param name="sourceSchema">The original source schema</param>
    /// <returns>True if the filter can be moved before the operation</returns>
    private static bool CanMoveFilterBefore(FilterOperation filter, IQueryOperation operation, Schema sourceSchema)
    {
        // Get columns referenced by the filter
        var filterColumns = GetReferencedColumns(filter.Condition);

        switch (operation.OperationType)
        {
            case "Select":
                // Filter can be pushed before select if all referenced columns are selected
                // Since we can't access internal properties, we'll be conservative
                return false;

            case "GroupBy":
                // Filter can be pushed before group by if all referenced columns exist in source
                return filterColumns.All(col => sourceSchema.HasColumn(col));

            case "Sort":
                // Filter can be pushed before sort if all referenced columns exist in source
                return filterColumns.All(col => sourceSchema.HasColumn(col));

            case "Join":
                // Be conservative with joins - don't push filters across join boundaries
                return false;

            default:
                // For unknown operations, be conservative
                return false;
        }
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

    /// <summary>
    /// Gets the column names selected by a select operation
    /// </summary>
    /// <param name="select">The select operation</param>
    /// <returns>A set of selected column names</returns>
    private static HashSet<string> GetSelectedColumns(IQueryOperation select)
    {
        // Since we can't access internal properties, return empty set
        // This makes the optimization conservative
        return new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    }
}