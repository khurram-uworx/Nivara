using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Optimizes query plans by applying various optimization techniques
/// </summary>
internal sealed class QueryOptimizer
{
    /// <summary>
    /// Optimizes a query plan by applying various optimization passes
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    /// <exception cref="ArgumentNullException">Thrown when plan is null</exception>
    public QueryPlan Optimize(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        try
        {
            var optimizedPlan = plan;

            // Apply optimization passes in order
            optimizedPlan = ApplyPredicatePushdown(optimizedPlan);
            optimizedPlan = ApplyOperationFusion(optimizedPlan);
            optimizedPlan = ApplyColumnElimination(optimizedPlan);
            optimizedPlan = ApplyOperationReordering(optimizedPlan);

            return optimizedPlan;
        }
        catch (Exception)
        {
            // If optimization fails, return the original plan
            // Optimization should never break a valid query
            return plan;
        }
    }

    /// <summary>
    /// Applies predicate pushdown optimization to move filter operations closer to data sources
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyPredicatePushdown(QueryPlan plan)
    {
        var operations = plan.Operations.ToList();
        var optimizedOperations = new List<IQueryOperation>();
        var pendingFilters = new List<FilterOperation>();

        // Collect all filter operations and move them as early as possible
        foreach (var operation in operations)
        {
            if (operation is FilterOperation filter)
            {
                pendingFilters.Add(filter);
            }
            else
            {
                // Add all pending filters before this operation if they don't depend on it
                var applicableFilters = new List<FilterOperation>();
                var remainingFilters = new List<FilterOperation>();

                foreach (var pendingFilter in pendingFilters)
                {
                    if (CanApplyFilterBefore(pendingFilter, operation, plan.Source.Schema))
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

    /// <summary>
    /// Determines if a filter can be applied before a given operation
    /// </summary>
    /// <param name="filter">The filter operation</param>
    /// <param name="operation">The operation to check against</param>
    /// <param name="sourceSchema">The original source schema</param>
    /// <returns>True if the filter can be applied before the operation</returns>
    private static bool CanApplyFilterBefore(FilterOperation filter, IQueryOperation operation, Schema sourceSchema)
    {
        // For Select operations, check if the filter references columns that will be eliminated
        if (operation is SelectOperation select)
        {
            var filterColumns = GetReferencedColumns(filter.Condition);
            var selectedColumns = select.Columns.OfType<ColumnReference>().Select(c => c.ColumnName).ToHashSet(StringComparer.OrdinalIgnoreCase);

            // If the select operation includes all columns referenced by the filter, we can push the filter down
            return filterColumns.All(col => selectedColumns.Contains(col) || sourceSchema.HasColumn(col));
        }

        // For GroupBy operations, filters can generally be applied before grouping
        if (operation is GroupByOperation)
        {
            var filterColumns = GetReferencedColumns(filter.Condition);
            return filterColumns.All(col => sourceSchema.HasColumn(col));
        }

        // For other operations, be conservative and don't push down
        return false;
    }

    /// <summary>
    /// Gets the column names referenced by a column expression
    /// </summary>
    /// <param name="expression">The expression to analyze</param>
    /// <returns>A set of column names referenced by the expression</returns>
    private static HashSet<string> GetReferencedColumns(ColumnExpression expression)
    {
        var columns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (expression is ColumnReference columnRef)
        {
            columns.Add(columnRef.ColumnName);
        }
        else if (expression is BinaryExpression binary)
        {
            columns.UnionWith(GetReferencedColumns(binary.Left));
            columns.UnionWith(GetReferencedColumns(binary.Right));
        }
        else if (expression is ComparisonExpression comparison)
        {
            columns.UnionWith(GetReferencedColumns(comparison.Left));
            // Right side might be a literal, so only add if it's a column reference
            if (comparison.Right is ColumnExpression rightExpr)
            {
                columns.UnionWith(GetReferencedColumns(rightExpr));
            }
        }
        else if (expression is ScalarExpression scalar)
        {
            columns.UnionWith(GetReferencedColumns(scalar.Column));
        }

        return columns;
    }

    /// <summary>
    /// Applies operation fusion to combine compatible operations
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyOperationFusion(QueryPlan plan)
    {
        var operations = plan.Operations.ToList();
        var fusedOperations = new List<IQueryOperation>();

        for (int i = 0; i < operations.Count; i++)
        {
            var currentOp = operations[i];

            // Check if we can fuse with the next operation
            if (i + 1 < operations.Count && CanFuseOperations(currentOp, operations[i + 1]))
            {
                var nextOp = operations[i + 1];
                var fusedOp = FuseOperations(currentOp, nextOp);

                if (fusedOp != null)
                {
                    fusedOperations.Add(fusedOp);
                    i++; // Skip the next operation since we fused it
                    continue;
                }
            }

            fusedOperations.Add(currentOp);
        }

        return new QueryPlan(plan.Source, fusedOperations);
    }

    /// <summary>
    /// Fuses two compatible operations into a single operation
    /// </summary>
    /// <param name="first">The first operation</param>
    /// <param name="second">The second operation</param>
    /// <returns>A fused operation, or null if fusion is not possible</returns>
    private static IQueryOperation? FuseOperations(IQueryOperation first, IQueryOperation second)
    {
        // Fuse multiple filter operations
        if (first is FilterOperation filter1 && second is FilterOperation filter2)
        {
            // Create a combined filter condition using AND logic
            var combinedCondition = new BinaryExpression(BinaryOperator.And, filter1.Condition, filter2.Condition);
            return new FilterOperation(combinedCondition);
        }

        // Fuse multiple select operations (take the columns from the second select)
        if (first is SelectOperation select1 && second is SelectOperation select2)
        {
            // The second select operation determines the final projection
            // We need to ensure the columns from select2 are available after select1
            var select1Columns = select1.Columns.OfType<ColumnReference>().Select(c => c.ColumnName).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var select2Columns = select2.Columns.OfType<ColumnReference>().ToList();

            // Only fuse if all columns in select2 are available from select1
            if (select2Columns.All(c => select1Columns.Contains(c.ColumnName)))
            {
                return new SelectOperation(select2.Columns.ToArray());
            }
        }

        return null; // Cannot fuse these operations
    }

    /// <summary>
    /// Applies column elimination to remove unused columns early in the pipeline
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyColumnElimination(QueryPlan plan)
    {
        // Analyze which columns are actually used throughout the pipeline
        var usedColumns = AnalyzeColumnUsage(plan);
        var sourceColumns = plan.Source.Schema.ColumnNames.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var unusedColumns = sourceColumns.Except(usedColumns, StringComparer.OrdinalIgnoreCase).ToList();

        // If there are unused columns and no explicit select operation at the beginning, add one
        if (unusedColumns.Count > 0 && plan.Operations.Count > 0 && plan.Operations[0] is not SelectOperation)
        {
            var usedColumnExpressions = usedColumns.Select(col => ColumnExpressions.Col(col)).ToArray();
            var selectOperation = new SelectOperation(usedColumnExpressions);

            var optimizedOperations = new List<IQueryOperation> { selectOperation };
            optimizedOperations.AddRange(plan.Operations);

            return new QueryPlan(plan.Source, optimizedOperations);
        }

        return plan; // No optimization needed
    }

    /// <summary>
    /// Analyzes which columns are used throughout the query pipeline
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>A set of column names that are actually used</returns>
    private static HashSet<string> AnalyzeColumnUsage(QueryPlan plan)
    {
        var usedColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var operation in plan.Operations)
        {
            if (operation is FilterOperation filter)
            {
                usedColumns.UnionWith(GetReferencedColumns(filter.Condition));
            }
            else if (operation is SelectOperation select)
            {
                foreach (var column in select.Columns)
                {
                    usedColumns.UnionWith(GetReferencedColumns(column));
                }
            }
            else if (operation is GroupByOperation groupBy)
            {
                foreach (var column in groupBy.GroupByColumns)
                {
                    usedColumns.UnionWith(GetReferencedColumns(column));
                }
            }
        }

        // If no operations use columns explicitly, assume all columns are used
        if (usedColumns.Count == 0)
        {
            usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
        }

        return usedColumns;
    }

    /// <summary>
    /// Applies operation reordering when it improves performance without changing semantics
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyOperationReordering(QueryPlan plan)
    {
        var operations = plan.Operations.ToList();
        var reorderedOperations = new List<IQueryOperation>();

        // Group operations by type for reordering
        var filters = new List<FilterOperation>();
        var selects = new List<SelectOperation>();
        var groupBys = new List<GroupByOperation>();
        var others = new List<IQueryOperation>();

        foreach (var operation in operations)
        {
            switch (operation)
            {
                case FilterOperation filter:
                    filters.Add(filter);
                    break;
                case SelectOperation select:
                    selects.Add(select);
                    break;
                case GroupByOperation groupBy:
                    groupBys.Add(groupBy);
                    break;
                default:
                    others.Add(operation);
                    break;
            }
        }

        // Optimal order: Filters first (reduce data early), then Selects (reduce columns), then GroupBy, then others
        // But we need to respect dependencies between operations

        // Add filters first (they're generally safe to move early)
        reorderedOperations.AddRange(filters);

        // Add selects (but be careful about dependencies)
        reorderedOperations.AddRange(selects);

        // Add group by operations
        reorderedOperations.AddRange(groupBys);

        // Add other operations
        reorderedOperations.AddRange(others);

        // Only return the reordered plan if it's different and safe
        if (!AreOperationListsEqual(operations, reorderedOperations) && IsReorderingSafe(operations, reorderedOperations, plan.Source.Schema))
        {
            return new QueryPlan(plan.Source, reorderedOperations);
        }

        return plan; // No safe reordering found
    }

    /// <summary>
    /// Checks if reordering operations is safe (doesn't change semantics)
    /// </summary>
    /// <param name="original">The original operation list</param>
    /// <param name="reordered">The reordered operation list</param>
    /// <param name="sourceSchema">The source schema</param>
    /// <returns>True if reordering is safe</returns>
    private static bool IsReorderingSafe(List<IQueryOperation> original, List<IQueryOperation> reordered, Schema sourceSchema)
    {
        // For now, use a conservative approach - only allow reordering if we have simple operations
        // A full implementation would do dependency analysis

        // Check that we have the same operations
        if (original.Count != reordered.Count)
            return false;

        var originalTypes = original.Select(op => op.OperationType).ToList();
        var reorderedTypes = reordered.Select(op => op.OperationType).ToList();

        // Allow reordering only if we have filters and selects (safe operations)
        var allowedTypes = new[] { "Filter", "Select" };
        return originalTypes.All(t => allowedTypes.Contains(t)) && reorderedTypes.All(t => allowedTypes.Contains(t));
    }

    /// <summary>
    /// Checks if two operation lists are equal
    /// </summary>
    /// <param name="list1">First operation list</param>
    /// <param name="list2">Second operation list</param>
    /// <returns>True if the lists contain the same operations in the same order</returns>
    private static bool AreOperationListsEqual(List<IQueryOperation> list1, List<IQueryOperation> list2)
    {
        if (list1.Count != list2.Count)
            return false;

        for (int i = 0; i < list1.Count; i++)
        {
            if (!ReferenceEquals(list1[i], list2[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Determines if two operations can be fused together
    /// </summary>
    /// <param name="first">The first operation</param>
    /// <param name="second">The second operation</param>
    /// <returns>True if the operations can be fused, false otherwise</returns>
    private static bool CanFuseOperations(IQueryOperation first, IQueryOperation second)
    {
        // Can fuse multiple filter operations
        if (first is FilterOperation && second is FilterOperation)
            return true;

        // Can fuse multiple select operations under certain conditions
        if (first is SelectOperation && second is SelectOperation)
            return true;

        return false;
    }

    /// <summary>
    /// Analyzes a query plan and provides optimization suggestions
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>A list of optimization suggestions</returns>
    public static IReadOnlyList<string> AnalyzeOptimizationOpportunities(QueryPlan plan)
    {
        if (plan == null)
            return Array.Empty<string>();

        var suggestions = new List<string>();

        // Check for multiple filter operations
        var filterCount = plan.Operations.Count(op => op.OperationType == "Filter");
        if (filterCount > 1)
        {
            suggestions.Add($"Found {filterCount} filter operations - consider combining them for better performance");
        }

        // Check for multiple select operations
        var selectCount = plan.Operations.Count(op => op.OperationType == "Select");
        if (selectCount > 1)
        {
            suggestions.Add($"Found {selectCount} select operations - consider combining projections");
        }

        // Check for predicate pushdown opportunities
        if (plan.Source.IsLazy && filterCount > 0)
        {
            suggestions.Add("Filter operations on lazy source detected - predicate pushdown optimization available");
        }

        // Check for column elimination opportunities
        var sourceColumnCount = plan.Source.Schema.ColumnNames.Count;
        var resultColumnCount = plan.ResultSchema.ColumnNames.Count;

        if (resultColumnCount < sourceColumnCount && selectCount == 0)
        {
            suggestions.Add($"Query uses {resultColumnCount} of {sourceColumnCount} columns - consider adding explicit column selection");
        }

        return suggestions;
    }
}