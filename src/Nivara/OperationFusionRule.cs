using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Optimization rule that fuses compatible operations to reduce intermediate allocations
/// </summary>
public sealed class OperationFusionRule : OptimizationRule
{
    /// <inheritdoc />
    public override string Name => "Operation Fusion";

    /// <inheritdoc />
    public override string Description => "Combines compatible operations to reduce intermediate allocations";

    /// <inheritdoc />
    public override int Priority => 80; // Medium priority

    /// <inheritdoc />
    public override bool CanApply(QueryPlan plan)
    {
        if (plan == null || plan.Operations.Count < 2)
            return false;

        var operations = plan.Operations.ToList();

        // Check if any adjacent operations can be fused
        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (CanFuseOperations(operations[i], operations[i + 1]))
            {
                return true;
            }
        }

        return false;
    }

    /// <inheritdoc />
    public override QueryPlan Apply(QueryPlan plan)
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

    /// <inheritdoc />
    protected override double CalculateEstimatedImprovement(QueryPlan originalPlan, QueryPlan optimizedPlan)
    {
        var operationReduction = originalPlan.Operations.Count - optimizedPlan.Operations.Count;

        if (originalPlan.Operations.Count == 0)
            return 0.0;

        // Operation fusion can provide significant performance improvements
        // by reducing intermediate allocations and improving cache locality
        var reductionPercentage = operationReduction / (double)originalPlan.Operations.Count;
        return reductionPercentage * 25.0; // Up to 25% improvement per fused operation
    }

    /// <summary>
    /// Determines if two operations can be fused together
    /// </summary>
    /// <param name="first">The first operation</param>
    /// <param name="second">The second operation</param>
    /// <returns>True if the operations can be fused</returns>
    private static bool CanFuseOperations(IQueryOperation first, IQueryOperation second)
    {
        return (first.OperationType, second.OperationType) switch
        {
            // Multiple filter operations can be fused
            ("Filter", "Filter") => true,

            // Multiple select operations can be fused under certain conditions
            ("Select", "Select") => true, // We'll validate this in FuseOperations

            // Filter followed by select can sometimes be fused
            ("Filter", "Select") => true,

            // Multiple sort operations can be fused
            ("Sort", "Sort") => true,

            _ => false
        };
    }

    /// <summary>
    /// Fuses two compatible operations into a single operation
    /// </summary>
    /// <param name="first">The first operation</param>
    /// <param name="second">The second operation</param>
    /// <returns>A fused operation, or null if fusion is not possible</returns>
    private static IQueryOperation? FuseOperations(IQueryOperation first, IQueryOperation second)
    {
        // For now, return null since we can't access internal operation details
        // A full implementation would need access to the internal operation properties
        // or a different architecture that exposes the necessary information
        return null;
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