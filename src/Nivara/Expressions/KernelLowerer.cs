using Nivara.Helpers;

namespace Nivara.Expressions;

/// <summary>
/// Lowers a validated <see cref="ColumnExpression"/> into a flat post-order
/// <see cref="KernelPlan"/> (issue #167). The IR is the single planning representation all fused
/// backends consume; expression trees are a planning/compilation detail. Promotion mirrors the
/// compiled <c>BuildNode</c> logic via <see cref="NumericPromoter"/>. Literal nodes keep their
/// original value so each backend coerces exactly once.
/// </summary>
internal static class KernelLowerer
{
    /// <summary>
    /// Lowers the expression into a kernel plan.
    /// </summary>
    /// <param name="expression">The validated expression to lower</param>
    /// <param name="plan">The inferred fused plan binding the expression's leaves</param>
    /// <returns>The lowered kernel plan</returns>
    /// <exception cref="NotSupportedException">Thrown when the expression contains a node the IR cannot represent</exception>
    public static KernelPlan Lower(ColumnExpression expression, FusedExpressionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(plan);

        var nodes = new List<KernelNode>();
        var leafIndex = new Dictionary<ColumnReference, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < plan.Columns.Count; i++)
            leafIndex[plan.Columns[i].Reference] = i;

        int LowerNode(ColumnExpression node)
        {
            switch (node)
            {
                case ColumnReference columnRef:
                    {
                        var leaf = leafIndex[columnRef];
                        var nodeIndex = nodes.Count;
                        nodes.Add(new KernelNode(KernelOp.Column, leaf, 0, null, plan.Columns[leaf].Column.ElementType));
                        return nodeIndex;
                    }

                case LiteralExpression literal:
                    {
                        var nodeIndex = nodes.Count;
                        nodes.Add(new KernelNode(KernelOp.Literal, 0, 0, literal.Value, literal.Value!.GetType()));
                        return nodeIndex;
                    }

                case ScalarExpression scalar:
                    return LowerScalar(scalar, nodes, LowerNode);

                case BinaryExpression binary:
                    return LowerBinary(binary, nodes, LowerNode);

                case ComparisonExpression comparison:
                    {
                        var left = LowerNode(comparison.Left);
                        var right = LowerNode(comparison.Right);
                        var nodeIndex = nodes.Count;
                        nodes.Add(new KernelNode(MapComparison(comparison.Operator), left, right, null, typeof(bool)));
                        return nodeIndex;
                    }

                case NotExpression not:
                    {
                        var operand = LowerNode(not.Operand);
                        var nodeIndex = nodes.Count;
                        nodes.Add(new KernelNode(KernelOp.Not, operand, 0, null, typeof(bool)));
                        return nodeIndex;
                    }

                case ConditionalExpression conditional:
                    {
                        var testIdx = LowerNode(conditional.Test);
                        var trueIdx = LowerNode(conditional.TrueValue);
                        var falseIdx = LowerNode(conditional.FalseValue);
                        var nodeIndex = nodes.Count;
                        // Left=test index, Right=trueValue index, Value=falseValue index (boxed int)
                        nodes.Add(new KernelNode(KernelOp.Conditional, testIdx, trueIdx, falseIdx, conditional.ResultType));
                        return nodeIndex;
                    }

                default:
                    throw new NotSupportedException($"Expression type {node.GetType().Name} is not supported by the fused evaluator");
            }
        }

        var root = LowerNode(expression);
        return new KernelPlan(nodes, plan.Columns, plan.ResultType, plan.HasNulls, plan.Signature, plan.IsGenericMath, root);
    }

    /// <summary>
    /// Lowers a scalar (column op literal) node as a binary node over a literal child, so the IR
    /// keeps a single binary shape for every arithmetic operation.
    /// </summary>
    static int LowerScalar(ScalarExpression scalar, List<KernelNode> nodes, Func<ColumnExpression, int> lowerNode)
    {
        var operand = lowerNode(scalar.Column);
        var scalarType = scalar.Scalar!.GetType();
        var promoted = NumericPromoter.GetPromotedType(nodes[operand].ComputeType, scalarType);
        if (promoted == null)
        {
            throw new NotSupportedException(
                $"Scalar operator {scalar.Operator} mixes non-promotable operand types in expression '{scalar.Name}'");
        }

        var literalIndex = nodes.Count;
        nodes.Add(new KernelNode(KernelOp.Literal, 0, 0, scalar.Scalar, scalarType));
        var nodeIndex = nodes.Count;
        nodes.Add(new KernelNode(MapArithmetic(scalar.Operator), operand, literalIndex, null, promoted));
        return nodeIndex;
    }

    static int LowerBinary(BinaryExpression binary, List<KernelNode> nodes, Func<ColumnExpression, int> lowerNode)
    {
        var left = lowerNode(binary.Left);
        var right = lowerNode(binary.Right);

        if (binary.Operator is BinaryOperator.And or BinaryOperator.Or)
        {
            var boolIndex = nodes.Count;
            nodes.Add(new KernelNode(binary.Operator == BinaryOperator.And ? KernelOp.And : KernelOp.Or, left, right, null, typeof(bool)));
            return boolIndex;
        }

        var promoted = NumericPromoter.GetPromotedType(nodes[left].ComputeType, nodes[right].ComputeType);
        if (promoted == null)
        {
            throw new NotSupportedException(
                $"Binary operator {binary.Operator} mixes non-promotable operand types in expression '{binary.Name}'");
        }

        var nodeIndex = nodes.Count;
        nodes.Add(new KernelNode(MapArithmetic(binary.Operator), left, right, null, promoted));
        return nodeIndex;
    }

    static KernelOp MapArithmetic(BinaryOperator op)
    {
        return op switch
        {
            BinaryOperator.Add => KernelOp.Add,
            BinaryOperator.Subtract => KernelOp.Subtract,
            BinaryOperator.Multiply => KernelOp.Multiply,
            BinaryOperator.Divide => KernelOp.Divide,
            BinaryOperator.Modulo => KernelOp.Modulo,
            _ => throw new NotSupportedException($"Binary operator {op} is not supported by the kernel IR")
        };
    }

    static KernelOp MapComparison(ComparisonOperator op)
    {
        return op switch
        {
            ComparisonOperator.Equal => KernelOp.Equal,
            ComparisonOperator.NotEqual => KernelOp.NotEqual,
            ComparisonOperator.GreaterThan => KernelOp.GreaterThan,
            ComparisonOperator.LessThan => KernelOp.LessThan,
            ComparisonOperator.GreaterThanOrEqual => KernelOp.GreaterThanOrEqual,
            ComparisonOperator.LessThanOrEqual => KernelOp.LessThanOrEqual,
            _ => throw new NotSupportedException($"Comparison operator {op} is not supported by the kernel IR")
        };
    }
}
