using Nivara.Exceptions;
using System.Globalization;
using System.Numerics;

namespace Nivara.Expressions;

/// <summary>
/// Node-tree fallback for the fused evaluator: interprets a generic-math expression over a uniform
/// element type in a single pass, JIT-monomorphized per (expression shape, <typeparamref name="T"/>).
/// Used only when the expression-tree-compiled target cannot be built; every leaf column must already
/// share the result element type so no per-element widening (and thus no boxing) is required. Value
/// results are written even at null positions (default(T)) while the caller's null mask marks them.
/// </summary>
internal static class FusedKernel
{
    /// <summary>
    /// Runs the expression over the bound leaves producing a single typed column.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="leaves">The bound leaf columns</param>
    /// <param name="mask">The precomputed null mask, or null when there are no nulls</param>
    /// <returns>A typed column with the evaluation results</returns>
    public static IColumn Evaluate<T>(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves, bool[]? mask)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(leaves);

        var length = leaves.Count == 0 ? 1 : leaves[0].Column.Length;
        var result = new T[length];
        for (int i = 0; i < length; i++)
            result[i] = EvalNode<T>(expression, leaves, i);

        return mask == null
            ? NivaraColumn<T>.Create(result)
            : NivaraColumn<T>.CreateFromSpans(result, mask);
    }

    static T EvalNode<T>(ColumnExpression node, IReadOnlyList<FusedColumnBinding> leaves, int index)
        where T : struct, INumber<T>
    {
        switch (node)
        {
            case ColumnReference columnRef:
                return ReadLeaf<T>(columnRef, leaves, index);

            case LiteralExpression literal:
                return CoerceLiteral<T>(literal.Value);

            case ScalarExpression scalar:
                return ApplyArithmetic(scalar.Operator, EvalNode<T>(scalar.Column, leaves, index), CoerceLiteral<T>(scalar.Scalar));

            case BinaryExpression binary when binary.Operator is not (BinaryOperator.And or BinaryOperator.Or):
                return ApplyArithmetic(binary.Operator, EvalNode<T>(binary.Left, leaves, index), EvalNode<T>(binary.Right, leaves, index));

            default:
                throw new NotSupportedException(
                    $"Expression type {node.GetType().Name} is not supported by the generic node-tree kernel");
        }
    }

    static T ReadLeaf<T>(ColumnReference columnRef, IReadOnlyList<FusedColumnBinding> leaves, int index)
        where T : struct
    {
        foreach (var leaf in leaves)
        {
            if (ReferenceEquals(leaf.Reference, columnRef))
                return ((NivaraColumn<T>)leaf.Column)[index];
        }

        throw new QueryExecutionException($"Column '{columnRef.ColumnName}' not found");
    }

    static T CoerceLiteral<T>(object? value)
        where T : struct, INumber<T>
    {
        if (value == null)
            throw new NotSupportedException("Null literals cannot run through the generic node-tree kernel");

        var converted = Convert.ChangeType(value, typeof(T), CultureInfo.InvariantCulture);
        return T.CreateChecked((T)converted);
    }

    static T ApplyArithmetic<T>(BinaryOperator op, T left, T right)
        where T : struct, INumber<T>
    {
        return op switch
        {
            BinaryOperator.Add => left + right,
            BinaryOperator.Subtract => left - right,
            BinaryOperator.Multiply => left * right,
            BinaryOperator.Divide => left / right,
            BinaryOperator.Modulo => left % right,
            _ => throw new NotSupportedException($"Binary operator {op} is not supported by the generic node-tree kernel")
        };
    }
}
