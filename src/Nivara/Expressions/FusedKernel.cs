using System.Globalization;
using System.Numerics;

namespace Nivara.Expressions;

/// <summary>
/// Span-kernel target for the fused evaluator: interprets a generic-math expression over a uniform
/// element type in a single fused pass over zero-copy leaf memory (the backing data holds
/// <c>default(T)</c> at null positions, so no snapshot is needed), writing values and the OR'd null
/// mask inline — no separate mask pass. JIT-monomorphized per (expression shape, <typeparamref name="T"/>).
/// Primary target for null-bearing uniform generic-math plans and fallback when the
/// expression-tree-compiled target cannot be built. Comparisons, And/Or/Not, and heterogeneous types
/// stay on the compiled path (bool is not generic math, so every bool-result expression runs through
/// the compiled target).
/// </summary>
internal static class FusedKernel
{
    /// <summary>
    /// Runs the expression over the bound leaves producing a single typed column. The null mask is
    /// computed inline from the leaf masks (OR semantics) in the same pass.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="leaves">The bound leaf columns</param>
    /// <returns>A typed column with the evaluation results</returns>
    public static IColumn Evaluate<T>(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(leaves);

        var length = leaves.Count == 0 ? 1 : leaves[0].Column.Length;

        var inputs = new ReadOnlyMemory<T>[leaves.Count];
        var leafMasks = new ReadOnlyMemory<bool>[leaves.Count];
        var hasNulls = false;
        for (int i = 0; i < leaves.Count; i++)
        {
            var column = (NivaraColumn<T>)leaves[i].Column;
            inputs[i] = column.Storage.Data;
            var mask = column.Storage.NullMaskMemory;
            leafMasks[i] = mask ?? default;
            if (mask.HasValue && mask.Value.Length > 0)
                hasNulls = true;
        }

        var output = new T[length];
        var outputMask = hasNulls ? new bool[length] : null;
        Execute<T>(expression, leaves, inputs, leafMasks, output, outputMask);

        return outputMask == null
            ? NivaraColumn<T>.Create(output)
            : NivaraColumn<T>.CreateFromSpans(output, outputMask);
    }

    /// <summary>
    /// Runs the expression over zero-copy leaf memory, writing values and (when present) the fused
    /// OR null mask in a single pass. Masked positions receive <c>default(T)</c>.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="leaves">The bound leaf columns (used only to resolve column references)</param>
    /// <param name="inputs">One zero-copy memory per leaf, matching <paramref name="leaves"/> order</param>
    /// <param name="masks">One null mask per leaf (empty when that leaf has no nulls)</param>
    /// <param name="output">The destination span; masked positions receive <c>default(T)</c></param>
    /// <param name="outputMask">The fused OR mask destination, or null when no leaf has nulls</param>
    internal static void Execute<T>(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves, ReadOnlyMemory<T>[] inputs, ReadOnlyMemory<bool>[] masks, Span<T> output, bool[]? outputMask)
        where T : struct, INumber<T>
    {
        var leafIndex = new Dictionary<ColumnReference, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < leaves.Count; i++)
            leafIndex[leaves[i].Reference] = i;

        var hasMask = new bool[masks.Length];
        for (int m = 0; m < masks.Length; m++)
            hasMask[m] = masks[m].Span.Length > 0;

        if (outputMask == null)
        {
            for (int i = 0; i < output.Length; i++)
                output[i] = EvalNode<T>(expression, leafIndex, inputs, i);
        }
        else
        {
            for (int i = 0; i < output.Length; i++)
            {
                var isNull = false;
                for (int m = 0; m < masks.Length; m++)
                {
                    if (hasMask[m] && masks[m].Span[i])
                    {
                        isNull = true;
                        break;
                    }
                }

                if (isNull)
                {
                    outputMask[i] = true;
                    output[i] = default;
                }
                else
                {
                    output[i] = EvalNode<T>(expression, leafIndex, inputs, i);
                }
            }
        }
    }

    static T EvalNode<T>(ColumnExpression node, IReadOnlyDictionary<ColumnReference, int> leafIndex, ReadOnlyMemory<T>[] inputs, int index)
        where T : struct, INumber<T>
    {
        switch (node)
        {
            case ColumnReference columnRef:
                return inputs[leafIndex[columnRef]].Span[index];

            case LiteralExpression literal:
                return CoerceLiteral<T>(literal.Value);

            case ScalarExpression scalar:
                return ApplyArithmetic(scalar.Operator, EvalNode<T>(scalar.Column, leafIndex, inputs, index), CoerceLiteral<T>(scalar.Scalar));

            case BinaryExpression binary when binary.Operator is not (BinaryOperator.And or BinaryOperator.Or):
                return ApplyArithmetic(binary.Operator, EvalNode<T>(binary.Left, leafIndex, inputs, index), EvalNode<T>(binary.Right, leafIndex, inputs, index));

            default:
                throw new NotSupportedException(
                    $"Expression type {node.GetType().Name} is not supported by the generic span kernel (bool comparisons and boolean operators run through the compiled target)");
        }
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
            _ => throw new NotSupportedException($"Binary operator {op} is not supported by the generic span kernel")
        };
    }
}
