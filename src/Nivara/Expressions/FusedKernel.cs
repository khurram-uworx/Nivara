using System.Globalization;
using System.Numerics;

namespace Nivara.Expressions;

/// <summary>
/// Span-kernel target for the fused evaluator (issue #167): interprets a uniform generic-math
/// <see cref="KernelPlan"/> over a uniform element type in a single fused pass over zero-copy leaf
/// memory (the backing data holds <c>default(T)</c> at null positions, so no snapshot is needed),
/// writing values and the OR'd null mask inline — no separate mask pass. JIT-monomorphized per
/// (plan shape, <typeparamref name="T"/>). The interpreter consumes the flat post-order IR with
/// hoisted literals and leaf indices resolved once (no per-element AST walk or dictionary lookups).
/// Primary target for null-bearing uniform generic-math plans and fallback when the
/// expression-tree-compiled target cannot be built. Comparisons, And/Or/Not, and heterogeneous
/// types stay on the compiled path (bool is not generic math, so every bool-result expression runs
/// through the compiled target).
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

        return Evaluate<T>(KernelLowerer.Lower(expression, BuildAdapterPlan<T>(expression, leaves)));
    }

    /// <summary>
    /// Runs a lowered uniform generic-math plan over its leaf columns in a single fused pass.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="plan">The lowered kernel plan</param>
    /// <returns>A typed column with the evaluation results</returns>
    public static IColumn Evaluate<T>(KernelPlan plan)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(plan);

        var length = plan.Columns.Count == 0 ? 1 : plan.Columns[0].Column.Length;

        var inputs = new ReadOnlyMemory<T>[plan.Columns.Count];
        var leafMasks = new ReadOnlyMemory<bool>[plan.Columns.Count];
        var hasNulls = false;
        for (int i = 0; i < plan.Columns.Count; i++)
        {
            var column = (NivaraColumn<T>)plan.Columns[i].Column;
            inputs[i] = column.Storage.Data;
            var mask = column.Storage.NullMaskMemory;
            if (mask.HasValue && mask.Value.Length > 0)
            {
                leafMasks[i] = mask.Value;
                hasNulls = true;
            }
        }

        var output = new T[length];
        var outputMask = hasNulls ? new bool[length] : null;
        Execute<T>(plan, inputs, leafMasks, output, outputMask is null ? default : outputMask.AsSpan());

        return outputMask == null
            ? NivaraColumn<T>.CreateFromOwnedArray(output)
            : NivaraColumn<T>.CreateFromSpans(output, outputMask);
    }

    /// <summary>
    /// Runs the expression over zero-copy leaf memory, writing values and (when present) the fused
    /// OR null mask in a single pass. Masked positions receive <c>default(T)</c>. Adapter over the
    /// IR-based core for the raw memory-based call shape.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="leaves">The bound leaf columns</param>
    /// <param name="inputs">One zero-copy memory per leaf, matching <paramref name="leaves"/> order</param>
    /// <param name="masks">One null mask per leaf (empty when that leaf has no nulls)</param>
    /// <param name="output">The destination span; masked positions receive <c>default(T)</c></param>
    /// <param name="outputMask">The fused OR mask destination, or null when no leaf has nulls</param>
    internal static void Execute<T>(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves, ReadOnlyMemory<T>[] inputs, ReadOnlyMemory<bool>[] masks, Span<T> output, bool[]? outputMask)
        where T : struct, INumber<T>
    {
        var plan = KernelLowerer.Lower(expression, BuildAdapterPlan<T>(expression, leaves));
        Execute<T>(plan, inputs, masks, output, outputMask is null ? default : outputMask.AsSpan());
    }

    /// <summary>
    /// Runs a lowered uniform generic-math plan over zero-copy leaf memory, writing values and (when
    /// present) the fused OR null mask in a single pass. Masked positions receive <c>default(T)</c>.
    /// Callers slice <paramref name="inputs"/>/<paramref name="masks"/>/<paramref name="output"/> per
    /// chunk to execute part of the plan (issue #167).
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="plan">The lowered kernel plan</param>
    /// <param name="inputs">One zero-copy memory per leaf, matching plan leaf order</param>
    /// <param name="masks">One null mask per leaf (empty when that leaf has no nulls)</param>
    /// <param name="output">The destination span; masked positions receive <c>default(T)</c></param>
    /// <param name="outputMask">The fused OR mask destination, or null when no leaf has nulls</param>
    internal static void Execute<T>(KernelPlan plan, ReadOnlyMemory<T>[] inputs, ReadOnlyMemory<bool>[] masks, Span<T> output, Span<bool> outputMask)
        where T : struct, INumber<T>
    {
        var nodes = plan.Nodes;
        var count = nodes.Count;
        var literals = new T[count];
        for (int n = 0; n < count; n++)
        {
            if (nodes[n].Op == KernelOp.Literal)
                literals[n] = CoerceLiteral<T>(nodes[n].Value);
        }

        var hasMask = new bool[masks.Length];
        for (int m = 0; m < masks.Length; m++)
            hasMask[m] = !masks[m].IsEmpty;

        var stack = new T[Math.Max(1, plan.MaxStackDepth)];

        if (outputMask.IsEmpty)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = EvalNodes<T>(nodes, inputs, literals, stack, i);
            }
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
                    outputMask[i] = false;
                    output[i] = EvalNodes<T>(nodes, inputs, literals, stack, i);
                }
            }
        }
    }

    /// <summary>
    /// Evaluates the post-order plan for one element using a stack of pending values.
    /// </summary>
    static T EvalNodes<T>(IReadOnlyList<KernelNode> nodes, ReadOnlyMemory<T>[] inputs, T[] literals, T[] stack, int index)
        where T : struct, INumber<T>
    {
        var sp = 0;
        for (int n = 0; n < nodes.Count; n++)
        {
            switch (nodes[n].Op)
            {
                case KernelOp.Column:
                    stack[sp++] = inputs[nodes[n].Left].Span[index];
                    break;

                case KernelOp.Literal:
                    stack[sp++] = literals[n];
                    break;

                case KernelOp.Add:
                    {
                        var right = stack[--sp];
                        var left = stack[--sp];
                        stack[sp++] = left + right;
                    }
                    break;

                case KernelOp.Subtract:
                    {
                        var right = stack[--sp];
                        var left = stack[--sp];
                        stack[sp++] = left - right;
                    }
                    break;

                case KernelOp.Multiply:
                    {
                        var right = stack[--sp];
                        var left = stack[--sp];
                        stack[sp++] = left * right;
                    }
                    break;

                case KernelOp.Divide:
                    {
                        var right = stack[--sp];
                        var left = stack[--sp];
                        stack[sp++] = left / right;
                    }
                    break;

                case KernelOp.Modulo:
                    {
                        var right = stack[--sp];
                        var left = stack[--sp];
                        stack[sp++] = left % right;
                    }
                    break;

                default:
                    throw new NotSupportedException(
                        $"Kernel op {nodes[n].Op} is not supported by the generic span kernel (bool comparisons and boolean operators run through the compiled target)");
            }
        }

        return stack[0];
    }

    /// <summary>
    /// Builds the minimal fused plan an adapter expression needs for lowering: the caller guarantees
    /// the uniform generic-math domain (<c>where T : struct, INumber&lt;T&gt;</c>).
    /// </summary>
    static FusedExpressionPlan BuildAdapterPlan<T>(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves)
        where T : struct, INumber<T>
    {
        var hasNulls = false;
        foreach (var leaf in leaves)
        {
            if (leaf.Column.HasNulls)
            {
                hasNulls = true;
                break;
            }
        }

        return new FusedExpressionPlan(typeof(T), true, hasNulls, expression.Name, leaves);
    }

    internal static T CoerceLiteral<T>(object? value)
        where T : struct, INumber<T>
    {
        if (value == null)
            throw new NotSupportedException("Null literals cannot run through the generic span kernel");

        var converted = Convert.ChangeType(value, typeof(T), CultureInfo.InvariantCulture);
        return T.CreateChecked((T)converted);
    }
}
