using Nivara.Helpers;
using System.Numerics;

namespace Nivara.Expressions;

/// <summary>
/// TensorPrimitives single-op backend for fused expression evaluation (issue #167): a uniform
/// numeric, null-free plan that is a single element-wise Add/Subtract/Multiply/Divide over leaf
/// columns and literals dispatches directly to the SIMD-vectorized <see cref="System.Numerics.Tensors.TensorPrimitives"/>
/// overloads — one fused call instead of the compiled per-element loop. Scalar-first Subtract and
/// Divide use the manual scalar-first <see cref="INumber{T}"/> kernels (no generic TensorPrimitives
/// scalar-first overload in the pinned BCL version). Works on spans, so chunked execution slices the
/// leaf and destination spans per chunk.
/// </summary>
internal static class TensorPrimitivesKernel
{
    /// <summary>
    /// Attempts to evaluate the plan with the TensorPrimitives single-op backend. Requires a
    /// dispatchable candidate: single Add/Subtract/Multiply/Divide with at least one leaf child and
    /// no null masks.
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="plan">The lowered kernel plan</param>
    /// <param name="leaves">The leaf columns in plan order</param>
    /// <param name="length">The number of elements to evaluate</param>
    /// <param name="result">The typed result column when dispatchable</param>
    /// <returns>True when the plan was dispatched to TensorPrimitives</returns>
    public static bool TryEvaluate<T>(KernelPlan plan, IReadOnlyList<NivaraColumn<T>> leaves, int length, out IColumn result)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(plan);

        if (!IsDispatchable(plan) || plan.HasNulls)
        {
            result = null!;
            return false;
        }

        var root = plan.Nodes[plan.RootNode];
        var leftNode = plan.Nodes[root.Left];
        var rightNode = plan.Nodes[root.Right];

        var leftIsColumn = leftNode.Op == KernelOp.Column;
        var rightIsColumn = rightNode.Op == KernelOp.Column;
        var leftScalar = leftIsColumn ? default : FusedKernel.CoerceLiteral<T>(leftNode.Value);
        var rightScalar = rightIsColumn ? default : FusedKernel.CoerceLiteral<T>(rightNode.Value);

        var inputs = new ReadOnlyMemory<T>[leaves.Count];
        for (int i = 0; i < leaves.Count; i++)
            inputs[i] = leaves[i].Storage.Data;

        var output = new T[length];
        RunChunk<T>(plan, inputs, output, 0, length);

        result = NivaraColumn<T>.CreateFromOwnedArray(output);
        return true;
    }

    /// <summary>
    /// Evaluates the plan in row-batches of <paramref name="chunkSize"/> into one shared output array:
    /// each chunk slices the zero-copy leaf memory and the destination span, so chunked results are
    /// bit-identical to whole-column evaluation (issue #167).
    /// </summary>
    /// <typeparam name="T">The uniform element type, constrained to generic math</typeparam>
    /// <param name="plan">The lowered kernel plan</param>
    /// <param name="leaves">The leaf columns in plan order</param>
    /// <param name="length">The number of elements to evaluate</param>
    /// <param name="chunkSize">The row-batch size</param>
    /// <param name="result">The typed result column when dispatchable</param>
    /// <returns>True when the plan was dispatched to TensorPrimitives</returns>
    public static bool TryEvaluateChunked<T>(KernelPlan plan, IReadOnlyList<NivaraColumn<T>> leaves, int length, int chunkSize, out IColumn result)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(plan);

        if (!IsDispatchable(plan) || plan.HasNulls)
        {
            result = null!;
            return false;
        }

        var inputs = new ReadOnlyMemory<T>[leaves.Count];
        for (int i = 0; i < leaves.Count; i++)
            inputs[i] = leaves[i].Storage.Data;

        var output = new T[length];
        for (var start = 0; start < length; start += chunkSize)
        {
            var count = Math.Min(chunkSize, length - start);
            RunChunk<T>(plan, inputs, output.AsSpan(start, count), start, count);
        }

        result = NivaraColumn<T>.CreateFromOwnedArray(output);
        return true;
    }

    /// <summary>
    /// Resolves the plan shape once per chunk and runs the single binary op over the sliced leaf data
    /// spans and literal scalars.
    /// </summary>
    static void RunChunk<T>(KernelPlan plan, ReadOnlyMemory<T>[] inputs, Span<T> destination, int start, int count)
        where T : struct, INumber<T>
    {
        var root = plan.Nodes[plan.RootNode];
        var leftNode = plan.Nodes[root.Left];
        var rightNode = plan.Nodes[root.Right];

        var leftIsColumn = leftNode.Op == KernelOp.Column;
        var rightIsColumn = rightNode.Op == KernelOp.Column;
        var leftScalar = leftIsColumn ? default : FusedKernel.CoerceLiteral<T>(leftNode.Value);
        var rightScalar = rightIsColumn ? default : FusedKernel.CoerceLiteral<T>(rightNode.Value);
        var leftData = leftIsColumn ? inputs[leftNode.Left].Slice(start, count).Span : default;
        var rightData = rightIsColumn ? inputs[rightNode.Left].Slice(start, count).Span : default;

        Execute<T>(root.Op, leftIsColumn, rightIsColumn, leftScalar, rightScalar, leftData, rightData, destination);
    }

    /// <summary>
    /// Runs the single binary op over leaf data spans and literal scalars into
    /// <paramref name="destination"/>. Slice the leaf data spans and the destination per chunk for
    /// chunked execution.
    /// </summary>
    internal static void Execute<T>(KernelOp op, bool leftIsColumn, bool rightIsColumn, T leftScalar, T rightScalar, ReadOnlySpan<T> leftData, ReadOnlySpan<T> rightData, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (leftIsColumn && rightIsColumn)
        {
            switch (op)
            {
                case KernelOp.Add:
                    NumericTensorKernels<T>.Add(leftData, rightData, destination);
                    break;
                case KernelOp.Subtract:
                    NumericTensorKernels<T>.Subtract(leftData, rightData, destination);
                    break;
                case KernelOp.Multiply:
                    NumericTensorKernels<T>.Multiply(leftData, rightData, destination);
                    break;
                case KernelOp.Divide:
                    NumericTensorKernels<T>.Divide(leftData, rightData, destination);
                    break;
                default:
                    throw NotSupported(op);
            }
        }
        else if (leftIsColumn)
        {
            switch (op)
            {
                case KernelOp.Add:
                    NumericTensorKernels<T>.Add(leftData, rightScalar, destination);
                    break;
                case KernelOp.Subtract:
                    NumericTensorKernels<T>.Subtract(leftData, rightScalar, destination);
                    break;
                case KernelOp.Multiply:
                    NumericTensorKernels<T>.Multiply(leftData, rightScalar, destination);
                    break;
                case KernelOp.Divide:
                    NumericTensorKernels<T>.Divide(leftData, rightScalar, destination);
                    break;
                default:
                    throw NotSupported(op);
            }
        }
        else
        {
            switch (op)
            {
                case KernelOp.Add:
                    NumericTensorKernels<T>.Add(rightData, leftScalar, destination);
                    break;
                case KernelOp.Subtract:
                    NumericTensorKernels<T>.SubtractFrom(leftScalar, rightData, destination);
                    break;
                case KernelOp.Multiply:
                    NumericTensorKernels<T>.Multiply(rightData, leftScalar, destination);
                    break;
                case KernelOp.Divide:
                    NumericTensorKernels<T>.DivideBy(leftScalar, rightData, destination);
                    break;
                default:
                    throw NotSupported(op);
            }
        }
    }

    /// <summary>
    /// Gets whether the plan is a single Add/Subtract/Multiply/Divide binary over at least one leaf
    /// child (literal-only plans stay on the compiled path).
    /// </summary>
    internal static bool IsDispatchable(KernelPlan plan)
    {
        if (!plan.IsTensorPrimitivesCandidate || plan.HasNulls)
            return false;

        var root = plan.Nodes[plan.RootNode];
        var leftNode = plan.Nodes[root.Left];
        var rightNode = plan.Nodes[root.Right];
        return leftNode.Op == KernelOp.Column || rightNode.Op == KernelOp.Column;
    }

    static NotSupportedException NotSupported(KernelOp op)
        => new($"Kernel op {op} is not supported by the TensorPrimitives single-op backend");
}
