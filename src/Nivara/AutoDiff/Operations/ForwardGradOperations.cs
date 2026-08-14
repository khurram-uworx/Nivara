using Nivara.Helpers;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Forward-mode automatic differentiation operations.
/// Each method computes the primal (forward value) and the tangent (directional derivative
/// via JVP — Jacobian-Vector Product) for a seeded tangent direction.
/// Mirrors <see cref="ReverseGradOperations"/> in structure and convention.
/// </summary>
public static class ForwardGradOperations
{
    #region Element-wise Operations

    /// <summary>
    /// Adds two tensors element-wise.
    /// JVP: t_out = t_a + t_b
    /// </summary>
    public static ForwardGradTensor<T> Add<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot add tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primal = a.Data + b.Data;
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = b.Tangent;
            else if (b.Tangent == null)
                tangent = a.Tangent;
            else
                tangent = a.Tangent + b.Tangent;
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Subtracts two tensors element-wise.
    /// JVP: t_out = t_a - t_b
    /// </summary>
    public static ForwardGradTensor<T> Subtract<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primalArr = new T[a.Length];
        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        TensorPrimitives.Subtract(aSpan, bSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
            {
                b.Tangent!.TryGetSpan(out var bTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Negate(bTanSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (b.Tangent == null)
                tangent = a.Tangent;
            else
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                b.Tangent.TryGetSpan(out var bTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Subtract(aTanSpan, bTanSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Multiplies two tensors element-wise.
    /// JVP: t_out = t_a * b + a * t_b
    /// </summary>
    public static ForwardGradTensor<T> Multiply<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot multiply tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primal = a.Data * b.Data;
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = a.Data * b.Tangent!;
            else if (b.Tangent == null)
                tangent = a.Tangent * b.Data;
            else
                tangent = a.Tangent * b.Data + a.Data * b.Tangent;
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Divides two tensors element-wise.
    /// JVP: t_out = (t_a - result * t_b) / b
    /// </summary>
    public static ForwardGradTensor<T> Divide<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot divide tensors with different lengths: {a.Length} vs {b.Length}");
        }

        for (int i = 0; i < b.Length; i++)
        {
            if (b[i] == T.Zero)
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var primalArr = new T[a.Length];
        TensorPrimitives.Divide(aSpan, bSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
            {
                b.Tangent!.TryGetSpan(out var bTanSpan);
                var numArr = new T[a.Length];
                TensorPrimitives.Multiply(primalArr, bTanSpan, numArr);
                TensorPrimitives.Negate(numArr, numArr);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(numArr, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (b.Tangent == null)
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(aTanSpan, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                b.Tangent.TryGetSpan(out var bTanSpan);
                var numArr = new T[a.Length];
                TensorPrimitives.Multiply(primalArr, bTanSpan, numArr);
                TensorPrimitives.Subtract(aTanSpan, numArr, numArr);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(numArr, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Divides a tensor by a scalar: result[i] = a[i] / scalar.
    /// The scalar is not wrapped in a tensor. Mirrors
    /// <see cref="ReverseGradOperations.DivideScalar{T}(ReverseGradTensor{T}, T)"/>.
    /// JVP: t_out = t_a / scalar
    /// </summary>
    public static ForwardGradTensor<T> DivideScalar<T>(ForwardGradTensor<T> a, T scalar)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (scalar == T.Zero)
            throw new DivideByZeroException($"Division by zero with scalar divisor");

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        TensorPrimitives.Divide(aSpan, scalar, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent)
        {
            a.Tangent!.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            TensorPrimitives.Divide(aTanSpan, scalar, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Multiplies two matrices: result = a @ b.
    /// JVP: t_out = t_a @ B + A @ t_b
    /// </summary>
    public static ForwardGradTensor<T> MatMul<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Rank != 2)
            throw new ArgumentException($"Left operand must be a matrix (rank 2), got rank {a.Rank}", nameof(a));
        if (b.Rank != 2)
            throw new ArgumentException($"Right operand must be a matrix (rank 2), got rank {b.Rank}", nameof(b));

        var aRows = a.shape[0];
        var aCols = a.shape[1];
        var bRows = b.shape[0];
        var bCols = b.shape[1];

        if (aCols != bRows)
            throw new ArgumentException(
                $"Matrix dimensions incompatible: a({aRows}x{aCols}) @ b({bRows}x{bCols}). " +
                $"a's column count ({aCols}) must equal b's row count ({bRows}).");

        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var primalArr = new T[aRows * bCols];
        GradKernels.MatMul(aSpan, bSpan, primalArr, aRows, aCols, bCols);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        var resultShape = new[] { aRows, bCols };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            var aTan = a.Tangent;
            var bTan = b.Tangent;

            if (aTan != null && bTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                var tAB = new T[aRows * bCols];
                GradKernels.MatMul(aTanSpan, bSpan, tAB, aRows, aCols, bCols);
                bTan.TryGetSpan(out var bTanSpan);
                var aT_B = new T[aRows * bCols];
                GradKernels.MatMul(aSpan, bTanSpan, aT_B, aRows, aCols, bCols);
                var tanArr = new T[aRows * bCols];
                TensorPrimitives.Add(tAB, aT_B, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (aTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMul(aTanSpan, bSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (bTan != null)
            {
                bTan.TryGetSpan(out var bTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMul(aSpan, bTanSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    /// <summary>
    /// Transposes a matrix.
    /// JVP: t_out = Transpose(t_a)
    /// </summary>
    public static ForwardGradTensor<T> Transpose<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Rank != 2)
            throw new ArgumentException($"Transpose requires a matrix (rank 2), got rank {a.Rank}", nameof(a));

        var rows = a.shape[0];
        var cols = a.shape[1];
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Transpose(aSpan, primalArr, rows, cols);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        var resultShape = new[] { cols, rows };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.Transpose(aTanSpan, tanArr, rows, cols);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    /// <summary>
    /// Multiplies a matrix by the transpose of another: result = a @ b^T,
    /// where b is [bCols, aCols] and the result is [aRows, bCols].
    /// JVP: t_out = t_a @ B^T + A @ t_b^T
    /// </summary>
    public static ForwardGradTensor<T> MatMulTransposedB<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMulTransposedB requires rank-2 operands.");

        var aRows = a.shape[0];
        var aCols = a.shape[1];
        var bCols = b.shape[0];
        if (b.shape[1] != aCols)
            throw new ArgumentException(
                $"MatMulTransposedB dimension mismatch: a is {aRows}x{aCols}, b is {bCols}x{b.shape[1]}. " +
                $"b's column count ({b.shape[1]}) must equal a's column count ({aCols}).");

        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var primalArr = new T[aRows * bCols];
        GradKernels.MatMulTransposedB(aSpan, bSpan, primalArr, aRows, aCols, bCols);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        var resultShape = new[] { aRows, bCols };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            var aTan = a.Tangent;
            var bTan = b.Tangent;

            if (aTan != null && bTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                bTan.TryGetSpan(out var bTanSpan);
                var tAB = new T[aRows * bCols];
                GradKernels.MatMulTransposedB(aTanSpan, bSpan, tAB, aRows, aCols, bCols);
                var aTB = new T[aRows * bCols];
                GradKernels.MatMulTransposedB(aSpan, bTanSpan, aTB, aRows, aCols, bCols);
                var tanArr = new T[aRows * bCols];
                TensorPrimitives.Add(tAB, aTB, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (aTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMulTransposedB(aTanSpan, bSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (bTan != null)
            {
                bTan.TryGetSpan(out var bTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMulTransposedB(aSpan, bTanSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    /// <summary>
    /// Transposes a rank-2 or rank-3 tensor by swapping two axes.
    /// JVP: t_out = TransposeAxes(t_a)
    /// </summary>
    public static ForwardGradTensor<T> TransposeAxes<T>(ForwardGradTensor<T> a, int axis1, int axis2)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.Rank < 2 || a.Rank > 3)
            throw new ArgumentException($"TransposeAxes supports rank 2–3, got rank {a.Rank}", nameof(a));
        if (axis1 < 0 || axis1 >= a.Rank) throw new ArgumentOutOfRangeException(nameof(axis1));
        if (axis2 < 0 || axis2 >= a.Rank) throw new ArgumentOutOfRangeException(nameof(axis2));
        if (axis1 == axis2) throw new ArgumentException("axis1 and axis2 must differ");

        var srcDims = a.shape;
        var dstDims = (int[])srcDims.Clone();
        (dstDims[axis1], dstDims[axis2]) = (dstDims[axis2], dstDims[axis1]);

        var srcData = new T[a.Length];
        a.Data.TryGetSpan(out var srcSpan);
        srcSpan.CopyTo(srcData);
        var dstData = TransposeAxesData(srcData, srcDims, dstDims, axis1, axis2);
        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(dstData);

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var tanSpan);
            var tanData = new T[a.Length];
            tanSpan.CopyTo(tanData);
            var tanDst = TransposeAxesData(tanData, srcDims, dstDims, axis1, axis2);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanDst);
        }

        return new ForwardGradTensor<T>(resultCol, tangent, dstDims);
    }

    #endregion

    #region Selection Operations

    /// <summary>
    /// Extracts a contiguous slice from a 1D or row-vector tensor.
    /// Input shape: [1, n] or [n]; output shape: [1, length] or [length].
    /// JVP: t_out = Slice(t_a, start, length)
    /// </summary>
    public static ForwardGradTensor<T> Slice<T>(ForwardGradTensor<T> a, int start, int length)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (start < 0) throw new ArgumentOutOfRangeException(nameof(start));
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        int fullDim = a.shape.Length == 2 ? a.shape[1] : a.Length;
        int batchDim = a.shape.Length == 2 ? a.shape[0] : 1;

        if (start + length > a.Length)
            throw new ArgumentException($"Slice ({start}..{start + length}) exceeds tensor length {a.Length}");
        if (start + length > fullDim)
            throw new ArgumentException($"Slice ({start}..{start + length}) exceeds dimension size {fullDim}");

        int resultLen = batchDim * length;
        var resultValues = new T[resultLen];
        var srcData = new T[a.Length];
        a.Data.CopyTo(srcData, default(T)!);
        for (int r = 0; r < batchDim; r++)
            Array.Copy(srcData, r * fullDim + start, resultValues, r * length, length);
        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultValues);
        var resultShape = batchDim == 1
            ? new[] { length }
            : new[] { batchDim, length };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var tanSpan);
            var tanValues = new T[resultLen];
            for (int r = 0; r < batchDim; r++)
                tanSpan.Slice(r * fullDim + start, length).CopyTo(tanValues.AsSpan(r * length));
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanValues);
        }

        return new ForwardGradTensor<T>(resultCol, tangent, resultShape);
    }

    /// <summary>
    /// Concatenates 1D or 2D tensors along an axis. For 2D input, axis 0 joins rows and
    /// axis 1 joins columns.
    /// JVP: t_out = Concat(t_a_i), where inputs without a tangent contribute zeros.
    /// </summary>
    public static ForwardGradTensor<T> Concat<T>(ForwardGradTensor<T>[] tensors, int axis = 0)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor is required for Concat.", nameof(tensors));
        if (tensors.Length == 1)
            return tensors[0];

        int rank = tensors[0].Rank;
        if (rank < 1 || rank > 2)
            throw new ArgumentException($"Concat supports 1D or 2D tensors, got rank {rank}.");

        for (int i = 1; i < tensors.Length; i++)
        {
            if (tensors[i].Rank != rank)
                throw new ArgumentException(
                    $"All tensors must have the same rank. Tensor 0 has rank {rank}, tensor {i} has rank {tensors[i].Rank}.");

            if (rank == 2 && axis == 1 && tensors[i].shape[0] != tensors[0].shape[0])
                throw new ArgumentException(
                    $"For axis=1 concatenation, all tensors must have the same number of rows. " +
                    $"Tensor 0 has {tensors[0].shape[0]} rows, tensor {i} has {tensors[i].shape[0]} rows.");

            if (rank == 2 && axis == 0 && tensors[i].shape[1] != tensors[0].shape[1])
                throw new ArgumentException(
                    $"For axis=0 concatenation, all tensors must have the same number of columns. " +
                    $"Tensor 0 has {tensors[0].shape[1]} columns, tensor {i} has {tensors[i].shape[1]} columns.");
        }

        var shapes = new int[tensors.Length][];
        for (int i = 0; i < tensors.Length; i++)
            shapes[i] = tensors[i].shape;

        var dataCols = new NivaraColumn<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            dataCols[i] = tensors[i].Data;

        var resultData = ConcatColumns(dataCols, shapes, axis, rank);
        var resultShape = rank == 1
            ? new[] { resultData.Length }
            : axis == 0
                ? new[] { tensors.Sum(t => t.shape[0]), tensors[0].shape[1] }
                : new[] { tensors[0].shape[0], tensors.Sum(t => t.shape[1]) };
        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultData);

        NivaraColumn<T>? tangent = null;
        if (tensors.Any(t => t.RequiresTangent))
        {
            var tanCols = new NivaraColumn<T>[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
            {
                var tan = tensors[i].Tangent;
                if (tan != null)
                    tanCols[i] = tan;
                else
                    tanCols[i] = NivaraColumn<T>.Create(new T[tensors[i].Length]);
            }
            var tanData = ConcatColumns(tanCols, shapes, axis, rank);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanData);
        }

        return new ForwardGradTensor<T>(resultCol, tangent, resultShape);
    }

    /// <summary>
    /// Selects rows from a source tensor by integer index along axis 0.
    /// source shape: [N, ...], indices length: L → result shape: [L, ...].
    /// JVP: t_out = Gather(t_source, indices, axis)
    /// </summary>
    public static ForwardGradTensor<T> Gather<T>(ForwardGradTensor<T> source, int[] indices, int axis = 0)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (axis != 0) throw new ArgumentOutOfRangeException(nameof(axis), "Only axis 0 is currently supported.");
        if (indices.Length == 0)
            return new ForwardGradTensor<T>(
                NivaraColumn<T>.Create(Array.Empty<T>()),
                tangent: null,
                new[] { 0 });

        int sourceRowCount = source.shape[0];
        int stride = source.Length / sourceRowCount;

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= sourceRowCount)
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Index at position {i} is {indices[i]}, must be in range [0, {sourceRowCount}).");
        }

        int resultLen = indices.Length * stride;
        var resultValues = new T[resultLen];

        source.Data.TryGetSpan(out var span);
        for (int i = 0; i < indices.Length; i++)
        {
            int srcOffset = indices[i] * stride;
            int dstOffset = i * stride;
            span.Slice(srcOffset, stride).CopyTo(resultValues.AsSpan(dstOffset, stride));
        }

        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultValues);

        var resultShape = new int[source.shape.Length];
        resultShape[0] = indices.Length;
        for (int d = 1; d < source.shape.Length; d++)
            resultShape[d] = source.shape[d];

        NivaraColumn<T>? tangent = null;
        if (source.RequiresTangent && source.Tangent != null)
        {
            source.Tangent.TryGetSpan(out var tanSpan);
            var tanValues = new T[resultLen];
            for (int i = 0; i < indices.Length; i++)
            {
                int srcOffset = indices[i] * stride;
                int dstOffset = i * stride;
                tanSpan.Slice(srcOffset, stride).CopyTo(tanValues.AsSpan(dstOffset, stride));
            }
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanValues);
        }

        return new ForwardGradTensor<T>(resultCol, tangent, resultShape);
    }

    /// <summary>
    /// Embedding-bag: sums the rows of a [numEmbeddings, embeddingDim] weight tensor
    /// selected by 2D integer indices, producing [batchSize, embeddingDim]. Positions
    /// equal to <paramref name="paddingIndex"/> are skipped. Indices are not differentiable;
    /// JVP: t_out = SparseEmbeddingBag(t_weight, indices, paddingIndex).
    /// </summary>
    public static ForwardGradTensor<T> SparseEmbeddingBag<T>(
        ForwardGradTensor<T> weight,
        ForwardGradTensor<T> indices,
        int paddingIndex = -1)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (weight.Rank != 2)
            throw new ArgumentException("SparseEmbeddingBag weight must be a 2D tensor.", nameof(weight));
        if (indices.Rank != 2)
            throw new ArgumentException("SparseEmbeddingBag indices must be a 2D tensor.", nameof(indices));

        int numEmbeddings = weight.shape[0];
        int embeddingDim = weight.shape[1];
        int batchSize = indices.shape[0];
        int maxActiveFeatures = indices.shape[1];

        var parsedIndices = new int[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            int index = int.CreateChecked(indices.Data[i]);
            if (index != paddingIndex && ((uint)index >= (uint)numEmbeddings))
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Index at position {i} is {index}, must be {paddingIndex} or in range [0, {numEmbeddings}).");

            parsedIndices[i] = index;
        }

        var resultValues = new T[batchSize * embeddingDim];
        var weightSpan = weight.Data.AsSpan();
        for (int batch = 0; batch < batchSize; batch++)
        {
            int indexBase = batch * maxActiveFeatures;
            int outputBase = batch * embeddingDim;

            for (int slot = 0; slot < maxActiveFeatures; slot++)
            {
                int index = parsedIndices[indexBase + slot];
                if (index == paddingIndex)
                    continue;

                int weightBase = index * embeddingDim;
                var src = weightSpan.Slice(weightBase, embeddingDim);
                var dst = resultValues.AsSpan().Slice(outputBase, embeddingDim);
                TensorPrimitives.Add(src, dst, dst);
            }
        }

        var resultColumn = NivaraColumn<T>.CreateFromOwnedArray(resultValues);

        NivaraColumn<T>? tangent = null;
        if (weight.RequiresTangent && weight.Tangent != null)
        {
            weight.Tangent.TryGetSpan(out var tanSpan);
            var tanValues = new T[batchSize * embeddingDim];
            for (int batch = 0; batch < batchSize; batch++)
            {
                int indexBase = batch * maxActiveFeatures;
                int outputBase = batch * embeddingDim;

                for (int slot = 0; slot < maxActiveFeatures; slot++)
                {
                    int index = parsedIndices[indexBase + slot];
                    if (index == paddingIndex)
                        continue;

                    int weightBase = index * embeddingDim;
                    var src = tanSpan.Slice(weightBase, embeddingDim);
                    var dst = tanValues.AsSpan().Slice(outputBase, embeddingDim);
                    TensorPrimitives.Add(src, dst, dst);
                }
            }
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanValues);
        }

        return new ForwardGradTensor<T>(resultColumn, tangent, new[] { batchSize, embeddingDim });
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements.
    /// JVP: t_out = sum(t_a)  (scalar)
    /// </summary>
    public static ForwardGradTensor<T> Sum<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        var sumValue = TensorPrimitives.Sum(a.AsSpan());
        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { sumValue });

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var tanSpan);
            var tanSum = TensorPrimitives.Sum(tanSpan);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(new T[] { tanSum });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    /// <summary>
    /// Computes the mean (average) of all elements.
    /// JVP: t_out = sum(t_a) / n  (scalar)
    /// </summary>
    public static ForwardGradTensor<T> Mean<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute mean of empty tensor");
        }

        var meanValue = TensorPrimitives.Sum(a.AsSpan()) / T.CreateChecked(a.Length);
        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { meanValue });

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var tanSpan);
            var tanSum = TensorPrimitives.Sum(tanSpan);
            var tanMean = tanSum / T.CreateChecked(a.Length);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(new T[] { tanMean });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Applies the ReLU activation: max(0, x).
    /// JVP: t_out = (a > 0) ? t_a : 0
    /// </summary>
    public static ForwardGradTensor<T> Relu<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Relu(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.ReluGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Gaussian error linear unit (tanh approximation).
    /// JVP: t_out = GeluGradient(a) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Gelu<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Gelu(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.GeluGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Sigmoid activation: σ(x) = 1 / (1 + e⁻ˣ).
    /// JVP: t_out = σ(a) * (1 - σ(a)) * t_a = result * (1 - result) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Sigmoid<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Sigmoid(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.SigmoidGradient(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Tanh activation.
    /// JVP: t_out = (1 - tanh²(a)) * t_a = (1 - result²) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Tanh<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Tanh(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.TanhGradient(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Negates all elements.
    /// JVP: t_out = -t_a
    /// </summary>
    public static ForwardGradTensor<T> Negate<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Negate(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.Negate(aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise absolute value.
    /// JVP: t_out = sign(a) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Abs<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Abs(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.AbsGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Clips values to [min, max].
    /// JVP: t_out = (a in [min, max]) ? t_a : 0
    /// </summary>
    public static ForwardGradTensor<T> Clip<T>(ForwardGradTensor<T> a, T min, T max)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Clamp(aSpan, min, max, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.ClipGradient(aSpan, aTanSpan, min, max, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies LeakyReLU activation: x if x > 0, else αx.
    /// JVP: t_out = (a > 0) ? t_a : α * t_a
    /// </summary>
    public static ForwardGradTensor<T> LeakyRelu<T>(ForwardGradTensor<T> a, T negativeSlope = default)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.LeakyRelu(aSpan, negativeSlope, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LeakyReluGradient(aSpan, aTanSpan, negativeSlope, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise exponential.
    /// JVP: t_out = e^a * t_a = result * t_a
    /// </summary>
    public static ForwardGradTensor<T> Exp<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Exp(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            TensorPrimitives.Multiply(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise natural logarithm.
    /// JVP: t_out = t_a / a
    /// </summary>
    public static ForwardGradTensor<T> Log<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Log(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LogGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Softmax function along the last dimension.
    /// JVP: s ⊙ (t_a - Σ(s * t_a)) where s = softmax(a)
    /// The Jacobian is symmetric, so SoftmaxGradient(result, t_a, dim) computes the JVP.
    /// </summary>
    public static ForwardGradTensor<T> Softmax<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Softmax(aSpan, primalArr, classCount);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.SoftmaxGradient(primalArr, aTanSpan, tanArr, classCount);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the LogSoftmax function along the last dimension.
    /// JVP: t_a - Σ(s * t_a) where s = softmax(a)
    /// </summary>
    public static ForwardGradTensor<T> LogSoftmax<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.LogSoftmax(aSpan, primalArr, classCount);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LogSoftmaxGradient(aSpan, aTanSpan, tanArr, classCount);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies dropout during training. In eval mode (isTraining=false) returns the input unchanged.
    /// JVP: mask * t_a * scale  (same mask used in forward)
    /// </summary>
    public static ForwardGradTensor<T> Dropout<T>(ForwardGradTensor<T> input, double probability, bool isTraining)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (probability < 0.0 || probability >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(probability), "Dropout probability must be in [0, 1).");

        if (!isTraining || probability <= 0.0)
            return input;

        var keepMask = new bool[input.Length];
        var random = Random.Shared;
        for (int i = 0; i < keepMask.Length; i++)
            keepMask[i] = random.NextDouble() >= probability;

        var scale = T.CreateChecked(1.0 / (1.0 - probability));
        return DropoutWithMask(input, keepMask, scale);
    }

    /// <summary>
    /// Applies dropout with a pre-generated mask.
    /// JVP: same mask applied to tangent with scaling.
    /// </summary>
    internal static ForwardGradTensor<T> DropoutWithMask<T>(ForwardGradTensor<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (keepMask.Length != input.Length)
            throw new ArgumentException(
                $"Dropout mask length ({keepMask.Length}) must match input length ({input.Length})",
                nameof(keepMask));

        var savedMask = keepMask.ToArray();
        var primal = GradOperationKernels.ApplyDropout(input.Data, savedMask, scale);
        NivaraColumn<T>? tangent = null;
        if (input.RequiresTangent && input.Tangent != null)
        {
            tangent = GradOperationKernels.ApplyDropoutGradient(input.Data, input.Tangent, savedMask, scale);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(input));
    }

    #endregion

    #region VAE Operations

    /// <summary>
    /// Computes the KL divergence between a diagonal Gaussian and N(0, 1).
    /// KL = -0.5 * sum(1 + logVar - mean² - exp(logVar))
    /// Output is a scalar (sum of per-element KL values).
    ///
    /// JVP: sum(mean * t_mean) + sum(0.5 * (exp(logVar) - 1) * t_logVar)
    /// </summary>
    public static ForwardGradTensor<T> KlDivergence<T>(ForwardGradTensor<T> mean, ForwardGradTensor<T> logVar)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        var klElements = GradOperationKernels.ApplyKlElementWise(mean.Data, logVar.Data);
        klElements.TryGetSpan(out var klSpan);
        var klSum = TensorPrimitives.Sum(klSpan);
        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { klSum });

        NivaraColumn<T>? tangent = null;
        if (mean.RequiresTangent || logVar.RequiresTangent)
        {
            var tanValue = T.Zero;

            if (mean.RequiresTangent && mean.Tangent != null)
            {
                mean.Data.TryGetSpan(out var mSpan);
                mean.Tangent.TryGetSpan(out var mTanSpan);
                var dMeanArr = new T[mean.Length];
                TensorPrimitives.Multiply(mSpan, mTanSpan, dMeanArr);
                tanValue += TensorPrimitives.Sum(dMeanArr);
            }

            if (logVar.RequiresTangent && logVar.Tangent != null)
            {
                logVar.Data.TryGetSpan(out var lvSpan);
                logVar.Tangent.TryGetSpan(out var lvTanSpan);
                var expLvArr = new T[logVar.Length];
                TensorPrimitives.Exp(lvSpan, expLvArr);
                var dLogVarArr = new T[logVar.Length];
                TensorPrimitives.Multiply(expLvArr, lvTanSpan, dLogVarArr);
                TensorPrimitives.Subtract(dLogVarArr, lvTanSpan, dLogVarArr);
                TensorPrimitives.Multiply(dLogVarArr, T.CreateChecked(0.5), dLogVarArr);
                tanValue += TensorPrimitives.Sum(dLogVarArr);
            }

            tangent = NivaraColumn<T>.CreateFromOwnedArray(new T[] { tanValue });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    /// <summary>
    /// Reparameterized sampling from a diagonal Gaussian: z = mean + exp(0.5 * logVar) * ε.
    /// JVP: t_z = t_mean + 0.5 * exp(0.5 * logVar) * ε * t_logVar
    /// </summary>
    public static ForwardGradTensor<T> SampleNormal<T>(ForwardGradTensor<T> mean, ForwardGradTensor<T> logVar, int? seed = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        int n = mean.Length;
        var epsilon = RandomGeneration.GenerateStandardNormal<T>(n, seed);
        var epsilonCol = NivaraColumn<T>.CreateFromOwnedArray(epsilon);
        var primal = GradOperationKernels.ApplySampleNormalForward(mean.Data, logVar.Data, epsilonCol);

        NivaraColumn<T>? tangent = null;
        if (mean.RequiresTangent || logVar.RequiresTangent)
        {
            if (mean.Tangent != null && logVar.Tangent != null)
            {
                var dLogVar = GradOperationKernels.ApplySampleNormalLogVarGradient(logVar.Data, logVar.Tangent, epsilonCol);
                tangent = mean.Tangent + dLogVar;
            }
            else if (mean.Tangent != null)
            {
                tangent = mean.Tangent;
            }
            else if (logVar.Tangent != null)
            {
                tangent = GradOperationKernels.ApplySampleNormalLogVarGradient(logVar.Data, logVar.Tangent, epsilonCol);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(mean, logVar));
    }

    #endregion

    #region Helper Methods

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        return a.shape;
    }

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        return a.shape;
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
    }

    static T[] TransposeAxesData<T>(T[] srcData, int[] srcDims, int[] dstDims, int axis1, int axis2)
        where T : struct, IFloatingPointIeee754<T>
    {
        var dstData = new T[srcData.Length];

        if (srcDims.Length == 2)
        {
            int rows = srcDims[0], cols = srcDims[1];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                {
                    int srcIdx = r * cols + c;
                    int dstIdx = c * rows + r;
                    dstData[dstIdx] = srcData[srcIdx];
                }
        }
        else
        {
            int d0 = srcDims[0], d1 = srcDims[1], d2 = srcDims[2];
            int nd1 = dstDims[1], nd2 = dstDims[2];
            for (int i0 = 0; i0 < d0; i0++)
                for (int i1 = 0; i1 < d1; i1++)
                    for (int i2 = 0; i2 < d2; i2++)
                    {
                        int srcIdx = i0 * d1 * d2 + i1 * d2 + i2;
                        var indices = new[] { i0, i1, i2 };
                        (indices[axis1], indices[axis2]) = (indices[axis2], indices[axis1]);
                        int dstIdx = indices[0] * nd1 * nd2 + indices[1] * nd2 + indices[2];
                        dstData[dstIdx] = srcData[srcIdx];
                    }
        }

        return dstData;
    }

    static T[] ConcatColumns<T>(NivaraColumn<T>[] cols, int[][] shapes, int axis, int rank)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (rank == 1)
        {
            int totalLen = cols.Sum(c => c.Length);
            var result = new T[totalLen];
            int offset = 0;
            for (int i = 0; i < cols.Length; i++)
            {
                cols[i].CopyTo(result.AsSpan(offset, cols[i].Length), default(T)!);
                offset += cols[i].Length;
            }
            return result;
        }

        if (axis == 1)
        {
            int rows = shapes[0][0];
            int totalCols = shapes.Sum(s => s[1]);
            var result = new T[rows * totalCols];
            int colOffset = 0;
            for (int i = 0; i < cols.Length; i++)
            {
                int tCols = shapes[i][1];
                var src = new T[cols[i].Length];
                cols[i].CopyTo(src, default(T)!);
                for (int r = 0; r < rows; r++)
                    Array.Copy(src, r * tCols, result, r * totalCols + colOffset, tCols);
                colOffset += tCols;
            }
            return result;
        }

        int totalRows = shapes.Sum(s => s[0]);
        int outCols = shapes[0][1];
        var resultRows = new T[totalRows * outCols];
        int rowOffset = 0;
        for (int i = 0; i < cols.Length; i++)
        {
            int tRows = shapes[i][0];
            var src = new T[cols[i].Length];
            cols[i].CopyTo(src, default(T)!);
            Array.Copy(src, 0, resultRows, rowOffset * outCols, tRows * outCols);
            rowOffset += tRows;
        }
        return resultRows;
    }

    #endregion
}
