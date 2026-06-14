using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.Tensors;

/// <summary>
/// Central tensor kernel helpers — the single file to check when upgrading
/// to a new .NET version. Each section documents the BCL API that should
/// replace the handwritten implementation below.
/// </summary>
static class TensorsHelper
{
    // ═══════════════════════════════════════════════════════════════
    //  MatMul / Transpose
    //  .NET future: Tensor.MatrixMultiply&lt;T&gt;
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Transpose a row-major matrix.
    /// .NET 11: Tensor.Transpose&lt;T&gt;(tensor)
    /// </summary>
    public static void Transpose<T>(ReadOnlySpan<T> src, Span<T> dst, int rows, int cols)
        where T : struct, INumber<T>
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dst[j * rows + i] = src[i * cols + j];
    }

    /// <summary>
    /// Null-aware transpose.
    /// </summary>
    public static void Transpose<T>(
        ReadOnlySpan<T> src, ReadOnlySpan<bool> srcNullMask,
        Span<T> dst, Span<bool> dstNullMask,
        int rows, int cols)
        where T : struct, INumber<T>
    {
        bool hasMask = srcNullMask.Length > 0;
        if (!hasMask)
        {
            dstNullMask.Clear();
            Transpose(src, dst, rows, cols);
            return;
        }

        int n = rows * cols;
        var filled = ArrayPool<T>.Shared.Rent(n);
        var maskCopy = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            src.CopyTo(filled.AsSpan(0, n));
            srcNullMask.CopyTo(maskCopy.AsSpan(0, n));

            for (int idx = 0; idx < n; idx++)
                if (maskCopy[idx]) filled[idx] = T.Zero;

            Transpose(filled.AsSpan(0, n), dst, rows, cols);

            // Transpose the mask using the copy to handle aliased spans
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    dstNullMask[j * rows + i] = maskCopy[i * cols + j];

            for (int idx = 0; idx < n; idx++)
                if (dstNullMask[idx]) dst[idx] = T.Zero;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(filled, clearArray: true);
            ArrayPool<bool>.Shared.Return(maskCopy, clearArray: true);
        }
    }

    /// <summary>
    /// PRIMARY — Tensor&lt;T&gt; level matmul. Swap target for Tensor.MatrixMultiply.
    /// </summary>
    public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b,
        int aRows, int aCols, int bCols)
        where T : unmanaged, INumber<T>
    {
        int aLen = (int)a.FlattenedLength;
        int bLen = (int)b.FlattenedLength;
        int resLen = aRows * bCols;

        var result = new T[resLen];
        Multiply(a, b, result, aRows, aCols, bCols);
        return Tensor.Create(result, new ReadOnlySpan<nint>([aRows, bCols]));
    }

    /// <summary>
    /// Dense (no-null) matmul on Tensor&lt;T&gt; inputs, writing raw T[] result.
    /// This overload's body is the swap target when Tensor.MatrixMultiply ships.
    /// </summary>
    public static void Multiply<T>(Tensor<T> a, Tensor<T> b, T[] result,
        int aRows, int aCols, int bCols)
        where T : unmanaged, INumber<T>
    {
        int aLen = (int)a.FlattenedLength;
        int bLen = (int)b.FlattenedLength;
        var aFlat = ArrayPool<T>.Shared.Rent(aLen);
        var bFlat = ArrayPool<T>.Shared.Rent(bLen);
        try
        {
            a.FlattenTo(aFlat.AsSpan(0, aLen));
            b.FlattenTo(bFlat.AsSpan(0, bLen));
            MultiplyCore(aFlat.AsSpan(0, aLen), bFlat.AsSpan(0, bLen), result, aRows, aCols, bCols);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aFlat, clearArray: true);
            ArrayPool<T>.Shared.Return(bFlat, clearArray: true);
        }
    }

    /// <summary>
    /// Core dense matmul on flat row-major spans — transpose(B) + Dot + Parallel.For.
    /// Swap target for Tensor.MatrixMultiply.
    /// </summary>
    public static void MultiplyCore<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] result,
        int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aLen = a.Length, bLen = b.Length;
        var aCopy = ArrayPool<T>.Shared.Rent(aLen);
        var bCopy = ArrayPool<T>.Shared.Rent(bLen);
        var bT = ArrayPool<T>.Shared.Rent(bLen);
        try
        {
            a.CopyTo(aCopy);
            b.CopyTo(bCopy);
            Transpose(bCopy.AsSpan(0, bLen), bT.AsSpan(0, bLen), aCols, bCols);

            Parallel.For(0, aRows, i =>
            {
                int aOff = i * aCols;
                for (int j = 0; j < bCols; j++)
                {
                    int bOff = j * aCols;
                    result[i * bCols + j] = TensorPrimitives.Dot(
                        aCopy.AsSpan(aOff, aCols), bT.AsSpan(bOff, aCols));
                }
            });
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aCopy, clearArray: true);
            ArrayPool<T>.Shared.Return(bCopy, clearArray: true);
            ArrayPool<T>.Shared.Return(bT, clearArray: true);
        }
    }

    /// <summary>
    /// Null-aware matmul: fill nulls with T.Zero, run dense kernel,
    /// compute result mask via boolean OR propagation.
    /// </summary>
    public static void Multiply<T>(
        ReadOnlySpan<T> a, ReadOnlySpan<bool> aNullMask,
        ReadOnlySpan<T> b, ReadOnlySpan<bool> bNullMask,
        T[] result, Span<bool> resultMask,
        int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        bool hasAMask = aNullMask.Length > 0;
        bool hasBMask = bNullMask.Length > 0;

        if (!hasAMask && !hasBMask)
        {
            resultMask.Clear();
            MultiplyCore(a, b, result, aRows, aCols, bCols);
            return;
        }

        int aLen = a.Length, bLen = b.Length;
        var aFilled = ArrayPool<T>.Shared.Rent(aLen);
        var bFilled = ArrayPool<T>.Shared.Rent(bLen);
        try
        {
            a.CopyTo(aFilled.AsSpan(0, aLen));
            b.CopyTo(bFilled.AsSpan(0, bLen));

            if (hasAMask)
                for (int idx = 0; idx < aLen; idx++)
                    if (aNullMask[idx]) aFilled[idx] = T.Zero;

            if (hasBMask)
                for (int idx = 0; idx < bLen; idx++)
                    if (bNullMask[idx]) bFilled[idx] = T.Zero;

            MultiplyCore(aFilled.AsSpan(0, aLen), bFilled.AsSpan(0, bLen), result, aRows, aCols, bCols);

            PropagateNullMask(aNullMask, bNullMask, resultMask, aRows, aCols, bCols);

            for (int idx = 0; idx < resultMask.Length; idx++)
                if (resultMask[idx]) result[idx] = T.Zero;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aFilled, clearArray: true);
            ArrayPool<T>.Shared.Return(bFilled, clearArray: true);
        }
    }

    /// <summary>
    /// Computes one Euclidean norm per row from a flat row-major matrix.
    /// Uses a batched <see cref="Vector{T}"/> kernel to avoid per-row
    /// <see cref="TensorPrimitives"/> dispatch overhead when the column
    /// count is large enough to benefit from SIMD.
    /// Null handling is caller-owned.
    /// </summary>
    public static void RowNorms<T>(ReadOnlySpan<T> rowMajor, Span<T> destination, int rows, int cols)
        where T : unmanaged, IRootFunctions<T>
    {
        if (destination.Length < rows)
            throw new ArgumentException($"Destination span length ({destination.Length}) must be at least {rows}", nameof(destination));
        if (rowMajor.Length < rows * cols)
            throw new ArgumentException($"Input span length ({rowMajor.Length}) must be at least {rows * cols}", nameof(rowMajor));

        if (rows == 0) return;
        if (cols == 0)
        {
            destination.Slice(0, rows).Clear();
            return;
        }

        bool canVectorize = Vector.IsHardwareAccelerated && Vector<T>.Count > 0
                            && cols >= Vector<T>.Count;

        for (int row = 0; row < rows; row++)
        {
            var rowSpan = rowMajor.Slice(row * cols, cols);

            if (canVectorize)
            {
                var acc = Vector<T>.Zero;
                int col = 0;
                for (; col <= rowSpan.Length - Vector<T>.Count; col += Vector<T>.Count)
                {
                    var v = new Vector<T>(rowSpan.Slice(col));
                    acc += v * v;
                }

                T sumSq = Vector.Sum(acc);
                for (; col < rowSpan.Length; col++)
                {
                    T val = rowSpan[col];
                    sumSq += val * val;
                }

                destination[row] = T.Sqrt(sumSq);
            }
            else
            {
                destination[row] = TensorPrimitives.Norm(rowSpan);
            }
        }
    }

    internal static void PropagateNullMask(
        ReadOnlySpan<bool> aNullMask, ReadOnlySpan<bool> bNullMask,
        Span<bool> resultMask, int aRows, int aCols, int bCols)
    {
        bool hasAMask = aNullMask.Length > 0;
        bool hasBMask = bNullMask.Length > 0;

        if (!hasAMask && !hasBMask)
        {
            resultMask.Clear();
            return;
        }

        var aRowHasNull = ArrayPool<bool>.Shared.Rent(aRows);
        var bColumnHasNull = ArrayPool<bool>.Shared.Rent(bCols);
        try
        {
            var aRowsSpan = aRowHasNull.AsSpan(0, aRows);
            var bColsSpan = bColumnHasNull.AsSpan(0, bCols);
            aRowsSpan.Clear();
            bColsSpan.Clear();

            if (hasAMask)
                for (int i = 0; i < aRows; i++)
                {
                    int aRowOffset = i * aCols;
                    for (int k = 0; k < aCols; k++)
                        if (aNullMask[aRowOffset + k])
                        {
                            aRowsSpan[i] = true;
                            break;
                        }
                }

            if (hasBMask)
                for (int k = 0; k < aCols; k++)
                {
                    int bRowOffset = k * bCols;
                    for (int j = 0; j < bCols; j++)
                        if (bNullMask[bRowOffset + j])
                            bColsSpan[j] = true;
                }

            for (int i = 0; i < aRows; i++)
            {
                bool rowHasNull = aRowsSpan[i];
                int resultRowOffset = i * bCols;
                for (int j = 0; j < bCols; j++)
                    resultMask[resultRowOffset + j] = rowHasNull || bColsSpan[j];
            }
        }
        finally
        {
            ArrayPool<bool>.Shared.Return(aRowHasNull, clearArray: true);
            ArrayPool<bool>.Shared.Return(bColumnHasNull, clearArray: true);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  SoftMax
    //  .NET 11: TensorPrimitives.SoftMax&lt;T&gt;(x, destination) for
    //          single-vector softmax; Tensor.SoftMax&lt;T&gt;(tensor) for
    //          tensor-level.
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Single-vector softmax: exp(x[i]) / sum(exp(x)))
    /// .NET 11: replace body with TensorPrimitives.SoftMax&lt;T&gt;(x, destination).
    /// </summary>
    public static void SoftMax<T>(ReadOnlySpan<T> x, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (typeof(T) == typeof(float))
        {
            var s = MemoryMarshal.Cast<T, float>(x);
            var d = MemoryMarshal.Cast<T, float>(destination);
            float max = float.NegativeInfinity;
            for (int i = 0; i < s.Length; i++)
                if (s[i] > max) max = s[i];
            TensorPrimitives.Subtract(s, max, d);
            TensorPrimitives.Exp(d, d);
            TensorPrimitives.Divide(d, TensorPrimitives.Sum(d), d);
        }
        else if (typeof(T) == typeof(double))
        {
            var s = MemoryMarshal.Cast<T, double>(x);
            var d = MemoryMarshal.Cast<T, double>(destination);
            double max = double.NegativeInfinity;
            for (int i = 0; i < s.Length; i++)
                if (s[i] > max) max = s[i];
            TensorPrimitives.Subtract(s, max, d);
            TensorPrimitives.Exp(d, d);
            TensorPrimitives.Divide(d, TensorPrimitives.Sum(d), d);
        }
        else
        {
            int n = x.Length;
            double max = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                var val = double.CreateChecked(x[i]);
                if (val > max) max = val;
            }
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                var exp = Math.Exp(double.CreateChecked(x[i]) - max);
                destination[i] = T.CreateChecked(exp);
                sum += exp;
            }
            if (sum > 0)
                for (int i = 0; i < n; i++)
                    destination[i] = T.CreateChecked(double.CreateChecked(destination[i]) / sum);
        }
    }

    /// <summary>
    /// Row-wise softmax (flat span with classCount elements per row).
    /// </summary>
    public static void SoftMax<T>(ReadOnlySpan<T> x, Span<T> destination, int classCount)
        where T : struct, INumber<T>
    {
        if (classCount <= 0 || classCount >= x.Length)
        {
            SoftMax(x, destination);
            return;
        }
        int rows = x.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            int start = r * classCount;
            SoftMax(x.Slice(start, classCount), destination.Slice(start, classCount));
        }
    }

    /// <summary>
    /// Null-aware row-wise softmax. Fills nulls with T.Zero, computes dense
    /// softmax, then restores null mask and zeros result at null positions.
    /// </summary>
    public static void SoftMax<T>(
        ReadOnlySpan<T> x, ReadOnlySpan<bool> xNullMask,
        Span<T> destination, Span<bool> resultMask, int classCount)
        where T : struct, INumber<T>
    {
        bool hasMask = xNullMask.Length > 0;
        if (!hasMask)
        {
            resultMask.Clear();
            SoftMax(x, destination, classCount);
            return;
        }

        int n = x.Length;
        var filled = ArrayPool<T>.Shared.Rent(n);
        try
        {
            x.CopyTo(filled.AsSpan(0, n));
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) filled[idx] = T.Zero;

            SoftMax(filled.AsSpan(0, n), destination, classCount);

            xNullMask.CopyTo(resultMask);
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) destination[idx] = T.Zero;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(filled, clearArray: true);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Sigmoid
    //  .NET 11: TensorPrimitives.Sigmoid&lt;T&gt;(x, destination)
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Element-wise sigmoid: 1 / (1 + exp(-x)).
    /// .NET 11: replace body with TensorPrimitives.Sigmoid&lt;T&gt;(x, destination).
    /// </summary>
    public static void Sigmoid<T>(ReadOnlySpan<T> x, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (typeof(T) == typeof(float))
        {
            var s = MemoryMarshal.Cast<T, float>(x);
            var d = MemoryMarshal.Cast<T, float>(destination);
            TensorPrimitives.Negate(s, d);
            TensorPrimitives.Exp(d, d);
            TensorPrimitives.Add(d, 1.0f, d);
            TensorPrimitives.Divide(1.0f, d, d);
        }
        else if (typeof(T) == typeof(double))
        {
            var s = MemoryMarshal.Cast<T, double>(x);
            var d = MemoryMarshal.Cast<T, double>(destination);
            TensorPrimitives.Negate(s, d);
            TensorPrimitives.Exp(d, d);
            TensorPrimitives.Add(d, 1.0, d);
            TensorPrimitives.Divide(1.0, d, d);
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
            {
                var val = double.CreateChecked(x[i]);
                destination[i] = T.CreateChecked(1.0 / (1.0 + Math.Exp(-val)));
            }
        }
    }

    /// <summary>
    /// Null-aware element-wise sigmoid.
    /// </summary>
    public static void Sigmoid<T>(
        ReadOnlySpan<T> x, ReadOnlySpan<bool> xNullMask,
        Span<T> destination, Span<bool> resultMask)
        where T : struct, INumber<T>
    {
        bool hasMask = xNullMask.Length > 0;
        if (!hasMask)
        {
            resultMask.Clear();
            Sigmoid(x, destination);
            return;
        }

        int n = x.Length;
        var filled = ArrayPool<T>.Shared.Rent(n);
        try
        {
            x.CopyTo(filled.AsSpan(0, n));
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) filled[idx] = T.Zero;

            Sigmoid(filled.AsSpan(0, n), destination);

            xNullMask.CopyTo(resultMask);
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) destination[idx] = T.Zero;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(filled, clearArray: true);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Tanh
    //  .NET 11: TensorPrimitives.Tanh&lt;T&gt;(x, destination)
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Element-wise hyperbolic tangent.
    /// .NET 11: replace body with TensorPrimitives.Tanh&lt;T&gt;(x, destination).
    /// </summary>
    public static void Tanh<T>(ReadOnlySpan<T> x, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Tanh(MemoryMarshal.Cast<T, float>(x), MemoryMarshal.Cast<T, float>(destination));
        }
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Tanh(MemoryMarshal.Cast<T, double>(x), MemoryMarshal.Cast<T, double>(destination));
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
            {
                var val = double.CreateChecked(x[i]);
                destination[i] = T.CreateChecked(Math.Tanh(val));
            }
        }
    }

    /// <summary>
    /// Null-aware element-wise hyperbolic tangent.
    /// </summary>
    public static void Tanh<T>(
        ReadOnlySpan<T> x, ReadOnlySpan<bool> xNullMask,
        Span<T> destination, Span<bool> resultMask)
        where T : struct, INumber<T>
    {
        bool hasMask = xNullMask.Length > 0;
        if (!hasMask)
        {
            resultMask.Clear();
            Tanh(x, destination);
            return;
        }

        int n = x.Length;
        var filled = ArrayPool<T>.Shared.Rent(n);
        try
        {
            x.CopyTo(filled.AsSpan(0, n));
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) filled[idx] = T.Zero;

            Tanh(filled.AsSpan(0, n), destination);

            xNullMask.CopyTo(resultMask);
            for (int idx = 0; idx < n; idx++)
                if (xNullMask[idx]) destination[idx] = T.Zero;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(filled, clearArray: true);
        }
    }
}
