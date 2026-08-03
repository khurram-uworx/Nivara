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
    /// Core dense matmul on flat row-major spans. Swap target for
    /// <c>Tensor.MatrixMultiply</c>.
    /// float/double use a transposed-B row kernel with
    /// <see cref="TensorPrimitives.Dot"/> inner accumulation (BCL-tuned SIMD);
    /// other vectorizable <c>INumber</c> types use a generic
    /// <see cref="Vector{T}"/> path; the rest fall back to
    /// <see cref="TensorPrimitives.Dot"/> scalar accumulation. Row parallelism is
    /// gated by <see cref="ShouldParallelize"/> to avoid small-matmul overhead.
    /// </summary>
    public static void MultiplyCore<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed = false)
        where T : struct, INumber<T>
    {
        if (aRows < 0) throw new ArgumentOutOfRangeException(nameof(aRows), "Row count must be non-negative.");
        if (aCols < 0) throw new ArgumentOutOfRangeException(nameof(aCols), "Column count must be non-negative.");
        if (bCols < 0) throw new ArgumentOutOfRangeException(nameof(bCols), "Column count must be non-negative.");
        if (a.Length < aRows * aCols)
            throw new ArgumentException($"Input span length ({a.Length}) must be at least {aRows * aCols}", nameof(a));
        if (b.Length < aCols * bCols)
            throw new ArgumentException($"Input span length ({b.Length}) must be at least {aCols * bCols}", nameof(b));
        if (result.Length < aRows * bCols)
            throw new ArgumentException($"Result length ({result.Length}) must be at least {aRows * bCols}", nameof(result));

        if (typeof(T) == typeof(float))
        {
            MultiplyCoreFloat(MemoryMarshal.Cast<T, float>(a), MemoryMarshal.Cast<T, float>(b),
                result, aRows, aCols, bCols, bTransposed);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            MultiplyCoreDouble(MemoryMarshal.Cast<T, double>(a), MemoryMarshal.Cast<T, double>(b),
                result, aRows, aCols, bCols, bTransposed);
            return;
        }
        MultiplyCoreGeneric(a, b, result, aRows, aCols, bCols, bTransposed);
    }

    static bool ShouldParallelize(int aRows, int aCols, int bCols)
        => aRows >= 4 && (long)aRows * aCols * bCols >= 2 << 20;

    const int MultiplyRowTile = 8;

    static void MultiplyCoreFloat<T>(ReadOnlySpan<float> a, ReadOnlySpan<float> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed)
        where T : struct, INumber<T>
    {
        int aLen = a.Length, bLen = b.Length;
        var aCopy = ArrayPool<float>.Shared.Rent(aLen);
        var bT = ArrayPool<float>.Shared.Rent(bLen);
        try
        {
            a.CopyTo(aCopy);
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            bool parallel = ShouldParallelize(aRows, aCols, bCols);
            if (parallel)
                Parallel.For(0, aRows, i => MultiplyRowFloat(aCopy, bT, result, i, aCols, bCols));
            else
                for (int i = 0; i < aRows; i++)
                    MultiplyRowFloat(aCopy, bT, result, i, aCols, bCols);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(aCopy, clearArray: true);
            ArrayPool<float>.Shared.Return(bT, clearArray: true);
        }
    }

    static void MultiplyCoreDouble<T>(ReadOnlySpan<double> a, ReadOnlySpan<double> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed)
        where T : struct, INumber<T>
    {
        int aLen = a.Length, bLen = b.Length;
        var aCopy = ArrayPool<double>.Shared.Rent(aLen);
        var bT = ArrayPool<double>.Shared.Rent(bLen);
        try
        {
            a.CopyTo(aCopy);
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            bool parallel = ShouldParallelize(aRows, aCols, bCols);
            if (parallel)
                Parallel.For(0, aRows, i => MultiplyRowDouble(aCopy, bT, result, i, aCols, bCols));
            else
                for (int i = 0; i < aRows; i++)
                    MultiplyRowDouble(aCopy, bT, result, i, aCols, bCols);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(aCopy, clearArray: true);
            ArrayPool<double>.Shared.Return(bT, clearArray: true);
        }
    }

    static void MultiplyCoreGeneric<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed)
        where T : struct, INumber<T>
    {
        int aLen = a.Length, bLen = b.Length;
        var aCopy = ArrayPool<T>.Shared.Rent(aLen);
        var bT = ArrayPool<T>.Shared.Rent(bLen);
        try
        {
            a.CopyTo(aCopy);
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            int vec = Vector<T>.Count;
            bool vectorized = Vector.IsHardwareAccelerated && vec > 1 && aCols >= vec;
            bool parallel = ShouldParallelize(aRows, aCols, bCols);

            if (vectorized)
            {
                if (parallel)
                    Parallel.For(0, aRows, i => MultiplyRowVectorizedGeneric(aCopy, bT, result, i, aCols, bCols, vec));
                else
                    for (int i = 0; i < aRows; i++)
                        MultiplyRowVectorizedGeneric(aCopy, bT, result, i, aCols, bCols, vec);
            }
            else
            {
                if (parallel)
                    Parallel.For(0, aRows, i => MultiplyRowScalar(aCopy, bT, result, i, aCols, bCols));
                else
                    for (int i = 0; i < aRows; i++)
                        MultiplyRowScalar(aCopy, bT, result, i, aCols, bCols);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aCopy, clearArray: true);
            ArrayPool<T>.Shared.Return(bT, clearArray: true);
        }
    }

    static void MultiplyRowFloat<T>(float[] aCopy, float[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        var aSpan = aCopy.AsSpan(aOff, aCols);
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = T.CreateChecked(TensorPrimitives.Dot(aSpan, bT.AsSpan(j * aCols, aCols)));
    }

    static void MultiplyRowDouble<T>(double[] aCopy, double[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        var aSpan = aCopy.AsSpan(aOff, aCols);
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = T.CreateChecked(TensorPrimitives.Dot(aSpan, bT.AsSpan(j * aCols, aCols)));
    }

    static void MultiplyRowScalar<T>(T[] aCopy, T[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = TensorPrimitives.Dot(aCopy.AsSpan(aOff, aCols), bT.AsSpan(j * aCols, aCols));
    }

    static void MultiplyRowVectorizedGeneric<T>(T[] aCopy, T[] bT, T[] result,
        int i, int aCols, int bCols, int vec)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        int kVecEnd = aCols - (aCols % vec);

        Span<Vector<T>> accs = stackalloc Vector<T>[MultiplyRowTile];
        int j = 0;
        for (; j + MultiplyRowTile <= bCols; j += MultiplyRowTile)
        {
            accs.Clear();
            int k = 0;
            for (; k < kVecEnd; k += vec)
            {
                var av = Vector.LoadUnsafe(ref aCopy[aOff + k]);
                for (int t = 0; t < MultiplyRowTile; t++)
                    accs[t] = Vector.Add(accs[t], Vector.Multiply(av, Vector.LoadUnsafe(ref bT[(j + t) * aCols + k])));
            }
            for (int t = 0; t < MultiplyRowTile; t++)
                result[outOff + j + t] = Vector.Sum(accs[t]) + TailGeneric(aCopy, bT, aOff, j + t, k, aCols);
        }
        for (; j < bCols; j++)
        {
            var acc = Vector<T>.Zero;
            int k = 0;
            for (; k < kVecEnd; k += vec)
                acc = Vector.Add(acc, Vector.Multiply(Vector.LoadUnsafe(ref aCopy[aOff + k]), Vector.LoadUnsafe(ref bT[j * aCols + k])));
            result[outOff + j] = Vector.Sum(acc) + TailGeneric(aCopy, bT, aOff, j, k, aCols);
        }
    }

    static T TailGeneric<T>(T[] aCopy, T[] bT, int aOff, int j, int k, int aCols)
        where T : struct, INumber<T>
    {
        T sum = T.Zero;
        int bOff = j * aCols;
        for (; k < aCols; k++)
            sum += aCopy[aOff + k] * bT[bOff + k];
        return sum;
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
