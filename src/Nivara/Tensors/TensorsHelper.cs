using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
    /// Transpose a row-major matrix using a cache-friendly tiled loop.
    /// .NET 11: Tensor.Transpose&lt;T&gt;(tensor)
    /// </summary>
    internal static void Transpose<T>(ReadOnlySpan<T> src, Span<T> dst, int rows, int cols)
        where T : struct, INumber<T>
    {
        const int tile = 32;
        for (int i0 = 0; i0 < rows; i0 += tile)
        {
            int iMax = Math.Min(i0 + tile, rows);
            for (int j0 = 0; j0 < cols; j0 += tile)
            {
                int jMax = Math.Min(j0 + tile, cols);
                for (int i = i0; i < iMax; i++)
                {
                    int srcRow = i * cols;
                    for (int j = j0; j < jMax; j++)
                        dst[j * rows + i] = src[srcRow + j];
                }
            }
        }
    }

    /// <summary>
    /// Null-aware transpose.
    /// </summary>
    internal static void Transpose<T>(
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
    /// Core dense matmul on flat row-major spans. Swap target for
    /// <c>Tensor.MatrixMultiply</c>.
    /// float/double use a transposed-B row kernel with
    /// <see cref="TensorPrimitives.Dot"/> inner accumulation (BCL-tuned SIMD);
    /// other vectorizable <c>INumber</c> types use a generic
    /// <see cref="Vector{T}"/> path; the rest fall back to
    /// <see cref="TensorPrimitives.Dot"/> scalar accumulation. Row parallelism is
    /// gated by <see cref="ShouldParallelize"/> to avoid small-matmul overhead.
    /// </summary>
    internal static void MultiplyCore<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] result,
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
        int bLen = b.Length;
        var bT = ArrayPool<float>.Shared.Rent(bLen);
        try
        {
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            bool parallel = ShouldParallelize(aRows, aCols, bCols);
            float[]? aCopy = parallel ? RentCopy(a) : null;
            try
            {
                if (parallel)
                    Parallel.For(0, aRows, i => MultiplyRowFloat(aCopy!, bT, result, i, aCols, bCols));
                else
                    for (int i = 0; i < aRows; i++)
                        MultiplyRowFloat(a, bT, result, i, aCols, bCols);
            }
            finally
            {
                if (aCopy != null)
                    ArrayPool<float>.Shared.Return(aCopy, clearArray: true);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(bT, clearArray: true);
        }
    }

    static void MultiplyCoreDouble<T>(ReadOnlySpan<double> a, ReadOnlySpan<double> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed)
        where T : struct, INumber<T>
    {
        int bLen = b.Length;
        var bT = ArrayPool<double>.Shared.Rent(bLen);
        try
        {
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            bool parallel = ShouldParallelize(aRows, aCols, bCols);
            double[]? aCopy = parallel ? RentCopy(a) : null;
            try
            {
                if (parallel)
                    Parallel.For(0, aRows, i => MultiplyRowDouble(aCopy!, bT, result, i, aCols, bCols));
                else
                    for (int i = 0; i < aRows; i++)
                        MultiplyRowDouble(a, bT, result, i, aCols, bCols);
            }
            finally
            {
                if (aCopy != null)
                    ArrayPool<double>.Shared.Return(aCopy, clearArray: true);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(bT, clearArray: true);
        }
    }

    static void MultiplyCoreGeneric<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] result,
        int aRows, int aCols, int bCols, bool bTransposed)
        where T : struct, INumber<T>
    {
        int bLen = b.Length;
        var bT = ArrayPool<T>.Shared.Rent(bLen);
        try
        {
            if (bTransposed)
                b.CopyTo(bT.AsSpan(0, bLen));
            else
                Transpose(b, bT.AsSpan(0, bLen), aCols, bCols);

            int vec = Vector<T>.Count;
            bool vectorized = Vector.IsHardwareAccelerated && vec > 1 && aCols >= vec;
            bool parallel = ShouldParallelize(aRows, aCols, bCols);
            T[]? aCopy = parallel ? RentCopy(a) : null;
            try
            {
                if (vectorized)
                {
                    if (parallel)
                        Parallel.For(0, aRows, i => MultiplyRowVectorizedGeneric(aCopy!, bT, result, i, aCols, bCols, vec));
                    else
                        for (int i = 0; i < aRows; i++)
                            MultiplyRowVectorizedGeneric(a, bT, result, i, aCols, bCols, vec);
                }
                else
                {
                    if (parallel)
                        Parallel.For(0, aRows, i => MultiplyRowScalar(aCopy!, bT, result, i, aCols, bCols));
                    else
                        for (int i = 0; i < aRows; i++)
                            MultiplyRowScalar(a, bT, result, i, aCols, bCols);
                }
            }
            finally
            {
                if (aCopy != null)
                    ArrayPool<T>.Shared.Return(aCopy, clearArray: true);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(bT, clearArray: true);
        }
    }

    static T[] RentCopy<T>(ReadOnlySpan<T> span)
    {
        var copy = ArrayPool<T>.Shared.Rent(span.Length);
        span.CopyTo(copy);
        return copy;
    }

    static void MultiplyRowFloat<T>(ReadOnlySpan<float> a, float[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        var aSpan = a.Slice(aOff, aCols);
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = T.CreateChecked(TensorPrimitives.Dot(aSpan, bT.AsSpan(j * aCols, aCols)));
    }

    static void MultiplyRowDouble<T>(ReadOnlySpan<double> a, double[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        var aSpan = a.Slice(aOff, aCols);
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = T.CreateChecked(TensorPrimitives.Dot(aSpan, bT.AsSpan(j * aCols, aCols)));
    }

    static void MultiplyRowScalar<T>(ReadOnlySpan<T> a, T[] bT, T[] result,
        int i, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        for (int j = 0; j < bCols; j++)
            result[outOff + j] = TensorPrimitives.Dot(a.Slice(aOff, aCols), bT.AsSpan(j * aCols, aCols));
    }

    static void MultiplyRowVectorizedGeneric<T>(ReadOnlySpan<T> a, T[] bT, T[] result,
        int i, int aCols, int bCols, int vec)
        where T : struct, INumber<T>
    {
        int aOff = i * aCols;
        int outOff = i * bCols;
        int kVecEnd = aCols - (aCols % vec);
        ref T aRef = ref MemoryMarshal.GetReference(a.Slice(aOff, aCols));

        Span<Vector<T>> accs = stackalloc Vector<T>[MultiplyRowTile];
        int j = 0;
        for (; j + MultiplyRowTile <= bCols; j += MultiplyRowTile)
        {
            accs.Clear();
            int k = 0;
            for (; k < kVecEnd; k += vec)
            {
                var av = Vector.LoadUnsafe(ref Unsafe.Add(ref aRef, k));
                for (int t = 0; t < MultiplyRowTile; t++)
                    accs[t] = Vector.Add(accs[t], Vector.Multiply(av, Vector.LoadUnsafe(ref bT[(j + t) * aCols + k])));
            }
            for (int t = 0; t < MultiplyRowTile; t++)
                result[outOff + j + t] = Vector.Sum(accs[t]) + TailGeneric(a, bT, aOff, j + t, k, aCols);
        }
        for (; j < bCols; j++)
        {
            var acc = Vector<T>.Zero;
            int k = 0;
            for (; k < kVecEnd; k += vec)
                acc = Vector.Add(acc, Vector.Multiply(Vector.LoadUnsafe(ref Unsafe.Add(ref aRef, k)), Vector.LoadUnsafe(ref bT[j * aCols + k])));
            result[outOff + j] = Vector.Sum(acc) + TailGeneric(a, bT, aOff, j, k, aCols);
        }
    }

    static T TailGeneric<T>(ReadOnlySpan<T> a, T[] bT, int aOff, int j, int k, int aCols)
        where T : struct, INumber<T>
    {
        T sum = T.Zero;
        int bOff = j * aCols;
        for (; k < aCols; k++)
            sum += a[aOff + k] * bT[bOff + k];
        return sum;
    }

    /// <summary>
    /// Null-aware matmul: fill nulls with T.Zero, run dense kernel,
    /// compute result mask via boolean OR propagation.
    /// </summary>
    internal static void Multiply<T>(
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
    //  Row-slice scoring kernels (#141)
    //  Score each row of a row-major buffer with the platform
    //  TensorPrimitives kernels. Row slices are contiguous spans into
    //  the materialized buffer — no per-row copy. Mask-first null
    //  semantics: a null in a row masks only that row's score; a null
    //  in the query masks all scores. The output mask is authoritative;
    //  placeholder output at masked positions is not valid.
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Computes <c>output[r] = dot(rowMajor[r, :], query)</c> for each row of a
    /// row-major buffer. Nulls in a row mask only that row; nulls in the query
    /// mask all scores.
    /// </summary>
    internal static void RowDot<T>(
        ReadOnlySpan<T> rowMajor, ReadOnlySpan<bool> rowMajorNullMask,
        ReadOnlySpan<T> query, ReadOnlySpan<bool> queryNullMask,
        Span<T> output, Span<bool> outputMask, int rows, int cols)
        where T : struct, INumber<T>
    {
        ValidateRowKernelArgs(rowMajor, rowMajorNullMask, query, queryNullMask, output, outputMask, rows, cols, requireQuery: true);

        if (AnyTrue(queryNullMask))
        {
            outputMask.Fill(true);
            output.Clear();
            return;
        }

        if (rowMajorNullMask.Length == 0)
        {
            outputMask.Clear();
            for (int r = 0; r < rows; r++)
                output[r] = TensorPrimitives.Dot(rowMajor.Slice(r * cols, cols), query);
            return;
        }

        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            bool nullRow = AnyTrue(rowMajorNullMask.Slice(off, cols));
            outputMask[r] = nullRow;
            output[r] = nullRow ? default : TensorPrimitives.Dot(rowMajor.Slice(off, cols), query);
        }
    }

    /// <summary>
    /// Computes <c>output[r] = cosineSimilarity(rowMajor[r, :], query)</c> for each
    /// row of a row-major buffer. Nulls in a row mask only that row; nulls in the
    /// query mask all scores.
    /// </summary>
    internal static void RowCosineSimilarity<T>(
        ReadOnlySpan<T> rowMajor, ReadOnlySpan<bool> rowMajorNullMask,
        ReadOnlySpan<T> query, ReadOnlySpan<bool> queryNullMask,
        Span<T> output, Span<bool> outputMask, int rows, int cols)
        where T : struct, IRootFunctions<T>
    {
        ValidateRowKernelArgs(rowMajor, rowMajorNullMask, query, queryNullMask, output, outputMask, rows, cols,
            requireQuery: true, requireNonZeroCols: true);

        if (AnyTrue(queryNullMask))
        {
            outputMask.Fill(true);
            output.Clear();
            return;
        }

        if (rowMajorNullMask.Length == 0)
        {
            outputMask.Clear();
            for (int r = 0; r < rows; r++)
                output[r] = TensorPrimitives.CosineSimilarity(rowMajor.Slice(r * cols, cols), query);
            return;
        }

        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            bool nullRow = AnyTrue(rowMajorNullMask.Slice(off, cols));
            outputMask[r] = nullRow;
            output[r] = nullRow ? default : TensorPrimitives.CosineSimilarity(rowMajor.Slice(off, cols), query);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Normalize / Standardize (population z-score)
    //  .NET future: TensorPrimitives.Mean&lt;T&gt; / StdDev&lt;T&gt; (net-11 names)
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// In-place population z-score normalization for IEEE-754 float types.
    /// Computes mean/stddev over the whole span and subtracts/divides in
    /// place. Returns <c>false</c> when the population standard deviation is
    /// zero (caller keeps the data unchanged).
    /// <see cref="TensorPrimitives.StdDev{T}"/> requires <c>IRootFunctions&lt;T&gt;</c>,
    /// satisfied by <c>IFloatingPointIeee754&lt;T&gt;</c>.
    /// </summary>
    internal static bool TryNormalizeInPlace<T>(Span<T> values)
        where T : struct, IFloatingPointIeee754<T>
    {
        var mean = TensorPrimitives.Average<T>(values);
        var stdDev = TensorPrimitives.StdDev<T>(values);
        if (stdDev == T.Zero) return false;

        TensorPrimitives.Subtract(values, mean, values);
        TensorPrimitives.Divide(values, stdDev, values);
        return true;
    }

    /// <summary>
    /// Population z-score normalization of any <c>INumber</c> span into a
    /// <c>double</c> destination. Converts via
    /// <see cref="TensorPrimitives.ConvertChecked{TFrom,TTo}"/> (vectorized
    /// <c>CreateChecked</c>), then reuses the double SIMD chain for statistics
    /// and transform. <c>INumber&lt;T&gt;</c> does not satisfy
    /// <c>IRootFunctions&lt;T&gt;</c>, so <c>StdDev</c> cannot run on the source
    /// type (and integer arithmetic would truncate); the double conversion
    /// gives both correctness and BCL-tuned numerical stability. Returns
    /// <c>false</c> when the population standard deviation is zero (the caller
    /// keeps the column); on success the destination holds the z-scores.
    /// </summary>
    internal static bool TryNormalizeToDouble<T>(ReadOnlySpan<T> values, Span<double> destination)
        where T : struct, INumber<T>
    {
        if (destination.Length < values.Length)
            throw new ArgumentException($"Destination length ({destination.Length}) must be at least {values.Length}.", nameof(destination));

        TensorPrimitives.ConvertChecked<T, double>(values, destination);
        var mean = TensorPrimitives.Average<double>(destination);
        var stdDev = TensorPrimitives.StdDev<double>(destination);
        if (stdDev == 0.0) return false;

        TensorPrimitives.Subtract(destination, mean, destination);
        TensorPrimitives.Divide(destination, stdDev, destination);
        return true;
    }

    static void ValidateRowKernelArgs<T>(
        ReadOnlySpan<T> rowMajor, ReadOnlySpan<bool> rowMajorNullMask,
        ReadOnlySpan<T> query, ReadOnlySpan<bool> queryNullMask,
        Span<T> output, Span<bool> outputMask, int rows, int cols,
        bool requireQuery, bool requireNonZeroCols = false)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows), "Row count must be non-negative.");
        if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols), "Column count must be non-negative.");
        if (requireNonZeroCols && cols <= 0) throw new ArgumentException("Column count must be at least 1 for this kernel.", nameof(cols));
        if ((long)rows * cols > rowMajor.Length)
            throw new ArgumentException($"Row-major span length ({rowMajor.Length}) must be at least {rows * cols}.", nameof(rowMajor));
        if (requireQuery && query.Length != cols)
            throw new ArgumentException($"Query length ({query.Length}) must equal the row width ({cols}).", nameof(query));
        if (requireQuery && queryNullMask.Length > 0 && queryNullMask.Length != cols)
            throw new ArgumentException($"Query null mask length ({queryNullMask.Length}) must equal the query length ({cols}).", nameof(queryNullMask));
        if (output.Length < rows)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least {rows}.", nameof(output));
        if (outputMask.Length < rows)
            throw new ArgumentException($"Output mask length ({outputMask.Length}) must be at least {rows}.", nameof(outputMask));
    }

    static bool AnyTrue(ReadOnlySpan<bool> mask)
    {
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) return true;
        return false;
    }

}
