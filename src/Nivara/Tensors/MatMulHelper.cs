using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Tensors;

/// <summary>
/// Central matrix-multiply helper. All AutoDiff MatMul callers route here.
///
/// .NET 10 implementation: transpose B + TensorPrimitives.Dot + Parallel.For.
/// When Tensor.MatrixMultiply&lt;T&gt; ships in a future .NET version, replace the
/// body of the Tensor&lt;T&gt; overload below — no callers change.
/// </summary>
static class MatMulHelper
{
    public static void Transpose<T>(ReadOnlySpan<T> src, Span<T> dst, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dst[j * rows + i] = src[i * cols + j];
    }

    /// <summary>
    /// PRIMARY — Tensor&lt;T&gt; level. Swap target for Tensor.MatrixMultiply.
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
    /// Core dense matmul on flat row-major spans. Copies to arrays first
    /// (ref structs can't be captured in Parallel.For lambdas), then runs
    /// transpose + TensorPrimitives.Dot + Parallel.For.
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
            {
                for (int idx = 0; idx < aLen; idx++)
                    if (aNullMask[idx]) aFilled[idx] = T.Zero;
            }

            if (hasBMask)
            {
                for (int idx = 0; idx < bLen; idx++)
                    if (bNullMask[idx]) bFilled[idx] = T.Zero;
            }

            MultiplyCore(aFilled.AsSpan(0, aLen), bFilled.AsSpan(0, bLen), result, aRows, aCols, bCols);

            PropagateNullMask(aNullMask, bNullMask, resultMask, aRows, aCols, bCols);

            for (int idx = 0; idx < resultMask.Length; idx++)
            {
                if (resultMask[idx])
                    result[idx] = T.Zero;
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aFilled, clearArray: true);
            ArrayPool<T>.Shared.Return(bFilled, clearArray: true);
        }
    }

    internal static void PropagateNullMask(
        ReadOnlySpan<bool> aNullMask,
        ReadOnlySpan<bool> bNullMask,
        Span<bool> resultMask,
        int aRows,
        int aCols,
        int bCols)
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
            {
                for (int i = 0; i < aRows; i++)
                {
                    int aRowOffset = i * aCols;
                    for (int k = 0; k < aCols; k++)
                    {
                        if (aNullMask[aRowOffset + k])
                        {
                            aRowsSpan[i] = true;
                            break;
                        }
                    }
                }
            }

            if (hasBMask)
            {
                for (int k = 0; k < aCols; k++)
                {
                    int bRowOffset = k * bCols;
                    for (int j = 0; j < bCols; j++)
                    {
                        if (bNullMask[bRowOffset + j])
                            bColsSpan[j] = true;
                    }
                }
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
}
