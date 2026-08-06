using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Span-level kernels for fused multi-head scaled dot-product attention.
///
/// Q/K/V are row-major [rows, D]; each head owns contiguous columns
/// [h*headDim, (h+1)*headDim). Head matrices are packed once into a
/// [numHeads, rows, headDim] contiguous layout so every matmul below feeds
/// the SIMD <see cref="TensorsHelper.MultiplyCore{T}"/> path with zero
/// per-head transposes (QK^T uses the transposed-B layout directly).
/// </summary>
internal static class AttentionKernels<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Gathers one head's contiguous columns from a [rows, D] matrix into a
    /// contiguous [rows, headDim] span.
    /// </summary>
    public static void GatherHead(ReadOnlySpan<T> src, Span<T> dst, int rows, int D, int head, int headDim)
    {
        int hs = head * headDim;
        for (int r = 0; r < rows; r++)
            src.Slice(r * D + hs, headDim).CopyTo(dst.Slice(r * headDim, headDim));
    }

    /// <summary>
    /// Scatters a contiguous [rows, headDim] span back into one head's columns
    /// of a [rows, D] matrix.
    /// </summary>
    public static void ScatterHead(ReadOnlySpan<T> src, Span<T> dst, int rows, int D, int head, int headDim)
    {
        int hs = head * headDim;
        for (int r = 0; r < rows; r++)
            src.Slice(r * headDim, headDim).CopyTo(dst.Slice(r * D + hs, headDim));
    }

    /// <summary>
    /// Packs a [rows, D] matrix into [numHeads, rows, headDim] head-major layout.
    /// </summary>
    public static void PackHeads(ReadOnlySpan<T> src, Span<T> dst, int rows, int numHeads, int headDim)
    {
        for (int h = 0; h < numHeads; h++)
            GatherHead(src, dst.Slice(h * rows * headDim, rows * headDim), rows, numHeads * headDim, h, headDim);
    }

    /// <summary>
    /// In-place row-wise softmax (max subtraction, exp, normalize). Delegates to
    /// <see cref="GradKernels.SoftmaxRowsInPlace{T}"/> so attention and the
    /// Softmax op share one kernel.
    /// </summary>
    public static void SoftmaxRows(Span<T> x, int rows, int cols)
        => GradKernels.SoftmaxRowsInPlace(x, rows, cols);

    /// <summary>
    /// In-place softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - dot(P_i, dP_i)).
    /// float/double use a two-pass <see cref="TensorPrimitives"/> chain
    /// (Multiply then MultiplyAdd); Half/BFloat16 use a scalar loop.
    /// </summary>
    public static void SoftmaxBackwardRows(ReadOnlySpan<T> weights, Span<T> dS, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var w = weights.Slice(r * cols, cols);
            var g = dS.Slice(r * cols, cols);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
            {
                T dot = TensorPrimitives.Dot(w, g);
                TensorPrimitives.Subtract(g, dot, g);
                TensorPrimitives.Multiply(g, w, g);
                continue;
            }

            double dotAcc = 0.0;
            for (int i = 0; i < cols; i++)
                dotAcc += double.CreateChecked(w[i]) * double.CreateChecked(g[i]);
            for (int i = 0; i < cols; i++)
                g[i] = T.CreateChecked(double.CreateChecked(w[i]) * (double.CreateChecked(g[i]) - dotAcc));
        }
    }
}
