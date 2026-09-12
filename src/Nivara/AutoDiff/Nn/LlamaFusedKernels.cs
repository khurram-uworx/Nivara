using Nivara.AutoDiff.Operations;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Runtime toggles for the inference-only fused Llama kernels plus the zero-allocation fused
/// kernel helpers they rely on. Mirroring <see cref="Nivara.Primitives.NivaraPrimitives.UseWidenSimd"/>,
/// the default is on so cached inference outside a <see cref="Utilities.GradientUtils.Grad"/> scope
/// routes through the fused per-token decoder-block path; parity tests flip it to <c>false</c> to
/// force the byte-compatible per-op chain for A/B comparisons. Grad-enabled training is unaffected
/// (the fused kernels never run inside a Grad scope).
/// </summary>
public static class LlamaFusedKernels
{
    /// <summary>
    /// Gets or sets whether cached inference (decode and prefill) routes through the fused
    /// single-token decoder-block kernel instead of the per-op chain.
    /// </summary>
    public static bool DecoderBlockFused { get; set; } = true;

    /// <summary>
    /// Fused RMS normalization forward over <c>rows</c> rows of <c>cols</c>, in place, followed by
    /// the per-dimension gamma multiply. Runs the same <see cref="RMSNormKernel{T}"/> forward kernel
    /// and per-row <see cref="TensorPrimitives.Multiply"/> sequence as <see cref="RMSNorm{T}.Forward"/>'s
    /// inference path, so the result is bit-identical while allocating nothing on the heap. Exposes
    /// the internal <see cref="RMSNormKernel{T}"/> to callers outside the core assembly (e.g. the
    /// Llama samples).
    /// </summary>
    /// <typeparam name="T">The numeric element type implementing <see cref="IFloatingPointIeee754{T}"/></typeparam>
    /// <param name="buffer">The row-major <c>[rows, cols]</c> normalized data; must hold <c>rows * cols</c> elements</param>
    /// <param name="gamma">The per-dimension scale, length <c>cols</c></param>
    /// <param name="rows">Number of rows to normalize</param>
    /// <param name="cols">Width of a row (the normalized dimension; must be positive)</param>
    /// <param name="eps">Stability term added to each row's mean-square</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="buffer"/> is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="rows"/> or <paramref name="cols"/> is invalid</exception>
    /// <exception cref="ArgumentException">Thrown when the buffer cannot hold the data or <paramref name="gamma"/> is not length <paramref name="cols"/></exception>
    public static void RMSNormForwardInPlace<T>(T[] buffer, ReadOnlySpan<T> gamma, int rows, int cols, double eps)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        if (buffer.Length < rows * cols)
            throw new ArgumentException($"Buffer length ({buffer.Length}) must hold rows * cols ({rows * cols}).", nameof(buffer));
        if (gamma.Length != cols)
            throw new ArgumentException($"Gamma length ({gamma.Length}) must equal the normalized dimension ({cols}).", nameof(gamma));

        RMSNormKernel<T>.PerRowRMSNormForwardKernel(buffer, buffer, rows, cols, eps);
        for (int i = 0; i < rows; i++)
        {
            int baseIdx = i * cols;
            TensorPrimitives.Multiply(buffer.AsSpan(baseIdx, cols), gamma, buffer.AsSpan(baseIdx, cols));
        }
    }

    /// <summary>
    /// Row-major <c>[aRows, aCols] @ [bCols, aCols]^T -> [aRows, bCols]</c> dense matmul into a
    /// caller-owned output buffer. Delegates to the same <see cref="GradKernels.MatMulTransposedB{T}"/>
    /// kernel the per-op <c>MatMulTransposedB</c> op runs, so results are bit-identical; with a
    /// single input row it takes the allocation-free BLAS2 GEMV path used by the fused Llama head.
    /// Exposes the internal kernel to callers outside the core assembly (e.g. the Llama samples).
    /// </summary>
    /// <typeparam name="T">The numeric element type implementing <see cref="IFloatingPointIeee754{T}"/></typeparam>
    /// <param name="a">The first operand, <c>[aRows * aCols]</c> row-major</param>
    /// <param name="b">The second operand, <c>[aCols * bCols]</c> row-major</param>
    /// <param name="output">The caller-owned result buffer, length at least <c>aRows * bCols</c></param>
    /// <param name="aRows">Number of rows in <paramref name="a"/></param>
    /// <param name="aCols">Number of columns in <paramref name="a"/> (must equal <paramref name="b"/>'s row length)</param>
    /// <param name="bCols">Number of columns in <paramref name="b"/></param>
    /// <exception cref="ArgumentException">Thrown when the spans or output buffer are too small</exception>
    public static void MatMulTransposedB<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] output, int aRows, int aCols, int bCols)
        where T : struct, IFloatingPointIeee754<T>
        => GradKernels.MatMulTransposedB(a, b, output, aRows, aCols, bCols);
}
