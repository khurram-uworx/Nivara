using ILGPU;
using ILGPU.Algorithms;

namespace Nivara.GpuProbe.Ilgpu;

/// <summary>
/// The three production SmolLM kernels as ILGPU device methods (implicitly-grouped,
/// <see cref="Index1D"/>-driven), fed the same byte-identical BF16 fixtures as every other
/// leg over the packed-2-per-uint transport shared with the D3D12 leg
/// (<c>D3d12.GemvKernels.PackBf16</c> / its in-shader <c>Widen</c>). BF16→f32 widening happens
/// in-shader via <see cref="Interop.IntAsFloat"/> (kernel-safe bit reinterpret, the ILGPU
/// mirror of HLSL <c>asfloat</c>); silu uses <see cref="XMath.Exp"/> from ILGPU.Algorithms
/// (compiles to the OpenCL <c>exp</c> intrinsic — core IntrinsicMath has no exp, see
/// docs/ILGPU.md). Every kernel bounds-checks its index because ILGPU's implicit grouping
/// pads the grid to group-size multiples (same reason the HLSL leg guards <c>dt.x &gt;= n</c>).
/// </summary>
internal static class IlgpuKernels
{
    /// <summary>BF16→f32 in-shader widen for the packed layout (element 2k in the HIGH half,
    /// 2k+1 in the LOW) — the ILGPU mirror of <c>D3d12.GemvKernels.Widen</c>.</summary>
    private static float Widen(uint packed, uint element) =>
        (element & 1u) == 0u
            ? Interop.IntAsFloat(packed & 0xFFFF0000u)
            : Interop.IntAsFloat((packed & 0xFFFFu) << 16);

    /// <summary>dot16: single thread, serial K=16 f32 accumulate over packed [a(K) | b(K)].</summary>
    internal static void Dot16Kernel(Index1D index, ArrayView<uint> input, ArrayView<float> output, int k)
    {
        int i = index;
        if (i > 0)
            return;
        float acc = 0f;
        for (int j = 0; j < k; j++)
            acc += Widen(input[j >> 1], (uint)j) * Widen(input[(k + j) >> 1], (uint)(k + j));
        output[0] = acc;
    }

    /// <summary>silu: per-element sigmoid-then-multiply (<c>x / (1 + exp(-x))</c>), like <c>Activation.Silu</c>.
    /// extent = <paramref name="n"/> elements.</summary>
    internal static void SiluKernel(Index1D index, ArrayView<uint> input, ArrayView<float> output, int n)
    {
        int i = index;
        if (i >= n)
            return;
        float x = Widen(input[i >> 1], (uint)i);
        output[i] = x / (1f + XMath.Exp(-x));
    }

    /// <summary>gemv: one thread per output row of y = W·x over packed [w(rows·cols) | x(cols)].
    /// extent = <paramref name="rows"/> threads, serial K = <paramref name="cols"/> each.</summary>
    internal static void GemvKernel(Index1D index, ArrayView<uint> input, ArrayView<float> output, int rows, int cols, int weightElems)
    {
        int row = index;
        if (row >= rows)
            return;
        float acc = 0f;
        for (int k = 0; k < cols; k++)
        {
            int w = row * cols + k;
            acc += Widen(input[w >> 1], (uint)w) * Widen(input[(weightElems + k) >> 1], (uint)(weightElems + k));
        }
        output[row] = acc;
    }
}