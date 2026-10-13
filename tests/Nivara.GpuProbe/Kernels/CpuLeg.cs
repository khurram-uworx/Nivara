using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AI;
using Nivara.Samples;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// CPU leg of the multi-leg gate: the production Nivara kernels exactly as
/// <c>NivaraInference</c> calls them for SmolLM, consumed read-only through public API.
/// This is the gold target - every GPU leg (L0 hand-authored, SYCL/oneAPI, DX12) is
/// gated against it, not against a hand-rolled or double-precision oracle; the
/// per-backend results are tabulated in [docs/ILGPU.md](../../../docs/ILGPU.md).
/// </summary>
internal static class CpuLeg
{
    public const double GateAbs = 1e-6;
    public const double GateRel = 1e-5;

    /// <summary>BF16→F32 once at load, via the production SIMD widen (lossless).</summary>
    public static float[] Widen(ReadOnlySpan<BFloat16> values)
    {
        var f32 = new float[values.Length];
        SafeTensorsLoader.WidenBf16ToF32(MemoryMarshal.Cast<BFloat16, ushort>(values), f32);
        return f32;
    }

    /// <summary>Pure dot through the production GEMV kernel (aRows=1, bCols=1).</summary>
    public static float Dot(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        var wa = Widen(a);
        var wb = Widen(b);
        var output = new float[1];
        LlamaFusedKernels.MatMulTransposedB(wa, wb, output, aRows: 1, aCols: a.Length, bCols: 1);
        return output[0];
    }

    /// <summary>
    /// GEMV y = W·x through the production kernel in the fused-head call shape
    /// (aRows=1 → the allocation-free BLAS2 GEMV path). <paramref name="w"/> is the
    /// row-major W ∈ [rows × cols] flattening.
    /// </summary>
    public static float[] Gemv(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> w, int rows, int cols)
    {
        var wx = Widen(x);
        var ww = Widen(w);
        var output = new float[rows];
        LlamaFusedKernels.MatMulTransposedB(wx, ww, output, aRows: 1, aCols: cols, bCols: rows);
        return output;
    }

    /// <summary>
    /// Production SiLU (sigmoid-then-multiply, <c>GradKernels.Silu</c>) via the public
    /// <see cref="Activation.Silu{T}"/> entry point. Runs outside any <c>Grad()</c> scope,
    /// so no graph nodes are created (inference default). Values are copied before the
    /// tensors are disposed.
    /// </summary>
    public static float[] Silu(ReadOnlySpan<BFloat16> values)
    {
        var f32 = Widen(values);
        using var input = ReverseGradTensor<float>.FromArray(f32);
        using var result = Activation.Silu(input);
        result.Data.TryGetSpan(out var span);
        return span.ToArray();
    }

    /// <summary>
    /// The <see cref="Silu"/> formula computed in f32 then rounded to Half once: the
    /// reference an F16-executed elementwise sigmoid+multiply must match. The BCL has
    /// no Half exponentiation (the same limitation recorded for the Silk.NET fp16
    /// diagnostic), so this is the f32-then-round form only. Derived from the same
    /// production widen the gold is computed from — never from a leg's own output.
    /// </summary>
    public static ushort[] SiluF16ReferenceBits(ReadOnlySpan<BFloat16> values)
    {
        var f32 = Widen(values);
        var bits = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
            bits[i] = BitConverter.HalfToUInt16Bits((Half)(f32[i] / (1f + MathF.Exp(-f32[i]))));
        return bits;
    }

    /// <summary>Ordered distance in the half lattice: sign-magnitude bits are mapped
    /// to a two's-complement order so a value and its neighbour differ by 1, including
    /// across zero.</summary>
    public static int HalfUlpDistance(ushort a, ushort b)
    {
        static int Ordered(ushort bits) => (bits & 0x8000) != 0 ? 0x8000 - (bits & 0x7FFF) : bits;
        return Math.Abs(Ordered(a) - Ordered(b));
    }

    public static bool WithinTolerance(double leg, double reference)
        => Math.Abs(leg - reference) <= GateAbs + GateRel * Math.Abs(reference);

    /// <summary>Diagnostic: |leg − reference| in f32 ULPs at the reference's magnitude.</summary>
    public static double UlpDistance(double leg, double reference)
    {
        float r32 = (float)reference;
        double ulp = Math.Max(
            Math.Abs((double)MathF.BitIncrement(r32) - (double)r32),
            Math.Abs((double)r32 - (double)MathF.BitDecrement(r32)));
        return ulp > 0 ? Math.Abs(leg - reference) / ulp : 0.0;
    }

    /// <summary>
    /// The CPU leg over the full fixture set — the gold target for all other legs,
    /// with per-kernel wall time in microseconds.
    /// </summary>
    public static LegResults ComputeLeg(KernelFixtures fixtures)
    {
        var sw = new System.Diagnostics.Stopwatch();

        sw.Restart();
        float dot = Dot(fixtures.Dot16A, fixtures.Dot16B);
        double dot16Us = sw.Elapsed.TotalMicroseconds;

        sw.Restart();
        float[] silu = Silu(fixtures.SiluX);
        double siluUs = sw.Elapsed.TotalMicroseconds;

        sw.Restart();
        float[] gemv = Gemv(fixtures.GemvX, fixtures.GemvW, KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize);
        double gemvUs = sw.Elapsed.TotalMicroseconds;

        return new LegResults(dot, silu, gemv, dot16Us, siluUs, gemvUs);
    }
}
