using System.Numerics;
using System.Runtime.InteropServices;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;

namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// CPU leg of the multi-leg gate: the production Nivara kernels exactly as
/// <c>NivaraInference</c> calls them for SmolLM, consumed read-only through public API.
/// This is the gold target — every GPU leg (L0 hand-authored, SYCL/oneAPI, DX12) is
/// gated against it (see docs/TODO.md), not against a hand-rolled or double-precision oracle.
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
    /// The CPU leg over the full fixture set — the gold target for all other legs.
    /// </summary>
    public static LegResults ComputeLeg(KernelFixtures fixtures) => new(
        Dot(fixtures.Dot16A, fixtures.Dot16B),
        Silu(fixtures.SiluX),
        Gemv(fixtures.GemvX, fixtures.GemvW, KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize));
}