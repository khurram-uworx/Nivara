using System.Numerics;

namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// Double-precision golden oracle for the probe kernels — the single
/// implementation-independent target every leg (CPU SIMD, L0/SPIR-V, DX12) is
/// compared against. BF16→double widen is exact (BF16 ⊂ float32 ⊂ double), double
/// products of BF16 (8-bit mantissa) operands are exact, and accumulation is serial
/// double — deliberately not a kernel, so it never depends on Nivara's reduction
/// order, the IGC codegen, or the HLSL expansion. Gate for every kernel/leg:
/// |leg − golden| ≤ 1e-6 + 1e-5·|golden|; per-leg worst ULP is a reported diagnostic.
/// </summary>
internal static class GoldenReferences
{
    public const double GateAbs = 1e-6;
    public const double GateRel = 1e-5;

    /// <summary>Serial double dot of two BF16 vectors — the strict three-way apple (K=16 / K=576).</summary>
    public static double GoldenDot(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += ToDouble(a[i]) * ToDouble(b[i]);
        return sum;
    }

    /// <summary>Row-major BF16 matvec y = W·x in double — the shape of every SmolLM linear.</summary>
    public static double[] GoldenGemv(ReadOnlySpan<BFloat16> w, ReadOnlySpan<BFloat16> x, int rows, int cols)
    {
        var result = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            int rowStart = r * cols;
            for (int c = 0; c < cols; c++)
                sum += ToDouble(w[rowStart + c]) * ToDouble(x[c]);
            result[r] = sum;
        }

        return result;
    }

    /// <summary>BF16 swish/SiLU x·σ(x) = x / (1 + exp(−x)) in double.</summary>
    public static double[] GoldenSilu(ReadOnlySpan<BFloat16> x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double v = ToDouble(x[i]);
            result[i] = v / (1.0 + Math.Exp(-v));
        }

        return result;
    }

    public static bool WithinTolerance(double leg, double golden)
        => Math.Abs(leg - golden) <= GateAbs + GateRel * Math.Abs(golden);

    /// <summary>Diagnostic: |leg − golden| expressed in f32 ULPs at the golden's magnitude.</summary>
    public static double UlpDistance(double leg, double golden)
    {
        float g32 = (float)golden;
        double ulp = Math.Max(
            Math.Abs((double)MathF.BitIncrement(g32) - (double)g32),
            Math.Abs((double)g32 - (double)MathF.BitDecrement(g32)));
        return ulp > 0 ? Math.Abs(leg - golden) / ulp : 0.0;
    }

    /// <summary>BF16→float is exact (same exponent width, superset mantissa); float→double is exact.</summary>
    private static double ToDouble(BFloat16 value) => (double)float.CreateChecked(value);
}