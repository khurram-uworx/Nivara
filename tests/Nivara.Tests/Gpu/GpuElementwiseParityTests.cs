using Nivara.AutoDiff.Operations;
using Nivara.Samples.Gpu;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the GPU path's scalar kernels against the CPU reference they are meant to mirror.
/// The GPU elementwise kernels carry their own A–S 7.1.26 erf port because ILGPU core and
/// <c>ILGPU.Algorithms</c> expose no erf, so a drift here silently changes inference output.
/// </summary>
[TestFixture]
public class GpuElementwiseParityTests
{
    /// <summary>
    /// The GPU erf port and the CPU <c>GradKernels.Erf</c> are the same polynomial but not the
    /// same instruction sequence: the CPU chains <c>T.FusedMultiplyAdd</c> while the GPU port
    /// does separate multiply-then-add, because XMath exposes no FMA. FMA keeps the intermediate
    /// product at full precision, so the two disagree by a few ULP on the polynomial — the
    /// agreement is close, not bit-exact, and this bound is what "a few ULP" measures out to.
    /// DistilBERT's f32 FFN already routes through this kernel, and the recorded GPU-vs-CPU gate
    /// (docs/BERT-GPU.md §3.6) passes at maxRel 3.2e-6 with it in the path.
    /// </summary>
    const float GeluRelativeTolerance = 1e-6f;

    [Test]
    public void GeluExact_GpuScalar_AgreesWithCpuKernel()
    {
        var input = new float[4096];
        for (int i = 0; i < input.Length; i++)
        {
            // Span the erf argument's interesting range: the sign flip at 0, the
            // near-cancellation around |z| ~ 1, and both tails where erf saturates to +/-1.
            float t = (i / (float)(input.Length - 1)) * 12f - 6f;
            input[i] = t;
        }

        var cpu = new float[input.Length];
        GradKernels.GeluExact<float>(input, cpu);

        float worst = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            float gpu = ElementwiseKernels.GeluExact(input[i]);
            float rel = Math.Abs(gpu - cpu[i]) / Math.Max(1f, Math.Abs(cpu[i]));
            if (rel > worst) worst = rel;
        }

        Assert.That(worst, Is.LessThan(GeluRelativeTolerance),
            $"GPU GeluExact deviates from CPU GradKernels.GeluExact by {worst:E3} relative, " +
            $"above the {GeluRelativeTolerance:E3} bound; the two erf ports have drifted apart.");
    }

    /// <summary>
    /// Guards the sign handling the A–S port gets wrong most easily. Because erf is odd,
    /// gelu splits into an even and an odd part: <c>gelu(v) + gelu(-v) = v·erf(v/√2)</c> and
    /// <c>gelu(v) − gelu(-v) = v</c>. The second identity is exact for every input — no erf
    /// saturation assumption — and it still discriminates: a dropped sign flip
    /// (<c>if (z &lt; 0) erf = -erf</c>) makes erf even, collapsing the difference to
    /// <c>v·(1 + erf(v/√2))</c> and failing here.
    /// </summary>
    [Test]
    public void GeluExact_GpuScalar_OddPartRecoversTheInput()
    {
        for (float v = -8f; v <= 8f; v += 0.125f)
        {
            float atV = ElementwiseKernels.GeluExact(v);
            float atNegativeV = ElementwiseKernels.GeluExact(-v);
            Assert.That(atV - atNegativeV, Is.EqualTo(v).Within(1e-4f),
                $"gelu({v}) - gelu({-v}) must recover the input through the odd part of erf.");
        }
    }
}
