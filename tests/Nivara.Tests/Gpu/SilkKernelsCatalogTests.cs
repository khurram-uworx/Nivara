using Nivara.Samples.Gpu;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Device-free check that the Silk translation unit and <see cref="SilkKernels.KernelNames"/>
/// name the same kernels. A name in the list that the source does not define fails at
/// <c>clCreateKernel</c> with a message that does not say which model needed it.
/// </summary>
public sealed class SilkKernelsCatalogTests
{
    [Test]
    public void KernelNames_EachName_IsDefinedInSource()
    {
        Assert.That(SilkKernels.KernelNames, Is.Unique);

        string source = SilkKernels.Source;
        Assert.Multiple(() =>
        {
            foreach (string name in SilkKernels.KernelNames)
                Assert.That(source, Does.Contain($"__kernel void {name}("), name);
        });
    }

    [Test]
    public void KernelNames_ModernBertAndLayaPorts_AreInTheCatalog()
    {
        string[] required =
        [
            SilkKernels.LayerNorm,
            SilkKernels.Add,
            SilkKernels.AddBias,
            SilkKernels.Gemm,
            SilkKernels.SplitColumns,
            SilkKernels.Rotary,
            SilkKernels.GeGlu,
        ];

        Assert.That(SilkKernels.KernelNames, Is.SupersetOf(required));

        string source = SilkKernels.Source;
        int geglu = source.IndexOf("y[idx] = gelu_exact(x[xBase + col])", StringComparison.Ordinal);
        Assert.That(geglu, Is.GreaterThanOrEqualTo(0));
        Assert.That(source.IndexOf("x[xBase + intermediate + col]", geglu, StringComparison.Ordinal), Is.GreaterThan(geglu));

        int gemm = source.IndexOf("void gemm_row4(", StringComparison.Ordinal);
        int gemmEnd = source.IndexOf(')', gemm);
        string gemmSignature = source[gemm..gemmEnd];
        Assert.That(gemmSignature, Does.Not.Contain("bias"));
    }
}
