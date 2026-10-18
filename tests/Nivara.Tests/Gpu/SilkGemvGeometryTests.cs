using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Device-free assertions on the fused decode GEMV's geometry: the host launch arithmetic in
/// <see cref="SilkKernels.GemvGlobalCols"/> against the <c>GEMVLANES</c>/<c>GEMVCOLS</c> macros the
/// kernel reads, the decoder's decode-only contract, and the per-width coverage the performance
/// argument rests on. The kernel-side half of the same arithmetic is pinned on a device by
/// <c>SilkRmsNormGemvTests</c> (bit-identity per slot, including tail widths).
/// </summary>
public sealed class SilkGemvGeometryTests
{
    [Test]
    public void GemvGeometry_AgreesWithTheSourceMacros()
    {
        string source = SilkKernels.Source;
        Assert.Multiple(() =>
        {
            Assert.That(source, Does.Contain($"#define GEMVLANES {SilkKernels.GemvGroupSize}"));
            Assert.That(source, Does.Contain($"#define GEMVCOLS {SilkKernels.GemvGroupCols}"));
            Assert.That(
                SilkKernels.GemvGroupCols,
                Is.EqualTo(SilkKernels.GemvGroupSize * SilkKernels.GemvBlockCols));

            // The decode and the tile kernel both read the one BLOCK macro, so their per-thread
            // column counts cannot drift apart without the source moving too.
            Assert.That(SilkKernels.GemvBlockCols, Is.EqualTo(SilkKernels.GemmBlockCols));
        });
    }

    [Test]
    public void GemvGlobalCols_EveryWidth_IsAWholeNumberOfGroups()
    {
        int[] widths = [1, 4, 24, 40, 96, 128, 192, 576, 896, 4864, 151936];
        foreach (int cols in widths)
        {
            int threads = SilkKernels.GemvGlobalCols(cols);
            Assert.Multiple(() =>
            {
                Assert.That(threads % SilkKernels.GemvGroupSize, Is.Zero, $"cols={cols}");
                Assert.That(
                    threads / SilkKernels.GemvGroupSize,
                    Is.EqualTo((cols + SilkKernels.GemvGroupCols - 1) / SilkKernels.GemvGroupCols),
                    $"cols={cols}");
                Assert.That(
                    threads * SilkKernels.GemvBlockCols - cols,
                    Is.LessThan(SilkKernels.GemvGroupCols),
                    $"cols={cols}");
            });
        }

        // A zero-width slot contributes no group: that is how trailing slots are omitted.
        Assert.That(SilkKernels.GemvGlobalCols(0), Is.Zero);
    }

    [Test]
    public void GemvGlobalCols_EveryModelledWidth_LeavesNoIdleLane()
    {
        // Qwen hidden, Qwen KV head dimension, Qwen intermediate, Qwen vocab.
        int[] widths = [896, 128, 4864, 151936];
        foreach (int cols in widths)
            Assert.That(
                SilkKernels.GemvGlobalCols(cols) * SilkKernels.GemvBlockCols,
                Is.EqualTo(cols),
                $"cols={cols}");
    }

    [Test]
    public void GemvGlobalCols_SmolLmWidths_OverlaunchOnlyTheTailGroup()
    {
        // SmolLM's hidden (576) and KV head dimension (192) are not whole groups; the kernel's
        // col >= bCols early return is what discards the surplus lanes.
        Assert.That(SilkKernels.GemvGlobalCols(576), Is.EqualTo(160));
        Assert.That(SilkKernels.GemvGlobalCols(576) * SilkKernels.GemvBlockCols, Is.EqualTo(640));
        Assert.That(SilkKernels.GemvGlobalCols(192), Is.EqualTo(64));
        Assert.That(SilkKernels.GemvGlobalCols(192) * SilkKernels.GemvBlockCols, Is.EqualTo(256));
    }

    [Test]
    public void RmsNormGemv_FusedDecodeKernel_IsSingleRow()
    {
        // The decode kernel must have no row dimension: aRows is gone from its parameter list, so
        // a one-row launch cannot silently become a TILE-row one. The tied-head tile kernel, which
        // prefill still uses, keeps its row dimension.
        string gemv = Signature(SilkKernels.RmsNormGemv);
        Assert.Multiple(() =>
        {
            Assert.That(gemv, Does.Not.Contain("aRows"));
            Assert.That(gemv, Does.Contain("aCols"));
            Assert.That(Signature(SilkKernels.RmsNormLmHead), Does.Contain("aRows"));
        });
    }

    [Test]
    public void RmsNormGemv_MultiRowArgument_IsRejected()
    {
        // The launcher keeps aRows so the call sites read the same before and after the decode
        // reshaping; passing a batch to the single-row kernel is a caller error, not a silent
        // one-row result.
        SilkLauncher.GemvSlot slot = new(Weight: 1, Bias: 0, Output: 2, Cols: 8);
        Assert.That(
            () => SilkLauncher.RmsNormGemv(
                runtime: null!, x: 0, gamma: 0, slots: [slot], hasBias: false, aRows: 2, aCols: 8, eps: 1e-5f),
            Throws.TypeOf<ArgumentOutOfRangeException>().With.Property("ParamName").EqualTo("aRows"));
    }

    static string Signature(string kernelName)
    {
        string source = SilkKernels.Source;
        int start = source.IndexOf($"__kernel void {kernelName}(", StringComparison.Ordinal);
        Assert.That(start, Is.GreaterThanOrEqualTo(0), $"SilkKernels.Source defines no {kernelName}");
        int end = source.IndexOf(')', start);
        Assert.That(end, Is.GreaterThan(start));
        return source[start..(end + 1)];
    }
}
