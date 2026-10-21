using Nivara.Silk;
using Nivara.Tests.NivaraTorch;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the Silk (OpenCL) attention kernels against the PyTorch reference fixtures generated
/// by <c>samples/NivaraTorch/gen_reference.py</c> (stored under
/// <c>samples/data/torch-comparison/</c>). Complements the existing CPU Torch-parity suites and
/// the kernel-level Silk vs ILGPU gate: these tests exercise the Silk path end-to-end against
/// an independent reference rather than against our own other backend.
/// </summary>
[TestFixture]
public sealed class SilkAttentionTorchParityTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk attention Torch parity."); throw; }
    }

    static nint Upload(SilkRuntime silk, float[] data)
    {
        var buf = silk.Allocate(data.Length);
        silk.Upload(buf, data);
        return buf;
    }

    static void ReleaseAll(SilkRuntime silk, params nint[] bufs)
    {
        foreach (var b in bufs) silk.Release(b);
    }

    // ---- batched_attention: banded bidirectional + padding -----------------

    [Test]
    public void BatchedAttention_BandedBidirectionalPadding_MatchesPyTorch()
    {
        var q = TestHelpers.LoadBin("attn_band_padding_q.bin");
        var k = TestHelpers.LoadBin("attn_band_padding_k.bin");
        var v = TestHelpers.LoadBin("attn_band_padding_v.bin");
        var expected = TestHelpers.LoadBin("attn_band_padding_output.bin");

        const int seqLen = 8;
        const int numHeads = 4;
        const int headDim = 4; // ModelDim 16 / 4
        const int validLength = 6;
        const int band = 1;
        const float scale = 0.5f;

        var mask = new float[seqLen];
        for (int j = 0; j < validLength; j++) mask[j] = 1f;

        using var silk = Device();
        nint sq = Upload(silk, q), sk = Upload(silk, k), sv = Upload(silk, v);
        nint sm = Upload(silk, mask);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Attention(silk, sq, sk, sv, sm, so, 1, seqLen, numHeads, headDim, band, scale);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Attention banded bidirectional + padding");
        ReleaseAll(silk, sq, sk, sv, sm, so);
    }

    [Test]
    public void BatchedAttention_BandedBidirectionalPadding_FullyMaskedRowIsZeros()
    {
        var q = TestHelpers.LoadBin("attn_band_padding_q.bin");
        var k = TestHelpers.LoadBin("attn_band_padding_k.bin");
        var v = TestHelpers.LoadBin("attn_band_padding_v.bin");

        const int seqLen = 8;
        const int modelDim = 16;
        const int numHeads = 4;
        const int headDim = 4;
        const int validLength = 6;
        const int band = 1;
        const float scale = 0.5f;
        const int fullyMaskedRow = 7;

        var mask = new float[seqLen];
        for (int j = 0; j < validLength; j++) mask[j] = 1f;

        using var silk = Device();
        nint sq = Upload(silk, q), sk = Upload(silk, k), sv = Upload(silk, v);
        nint sm = Upload(silk, mask);
        nint so = silk.Allocate(q.Length);
        SilkLauncher.Attention(silk, sq, sk, sv, sm, so, 1, seqLen, numHeads, headDim, band, scale);
        silk.Finish();
        var actual = silk.Download(so, q.Length);

        var row = new float[modelDim];
        Array.Copy(actual, fullyMaskedRow * modelDim, row, 0, modelDim);
        Assert.That(row, Is.All.EqualTo(0f), "a fully-masked query row must be exactly zeros");
        for (int i = 0; i < actual.Length; i++)
            Assert.That(float.IsNaN(actual[i]), Is.False, "no NaN may reach the output");

        ReleaseAll(silk, sq, sk, sv, sm, so);
    }

    [Test]
    public void BatchedAttention_GlobalBand_MatchesSelfAttentionPyTorch()
    {
        var q = TestHelpers.LoadBin("attn_self_q.bin");
        var k = TestHelpers.LoadBin("attn_self_k.bin");
        var v = TestHelpers.LoadBin("attn_self_v.bin");
        var expected = TestHelpers.LoadBin("attn_self_output.bin");

        const int seqLen = 4;
        const int numHeads = 4;
        const int headDim = 4; // ModelDim 16 / 4
        const float scale = 0.5f;

        var mask = new float[seqLen];
        for (int j = 0; j < seqLen; j++) mask[j] = 1f;

        using var silk = Device();
        nint sq = Upload(silk, q), sk = Upload(silk, k), sv = Upload(silk, v);
        nint sm = Upload(silk, mask);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Attention(silk, sq, sk, sv, sm, so, 1, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Attention global band (attn_self)");
        ReleaseAll(silk, sq, sk, sv, sm, so);
    }

    // ---- causal_gqa_attention: multi-head attention fixtures ----------------

    [Test]
    public void CausalGqaAttention_SelfCausal_MatchesPyTorch()
    {
        var q = TestHelpers.LoadBin("attn_self_causal_q.bin");
        var k = TestHelpers.LoadBin("attn_self_causal_k.bin");
        var v = TestHelpers.LoadBin("attn_self_causal_v.bin");
        var expected = TestHelpers.LoadBin("attn_self_causal_output.bin");

        const int seqLen = 4;
        const int numHeads = 4;
        const int numKvHeads = 4; // MHA: no GQA repeat
        const int headDim = 4; // ModelDim 16 / 4
        const float scale = 0.5f;

        using var silk = Device();
        nint sq = Upload(silk, q), sk = Upload(silk, k), sv = Upload(silk, v);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.CausalGqaAttention(silk, sq, sk, sv, so, seqLen, numHeads, numKvHeads, headDim, scale);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk CausalGqaAttention self-causal");
        ReleaseAll(silk, sq, sk, sv, so);
    }
}