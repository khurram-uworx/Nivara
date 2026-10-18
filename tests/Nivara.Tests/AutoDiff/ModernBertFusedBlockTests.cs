using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class ModernBertFusedBlockTests
{
    static ModernBertConfig TinyConfig => new()
    {
        HiddenSize = 8,
        NumAttentionHeads = 2,
        NumHiddenLayers = 3,
        IntermediateSize = 4,
        VocabSize = 32,
        LocalAttention = 6,
        GlobalAttnEveryNLayers = 2,
        NormEps = 1e-5f,
    };

    static float[] Filled(int n, int seed)
    {
        var rnd = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = (float)(rnd.NextDouble() * 2 - 1);
        return arr;
    }

    static void AssertClose(float[] a, float[] b, float tol, string what)
    {
        Assert.That(b.Length, Is.EqualTo(a.Length), $"{what}: length mismatch.");
        for (int i = 0; i < a.Length; i++)
            Assert.That(Math.Abs(a[i] - b[i]), Is.LessThanOrEqualTo(tol),
                $"{what}[{i}]: expected {a[i]}, got {b[i]}.");
    }

    [Test]
    public void ForwardFused_MatchesPerOpChain_Within1e5()
    {
        var config = TinyConfig;
        foreach (int layerIndex in new[] { 0, 1 })
        {
            var layer = new ModernBertLayer<float>(config, layerIndex);
            int seqLen = 5;
            int hidden = config.HiddenSize;
            var input = Filled(seqLen * hidden, 7 + layerIndex);
            var inTensor = ReverseGradTensor<float>.FromMatrix(input, seqLen, hidden, requiresGrad: false);
            int band = config.IsFullAttention(layerIndex) ? -1 : config.SlidingWindow;
            int validLength = seqLen - 1;

            var perOp = layer.Forward(inTensor, band, validLength);
            var fused = new float[seqLen * hidden];
            layer.ForwardFused(input, fused, band, validLength);

            perOp.Data.TryGetSpan(out var perSpan);
            AssertClose(perSpan.ToArray(), fused, 1e-5f, $"layer {layerIndex} output");
        }
    }

    [Test]
    public void ForwardFused_Layer0_MissingAttnNorm_Works()
    {
        var config = TinyConfig;
        var layer = new ModernBertLayer<float>(config, 0);
        Assert.That(layer.attnNorm, Is.Null);
        int seqLen = 4;
        int hidden = config.HiddenSize;
        var input = Filled(seqLen * hidden, 11);
        var fused = new float[seqLen * hidden];
        layer.ForwardFused(input, fused, -1, seqLen);
        foreach (var v in fused)
            Assert.That(float.IsFinite(v), Is.True);
    }

    [Test]
    public void ForwardFused_FullyMaskedRow_ProducesZeros_NotNaN()
    {
        var config = TinyConfig;
        // sliding layer (layer 1)
        var layer = new ModernBertLayer<float>(config, 1);
        int seqLen = 8;
        int hidden = config.HiddenSize;
        var input = Filled(seqLen * hidden, 21);
        int band = config.SlidingWindow; // 3
        int validLength = 2; // rows >= 5 fully masked
        var fused = new float[seqLen * hidden];
        layer.ForwardFused(input, fused, band, validLength);
        // row 5 (0-based) should be finite (no NaN) — zeros in attention context but residual/MLP finite
        for (int i = 0; i < fused.Length; i++)
            Assert.That(float.IsNaN(fused[i]) || float.IsInfinity(fused[i]), Is.False,
                $"fused[{i}] must be finite.");
    }

    [Test]
    public void ForwardFused_ThrowsInsideGrad()
    {
        var config = TinyConfig;
        var layer = new ModernBertLayer<float>(config, 1);
        int hidden = config.HiddenSize;
        var input = new float[4 * hidden];
        var output = new float[4 * hidden];
        using (GradientUtils.Grad())
        {
            Assert.Throws<InvalidOperationException>(() =>
                layer.ForwardFused(input, output, config.SlidingWindow, 4));
        }
    }
}
