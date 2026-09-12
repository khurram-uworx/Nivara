using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaDecoderBlockTests
{
    // No global Grad() scope: the inference guard is that no graph nodes are built outside Grad().

    [Test]
    public void Inference_OutsideGrad_PreservesShapeAndBuildsNoGraph()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024, seqLen = 6;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(3);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);

        var output = block.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, hidden }));
        Assert.That(output.IsLeaf, Is.True, "Decoder block inference outside Grad() must not build graph nodes.");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesFiniteGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, intermediate = 512, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(5);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = block.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(seqLen * hidden));
        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Block input gradient must be finite.");
    }

    [Test]
    public void Constructor_QkvBiasTrue_CreatesAttentionQkvBiasesOnly()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, qkvBias: true);

        Assert.That(block.Attention.QProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.KProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.VProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.OProj.Bias, Is.Null);
        Assert.That(block.GateProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
        Assert.That(block.UpProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
        Assert.That(block.DownProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
    }

    [Test]
    public void Constructor_QkvBiasFalse_AttentionBiasFree()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, qkvBias: false);

        Assert.That(block.Attention.QProj.Bias, Is.Null);
        Assert.That(block.Attention.KProj.Bias, Is.Null);
        Assert.That(block.Attention.VProj.Bias, Is.Null);
        Assert.That(block.Attention.OProj.Bias, Is.Null);
    }

    [Test]
    public void Forward_InsideGrad_QkvBiasTrue_AccumulatesFiniteBiasGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, intermediate = 512, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(17);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = block.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(block.Attention.QProj.Bias!.Tensor.Grad, Is.Not.Null, "q_proj bias gradient must flow");
        Assert.That(block.Attention.KProj.Bias!.Tensor.Grad, Is.Not.Null, "k_proj bias gradient must flow");
        Assert.That(block.Attention.VProj.Bias!.Tensor.Grad, Is.Not.Null, "v_proj bias gradient must flow");
        foreach (var g in new[] { block.Attention.QProj.Bias!.Tensor.Grad!, block.Attention.KProj.Bias!.Tensor.Grad!, block.Attention.VProj.Bias!.Tensor.Grad! })
            foreach (var v in g)
                Assert.That(float.IsNaN(v) || float.IsInfinity(v), Is.False, "Block bias gradient must be finite.");
    }

    [Test]
    public void Forward_ResidualAdds_ChangeOutputFromRawNormPath()
    {
        // With all weights identity-ish (Linear defaults are small Kaiming), the residual
        // adds still guarantee the output is a finite, non-zero tensor different from pure
        // attention alone. This is a structural smoke check, not a numeric reference.
        const int hidden = 64, numHeads = 4, numKvHeads = 2, intermediate = 128, seqLen = 3;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 8);

        var inputData = new float[seqLen * hidden];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = 1f;
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);
        var output = block.Forward(input);

        Assert.That(output.Length, Is.EqualTo(seqLen * hidden));
        Assert.That(output[0], Is.Not.EqualTo(0f));
    }

    static int KvWidth(int heads, int kvHeads, int hidden) => kvHeads * (hidden / heads);

    static float[] Filled(int n, int seed)
    {
        var rnd = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = (float)(rnd.NextDouble() * 2 - 1);
        return arr;
    }

    static void SeedCaches(float[] kCache, float[] vCache, int kvLen, int kvWidth, int seed)
    {
        var rnd = new Random(seed);
        for (int i = 0; i < kvLen * kvWidth; i++)
        {
            kCache[i] = (float)(rnd.NextDouble() * 2 - 1);
            vCache[i] = (float)(rnd.NextDouble() * 2 - 1);
        }
    }

    static void AssertWithin(float[] a, float[] b, float tol, string what)
    {
        Assert.That(b.Length, Is.EqualTo(a.Length), $"{what}: length mismatch.");
        for (int i = 0; i < a.Length; i++)
            Assert.That(Math.Abs(a[i] - b[i]), Is.LessThanOrEqualTo(tol),
                $"{what}[{i}]: expected {a[i]}, got {b[i]}.");
    }

    [Test]
    public void ForwardCachedFused_MatchesPerOpChain_Within1e5()
    {
        // The fused single-token kernel (GEMV + DecodeAttention + RMSNorm + RoPE + SiLU) must
        // reproduce the per-op block.ForwardCached chain within 1e-5, with and without qkvBias.
        foreach (bool qkvBias in new[] { false, true })
        {
            const int hidden = 64, numHeads = 4, numKvHeads = 2, kvLen = 8, maxPos = 32;
            const int intermediate = 192;
            using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate,
                rmsNormEps: 1e-5f, maxPositionEmbeddings: maxPos, qkvBias: qkvBias);
            int kvWidth = KvWidth(numHeads, numKvHeads, hidden);

            var perOpK = new float[(kvLen + 1) * kvWidth];
            var perOpV = new float[(kvLen + 1) * kvWidth];
            var fusedK = new float[(kvLen + 1) * kvWidth];
            var fusedV = new float[(kvLen + 1) * kvWidth];
            SeedCaches(perOpK, perOpV, kvLen, kvWidth, 101);
            SeedCaches(fusedK, fusedV, kvLen, kvWidth, 101);
            var input = Filled(hidden, 7);

            var inTensor = ReverseGradTensor<float>.FromMatrix(input, 1, hidden, requiresGrad: false);
            var perOp = block.ForwardCached(inTensor, kvLen, perOpK, perOpV, kvLen);
            var fused = new float[hidden];
            block.ForwardCachedFused(input, fused, kvLen, fusedK, fusedV, kvLen);

            perOp.Data.TryGetSpan(out var perSpan);
            var label = qkvBias ? "fused decode output (qkvBias)" : "fused decode output";
            AssertWithin(perSpan.ToArray(), fused, 1e-5f, label);
            AssertWithin(perOpK, fusedK, 1e-5f, $"K cache rows ({label})");
            AssertWithin(perOpV, fusedV, 1e-5f, $"V cache rows ({label})");
        }
    }

    [Test]
    public void ForwardPrefillFused_MatchesPerOpPrefill_Within1e5()
    {
        // The fused batched kernel must reproduce block.ForwardPrefill within 1e-5 across the
        // GQA ratios 4/2 and 8/2 (headDim 16 via hidden/heads).
        foreach (var (hidden, heads, kvHeads) in new[] { (64, 4, 2), (128, 8, 2) })
        {
            using var block = new LlamaDecoderBlock<float>(hidden, heads, kvHeads, 3 * hidden);
            int kvWidth = KvWidth(heads, kvHeads, hidden);
            int L = 6;
            var perOpK = new float[L * kvWidth];
            var perOpV = new float[L * kvWidth];
            var fusedK = new float[L * kvWidth];
            var fusedV = new float[L * kvWidth];
            var rnd = new Random(23);
            var input = new float[L * hidden];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rnd.NextDouble() * 2 - 1);

            var inTensor = ReverseGradTensor<float>.FromMatrix(input, L, hidden, requiresGrad: false);
            var perOp = block.ForwardPrefill(inTensor, 0, perOpK, perOpV);
            var fused = new float[L * hidden];
            block.ForwardPrefillFused(input, fused, 0, fusedK, fusedV);

            perOp.Data.TryGetSpan(out var perSpan);
            AssertWithin(perSpan.ToArray(), fused, 1e-5f, $"prefill output (hidden={hidden})");
            AssertWithin(perOpK, fusedK, 1e-5f, $"prefill K cache (hidden={hidden})");
            AssertWithin(perOpV, fusedV, 1e-5f, $"prefill V cache (hidden={hidden})");
        }
    }

    [Test]
    public void ForwardCachedFused_TwoStep_MatchesPerOpWalk()
    {
        // Two successive fused decode steps reusing the same per-block scratch must match the
        // per-op walk row-for-row (scratch-aliasing guard: stale buffer values must never leak
        // into a later step's cache rows or output).
        const int hidden = 64, numHeads = 4, numKvHeads = 2, cacheLen0 = 2, maxPos = 32;
        const int intermediate = 192;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate,
            qkvBias: true, maxPositionEmbeddings: maxPos);
        int kvWidth = KvWidth(numHeads, numKvHeads, hidden);
        int capacity = cacheLen0 + 2;
        var perOpK = new float[capacity * kvWidth];
        var perOpV = new float[capacity * kvWidth];
        var fusedK = new float[capacity * kvWidth];
        var fusedV = new float[capacity * kvWidth];
        SeedCaches(perOpK, perOpV, cacheLen0, kvWidth, 202);
        SeedCaches(fusedK, fusedV, cacheLen0, kvWidth, 202);
        var step0 = Filled(hidden, 41);
        var step1 = Filled(hidden, 42);

        var p0 = block.ForwardCached(ReverseGradTensor<float>.FromMatrix(step0, 1, hidden, requiresGrad: false),
            cacheLen0, perOpK, perOpV, cacheLen0);
        var p1 = block.ForwardCached(ReverseGradTensor<float>.FromMatrix(step1, 1, hidden, requiresGrad: false),
            cacheLen0 + 1, perOpK, perOpV, cacheLen0 + 1);

        var f0 = new float[hidden];
        var f1 = new float[hidden];
        block.ForwardCachedFused(step0, f0, cacheLen0, fusedK, fusedV, cacheLen0);
        block.ForwardCachedFused(step1, f1, cacheLen0 + 1, fusedK, fusedV, cacheLen0 + 1);

        p0.Data.TryGetSpan(out var p0Span);
        p1.Data.TryGetSpan(out var p1Span);
        AssertWithin(p0Span.ToArray(), f0, 1e-5f, "step-0 fused output");
        AssertWithin(p1Span.ToArray(), f1, 1e-5f, "step-1 fused output");
        AssertWithin(perOpK, fusedK, 1e-5f, "K cache after both steps");
        AssertWithin(perOpV, fusedV, 1e-5f, "V cache after both steps");
    }

    [Test]
    public void ForwardCachedFused_SteadyState_AllocatesNothing()
    {
        // Qwen2.5-0.5B block shapes: the fused kernel must be allocation-free per call in
        // steady state (per-layer scratch reused, DecodeAttention score buffer pooled).
        const int hidden = 896, numHeads = 14, numKvHeads = 2, intermediate = 4864, kvLen = 64;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate);
        int kvWidth = KvWidth(numHeads, numKvHeads, hidden);
        var kCache = new float[(kvLen + 1) * kvWidth];
        var vCache = new float[(kvLen + 1) * kvWidth];
        SeedCaches(kCache, vCache, kvLen, kvWidth, 31);
        var input = Filled(hidden, 9);
        var output = new float[hidden];

        // Warm the RoPE tables, the per-block scratch, and the ArrayPool score buffer.
        for (int i = 0; i < 3; i++)
            block.ForwardCachedFused(input, output, kvLen, kCache, vCache, kvLen);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++)
            block.ForwardCachedFused(input, output, kvLen, kCache, vCache, kvLen);
        long after = GC.GetAllocatedBytesForCurrentThread();

        Assert.That(after - before, Is.LessThan(2048),
            "Fused decode block must not allocate per call in steady state.");
    }

    [Test]
    public void FusedKernels_InsideGrad_Throw()
    {
        const int hidden = 64, numHeads = 4, numKvHeads = 2, kvLen = 2, maxPos = 32;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, 3 * hidden,
            maxPositionEmbeddings: maxPos);
        int kvWidth = KvWidth(numHeads, numKvHeads, hidden);
        var kCache = new float[(kvLen + 1) * kvWidth];
        var vCache = new float[(kvLen + 1) * kvWidth];
        var input = Filled(hidden, 1);
        var output = new float[hidden];

        using (GradientUtils.Grad())
        {
            Assert.Throws<InvalidOperationException>(
                () => block.ForwardCachedFused(input, output, kvLen, kCache, vCache, kvLen),
                "fused decode inside Grad scope");
            Assert.Throws<InvalidOperationException>(
                () => block.ForwardPrefillFused(input, output, 0, kCache, vCache),
                "fused prefill inside Grad scope");
        }
    }
}
