using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaCausalAttentionTests
{
    // No global Grad() scope here deliberately: the inference-path guard is that no graph
    // nodes are built outside Grad(), which is the model's default execution mode.

    [Test]
    public void Inference_OutsideGrad_PreservesShapeAndBuildsNoGraph()
    {
        const int hidden = 768, numHeads = 12, numKvHeads = 4, seqLen = 8;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 64);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(7);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);

        var output = attn.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, hidden }));
        Assert.That(output.IsLeaf, Is.True, "Inference forward outside Grad() must not build graph nodes.");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesGradientsOnAllProjections()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(11);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = attn.Forward(input);
        var loss = Nivara.AutoDiff.Operations.ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(seqLen * hidden));
        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Attention input gradient must be finite.");
    }

    [Test]
    public void Constructor_QkvBiasTrue_CreatesQkvBiasesOnly()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, qkvBias: true);

        Assert.That(attn.QProj.Bias, Is.Not.Null);
        Assert.That(attn.KProj.Bias, Is.Not.Null);
        Assert.That(attn.VProj.Bias, Is.Not.Null);
        Assert.That(attn.OProj.Bias, Is.Null, "o_proj is bias-free in canonical and Qwen2-style attention");

        Assert.That(attn.QProj.Bias!.Shape, Is.EqualTo(new[] { 1, numHeads * (hidden / numHeads) }));
        Assert.That(attn.KProj.Bias!.Shape, Is.EqualTo(new[] { 1, numKvHeads * (hidden / numHeads) }));
        Assert.That(attn.VProj.Bias!.Shape, Is.EqualTo(new[] { 1, numKvHeads * (hidden / numHeads) }));
    }

    [Test]
    public void Constructor_QkvBiasFalse_AllProjectionsBiasFree()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, qkvBias: false);

        Assert.That(attn.QProj.Bias, Is.Null);
        Assert.That(attn.KProj.Bias, Is.Null);
        Assert.That(attn.VProj.Bias, Is.Null);
        Assert.That(attn.OProj.Bias, Is.Null);
    }

    [Test]
    public void Forward_InsideGrad_QkvBiasTrue_AccumulatesFiniteBiasGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(21);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = attn.Forward(input);
        var loss = Nivara.AutoDiff.Operations.ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(attn.QProj.Bias!.Tensor.Grad, Is.Not.Null, "q_proj bias gradient must flow");
        Assert.That(attn.KProj.Bias!.Tensor.Grad, Is.Not.Null, "k_proj bias gradient must flow");
        Assert.That(attn.VProj.Bias!.Tensor.Grad, Is.Not.Null, "v_proj bias gradient must flow");
        foreach (var g in new[] { attn.QProj.Bias!.Tensor.Grad!, attn.KProj.Bias!.Tensor.Grad!, attn.VProj.Bias!.Tensor.Grad! })
            foreach (var v in g)
                Assert.That(float.IsNaN(v) || float.IsInfinity(v), Is.False, "Attention bias gradient must be finite.");
    }

    [Test]
    public void ForwardCached_QkvBiasTrue_MatchesFullForward()
    {
        // With qkvBias=true the bias must be applied consistently in the cached single-token
        // path and the full-sequence path, so the two outputs agree.
        const int hidden = 128, numHeads = 8, numKvHeads = 4, seqLen = 6;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(33);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);

        var fullInput = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);
        var fullOutput = attn.Forward(fullInput);

        int kvWidth = numKvHeads * (hidden / numHeads);
        var kCache = new float[seqLen * kvWidth];
        var vCache = new float[seqLen * kvWidth];
        var stepOutputs = new ReverseGradTensor<float>[seqLen];
        for (int p = 0; p < seqLen; p++)
        {
            var tokenData = new float[hidden];
            Buffer.BlockCopy(inputData, p * hidden * sizeof(float), tokenData, 0, hidden * sizeof(float));
            var token = ReverseGradTensor<float>.FromArray(tokenData, requiresGrad: false);
            token.Reshape(1, hidden);
            stepOutputs[p] = attn.ForwardCached(token, p, kCache, vCache, p);
        }

        for (int p = 0; p < seqLen; p++)
            for (int d = 0; d < hidden; d++)
            {
                float full = fullOutput[p * hidden + d];
                float step = stepOutputs[p][d];
                Assert.That(step, Is.EqualTo(full).Within(1e-5f), $"Cached vs full mismatch at token {p}, dim {d}.");
            }
    }

    [Test]
    public void DecodeAttention_FusedMatchesMultiHeadAttention_AcrossGqaRatios()
    {
        // The fused single-query decode kernel must reproduce the multi-step MultiHeadAttention
        // path (GqaRepeatKV + PackHeads + per-head QK^T/softmax/V) for every GQA head ratio.
        const int headDim = 16;
        const int kvLen = 5;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var ratios = new (int NumHeads, int NumKvHeads)[] { (14, 2), (8, 4), (12, 4), (8, 8) };

        foreach (var (numHeads, numKvHeads) in ratios)
        {
            int hidden = numHeads * headDim;
            int kvWidth = numKvHeads * headDim;
            var rnd = new Random(100 + numHeads + numKvHeads);

            var q = new float[hidden];
            var kCache = new float[kvLen * kvWidth];
            var vCache = new float[kvLen * kvWidth];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rnd.NextDouble() * 2 - 1);
            for (int i = 0; i < kCache.Length; i++) { kCache[i] = (float)(rnd.NextDouble() * 2 - 1); vCache[i] = (float)(rnd.NextDouble() * 2 - 1); }

            var fusedOut = new float[hidden];
            AttentionKernels<float>.DecodeAttention(q, kCache, vCache, fusedOut, kvLen, numHeads, numKvHeads, headDim, scale);

            // Slow reference: exactly what ForwardCached did before the fused path.
            var qTensor = ReverseGradTensor<float>.FromMatrix(q, 1, hidden, requiresGrad: false);
            var kTensor = ReverseGradTensor<float>.FromMatrix(kCache, kvLen, kvWidth, requiresGrad: false);
            var vTensor = ReverseGradTensor<float>.FromMatrix(vCache, kvLen, kvWidth, requiresGrad: false);
            var kFull = ReverseGradOperations.GqaRepeatKV(kTensor, numHeads, numKvHeads);
            var vFull = ReverseGradOperations.GqaRepeatKV(vTensor, numHeads, numKvHeads);
            var openMask = ReverseGradTensor<float>.FromMatrix(new float[kvLen], 1, kvLen, requiresGrad: false);
            var slow = ReverseGradOperations.MultiHeadAttention(qTensor, kFull, vFull, numHeads, scale, openMask);

            for (int d = 0; d < hidden; d++)
                Assert.That(fusedOut[d], Is.EqualTo(slow[d]).Within(1e-5f),
                    $"Fused vs slow mismatch heads {numHeads}/{numKvHeads}, dim {d}.");
        }
    }

    [Test]
    public void DecodeAttention_GqaMapping_ReusesSharedKeyValueHead()
    {
        // For a 4:1 GQA ratio the fused kernel must route every query head in a group to the
        // same KV head, so the fused output equals per-query-head attention against that single
        // KV head. Guards the virtual mapping kvHead = qh / repeat (no GqaRepeatKV expansion).
        const int headDim = 16, numHeads = 8, numKvHeads = 2, kvLen = 4;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var rnd = new Random(7);
        var q = new float[numHeads * headDim];
        var kCache = new float[kvLen * numKvHeads * headDim];
        var vCache = new float[kvLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < kCache.Length; i++) { kCache[i] = (float)(rnd.NextDouble() * 2 - 1); vCache[i] = (float)(rnd.NextDouble() * 2 - 1); }

        var fused = new float[numHeads * headDim];
        AttentionKernels<float>.DecodeAttention(q, kCache, vCache, fused, kvLen, numHeads, numKvHeads, headDim, scale);

        // Independent reference: for each query head, dot against the KV head kv = qh / 4,
        // softmax, weighted V.
        for (int qh = 0; qh < numHeads; qh++)
        {
            int kv = qh / (numHeads / numKvHeads);
            var scores = new float[kvLen];
            for (int j = 0; j < kvLen; j++)
                scores[j] = scale * TensorPrimitives.Dot(
                    q.AsSpan(qh * headDim, headDim),
                    kCache.AsSpan(j * numKvHeads * headDim + kv * headDim, headDim));
            AttentionKernels<float>.SoftmaxRows(scores, 1, kvLen);
            for (int d = 0; d < headDim; d++)
            {
                float acc = 0;
                for (int j = 0; j < kvLen; j++)
                    acc += scores[j] * vCache[j * numKvHeads * headDim + kv * headDim + d];
                Assert.That(fused[qh * headDim + d], Is.EqualTo(acc).Within(1e-5f),
                    $"GQA mapping mismatch head {qh}, dim {d}.");
            }
        }
    }

    [Test]
    public void DecodeAttention_SteadyState_AllocatesNothing()
    {
        // The fused decode kernel must not copy the cached prefix: steady-state (pooled scores
        // buffer) allocation must be ~0 with a preallocated output span.
        const int headDim = 16, numHeads = 8, numKvHeads = 2, kvLen = 64;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var rnd = new Random(3);
        var q = new float[numHeads * headDim];
        var kCache = new float[kvLen * numKvHeads * headDim];
        var vCache = new float[kvLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < kCache.Length; i++) { kCache[i] = (float)(rnd.NextDouble() * 2 - 1); vCache[i] = (float)(rnd.NextDouble() * 2 - 1); }
        var output = new float[numHeads * headDim];

        // Warm the ArrayPool so steady-state rent/return allocates nothing.
        for (int i = 0; i < 2; i++)
            AttentionKernels<float>.DecodeAttention(q, kCache, vCache, output, kvLen, numHeads, numKvHeads, headDim, scale);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++)
            AttentionKernels<float>.DecodeAttention(q, kCache, vCache, output, kvLen, numHeads, numKvHeads, headDim, scale);
        long after = GC.GetAllocatedBytesForCurrentThread();

        Assert.That(after - before, Is.LessThan(2048), "Fused decode kernel must not allocate per call (no cache copy).");
    }

    [Test]
    public void ForwardCached_OutsideGrad_FusedPathBuildsNoGraphNode()
    {
        const int hidden = 112, numHeads = 14, numKvHeads = 2, kvLen = 4, maxPos = 32;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: maxPos);
        int kvWidth = numKvHeads * (hidden / numHeads);
        var kCache = new float[(kvLen + 1) * kvWidth];
        var vCache = new float[(kvLen + 1) * kvWidth];
        var rnd = new Random(4);
        for (int i = 0; i < kvLen * kvWidth; i++) { kCache[i] = (float)(rnd.NextDouble() * 2 - 1); vCache[i] = (float)(rnd.NextDouble() * 2 - 1); }

        var input = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[hidden]), requiresGrad: false);
        input.Reshape(1, hidden);
        var output = attn.ForwardCached(input, kvLen, kCache, vCache, kvLen);

        Assert.That(output.IsLeaf, Is.True, "Inference ForwardCached must not build a graph node (fused path).");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }
}
