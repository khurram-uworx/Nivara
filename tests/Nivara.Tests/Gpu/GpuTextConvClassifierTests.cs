using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Utilities;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the device forward (#901 groundwork) of <see cref="GpuTextConvClassifier"/> against the
/// real CPU model (<c>ConvTextClassifierModel&lt;float&gt;</c>, Dropout identity in Eval) on
/// identical weights, and pins the device backward (#900) against a reverse-mode autograd pass on
/// the same batch.
/// </summary>
/// <remarks>
/// <para>
/// Tolerances are used for both assertions, deliberately. The device path reassociates across
/// three independently accumulated GEMMs plus the OpenCL conv kernel, and the autograd reverse
/// pass accumulates in a different order again — these are different summation orders of the
/// same mathematics, not a bit-reproduction claim, so an exactness assertion would fail on any
/// legal float reassociation. The bounds stay tight (forward ~2e-4 on unit-scale logits,
/// gradients ~5e-4), enough to catch a transposed weight block or a wrong padding window.
/// </para>
/// <para>
/// The FC weights arrive in the CPU <c>[out, in]</c> layout through <c>SetWeightsCpuLayout</c>,
/// so the tests also pin that adapter — a transposed FC1 branch block or a stale conv weight row
/// would show up as a parity failure on this fixture rather than as a silently wrong model.
/// </para>
/// </remarks>
public sealed class GpuTextConvClassifierTests
{
    const int Vocab = 37;
    const int Embed = 8;
    const int Hidden = 10;
    const int SeqLen = 12;
    const float ForwardEps = 2e-4f;
    const float GradEps = 5e-4f;

    [Test]
    public void Forward_Batch_MatchesAutoDiffReference()
    {
        using var gpu = DeviceGated(7, out var reference);
        int batch = 4;
        int[] tokens = Tokens(batch, 11);

        var logits = gpu.ForwardBatch(tokens, batch);
        var logitsRef = ForwardReference(reference, tokens, batch);

        float worst = 0f;
        for (int i = 0; i < logitsRef.Length; i++)
            worst = Math.Max(worst, Math.Abs(logits[i] - logitsRef[i]));
        Assert.That(worst, Is.LessThan(ForwardEps), $"batch={batch}, worst logit diff {worst}");
    }

    [Test]
    public void Gradients_Batch_MatchReverseAutograd()
    {
        using var gpu = DeviceGated(13, out var reference);
        int batch = 3;
        int[] tokens = Tokens(batch, 17);
        int[] labels = Labels(batch, 19);

        var logits = gpu.ForwardBatch(tokens, batch);
        gpu.Backward(logits, tokens, labels, batch);
        var (devC1w, devC1b, devC2w, devC2b, devC3w, devC3b, devFc1w, devFc1b, devFc2w, devFc2b) =
            gpu.DownloadGradients();

        var lossFn = new CrossEntropyLoss<float>();
        float lossRef;
        float[] gradEmb, gradC1w, gradC1b, gradC2w, gradC2b, gradC3w, gradC3b, refFc1wCpu, refFc1b, refFc2wCpu, refFc2b;
        using (var scope = GradientUtils.Grad())
        {
            var data = new float[tokens.Length];
            for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
            var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
            var lossTensor = lossFn.Forward(reference.Forward(input), labels);
            lossRef = lossTensor.Data.ToArray()[0];
            lossTensor.Backward();

            var ps = reference.GetParameters().Values.ToArray();
            refFc1wCpu = GradOf(ps[7]); // CPU layout [out, in] = [Hidden, convHidden]
            refFc2wCpu = GradOf(ps[9]); // CPU layout [out, in] = [2, Hidden]
            gradEmb = GradOf(ps[0]);
            gradC1w = GradOf(ps[1]);
            gradC1b = GradOf(ps[2]);
            gradC2w = GradOf(ps[3]);
            gradC2b = GradOf(ps[4]);
            gradC3w = GradOf(ps[5]);
            gradC3b = GradOf(ps[6]);
            refFc1b = GradOf(ps[8]);
            refFc2b = GradOf(ps[10]);
        }

        // Device FC weight grads arrive in the GPU [in, out] layout; transpose into the CPU
        // [out, in] layout for comparison.
        int convHidden = 64 * 3 * SeqLen;
        var devFc1wCpu = new float[Hidden * convHidden];
        for (int j = 0; j < convHidden; j++)
            for (int h = 0; h < Hidden; h++)
                devFc1wCpu[h * convHidden + j] = devFc1w[j * Hidden + h];
        var devFc2wCpu = new float[2 * Hidden];
        for (int h = 0; h < Hidden; h++)
            for (int c = 0; c < 2; c++)
                devFc2wCpu[c * Hidden + h] = devFc2w[h * 2 + c];

        Assert.That(lossRef, Is.LessThan(10f), $"unexpected loss {lossRef}");
        AssertGrad("embedding", gpu.gradEmb, gradEmb);
        AssertGrad("conv1W", devC1w, gradC1w);
        AssertGrad("conv1B", devC1b, gradC1b);
        AssertGrad("conv2W", devC2w, gradC2w);
        AssertGrad("conv2B", devC2b, gradC2b);
        AssertGrad("conv3W", devC3w, gradC3w);
        AssertGrad("conv3B", devC3b, gradC3b);
        AssertGrad("fc1W", devFc1wCpu, refFc1wCpu);
        AssertGrad("fc1B", devFc1b, refFc1b);
        AssertGrad("fc2W", devFc2wCpu, refFc2wCpu);
        AssertGrad("fc2B", devFc2b, refFc2b);
    }

    [Test]
    public void SetWeightsCpuLayout_ExportCpuLayout_HasConsistentShapes()
    {
        using var gpu = DeviceGated(23, out _);
        var round = gpu.ExportCpuLayout();
        Assert.That(round.Embedding.Length, Is.EqualTo(Vocab * Embed));
        Assert.That(round.Conv1W.Length, Is.EqualTo(64 * Embed * 3));
        Assert.That(round.Conv2W.Length, Is.EqualTo(64 * Embed * 5));
        Assert.That(round.Conv3W.Length, Is.EqualTo(64 * Embed * 7));
        Assert.That(round.Fc1W.Length % Hidden, Is.Zero, "FC1 CPU weight rows must be hiddenDim.");
        int convHidden = 64 * 3 * SeqLen;
        Assert.That(round.Fc1W.Length, Is.EqualTo(Hidden * convHidden));
        Assert.That(gpu.ExportFc1Weight().Length, Is.EqualTo(convHidden * Hidden));
        Assert.That(round.Fc2W.Length, Is.EqualTo(2 * Hidden));
    }

    [Test]
    public void Train_LossDecreases_OnSeparableData()
    {
        using var gpu = DeviceGated(29, out _);
        const int samples = 48;
        var rng = new Random(31);
        var tokens = new int[samples * SeqLen];
        var labels = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            labels[i] = i % 2;
            tokens[i * SeqLen] = labels[i] == 0 ? 2 : 5;
            for (int s = 1; s < SeqLen; s++) tokens[i * SeqLen + s] = rng.Next(1, Vocab);
        }

        var epochs = gpu.Train(tokens, labels, batchSize: 8, seqLen: SeqLen, epochs: 6, learningRate: 0.05f, seed: 37);

        Assert.That(epochs.Count, Is.EqualTo(6));
        for (int i = 0; i < epochs.Count; i++)
        {
            Assert.That(epochs[i].Epoch, Is.EqualTo(i));
            Assert.That(float.IsFinite(epochs[i].Loss), Is.True, $"epoch {i} loss not finite");
        }
        Assert.That(epochs[^1].Loss, Is.LessThan(0.7f * epochs[0].Loss),
            $"loss did not fall: {epochs[0].Loss} -> {epochs[^1].Loss}");

        // The host mirrors must reflect the trained device weights after Train returns.
        var round = gpu.ExportCpuLayout();
        foreach (var weights in new[] { round.Embedding, round.Conv1W, round.Conv1B, round.Conv2W,
            round.Conv2B, round.Conv3W, round.Conv3B, round.Fc1W, round.Fc1B, round.Fc2W, round.Fc2B })
            foreach (float w in weights)
                Assert.That(float.IsFinite(w), Is.True, "host mirrors must stay finite after Train");

        // Trained weights must classify the training signal correctly.
        var probeTokens = new[] { 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Assert.That(gpu.Predict(probeTokens, SeqLen)[0], Is.EqualTo(0));
        var probeTokens2 = new[] { 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Assert.That(gpu.Predict(probeTokens2, SeqLen)[0], Is.EqualTo(1));
    }

    [Test]
    public void Members_AfterDispose_ThrowObjectDisposed()
    {
        var gpu = DeviceGated(41, out _);
        gpu.Dispose();
        gpu.Dispose(); // idempotent

        float[] emb = new float[Vocab * Embed];
        Assert.Throws<ObjectDisposedException>(() => gpu.Forward(Tokens(2, 3), 2));
        Assert.Throws<ObjectDisposedException>(() => gpu.Predict(Tokens(2, 3), SeqLen));
        Assert.Throws<ObjectDisposedException>(() => gpu.ExportEmbedding());
        Assert.Throws<ObjectDisposedException>(() => gpu.ExportConv1Weight());
        Assert.Throws<ObjectDisposedException>(() => gpu.ExportFc1Weight());
        Assert.Throws<ObjectDisposedException>(() => gpu.SetWeights(emb,
            new float[64 * Embed * 3], new float[64], new float[64 * Embed * 5], new float[64],
            new float[64 * Embed * 7], new float[64], new float[Hidden * (64 * 3 * SeqLen)],
            new float[Hidden], new float[Hidden * 2], new float[2]));
        Assert.Throws<ObjectDisposedException>(() =>
            gpu.Train(new int[2 * SeqLen], new int[2], 2, SeqLen, 1, 0.1f, 1));
    }

    static void AssertGrad(string name, float[] actual, float[] expected)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), name);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        Assert.That(worst, Is.LessThan(GradEps), $"{name}: worst gradient diff {worst}");
    }

    static GpuTextConvClassifier DeviceGated(int seed, out ConvTextClassifierModel<float> reference)
    {
        GpuTextConvClassifier gpu;
        try
        {
            gpu = new GpuTextConvClassifier(Vocab, Embed, Hidden, SeqLen, seed);
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping GpuTextConvClassifier verification.");
            throw;
        }
        reference = new ConvTextClassifierModel<float>(Vocab, Embed, Hidden, 2, SeqLen);

        var rng = new Random(seed + 1);
        float[] emb = Randoms(Vocab * Embed, rng);
        float[] c1w = Randoms(64 * Embed * 3, rng);
        float[] c1b = Randoms(64, rng);
        float[] c2w = Randoms(64 * Embed * 5, rng);
        float[] c2b = Randoms(64, rng);
        float[] c3w = Randoms(64 * Embed * 7, rng);
        float[] c3b = Randoms(64, rng);
        int convHidden = 64 * 3 * SeqLen;
        float[] fc1w = Randoms(Hidden * convHidden, rng);
        float[] fc1b = Randoms(Hidden, rng);
        float[] fc2w = Randoms(2 * Hidden, rng);
        float[] fc2b = Randoms(2, rng);

        SeedReference(reference, emb, c1w, c1b, c2w, c2b, c3w, c3b, fc1w, fc1b, fc2w, fc2b);
        reference.Eval();
        gpu.SetWeightsCpuLayout(Vocab, Embed, SeqLen,
            emb, c1w, c1b, c2w, c2b, c3w, c3b, fc1w, fc1b, fc2w, fc2b);
        return gpu;
    }

    static float[] Randoms(int count, Random rng)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return values;
    }

    /// <summary>Seeds the reference model's parameters in place with <c>requiresGrad: true</c>;
    /// its <c>SetWeights</c> would attach non-differentiable tensors. Parameter order matches
    /// <see cref="ConvTextClassifierModel{T}.SetWeights"/>: embedding, conv1..3 (w, b), fc1, fc2.</summary>
    static void SeedReference(ConvTextClassifierModel<float> reference,
        float[] emb, float[] c1w, float[] c1b, float[] c2w, float[] c2b, float[] c3w, float[] c3b,
        float[] fc1w, float[] fc1b, float[] fc2w, float[] fc2b)
    {
        var ps = reference.GetParameters().Values.ToArray();
        ps[0].Tensor = ReverseGradTensor<float>.FromMatrix(emb, Vocab, Embed, requiresGrad: true);
        ps[1].Tensor = ReverseGradTensor<float>.FromMatrix(c1w, 64, Embed * 3, requiresGrad: true);
        ps[2].Tensor = ReverseGradTensor<float>.FromArray(c1b, requiresGrad: true);
        ps[3].Tensor = ReverseGradTensor<float>.FromMatrix(c2w, 64, Embed * 5, requiresGrad: true);
        ps[4].Tensor = ReverseGradTensor<float>.FromArray(c2b, requiresGrad: true);
        ps[5].Tensor = ReverseGradTensor<float>.FromMatrix(c3w, 64, Embed * 7, requiresGrad: true);
        ps[6].Tensor = ReverseGradTensor<float>.FromArray(c3b, requiresGrad: true);
        int convHidden = 64 * 3 * SeqLen;
        ps[7].Tensor = ReverseGradTensor<float>.FromMatrix(fc1w, Hidden, convHidden, requiresGrad: true);
        ps[8].Tensor = ReverseGradTensor<float>.FromMatrix(fc1b, 1, Hidden, requiresGrad: true);
        ps[9].Tensor = ReverseGradTensor<float>.FromMatrix(fc2w, 2, Hidden, requiresGrad: true);
        ps[10].Tensor = ReverseGradTensor<float>.FromMatrix(fc2b, 1, 2, requiresGrad: true);
    }

    static float[] GradOf(Nivara.AutoDiff.Nn.Parameter<float> p)
    {
        var grad = p.Tensor.Grad;
        var arr = new float[grad?.Length ?? 0];
        grad?.CopyTo(arr, 0.0f);
        return arr;
    }

    static int[] Tokens(int batch, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[batch * SeqLen];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = rng.Next(1, Vocab);
        return tokens;
    }

    static int[] Labels(int batch, int seed)
    {
        var rng = new Random(seed);
        var labels = new int[batch];
        for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);
        return labels;
    }

    static float[] ForwardReference(ConvTextClassifierModel<float> reference, int[] tokens, int batch)
    {
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
        var logits = reference.Forward(input);
        var arr = new float[logits.Length];
        logits.Data.CopyTo(arr, 0.0f);
        return arr;
    }

}
