using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins <see cref="GpuTextLinearClassifier"/> against the real CPU model the issue names
/// (<c>TextClassifierModel&lt;float&gt;</c>): device forward parity on identical weights, device
/// gradient parity (#900) against a reverse-mode autograd pass on the same batch, the CPU-layout
/// adapters, a training smoke run and the disposal guards.
/// </summary>
/// <remarks>
/// Tolerances are deliberate, as in the conv fixture: the device GEMM reassociates the dot
/// products and the autograd reverse pass accumulates in a different order again — different
/// summation orders of the same mathematics, not a bit-reproduction claim. The bounds stay tight
/// (forward ~2e-4 on unit-scale logits, gradients ~5e-4), enough that a wrong transposition, a
/// stale layout or a missing mask would fail by orders of magnitude instead.
/// </remarks>
public sealed class GpuTextLinearClassifierTests
{
    const int Vocab = 31;
    const int Embed = 8;
    const int Hidden = 12;
    const int SeqLen = 10;
    const float ForwardEps = 2e-4f;
    const float GradEps = 5e-4f;

    [Test]
    public void Forward_Batch_MatchesCpuTextClassifierModel()
    {
        using var gpu = DeviceGated(7, out var reference);
        SeedBoth(gpu, reference, 3);
        int batch = 5;
        int[] tokens = Tokens(batch, 11);

        var logits = gpu.Forward(tokens, batch, SeqLen);
        var logitsRef = ForwardReference(reference, tokens, batch);

        Assert.That(logits.Length, Is.EqualTo(logitsRef.Length));
        float worst = 0f;
        for (int i = 0; i < logitsRef.Length; i++)
            worst = Math.Max(worst, Math.Abs(logits[i] - logitsRef[i]));
        Assert.That(worst, Is.LessThan(ForwardEps), $"batch={batch}, worst logit diff {worst}");

        for (int b = 0; b < batch; b++)
        {
            int got = logits[b * 2 + 1] > logits[b * 2] ? 1 : 0;
            int want = logitsRef[b * 2 + 1] > logitsRef[b * 2] ? 1 : 0;
            Assert.That(got, Is.EqualTo(want), $"argmax row {b}");
        }
    }

    [Test]
    public void Gradients_Batch_MatchReverseAutograd()
    {
        using var gpu = DeviceGated(13, out var reference);
        SeedBoth(gpu, reference, 5);
        int batch = 4;
        int[] tokens = Tokens(batch, 17);
        int[] labels = Labels(batch, 19);

        var logits = gpu.Forward(tokens, batch, SeqLen);
        gpu.Backward(logits, tokens, labels, batch, SeqLen);
        var (dW1, db1, dW2, db2) = gpu.DownloadGradients();

        var lossFn = new CrossEntropyLoss<float>();
        float[] gradEmbRef, gradFc1wCpu, gradFc1bRef, gradFc2wCpu, gradFc2bRef;
        using (var scope = GradientUtils.Grad())
        {
            var data = new float[tokens.Length];
            for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
            var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
            var lossTensor = lossFn.Forward(reference.Forward(input), labels);
            lossTensor.Backward();

            var ps = reference.GetParameters().Values.ToArray();
            gradEmbRef = GradOf(ps[0]);
            gradFc1wCpu = GradOf(ps[1]); // CPU layout [out, in] = [Hidden, Embed]
            gradFc1bRef = GradOf(ps[2]);
            gradFc2wCpu = GradOf(ps[3]); // CPU layout [2, Hidden]
            gradFc2bRef = GradOf(ps[4]);
        }

        AssertGrad("embedding", gpu.gradEmb, gradEmbRef);

        // Transpose device FC grads (GPU [in, out]) into the CPU layout for comparison.
        var dW1Cpu = new float[Hidden * Embed];
        for (int e = 0; e < Embed; e++)
            for (int h = 0; h < Hidden; h++)
                dW1Cpu[h * Embed + e] = dW1[e * Hidden + h];
        AssertGrad("fc1W", dW1Cpu, gradFc1wCpu);
        AssertGrad("fc1B", db1, gradFc1bRef);

        var dW2Cpu = new float[2 * Hidden];
        for (int h = 0; h < Hidden; h++)
            for (int c = 0; c < 2; c++)
                dW2Cpu[c * Hidden + h] = dW2[h * 2 + c];
        AssertGrad("fc2W", dW2Cpu, gradFc2wCpu);
        AssertGrad("fc2B", db2, gradFc2bRef);
    }

    [Test]
    public void SetWeightsCpuLayout_ExportCpuLayout_RoundTripsExactly()
    {
        using var gpu = DeviceGated(23, out _);
        var rng = new Random(29);
        float[] emb = Randoms(Vocab * Embed, rng);
        float[] fc1Cpu = Randoms(Hidden * Embed, rng);
        float[] fc1B = Randoms(Hidden, rng);
        float[] fc2Cpu = Randoms(2 * Hidden, rng);
        float[] fc2B = Randoms(2, rng);

        gpu.SetWeightsCpuLayout(Vocab, Embed, SeqLen, emb, fc1Cpu, fc1B, fc2Cpu, fc2B);
        var round = gpu.ExportCpuLayout();

        Assert.That(round.Embedding, Is.EqualTo(emb));
        Assert.That(round.Fc1W, Is.EqualTo(fc1Cpu));
        Assert.That(round.Fc1B, Is.EqualTo(fc1B));
        Assert.That(round.Fc2W, Is.EqualTo(fc2Cpu));
        Assert.That(round.Fc2B, Is.EqualTo(fc2B));
    }

    [Test]
    public void Train_LossDecreases_OnSeparableData()
    {
        using var gpu = DeviceGated(31, out _);
        const int samples = 64;
        var rng = new Random(37);
        var tokens = new int[samples * SeqLen];
        var labels = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            labels[i] = i % 2;
            tokens[i * SeqLen] = labels[i] == 0 ? 2 : 5;
            for (int s = 1; s < SeqLen; s++) tokens[i * SeqLen + s] = rng.Next(1, Vocab);
        }

        var epochs = gpu.Train(tokens, labels, batchSize: 8, seqLen: SeqLen, epochs: 8, learningRate: 0.05f, seed: 41);

        Assert.That(epochs.Count, Is.EqualTo(8));
        for (int i = 0; i < epochs.Count; i++)
        {
            Assert.That(epochs[i].Epoch, Is.EqualTo(i));
            Assert.That(float.IsFinite(epochs[i].Loss), Is.True, $"epoch {i} loss not finite");
        }
        Assert.That(epochs[^1].Loss, Is.LessThan(0.5f * epochs[0].Loss),
            $"loss did not halve: {epochs[0].Loss} -> {epochs[^1].Loss}");

        var round = gpu.ExportCpuLayout();
        foreach (var weights in new[] { round.Embedding, round.Fc1W, round.Fc1B, round.Fc2W, round.Fc2B })
            foreach (float w in weights)
                Assert.That(float.IsFinite(w), Is.True, "host mirrors must stay finite after Train");
    }

    [Test]
    public void Predict_MatchesPerSamplePredictions_BatchInvariance()
    {
        using var gpu = DeviceGated(43, out _);
        int batch = 4;
        int[] tokens = Tokens(batch, 47);

        var batched = gpu.Predict(tokens, SeqLen);
        Assert.That(batched.Length, Is.EqualTo(batch));
        for (int b = 0; b < batch; b++)
        {
            var single = new int[SeqLen];
            Array.Copy(tokens, b * SeqLen, single, 0, SeqLen);
            Assert.That(gpu.Predict(single, SeqLen)[0], Is.EqualTo(batched[b]), $"row {b}");
        }
    }

    [Test]
    public void Members_AfterDispose_ThrowObjectDisposed()
    {
        var gpu = DeviceGated(53, out _);
        gpu.Dispose();
        gpu.Dispose(); // idempotent

        float[] emb = new float[Vocab * Embed];
        Assert.Throws<ObjectDisposedException>(() => gpu.Forward(Tokens(2, 3), 2, SeqLen));
        Assert.Throws<ObjectDisposedException>(() => gpu.Predict(Tokens(2, 3), SeqLen));
        Assert.Throws<ObjectDisposedException>(() => gpu.ExportEmbedding());
        Assert.Throws<ObjectDisposedException>(() => gpu.ExportFc1Weight());
        Assert.Throws<ObjectDisposedException>(() => gpu.SetWeights(emb, new float[Embed * Hidden],
            new float[Hidden], new float[Hidden * 2], new float[2]));
        Assert.Throws<ObjectDisposedException>(() => gpu.Train(new int[2 * SeqLen], new int[2], 2, SeqLen, 1, 0.1f, 1));
    }

    static GpuTextLinearClassifier DeviceGated(int seed, out TextClassifierModel<float> reference)
    {
        GpuTextLinearClassifier gpu;
        try
        {
            gpu = new GpuTextLinearClassifier(Vocab, Embed, Hidden, seed);
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping GpuTextLinearClassifier verification.");
            throw;
        }
        reference = new TextClassifierModel<float>(Vocab, Embed, Hidden, 2, SeqLen);
        return gpu;
    }

    static float[] Randoms(int count, Random rng)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return values;
    }

    /// <summary>Generates one weight set and seeds both classifiers with it — the parity tests
    /// compare the same weights, not two independent initializations.</summary>
    static void SeedBoth(GpuTextLinearClassifier gpu, TextClassifierModel<float> reference, int seed)
    {
        var rng = new Random(seed);
        float[] emb = Randoms(Vocab * Embed, rng);
        float[] fc1Cpu = Randoms(Hidden * Embed, rng);
        float[] fc1B = Randoms(Hidden, rng);
        float[] fc2Cpu = Randoms(2 * Hidden, rng);
        float[] fc2B = Randoms(2, rng);

        gpu.SetWeightsCpuLayout(Vocab, Embed, SeqLen, emb, fc1Cpu, fc1B, fc2Cpu, fc2B);
        SeedReference(reference, emb, fc1Cpu, fc1B, fc2Cpu, fc2B);
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

    /// <summary>Seeds the reference model's parameters in place with <c>requiresGrad: true</c>;
    /// <c>SetWeights</c> would attach non-differentiable tensors.</summary>
    static void SeedReference(TextClassifierModel<float> reference,
        float[] emb, float[] fc1Cpu, float[] fc1B, float[] fc2Cpu, float[] fc2B)
    {
        var ps = reference.GetParameters().Values.ToArray();
        ps[0].Tensor = ReverseGradTensor<float>.FromMatrix(emb, Vocab, Embed, requiresGrad: true);
        ps[1].Tensor = ReverseGradTensor<float>.FromMatrix(fc1Cpu, fc1Cpu.Length / Embed, Embed, requiresGrad: true);
        ps[2].Tensor = ReverseGradTensor<float>.FromMatrix(fc1B, 1, fc1B.Length, requiresGrad: true);
        ps[3].Tensor = ReverseGradTensor<float>.FromMatrix(fc2Cpu, 2, fc2Cpu.Length / 2, requiresGrad: true);
        ps[4].Tensor = ReverseGradTensor<float>.FromMatrix(fc2B, 1, fc2B.Length, requiresGrad: true);
    }

    static float[] ForwardReference(TextClassifierModel<float> reference, int[] tokens, int batch)
    {
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
        var logits = reference.Forward(input);
        var arr = new float[logits.Length];
        logits.Data.CopyTo(arr, 0.0f);
        return arr;
    }

    static float[] GradOf(Nivara.AutoDiff.Nn.Parameter<float> p)
    {
        var grad = p.Tensor.Grad;
        var arr = new float[grad?.Length ?? 0];
        grad?.CopyTo(arr, 0.0f);
        return arr;
    }

    static void AssertGrad(string name, float[] actual, float[] expected)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), name);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        Assert.That(worst, Is.LessThan(GradEps), $"{name}: worst gradient diff {worst}");
    }
}
