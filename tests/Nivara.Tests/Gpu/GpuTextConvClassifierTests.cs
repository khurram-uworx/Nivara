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
/// identical weights, and pins the host backward against a reverse-mode autograd pass on the same
/// batch.
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

        var (logits, _, _) = gpu.ForwardBatch(tokens, batch, collectBackwardState: false);
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

        var (logits, _, _) = gpu.ForwardBatch(tokens, batch, collectBackwardState: true);
        gpu.BackwardAndSetGradients(logits, tokens, labels, batch);

        var lossFn = new CrossEntropyLoss<float>();
        float lossRef;
        float[] gradEmb, gradC1w, gradC1b, gradC2w, gradC2b, gradC3w, gradC3b, gradFc1wCpu, gradFc1b, gradFc2wCpu, gradFc2b, rawGradFc1w, rawGradFc2w;
        using (var scope = GradientUtils.Grad())
        {
            var data = new float[tokens.Length];
            for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
            var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
            var lossTensor = lossFn.Forward(reference.Forward(input), labels);
            lossRef = lossTensor.Data.ToArray()[0];
            lossTensor.Backward();

            var ps = reference.GetParameters().Values.ToArray();
            rawGradFc1w = GradOf(ps[7]); // CPU layout [out, in] = [Hidden, convHidden]
            rawGradFc2w = GradOf(ps[9]); // CPU layout [out, in] = [2, Hidden]
            (gradEmb, gradC1w, gradC1b, gradC2w, gradC2b, gradC3w, gradC3b, rawGradFc1w, gradFc1b, rawGradFc2w, gradFc2b) =
                (GradOf(ps[0]), GradOf(ps[1]), GradOf(ps[2]), GradOf(ps[3]), GradOf(ps[4]), GradOf(ps[5]),
                 GradOf(ps[6]), GradOf(ps[7]), GradOf(ps[8]), GradOf(ps[9]), GradOf(ps[10]));

            // Transpose device FC grads (GPU [in, out]) into CPU layout ([out, in]) for comparison.
            int convHidden = 64 * 3 * SeqLen;
            gradFc1wCpu = new float[Hidden * convHidden];
            var devFc1 = gpu.gradFc1.AsSpan(0, Hidden * convHidden).ToArray();
            for (int j = 0; j < convHidden; j++)
                for (int h = 0; h < Hidden; h++)
                    gradFc1wCpu[h * convHidden + j] = devFc1[j * Hidden + h];

            gradFc2wCpu = new float[2 * Hidden];
            var devFc2 = gpu.gradFc2.AsSpan(0, 2 * Hidden).ToArray();
            for (int h = 0; h < Hidden; h++)
                for (int c = 0; c < 2; c++)
                    gradFc2wCpu[c * Hidden + h] = devFc2[h * 2 + c];
        }

        Assert.That(lossRef, Is.LessThan(10f), $"unexpected loss {lossRef}");
        AssertGrad("embedding", gpu.gradEmb, gradEmb);
        AssertGrad("conv1W", gpu.gradConv.AsSpan(0, gradC1w.Length).ToArray(), gradC1w);
        AssertGrad("conv1B", gpu.gradConv.AsSpan(gradC1w.Length, gradC1b.Length).ToArray(), gradC1b);
        AssertGrad("conv2W", gpu.gradConv.AsSpan(gradC1w.Length + gradC1b.Length, gradC2w.Length).ToArray(), gradC2w);
        AssertGrad("conv2B", gpu.gradConv.AsSpan(gradC1w.Length + gradC1b.Length + gradC2w.Length, gradC2b.Length).ToArray(), gradC2b);
        AssertGrad("conv3W", gpu.gradConv.AsSpan(gradC1w.Length + gradC1b.Length + gradC2w.Length + gradC2b.Length, gradC3w.Length).ToArray(), gradC3w);
        AssertGrad("conv3B", gpu.gradConv.AsSpan(gradC1w.Length + gradC1b.Length + gradC2w.Length + gradC2b.Length + gradC3w.Length, gradC3b.Length).ToArray(), gradC3b);
        AssertGrad("fc1W", gradFc1wCpu, rawGradFc1w);
        AssertGrad("fc1B", gpu.gradFc1.AsSpan(Hidden * (64 * 3 * SeqLen), Hidden).ToArray(), gradFc1b);
        AssertGrad("fc2W", gradFc2wCpu, rawGradFc2w);
        AssertGrad("fc2B", gpu.gradFc2.AsSpan(2 * Hidden, 2).ToArray(), gradFc2b);
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
