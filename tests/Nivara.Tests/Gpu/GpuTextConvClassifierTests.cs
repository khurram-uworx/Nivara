using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the device forward (#901 groundwork) of <see cref="GpuTextConvClassifier"/> against the
/// AutoDiff reference (Embedding -> Conv1d branches -> concat -> FC1 -> ReLU -> FC2) on identical
/// weights, and pins the host backward against a reverse-mode autograd pass on the same batch.
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

    static GpuTextConvClassifier DeviceGated(int seed, out ReferenceModel reference)
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
        reference = new ReferenceModel(seed);
        gpu.SetWeightsCpuLayout(
            Vocab, Embed, SeqLen,
            reference.Embedding, reference.Conv1W, reference.Conv1B, reference.Conv2W,
            reference.Conv2B, reference.Conv3W, reference.Conv3B,
            reference.Fc1W, reference.Fc1B, reference.Fc2W, reference.Fc2B);
        return gpu;
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

    static float[] ForwardReference(ReferenceModel reference, int[] tokens, int batch)
    {
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++) data[i] = tokens[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, batch, SeqLen, requiresGrad: false);
        var logits = reference.Forward(input);
        var arr = new float[logits.Length];
        logits.Data.CopyTo(arr, 0.0f);
        return arr;
    }

    /// <summary>Autograd mirror of the device forward, seeded with the same weights.</summary>
    sealed class ReferenceModel : Module<float>
    {
        readonly Embedding<float> embedding;
        readonly Conv1d<float> conv1;
        readonly Conv1d<float> conv2;
        readonly Conv1d<float> conv3;
        readonly Linear<float> fc1;
        readonly Linear<float> fc2;

        public ReferenceModel(int seed)
        {
            var rng = new Random(seed + 1);
            embedding = new Embedding<float>(Vocab, Embed);
            conv1 = new Conv1d<float>(Embed, 64, kernelSize: 3, padding: 1);
            conv2 = new Conv1d<float>(Embed, 64, kernelSize: 5, padding: 2);
            conv3 = new Conv1d<float>(Embed, 64, kernelSize: 7, padding: 3);
            fc1 = new Linear<float>(64 * 3 * SeqLen, Hidden, bias: true);
            fc2 = new Linear<float>(Hidden, 2, bias: true);
            RegisterModules(embedding, conv1, conv2, conv3, fc1, fc2);

            SeedWeights(
                Randoms(Vocab * Embed, rng),
                Randoms(64 * Embed * 3, rng), Randoms(64, rng),
                Randoms(64 * Embed * 5, rng), Randoms(64, rng),
                Randoms(64 * Embed * 7, rng), Randoms(64, rng),
                Randoms(Hidden * (64 * 3 * SeqLen), rng), Randoms(Hidden, rng),
                Randoms(2 * Hidden, rng), Randoms(2, rng));
        }

        public float[] Embedding { get; private set; } = [];
        public float[] Conv1W { get; private set; } = [];
        public float[] Conv1B { get; private set; } = [];
        public float[] Conv2W { get; private set; } = [];
        public float[] Conv2B { get; private set; } = [];
        public float[] Conv3W { get; private set; } = [];
        public float[] Conv3B { get; private set; } = [];
        public float[] Fc1W { get; private set; } = [];
        public float[] Fc1B { get; private set; } = [];
        public float[] Fc2W { get; private set; } = [];
        public float[] Fc2B { get; private set; } = [];

        void SeedWeights(float[] embedding, float[] conv1W, float[] conv1B, float[] conv2W, float[] conv2B,
            float[] conv3W, float[] conv3B, float[] fc1W, float[] fc1B, float[] fc2W, float[] fc2B)
        {
            Embedding = embedding;
            Conv1W = conv1W; Conv1B = conv1B;
            Conv2W = conv2W; Conv2B = conv2B;
            Conv3W = conv3W; Conv3B = conv3B;
            Fc1W = fc1W; Fc1B = fc1B;
            Fc2W = fc2W; Fc2B = fc2B;

            var ps = GetParameters().Values.ToArray();
            ps[0].Tensor = ReverseGradTensor<float>.FromMatrix(Embedding, Vocab, Embed, requiresGrad: true);
            ps[1].Tensor = ReverseGradTensor<float>.FromMatrix(Conv1W, 64, Embed * 3, requiresGrad: true);
            ps[2].Tensor = ReverseGradTensor<float>.FromArray(Conv1B, requiresGrad: true);
            ps[3].Tensor = ReverseGradTensor<float>.FromMatrix(Conv2W, 64, Embed * 5, requiresGrad: true);
            ps[4].Tensor = ReverseGradTensor<float>.FromArray(Conv2B, requiresGrad: true);
            ps[5].Tensor = ReverseGradTensor<float>.FromMatrix(Conv3W, 64, Embed * 7, requiresGrad: true);
            ps[6].Tensor = ReverseGradTensor<float>.FromArray(Conv3B, requiresGrad: true);
            int convHidden = 64 * 3 * SeqLen;
            ps[7].Tensor = ReverseGradTensor<float>.FromMatrix(Fc1W, Hidden, convHidden, requiresGrad: true);
            ps[8].Tensor = ReverseGradTensor<float>.FromMatrix(Fc1B, 1, Hidden, requiresGrad: true);
            ps[9].Tensor = ReverseGradTensor<float>.FromMatrix(Fc2W, 2, Hidden, requiresGrad: true);
            ps[10].Tensor = ReverseGradTensor<float>.FromMatrix(Fc2B, 1, 2, requiresGrad: true);
        }

        static float[] Randoms(int count, Random rng)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            return values;
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        {
            var emb = embedding.Forward(input);
            var ncl = ReverseGradOperations.TransposeAxes(emb, 1, 2);
            var c1 = ReverseGradOperations.Relu(conv1.Forward(ncl));
            var c2 = ReverseGradOperations.Relu(conv2.Forward(ncl));
            var c3 = ReverseGradOperations.Relu(conv3.Forward(ncl));
            int b = c1.Shape[0];
            c1.Reshape(b, c1.Shape[1] * c1.Shape[2]);
            c2.Reshape(b, c2.Shape[1] * c2.Shape[2]);
            c3.Reshape(b, c3.Shape[1] * c3.Shape[2]);
            var cat = ReverseGradOperations.Concat([c1, c2, c3], axis: 1);
            var h = ReverseGradOperations.Relu(fc1.Forward(cat));
            return fc2.Forward(h);
        }

        public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName) => inputName;
    }
}
