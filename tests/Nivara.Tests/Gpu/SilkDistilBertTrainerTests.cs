using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins <see cref="SilkDistilBertTrainer"/> against the CPU reverse-mode model on a tiny
/// DistilBERT shape: device forward parity and device gradient parity against
/// <c>DistilBertForSequenceClassification</c> driven through <c>CrossEntropyLoss</c>. The issue's
/// acceptance (25 examples / 1 epoch) cannot distinguish a correct backward from a broken one, so
/// these are the real gates.
/// </summary>
[Category("Gpu")]
public sealed class SilkDistilBertTrainerTests
{
    const int Layers = 2;
    const int Hidden = 32;
    const int Heads = 4;
    const int Intermediate = 64;
    const int Vocab = 64;
    const int MaxPos = 16;
    const int SeqLen = 8;
    const float ForwardEps = 2e-4f;
    const float GradEps = 5e-4f;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk DistilBERT trainer verification."); throw; }
    }

    static DistilBertConfig TinyDistilConfig() => new()
    {
        Dim = Hidden,
        NLayers = Layers,
        NHeads = Heads,
        HiddenDim = Intermediate,
        MaxPosition = MaxPos,
        Eps = 1e-4f,
        VocabSize = Vocab,
    };

    static BertConfig TinyBertConfig() => TinyDistilConfig().ToBertConfig();

    static (SilkRuntime runtime, SilkDistilBertTrainer trainer, DistilBertForSequenceClassification<float> model) Make()
    {
        var runtime = Device();
        var distilConfig = TinyDistilConfig();
        var config = distilConfig.ToBertConfig();
        var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        var tensors = model.ToSilkTensors();
        var trainer = new SilkDistilBertTrainer(runtime, config, tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);
        return (runtime, trainer, model);
    }

    [Test]
    public void Forward_Batch_MatchesCpuDistilBert()
    {
        var (runtime, trainer, model) = Make();
        using (runtime)
        using (trainer)
        {
            var rng = new Random(7);
            int batch = 2;
            int[] tokens = new int[batch * SeqLen];
            float[] mask = new float[batch * SeqLen];
            for (int i = 0; i < tokens.Length; i++)
            {
                tokens[i] = rng.Next(Vocab);
                mask[i] = 1f;
            }

            var deviceLogits = trainer.Forward(tokens, mask, batch, SeqLen);

            // CPU forward.
            float[] cpuLogits;
            using (var scope = GradientUtils.Grad())
            {
                var maskTensor = ReverseGradTensor<float>.FromArray(mask, requiresGrad: false);
                var logits = model.Forward(tokens, maskTensor, batch, SeqLen);
                var arr = new float[logits.Length];
                logits.Data.CopyTo(arr, 0.0f);
                cpuLogits = arr;
            }

            Assert.That(deviceLogits.Length, Is.EqualTo(cpuLogits.Length));
            float worst = 0f;
            for (int i = 0; i < cpuLogits.Length; i++)
                worst = Math.Max(worst, Math.Abs(deviceLogits[i] - cpuLogits[i]));
            Assert.That(worst, Is.LessThan(ForwardEps), $"worst logit diff {worst}");
        }
    }

    [Test]
    public void Gradients_Batch_MatchReverseAutograd()
    {
        var (runtime, trainer, model) = Make();
        using (runtime)
        using (trainer)
        {
            var rng = new Random(11);
            int batch = 2;
            int[] tokens = new int[batch * SeqLen];
            float[] mask = new float[batch * SeqLen];
            int[] labels = new int[batch];
            for (int i = 0; i < tokens.Length; i++)
            {
                tokens[i] = rng.Next(Vocab);
                mask[i] = 1f;
            }
            for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);

            var deviceLogits = trainer.Forward(tokens, mask, batch, SeqLen);
            var batchGradient = new float[batch * 2];
            SilkLoss.InputGradient(deviceLogits, labels, batch, 2, 1f / batch, batchGradient);
            trainer.Backward(batch, SeqLen, batchGradient);
            var deviceGrads = trainer.ExportGradients();

            // CPU reverse pass.
            var tensors = model.ToSilkTensors();
            var keys = new List<string>(tensors.Keys);
            var cpuGrads = new Dictionary<string, float[]>(StringComparer.Ordinal);
            using (var scope = GradientUtils.Grad())
            {
                var maskTensor = ReverseGradTensor<float>.FromArray(mask, requiresGrad: false);
                var logits = model.Forward(tokens, maskTensor, batch, SeqLen);
                var lossFn = new CrossEntropyLoss<float>();
                var lossTensor = lossFn.Forward(logits, labels);
                lossTensor.Backward();

                var ps = model.GetParameters().Values.ToArray();
                for (int i = 0; i < keys.Count; i++)
                    cpuGrads[keys[i]] = GradOf(ps[i]);
            }

            Assert.That(deviceGrads.Keys.Count, Is.EqualTo(keys.Count));
            float worst = 0f;
            string worstKey = "";
            foreach (var key in keys)
            {
                Assert.That(deviceGrads.ContainsKey(key), Is.True, $"missing device grad for {key}");
                var dev = deviceGrads[key].Data;
                var cpu = cpuGrads[key];
                Assert.That(dev.Length, Is.EqualTo(cpu.Length), key);
                float localWorst = 0f;
                for (int i = 0; i < cpu.Length; i++)
                    localWorst = Math.Max(localWorst, Math.Abs(dev[i] - cpu[i]));
                if (localWorst > worst) { worst = localWorst; worstKey = key; }
            }
            Assert.That(worst, Is.LessThan(GradEps), $"worst gradient diff {worst} at {worstKey}");
        }
    }

    static float[] GradOf(Nivara.AutoDiff.Nn.Parameter<float> p)
    {
        var grad = p.Tensor.Grad;
        var arr = new float[grad?.Length ?? 0];
        grad?.CopyTo(arr, 0.0f);
        return arr;
    }

    [Test]
    public void Train_LossDecreases()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            const int samples = 64;
            var rng = new Random(37);
            var tokens = new int[samples * SeqLen];
            var mask = new float[samples * SeqLen];
            var labels = new int[samples];
            for (int i = 0; i < samples; i++)
            {
                labels[i] = i % 2;
                tokens[i * SeqLen] = labels[i] == 0 ? 2 : 5;
                for (int s = 1; s < SeqLen; s++)
                    tokens[i * SeqLen + s] = rng.Next(1, Vocab);
                for (int s = 0; s < SeqLen; s++)
                    mask[i * SeqLen + s] = 1f;
            }

            var epochs = trainer.Train(tokens, mask, labels,
                batchSize: 4, seqLen: SeqLen, epochs: 8,
                learningRate: 0.1f, weightDecay: 0.01f, seed: 41);

            Assert.That(epochs.Count, Is.EqualTo(8));
            for (int i = 0; i < epochs.Count; i++)
            {
                Assert.That(epochs[i].Epoch, Is.EqualTo(i));
                Assert.That(float.IsFinite(epochs[i].Loss), Is.True, $"epoch {i} loss not finite");
            }
            Assert.That(epochs[^1].Loss, Is.LessThan(0.75f * epochs[0].Loss),
                $"loss did not decrease: {epochs[0].Loss} -> {epochs[^1].Loss}");
        }
    }
}
