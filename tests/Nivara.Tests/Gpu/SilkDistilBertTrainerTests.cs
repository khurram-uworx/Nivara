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

    static (SilkRuntime runtime, SilkDistilBertTrainer trainer, DistilBertForSequenceClassification<float> model) Make(int trainableLayerCount = -1)
    {
        var runtime = Device();
        var distilConfig = TinyDistilConfig();
        var config = distilConfig.ToBertConfig();
        var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        var tensors = model.ToSilkTensors();
        var trainer = new SilkDistilBertTrainer(runtime, config, tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16, trainableLayerCount: trainableLayerCount);
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
    public void Gradients_HeadOnly_MatchTrainableSubsetAndFrozenRemainZero()
    {
        var (runtime, trainer, model) = Make(0);
        using (runtime)
        using (trainer)
        {
            var rng = new Random(11);
            int batch = 2;
            int[] tokens = new int[batch * SeqLen];
            float[] mask = new float[batch * SeqLen];
            int[] labels = new int[batch];
            for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
            for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);

            var batchGradient = new float[batch * 2];
            var deviceLogits = trainer.Forward(tokens, mask, batch, SeqLen);
            SilkLoss.InputGradient(deviceLogits, labels, batch, 2, 1f / batch, batchGradient);
            trainer.Backward(batch, SeqLen, batchGradient);
            var deviceGrads = trainer.ExportGradients();

            // CPU full reverse pass keeps every parameter trainable; compare only the head subset.
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
                for (int i = 0; i < keys.Count; i++) cpuGrads[keys[i]] = GradOf(ps[i]);
            }

            foreach (var key in keys)
            {
                if (!deviceGrads.ContainsKey(key)) continue;
                var dev = deviceGrads[key].Data;
                var cpu = cpuGrads[key];
                if (key.Contains("pre_classifier", StringComparison.Ordinal)
                    || key.Contains("classifier", StringComparison.Ordinal))
                {
                    float localWorst = 0f;
                    for (int i = 0; i < cpu.Length; i++)
                        localWorst = Math.Max(localWorst, Math.Abs(dev[i] - cpu[i]));
                    Assert.That(localWorst, Is.LessThan(GradEps), $"{key}: {localWorst}");
                }
                else
                {
                    float max = 0f;
                    for (int i = 0; i < dev.Length; i++) max = Math.Max(max, Math.Abs(dev[i]));
                    Assert.That(max, Is.LessThan(1e-12f), $"frozen param {key} has nonzero gradient {max}");
                }
            }
        }
    }

    [Test]
    public void Gradients_TopOneLayer_MatchTrainableSubsetAndLowerLayersFrozen()
    {
        const int trainableLayerCount = 1;
        var (runtime, trainer, model) = Make(trainableLayerCount);
        using (runtime)
        using (trainer)
        {
            var rng = new Random(11);
            int batch = 2;
            int[] tokens = new int[batch * SeqLen];
            float[] mask = new float[batch * SeqLen];
            int[] labels = new int[batch];
            for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
            for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);

            var batchGradient = new float[batch * 2];
            var deviceLogits = trainer.Forward(tokens, mask, batch, SeqLen);
            SilkLoss.InputGradient(deviceLogits, labels, batch, 2, 1f / batch, batchGradient);
            trainer.Backward(batch, SeqLen, batchGradient);
            var deviceGrads = trainer.ExportGradients();

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
                for (int i = 0; i < keys.Count; i++) cpuGrads[keys[i]] = GradOf(ps[i]);
            }

            int trainableLayerIndex = Layers - 1;
            foreach (var key in keys)
            {
                if (!deviceGrads.ContainsKey(key)) continue;
                var dev = deviceGrads[key].Data;
                var cpu = cpuGrads[key];
                bool trainable = key.Contains("pre_classifier", StringComparison.Ordinal)
                    || key.Contains("classifier", StringComparison.Ordinal)
                    || key.Contains($"transformer.layer.{trainableLayerIndex}.", StringComparison.Ordinal);
                if (trainable)
                {
                    float localWorst = 0f;
                    for (int i = 0; i < cpu.Length; i++)
                        localWorst = Math.Max(localWorst, Math.Abs(dev[i] - cpu[i]));
                    Assert.That(localWorst, Is.LessThan(GradEps), $"{key}: {localWorst}");
                }
                else
                {
                    float max = 0f;
                    for (int i = 0; i < dev.Length; i++) max = Math.Max(max, Math.Abs(dev[i]));
                    Assert.That(max, Is.LessThan(1e-12f), $"frozen param {key} has nonzero gradient {max}");
                }
            }
        }
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

    [Test]
    public void TrainableLayerCount_Aliases_ReturnSameValue()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            foreach (int value in new[] { 0, 1, Layers })
            {
                trainer.TrainableLayerCount = value;
                Assert.That(trainer.NumTrainableLayers, Is.EqualTo(value));
                Assert.That(trainer.TrainableLayers, Is.EqualTo(value));
                Assert.That(trainer.NumTrainableLayers, Is.EqualTo(trainer.TrainableLayers));
            }
        }
    }

    [Test]
    public void TrainableLayerCount_ClampsNegativeAndOversized()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            trainer.TrainableLayerCount = -1;
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));

            trainer.TrainableLayerCount = -100;
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));

            trainer.TrainableLayerCount = Layers + 5;
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers));

            trainer.TrainableLayerCount = int.MaxValue;
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers));
        }
    }

    [Test]
    public void FreezeEncoder_SetsZeroTrainableLayers()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            trainer.TrainableLayerCount = Layers;
            trainer.FreezeEncoder();
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));
            Assert.That(trainer.NumTrainableLayers, Is.EqualTo(0));
            Assert.That(trainer.TrainableLayers, Is.EqualTo(0));
        }
    }

    [Test]
    public void SetTrainableLayers_ClampsAndUpdatesAliases()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            trainer.SetTrainableLayers(1);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(1));
            Assert.That(trainer.NumTrainableLayers, Is.EqualTo(1));
            Assert.That(trainer.TrainableLayers, Is.EqualTo(1));

            trainer.SetTrainableLayers(Layers + 3);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers));

            trainer.SetTrainableLayers(-2);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));
        }
    }

    [Test]
    public void FreezeLayers_SemanticsAndClamping()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            trainer.FreezeLayers(0);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers));

            trainer.FreezeLayers(1);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers - 1));

            trainer.FreezeLayers(Layers);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));

            trainer.FreezeLayers(Layers + 5);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));

            trainer.FreezeLayers(-3);
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(Layers));
        }
    }

    [Test]
    public void Freeze_Alias_MatchesFreezeEncoder()
    {
        var (runtime, trainer, _) = Make();
        using (runtime)
        using (trainer)
        {
            trainer.TrainableLayerCount = Layers;
            trainer.Freeze();
            Assert.That(trainer.TrainableLayerCount, Is.EqualTo(0));
        }
    }

    [Test]
    public void Head_RemainsTrainableInAllModes()
    {
        foreach (int trainable in new[] { 0, 1, Layers })
        {
            var (runtime, trainer, model) = Make(trainable);
            using (runtime)
            using (trainer)
            {
                var rng = new Random(11);
                int batch = 2;
                int[] tokens = new int[batch * SeqLen];
                float[] mask = new float[batch * SeqLen];
                int[] labels = new int[batch];
                for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
                for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);

                var batchGradient = new float[batch * 2];
                var deviceLogits = trainer.Forward(tokens, mask, batch, SeqLen);
                SilkLoss.InputGradient(deviceLogits, labels, batch, 2, 1f / batch, batchGradient);
                trainer.Backward(batch, SeqLen, batchGradient);
                var deviceGrads = trainer.ExportGradients();

                foreach (var key in deviceGrads.Keys)
                {
                    if (key.Contains("pre_classifier", StringComparison.Ordinal)
                        || key.Contains("classifier", StringComparison.Ordinal))
                    {
                        float max = 0f;
                        foreach (var v in deviceGrads[key].Data) max = Math.Max(max, Math.Abs(v));
                        Assert.That(max, Is.GreaterThan(0f), $"head param {key} should be trainable in mode {trainable}");
                    }
                }
            }
        }
    }
}
