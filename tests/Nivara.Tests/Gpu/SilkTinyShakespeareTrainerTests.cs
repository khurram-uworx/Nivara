using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using Nivara.Silk;
using NivaraChat.Transformer;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public class SilkTinyShakespeareTrainerTests
{
    const int Vocab = 32;
    const int NEmbd = 16;
    const int NLayer = 1;
    const int NHead = 2;
    const int Block = 8;

    static int[] SeededWeights(int seed)
    {
        return null!;
    }

    [Test]
    public void Step_GradientsMatchCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available."); }

        using var modelCpu = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        var initial = modelCpu.ToSilkWeights();

        int[] input = [1];
        int[] target = [2];

        // CPU backward
        float[] cpuGrad;
        using (var gradScope = GradientUtils.Grad())
        {
            var cpuInput = BuildInput(input);
            var logits = modelCpu.Forward(cpuInput);
            var targets = (int[])target.Clone();
            var lossFn = new CrossEntropyLoss<float>();
            var loss = lossFn.Forward(logits, targets);
            loss.Backward();
            var g = modelCpu.TokenEmbedding.Weight!.Tensor.Grad!;
            cpuGrad = new float[g.Length]; // g is NivaraColumn? Need copy
            // NivaraColumn has CopyTo?
        }

        // Silk backward
        var config = new TinyShakespeareConfig { HiddenSize = NEmbd, NumHiddenLayers = NLayer, NumAttentionHeads = NHead, VocabSize = Vocab, MaxSeqLen = Block };
        using var runtime = new SilkRuntime();
        using var trainer = new SilkTinyShakespeareTrainer(runtime, config, initial, maxBatch: 4, maxSeqLen: Block);
        trainer.Forward(input, input.Length);
        trainer.Backward(target, target.Length, 1f / input.Length);
        var silkGrad = trainer.ExportGradients()["tok_emb.weight"].Data;

        // copy cpu grad
        var cpuGrads = modelCpu.ToSilkGradients();
        var silkGrads = trainer.ExportGradients();

        float worst = 0;
        string worstKey = "";
        foreach (var (key, (cpuData, _)) in cpuGrads)
        {
            if (!silkGrads.TryGetValue(key, out var silkEntry)) continue;
            var silkData = silkEntry.Data;
            float ma = 0;
            for (int i = 0; i < cpuData.Length; i++) ma = MathF.Max(ma, MathF.Abs(cpuData[i] - silkData[i]));
            Console.WriteLine($"{key}: grad maxAbs={ma}");
            if (ma > worst) { worst = ma; worstKey = key; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(1e-3f), $"grad diverged: {worstKey}");
    }

    [Test]
    public void Step_WeightsMatchCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available."); }

        using var modelCpu = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        var initial = modelCpu.ToSilkWeights();

        using var modelSilk = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        modelSilk.LoadFromSilkWeights(initial);

        int[] input = [1, 2, 3, 4, 5, 6, 7, 0];
        int[] target = [2, 3, 4, 5, 6, 7, 0, 1];

        using (var gradScope = GradientUtils.Grad())
        {
            var cpuInput = BuildInput(input);
            var logits = modelCpu.Forward(cpuInput);
            var targets = (int[])target.Clone();
            var lossFn = new CrossEntropyLoss<float>();
            var loss = lossFn.Forward(logits, targets);
            loss.Backward();
            var opt = new Adam<float>(1e-3f, 0.9f, 0.95f);
            opt.AddParameterGroup(modelCpu.GetParameters().Values);
            opt.Step();
            opt.ZeroGrad();
        }
        var cpuWeights = modelCpu.ToSilkWeights();

        var config = new TinyShakespeareConfig { HiddenSize = NEmbd, NumHiddenLayers = NLayer, NumAttentionHeads = NHead, VocabSize = Vocab, MaxSeqLen = Block };
        using var runtime = new SilkRuntime();
        using var trainer = new SilkTinyShakespeareTrainer(runtime, config, initial, maxBatch: 4, maxSeqLen: Block);
        trainer.ZeroGradients();
        trainer.Forward(input, input.Length);
        trainer.Backward(target, target.Length, 1f / input.Length);
        trainer.AdamUpdateAll(1e-3f, 0f, 0.9f, 0.95f);
        var silkWeights = trainer.ExportCheckpoint();

        float worst = 0; string worstKey = "";
        foreach (var (key, (cpuData, _)) in cpuWeights)
        {
            var silkData = silkWeights[key].Data;
            float ma = 0;
            for (int i = 0; i < cpuData.Length; i++) ma = MathF.Max(ma, MathF.Abs(cpuData[i] - silkData[i]));
            if (ma > worst) { worst = ma; worstKey = key; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(1e-2f), $"weights diverged: {worstKey}");
    }

    [Test]
    public void Forward_MatchesCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available."); }

        using var model = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        int[] input = [1, 2, 3, 4];

        var cpuTensor = model.Forward(BuildInput(input));
        var cpuLogits = TensorToFloats(cpuTensor);

        var weights = model.ToSilkWeights();
        var config = new TinyShakespeareConfig { HiddenSize = NEmbd, NumHiddenLayers = NLayer, NumAttentionHeads = NHead, VocabSize = Vocab, MaxSeqLen = Block };
        using var runtime = new SilkRuntime();
        using var trainer = new SilkTinyShakespeareTrainer(runtime, config, weights, maxBatch: 4, maxSeqLen: Block);
        var silkLogits = trainer.Forward(input, input.Length);

        Assert.That(silkLogits.Length, Is.EqualTo(cpuLogits.Length));
        float maxAbs = 0;
        for (int i = 0; i < cpuLogits.Length; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuLogits[i] - silkLogits[i]));
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-4f));
    }

    static ReverseGradTensor<float> BuildInput(int[] ids)
    {
        var data = new float[ids.Length];
        for (int i = 0; i < ids.Length; i++) data[i] = ids[i];
        var col = NivaraColumn<float>.Create(data);
        var t = new ReverseGradTensor<float>(col, requiresGrad: false);
        t.Reshape(1, ids.Length);
        return t;
    }

    static float[] TensorToFloats(ReverseGradTensor<float> tensor)
    {
        var values = new float[tensor.Length];
        tensor.Data.CopyTo(values, 0f);
        return values;
    }
}
