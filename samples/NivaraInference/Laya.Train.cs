using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using Nivara.Silk;

namespace NivaraInference;

public static partial class Laya
{
    /// <summary>
    /// Runs a minimal Laya decision-head training loop on the Silk (OpenCL) backend: one forward,
    /// one backward and one Adam step on a seeded fixture, then exports the checkpoint and reloads
    /// it through <see cref="LayaDecisionHead{TModel,TWeight}.LoadWeights"/>. Demonstrates that
    /// <see cref="SilkLayaHeadTrainer"/> is reachable end to end. F32 only.
    /// </summary>
    public static int RunTrainGpu(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        Console.WriteLine("=== Laya Head Training (Silk) ===");
        Console.WriteLine("One forward / backward / Adam step on a seeded fixture, then checkpoint export + reload.");
        Console.WriteLine();

        using var runtime = new SilkRuntime();
        using var trainer = new SilkLayaHeadTrainer(runtime, tensors,
            headLayerCount: 1, actClassCount: 2, maxSeqLen: 8);

        Console.WriteLine($"Device: {trainer.DeviceName}");

        int hiddenDim = tensors["type_emb.weight"].Shape[1];
        const int seqLen = 6;
        const int validLength = 6;

        var rng = new Random(42);
        var hidden = new float[seqLen * hiddenDim];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        IReadOnlyList<int> markers = new[] { 2, 4 };

        var rng2 = new Random(3);
        var sg = new float[markers.Count];
        var ag = new float[2];
        for (int i = 0; i < sg.Length; i++) sg[i] = (float)(rng2.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < ag.Length; i++) ag[i] = (float)(rng2.NextDouble() * 2 - 1) * 0.1f;

        var output = trainer.Forward(hidden, LayaQuestionType.Choice, markers, validLength);
        Console.WriteLine($"  forward logits: {string.Join(", ", output.Logits.Select(v => v.ToString("F4")))}");

        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        Console.WriteLine($"  exported {exported.Count} tensors.");

        var reloaded = LayaDecisionHead<float, float>.LoadWeights(exported, 1, 2);
        Console.WriteLine($"  reloaded head: hidden={reloaded.HiddenSize}, layers={reloaded.HeadLayerCount}.");

        Console.WriteLine();
        Console.WriteLine("Training step + checkpoint round-trip: OK");
        return 0;
    }

    /// <summary>
    /// CPU AutoDiff counterpart of <see cref="RunTrainGpu"/>: one forward, one backward and one Adam
    /// step on the same seeded fixture, then checkpoint export + reload through
    /// <see cref="LayaDecisionHead{TModel,TWeight}.LoadWeights"/>. F32 only. Mirrors
    /// NivaraChat's adaptive-head training approach so the Silk path can be parity-checked against it.
    /// </summary>
    public static int RunTrain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        Console.WriteLine("=== Laya Head Training (CPU, F32) ===");
        Console.WriteLine("One forward / backward / Adam step on a seeded fixture, then checkpoint export + reload.");
        Console.WriteLine();

        int hiddenDim = tensors["type_emb.weight"].Shape[1];
        var head = LayaDecisionHead<float, float>.LoadWeights(tensors, 1, 2);

        const int seqLen = 6;
        const int validLength = 6;

        var rng = new Random(42);
        var hidden = new float[seqLen * hiddenDim];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        IReadOnlyList<int> markers = new[] { 2, 4 };

        var rng2 = new Random(3);
        var sg = new float[markers.Count];
        var ag = new float[2];
        for (int i = 0; i < sg.Length; i++) sg[i] = (float)(rng2.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < ag.Length; i++) ag[i] = (float)(rng2.NextDouble() * 2 - 1) * 0.1f;

        var hiddenTensor0 = ReverseGradTensor<float>.FromMatrix((float[])hidden.Clone(), seqLen, hiddenDim);
        var output = head.Forward(hiddenTensor0, LayaQuestionType.Choice, markers, validLength);
        Console.WriteLine($"  forward logits: {string.Join(", ", output.Logits.Select(v => v.ToString("F4")))}");

        using var optimizer = new Adam<float>(1e-3f);
        optimizer.AddParameterGroup(head.GetParameters().Values);

        optimizer.ZeroGrad();
        using (GradientUtils.Grad())
        {
            var hiddenT = ReverseGradTensor<float>.FromMatrix((float[])hidden.Clone(), seqLen, hiddenDim);
            var choiceLogits = head.ForwardChoiceLogits(hiddenT, LayaQuestionType.Choice, markers, validLength);
            var actLogits = head.ForwardActLogits(hiddenT, LayaQuestionType.Choice, markers, validLength);

            int k = markers.Count;
            var gradChoice = ReverseGradTensor<float>.FromMatrix((float[])sg.Clone(), 1, k, requiresGrad: false);
            var gradAct = ReverseGradTensor<float>.FromMatrix((float[])ag.Clone(), 1, 2, requiresGrad: false);

            var weightedChoice = ReverseGradOperations.Multiply(choiceLogits, gradChoice);
            var sumChoice = ReverseGradOperations.Sum(weightedChoice);
            var weightedAct = ReverseGradOperations.Multiply(actLogits, gradAct);
            var sumAct = ReverseGradOperations.Sum(weightedAct);
            var total = ReverseGradOperations.Add(sumChoice, sumAct);
            total.Backward();
        }
        optimizer.Step();

        var exported = ExportCheckpoint(head);
        Console.WriteLine($"  exported {exported.Count} tensors.");

        var reloaded = LayaDecisionHead<float, float>.LoadWeights(exported, 1, 2);
        Console.WriteLine($"  reloaded head: hidden={reloaded.HiddenSize}, layers={reloaded.HeadLayerCount}.");

        Console.WriteLine();
        Console.WriteLine("Training step + checkpoint round-trip: OK");
        return 0;
    }

    static Dictionary<string, (float[] Data, int[] Shape)> ExportCheckpoint(LayaDecisionHead<float, float> head)
    {
        var exported = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);
        int hidden = head.HiddenSize;

        if (head.typeEmb.Weight is not null && head.typeEmb.Weight.Tensor.Data.TryGetSpan(out var typeSpan))
        {
            var arr = new float[typeSpan.Length];
            typeSpan.CopyTo(arr);
            exported["type_emb.weight"] = (arr, new[] { 3, hidden });
        }

        for (int i = 0; i < head.HeadLayers.Count; i++)
        {
            var layer = head.HeadLayers[i];
            string prefix = $"head.layers.{i}";
            string attnPrefix = $"{prefix}.self_attn";

            ExportLayerNorm(exported, $"{prefix}.norm1", layer.norm1, hidden);
            ExportFusedAttention(exported, attnPrefix, layer.attn, hidden);
            ExportLayerNorm(exported, $"{prefix}.norm2", layer.norm2, hidden);
            ExportLinear(exported, $"{prefix}.linear1", layer.linear1);
            ExportLinear(exported, $"{prefix}.linear2", layer.linear2);
        }

        ExportLayerNorm(exported, "scorer.0", head.scorerNorm, hidden);
        ExportLinear(exported, "scorer.1", head.scorerHidden);
        ExportLinear(exported, "scorer.3", head.scorerOut);

        ExportLinear(exported, "act_head.0", head.actInput);
        ExportLinear(exported, "act_head.2", head.actOut);

        var temp = new float[head.CheckpointTemperature.Count];
        for (int i = 0; i < temp.Length; i++) temp[i] = head.CheckpointTemperature[i];
        exported["temperature"] = (temp, new[] { temp.Length });

        return exported;
    }

    static void ExportLayerNorm(Dictionary<string, (float[] Data, int[] Shape)> dict, string prefix, LayerNorm<float> ln, int hidden)
    {
        if (ln.Weight is not null && ln.Weight.Tensor.Data.TryGetSpan(out var wspan))
        {
            var warr = new float[wspan.Length];
            wspan.CopyTo(warr);
            dict[$"{prefix}.weight"] = (warr, new[] { hidden });
        }
        if (ln.Bias is not null && ln.Bias.Tensor.Data.TryGetSpan(out var bspan))
        {
            var barr = new float[bspan.Length];
            bspan.CopyTo(barr);
            dict[$"{prefix}.bias"] = (barr, new[] { hidden });
        }
    }

    static void ExportLinear(Dictionary<string, (float[] Data, int[] Shape)> dict, string prefix, Linear<float, float> linear)
    {
        if (linear.Weight is not null && linear.Weight.Tensor.Data.TryGetSpan(out var wspan))
        {
            var warr = new float[wspan.Length];
            wspan.CopyTo(warr);
            dict[$"{prefix}.weight"] = (warr, new[] { linear.OutFeatures, linear.InFeatures });
        }
        if (linear.Bias is not null && linear.Bias.Tensor.Data.TryGetSpan(out var bspan))
        {
            var barr = new float[bspan.Length];
            bspan.CopyTo(barr);
            dict[$"{prefix}.bias"] = (barr, new[] { barr.Length });
        }
    }

    static void ExportFusedAttention(Dictionary<string, (float[] Data, int[] Shape)> dict, string attnPrefix, BertSelfAttention<float, float> attn, int hidden)
    {
        var fusedW = new float[3 * hidden * hidden];
        var fusedB = new float[3 * hidden];
        var linears = new[] { attn.qProj, attn.kProj, attn.vProj };
        for (int b = 0; b < 3; b++)
        {
            var lin = linears[b];
            if (lin.Weight is not null && lin.Weight.Tensor.Data.TryGetSpan(out var wspan))
            {
                for (int r = 0; r < hidden; r++)
                    for (int c = 0; c < hidden; c++)
                        fusedW[(b * hidden + r) * hidden + c] = wspan[r * hidden + c];
            }
            if (lin.Bias is not null && lin.Bias.Tensor.Data.TryGetSpan(out var bspan))
            {
                for (int r = 0; r < hidden; r++)
                    fusedB[b * hidden + r] = bspan[r];
            }
        }
        dict[$"{attnPrefix}.in_proj_weight"] = (fusedW, new[] { 3 * hidden, hidden });
        dict[$"{attnPrefix}.in_proj_bias"] = (fusedB, new[] { 3 * hidden });
        ExportLinear(dict, $"{attnPrefix}.out_proj", attn.oProj);
    }
}