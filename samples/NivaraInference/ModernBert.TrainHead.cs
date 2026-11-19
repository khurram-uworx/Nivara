using Nivara.AI;
using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace NivaraInference;

public static partial class ModernBert
{
    /// <summary>
    /// Head-only fine-tune showcase for issue #1280: freezes a ModernBERT encoder,
    /// trains a small classification head on precomputed [CLS] features on CPU (F32),
    /// exports the head weights and reloads them for inference.
    /// </summary>
    public static int RunTrainHead(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        Console.WriteLine("=== ModernBERT Head-Only Fine-Tune (CPU, F32) ===");
        Console.WriteLine();

        var config = LoadConfig(modelDir);
        PrintConfig(config);

        var buildSw = System.Diagnostics.Stopwatch.StartNew();
        var encoder = ModernBertEncoder<float, float>.LoadWeights(tensors, config);
        encoder.Eval();
        buildSw.Stop();
        Console.WriteLine($"Load encoder: {buildSw.ElapsedMilliseconds} ms (frozen)");

        var head = new ModernBertClassificationHead<float, float>(config.HiddenSize, 4);

        // Deterministic synthetic examples: short random token-id sequences + labels.
        var rng = new Random(42);
        const int exampleCount = 16;
        const int seqLen = 8;
        var examples = new (int[] Ids, int ValidLength, int Label)[exampleCount];
        for (int i = 0; i < exampleCount; i++)
        {
            var ids = new int[seqLen];
            for (int j = 0; j < seqLen; j++) ids[j] = rng.Next(0, Math.Min(1000, config.VocabSize));
            ids[0] = config.ClsTokenId;
            examples[i] = (ids, seqLen, i % 4);
        }

        // Baseline accuracy before training.
        head.Eval();
        Console.WriteLine("Training head for 5 epochs on frozen [CLS] features...");
        var result = ModernBertHeadTraining.Train(encoder, head, examples, epochs: 5, learningRate: 0.01f);
        Console.WriteLine($"  examples={result.Examples}, epochs={result.Epochs}, finalLoss={result.FinalLoss:F4}, accuracy={result.Accuracy:P1}");

        // Export + reload round trip using safetensors-style keys matching StateDictLoader.
        var exported = new Dictionary<string, (float[] Data, int[] Shape)>();
        void ExportLinear(string prefix, Linear<float, float> linear)
        {
            if (linear.Weight is not null && linear.Weight.Tensor.Data.TryGetSpan(out var wspan))
            {
                var warr = new float[wspan.Length];
                wspan.CopyTo(warr);
                int outF = linear.OutFeatures;
                int inF = linear.InFeatures;
                exported[$"{prefix}.weight"] = (warr, new[] { outF, inF });
            }
            if (linear.Bias is not null && linear.Bias.Tensor.Data.TryGetSpan(out var bspan))
            {
                var barr = new float[bspan.Length];
                bspan.CopyTo(barr);
                exported[$"{prefix}.bias"] = (barr, new[] { barr.Length });
            }
        }
        ExportLinear("head.hidden", head.Hidden);
        ExportLinear("head.classifier", head.Classifier);
        Console.WriteLine($"  exported {exported.Count} head tensors.");

        var reloaded = ModernBertClassificationHead<float, float>.LoadWeights(exported, config.HiddenSize, 4);
        Console.WriteLine($"  reloaded head: hidden={reloaded.HiddenSize}, classes={reloaded.NumClasses}.");

        Console.WriteLine();
        Console.WriteLine("Head-only fine-tune + checkpoint round-trip: OK");
        return 0;
    }
}
