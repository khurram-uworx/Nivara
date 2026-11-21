using Nivara.AI;
using Nivara.Silk;

namespace NivaraInference;

public static partial class ModernBert
{
    public static int RunTrainHeadGpu(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string modelDir,
        Nivara.AI.OpenModernBertSession open)
    {
        Console.WriteLine("=== ModernBERT Head-Only Fine-Tune (Silk, F32) ===");
        Console.WriteLine();

        using var runtime = new SilkRuntime();
        Console.WriteLine($"Device: {runtime.DeviceName}");

        var config = LoadConfig(modelDir);
        using var runner = new SilkModernBertGpuRunner(runtime, config, tensors);

        // encode a short sentence -> [CLS] features
        var tokenizer = LoadTokenizer(modelDir, config);
        var ids = Encode(tokenizer, SampleSentences[0]);
        int validLength = ids.Length;
        var hidden = runner.Forward(ids, validLength);
        var cls = new float[config.HiddenSize];
        Array.Copy(hidden, 0, cls, 0, config.HiddenSize);

        using var trainer = new SilkModernBertHeadTrainer(runtime, config.HiddenSize, 4);
        var logits = trainer.Forward(cls);
        Console.WriteLine($"  forward logits: {string.Join(", ", logits.Select(v => v.ToString("F4")))}");

        // one-hot gradient for label 0
        var dLogits = new float[4];
        for (int c = 0; c < 4; c++) dLogits[c] = (c == 0 ? 1f : 0f) - Softmax(logits)[c];
        trainer.Backward(dLogits);
        trainer.AdamUpdate(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        Console.WriteLine($"  exported {exported.Count} head tensors.");

        var reloaded = ModernBertClassificationHead<float, float>.LoadWeights(exported, config.HiddenSize, 4);
        Console.WriteLine($"  reloaded head: hidden={reloaded.HiddenSize}, classes={reloaded.NumClasses}.");

        Console.WriteLine();
        Console.WriteLine("Silk head-only fine-tune + checkpoint round-trip: OK");
        return 0;
    }

    static float[] Softmax(float[] x)
    {
        float max = x.Max();
        var e = x.Select(v => MathF.Exp(v - max)).ToArray();
        float s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }
}
