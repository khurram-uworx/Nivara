using Nivara.AI;
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
}
