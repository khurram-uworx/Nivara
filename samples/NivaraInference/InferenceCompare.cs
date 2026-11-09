using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace NivaraInference;

/// <summary>
/// Shared compare/parity entry points extracted from the sample CLI so the performance harness
/// (and, eventually, other callers) can run them without going through the visitor-facing CLI
/// (issue #1097, follow-up to #1051).
/// </summary>
public static class InferenceCompare
{
    public static int VisionCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
        => RunCompare(tensors, modelType);

    public static int VisionCompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
        => RunCompareDiag(tensors, modelType);

    public static int ModernBertCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
        => ModernBert.Compare(tensors, modelDir);

    public static int ModernBertCompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
        => ModernBert.CompareDiag(tensors, modelDir);

    public static int ModernBertCompareGpu(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir,
        Nivara.AI.OpenModernBertSession open)
        => ModernBert.CompareGpu(tensors, modelDir, open);

    public static int LayaCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
        => Laya.Compare(tensors, modelDir);

    public static int LayaCompareGpu(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir,
        Nivara.AI.OpenLayaSession open)
        => Laya.CompareGpu(tensors, modelDir, open);

    public static int MiniLMCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => ProgramRunMiniLMCompare(tensors);

    public static int DistilBertCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => ProgramRunDistilBertCompare(tensors);

    public static int DistilBertSstCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => ProgramRunDistilBertSstCompare(tensors);

    // Moved bodies live below as private helpers; delegates target public/accessible shims added to Program.

    // Placeholder shim methods so the above delegates compile before Program exposes them.
    private static int ProgramRunMiniLMCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => Program.RunMiniLMCompare(tensors);

    private static int ProgramRunDistilBertCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => Program.RunDistilBertCompare(tensors);

    private static int ProgramRunDistilBertSstCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => Program.RunDistilBertSstCompare(tensors);

    private static int RunCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
    {
        string inputPath = Path.Combine("samples", "data", "compare_input.bin");
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/generate_input.py");
            return 1;
        }

        Console.WriteLine($"Reading input from {inputPath}...");
        var rawBytes = File.ReadAllBytes(inputPath);
        float[] inputData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, inputData, 0, rawBytes.Length);
        Console.WriteLine($"  {inputData.Length} floats, mean={TensorPrimitives.Average(inputData.AsSpan()):F6}");

        var input = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        Nivara.AutoDiff.Nn.Module<float> model = modelType switch
        {
            "mobilenet_v2" => MobileNetV2.LoadWeights(tensors),
            "resnet18" => ResNet18.LoadWeights(tensors),
            _ => throw new ArgumentException($"Unknown model: {modelType}")
        };

        Console.WriteLine("Running forward pass...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass: {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        int numClasses = output.Shape[^1];
        var logits = new float[numClasses];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
            outSpan.CopyTo(logits);

        Console.WriteLine($"Raw logits (first 10):");
        Console.Write("  [");
        for (int i = 0; i < Math.Min(10, numClasses); i++)
        {
            Console.Write($"{logits[i]:F6}");
            if (i < 9) Console.Write(", ");
        }
        Console.WriteLine("]");

        Console.WriteLine($"Logits stats: min={TensorPrimitives.Min(logits.AsSpan()):F6}, max={TensorPrimitives.Max(logits.AsSpan()):F6}, mean={TensorPrimitives.Average(logits.AsSpan()):F6}");

        Program.PrintTopK(output);

        var logitsPath = Path.Combine("samples", "data", "compare_logits_cs.bin");
        using (var fs = File.Create(logitsPath))
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
        Console.WriteLine($"Saved logits to {logitsPath}");

        return 0;
    }

    private static int RunCompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
    {
        string inputPath = Path.Combine("samples", "data", "compare_input.bin");
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            return 1;
        }

        string diagDir = Path.Combine("samples", "data", "diag");
        Directory.CreateDirectory(diagDir);

        Console.WriteLine($"Reading input from {inputPath}...");
        var rawBytes = File.ReadAllBytes(inputPath);
        float[] inputData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, inputData, 0, rawBytes.Length);
        Console.WriteLine($"  {inputData.Length} floats, mean={TensorPrimitives.Average(inputData.AsSpan()):F6}");

        var input = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        if (modelType == "resnet18")
            Program.RunResNet18Diag(tensors, input, diagDir);
        else if (modelType == "mobilenet_v2")
            Program.RunMobileNetV2Diag(tensors, input, diagDir);
        else
        {
            Console.Error.WriteLine($"Unknown model type: {modelType}");
            return 1;
        }

        Console.WriteLine($"Saved diagnostics to {diagDir}/");
        return 0;
    }
}
