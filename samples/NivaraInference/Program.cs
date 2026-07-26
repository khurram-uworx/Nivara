using System.Diagnostics;
using Nivara.AutoDiff.Nn;

namespace NivaraInference;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Nivara HuggingFace Inference ===");
        Console.WriteLine();

        string modelType = args.Length > 0 ? args[0] : "";
        string imagePath = args.Length > 1 ? args[1] : "";

        if (string.IsNullOrEmpty(modelType) || modelType is "-h" or "--help")
        {
            Console.WriteLine("Usage: NivaraInference <mobilenet_v2|resnet18> [image-path]");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  NivaraInference mobilenet_v2 samples/data/cat.jpg");
            Console.WriteLine("  NivaraInference resnet18 samples/data/dog.jpg");
            return 1;
        }

        string modelDir = Path.Combine("samples", "data", modelType);
        string modelPath = Path.Combine(modelDir, "model.safetensors");

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Model file not found: {modelPath}");
            return 1;
        }

        var fileInfo = new FileInfo(modelPath);
        Console.WriteLine($"Loading weights from {fileInfo.Name} ({fileInfo.Length / (1024.0 * 1024.0):F1} MB)...");

        var sw = Stopwatch.StartNew();
        var tensors = SafeTensorsLoader.Read(modelPath);
        sw.Stop();
        Console.WriteLine($"Loaded {tensors.Count} tensors in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        switch (modelType)
        {
            case "mobilenet_v2":
                RunMobileNetV2(tensors, imagePath);
                break;
            case "resnet18":
                RunResNet18(tensors, imagePath);
                break;
            default:
                Console.Error.WriteLine($"Unknown model type: {modelType}");
                return 1;
        }

        return 0;
    }

    static void RunMobileNetV2(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string imagePath)
    {
        Console.WriteLine("Building MobileNetV2 model...");
        var sw = Stopwatch.StartNew();
        var model = MobileNetV2.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        RunInference(model, "MobileNetV2");
    }

    static void RunResNet18(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string imagePath)
    {
        Console.WriteLine("Building ResNet-18 model...");
        var sw = Stopwatch.StartNew();
        var model = ResNet18.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        RunInference(model, "ResNet-18");
    }

    static void RunInference(Module<float> model, string modelName)
    {
        int n = 1, c = 3, h = 224, w = 224;
        int inputSize = n * c * h * w;
        var inputData = new float[inputSize];
        var rng = new Random(42);
        for (int i = 0; i < inputSize; i++)
            inputData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var input = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(inputData, n, c * h * w);
        input.Reshape(n, c, h, w);

        Console.WriteLine($"Running forward pass with input [{n},{c},{h},{w}]...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass completed in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
        Console.WriteLine($"Output length: {output.Length}");

        int numClasses = output.Shape[^1];
        int topK = Math.Min(5, numClasses);
        var scores = new float[numClasses];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
            outSpan.CopyTo(scores);

        var topIndices = Enumerable.Range(0, numClasses)
            .OrderByDescending(i => scores[i])
            .Take(topK)
            .ToArray();

        Console.WriteLine();
        Console.WriteLine($"Top-{topK} predictions:");
        for (int i = 0; i < topK; i++)
        {
            int idx = topIndices[i];
            Console.WriteLine($"  #{i + 1}: class {idx,5}  score={scores[idx]:F6}");
        }
    }
}
