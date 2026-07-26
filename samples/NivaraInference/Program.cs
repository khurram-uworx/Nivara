using System.Diagnostics;
using System.Drawing;
using Nivara.AutoDiff.Nn;

namespace NivaraInference;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Nivara HuggingFace Inference ===");
        Console.WriteLine();

        string modelType = args.Length > 0 ? args[0] : "";
        string mode = args.Length > 1 ? args[1] : "";

        if (string.IsNullOrEmpty(modelType) || modelType is "-h" or "--help")
        {
            Console.WriteLine("Usage: NivaraInference <mobilenet_v2|resnet18> [benchmark|image-path]");
            Console.WriteLine();
            Console.WriteLine("Modes:");
            Console.WriteLine("  benchmark         Run 10 inference passes on synthetic data + real images");
            Console.WriteLine("  <image-path>      Run inference on a single image");
            return 1;
        }

        string modelDir = Path.Combine("samples", "data", modelType);
        string modelPath = Path.Combine(modelDir, "model.safetensors");

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Model file not found: {modelPath}");
            return 1;
        }

        Console.WriteLine($"Loading weights from {Path.GetFileName(modelPath)}...");
        var loadSw = Stopwatch.StartNew();
        var tensors = SafeTensorsLoader.Read(modelPath);
        loadSw.Stop();
        Console.WriteLine($"  SafeTensors parse: {loadSw.ElapsedMilliseconds} ms ({tensors.Count} tensors)");
        Console.WriteLine();

        bool benchmark = mode == "benchmark";

        switch (modelType)
        {
            case "mobilenet_v2":
                return benchmark ? RunMobileNetV2Benchmark(tensors) : RunMobileNetV2Inference(tensors, mode);
            case "resnet18":
                return benchmark ? RunResNet18Benchmark(tensors) : RunResNet18Inference(tensors, mode);
            default:
                Console.Error.WriteLine($"Unknown model type: {modelType}");
                return 1;
        }
    }

    static int RunMobileNetV2Benchmark(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MobileNetV2 Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        var model = MobileNetV2.LoadWeights(tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int paramCount = MobileNetV2.CountParameters(tensors);
        Console.WriteLine($"Parameters: {paramCount:N0}");
        Console.WriteLine();

        int n = 1, c = 3, h = 224, w = 224;
        Console.WriteLine("Warmup (3 passes)...");
        for (int i = 0; i < 3; i++)
        {
            var dummy = CreateRandomInput(n, c, h, w);
            model.Forward(dummy);
        }
        Console.WriteLine();

        Console.WriteLine($"Benchmark: synthetic {w}x{h} input (10 passes)...");
        var times = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var input = CreateRandomInput(n, c, h, w);
            var sw = Stopwatch.StartNew();
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            times.Add(ms);
            Console.WriteLine($"  Run {i + 1,2}: {ms:F1} ms");
        }
        Console.WriteLine($"  Average: {times.Average():F1} ms  (min={times.Min():F1}, max={times.Max():F1})");

        var lastInput = CreateRandomInput(n, c, h, w);
        var lastOutput = model.Forward(lastInput);
        PrintTopK(lastOutput);
        Console.WriteLine();

        RunImageBenchmarks(model);
        return 0;
    }

    static int RunResNet18Benchmark(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== ResNet-18 Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        var model = ResNet18.LoadWeights(tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int paramCount = ResNet18.CountParameters(tensors);
        Console.WriteLine($"Parameters: {paramCount:N0}");
        Console.WriteLine();

        int n = 1, c = 3, h = 224, w = 224;
        Console.WriteLine("Warmup (3 passes)...");
        for (int i = 0; i < 3; i++)
        {
            var dummy = CreateRandomInput(n, c, h, w);
            model.Forward(dummy);
        }
        Console.WriteLine();

        Console.WriteLine($"Benchmark: synthetic {w}x{h} input (10 passes)...");
        var times = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var input = CreateRandomInput(n, c, h, w);
            var sw = Stopwatch.StartNew();
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            times.Add(ms);
            Console.WriteLine($"  Run {i + 1,2}: {ms:F1} ms");
        }
        Console.WriteLine($"  Average: {times.Average():F1} ms  (min={times.Min():F1}, max={times.Max():F1})");

        var lastInput = CreateRandomInput(n, c, h, w);
        var lastOutput = model.Forward(lastInput);
        PrintTopK(lastOutput);
        Console.WriteLine();

        RunImageBenchmarks(model);
        return 0;
    }

    static void RunImageBenchmarks(Module<float> model)
    {
        string imageDir = Path.Combine("samples", "data", "images");
        if (!Directory.Exists(imageDir))
        {
            Console.WriteLine("No images directory found, skipping image benchmarks.");
            return;
        }

        var imageFiles = Directory.GetFiles(imageDir, "*.jpg").OrderBy(f => f).ToArray();
        if (imageFiles.Length == 0)
        {
            Console.WriteLine("No .jpg images found, skipping image benchmarks.");
            return;
        }

        Console.WriteLine($"Benchmark: real images ({imageFiles.Length} images)...");
        foreach (var path in imageFiles)
        {
            using var img = new Bitmap(path);
            var sw = Stopwatch.StartNew();
            var input = PreprocessImage(img, 224);
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            Console.WriteLine($"  {Path.GetFileName(path)} ({img.Width}x{img.Height}): {ms:F1} ms");
            PrintTopK(output, k: 3);
            Console.WriteLine();
        }
    }

    static int RunMobileNetV2Inference(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string mode)
    {
        Console.WriteLine("Building MobileNetV2 model...");
        var sw = Stopwatch.StartNew();
        var model = MobileNetV2.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        if (string.IsNullOrEmpty(mode))
            RunInference(model);
        else
            RunImageInference(model, mode);
        return 0;
    }

    static int RunResNet18Inference(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string mode)
    {
        Console.WriteLine("Building ResNet-18 model...");
        var sw = Stopwatch.StartNew();
        var model = ResNet18.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        if (string.IsNullOrEmpty(mode))
            RunInference(model);
        else
            RunImageInference(model, mode);
        return 0;
    }

    static void RunInference(Module<float> model)
    {
        var input = CreateRandomInput(1, 3, 224, 224);

        Console.WriteLine($"Running forward pass with input [1,3,224,224]...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass completed in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        PrintTopK(output);
    }

    static void RunImageInference(Module<float> model, string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            Console.Error.WriteLine($"Image not found: {imagePath}");
            return;
        }

        Console.WriteLine($"Loading image: {imagePath}");
        using var img = new Bitmap(imagePath);
        Console.WriteLine($"  Original size: {img.Width}x{img.Height}");

        var input = PreprocessImage(img, 224);
        Console.WriteLine($"  Preprocessed to [1,3,224,224]");

        Console.WriteLine($"Running forward pass...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass completed in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        PrintTopK(output);
    }

    static Nivara.AutoDiff.ReverseGradTensor<float> CreateRandomInput(int n, int c, int h, int w)
    {
        int total = n * c * h * w;
        var data = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var input = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(data, n, c * h * w);
        input.Reshape(n, c, h, w);
        return input;
    }

    static Nivara.AutoDiff.ReverseGradTensor<float> PreprocessImage(Bitmap img, int size)
    {
        using var resized = new Bitmap(img, new Size(size, size));

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];
        var data = new float[3 * size * size];

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                var pixel = resized.GetPixel(x, y);
                int spatialIdx = y * size + x;
                data[0 * size * size + spatialIdx] = (pixel.R / 255.0f - mean[0]) / std[0];
                data[1 * size * size + spatialIdx] = (pixel.G / 255.0f - mean[1]) / std[1];
                data[2 * size * size + spatialIdx] = (pixel.B / 255.0f - mean[2]) / std[2];
            }
        }

        var input = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(data, 1, 3 * size * size);
        input.Reshape(1, 3, size, size);
        return input;
    }

    static void PrintTopK(Nivara.AutoDiff.ReverseGradTensor<float> output, int k = 5)
    {
        int numClasses = output.Shape[^1];
        k = Math.Min(k, numClasses);
        var scores = new float[numClasses];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
            outSpan.CopyTo(scores);

        var topIndices = Enumerable.Range(0, numClasses)
            .OrderByDescending(i => scores[i])
            .Take(k)
            .ToArray();

        Console.WriteLine($"Top-{k} predictions:");
        for (int i = 0; i < k; i++)
        {
            int idx = topIndices[i];
            Console.WriteLine($"  #{i + 1}: class {idx,5}  score={scores[idx]:F6}");
        }
    }
}
