using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;

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
            Console.WriteLine("Usage: NivaraInference <mobilenet_v2|resnet18|minilm> [benchmark|similarity|compare|compare_diag|image-path]");
            Console.WriteLine();
            Console.WriteLine("Modes:");
            Console.WriteLine("  benchmark         Run 10 inference passes on synthetic data + real images");
            Console.WriteLine("  compare           Run forward pass on shared input, print logits for Python comparison");
            Console.WriteLine("  compare_diag      Step-by-step diagnostics, save intermediates to samples/data/diag/");
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
        bool compare = mode == "compare";
        bool compareDiag = mode == "compare_diag";

        switch (modelType)
        {
            case "mobilenet_v2":
                if (compareDiag) return RunCompareDiag(tensors, "mobilenet_v2");
                if (compare) return RunCompare(tensors, "mobilenet_v2");
                return benchmark ? RunMobileNetV2Benchmark(tensors) : RunMobileNetV2Inference(tensors, mode);
            case "resnet18":
                if (compareDiag) return RunCompareDiag(tensors, "resnet18");
                if (compare) return RunCompare(tensors, "resnet18");
                return benchmark ? RunResNet18Benchmark(tensors) : RunResNet18Inference(tensors, mode);
            case "minilm":
                bool similarity = mode == "similarity";
                return similarity ? RunMiniLMSimilarity(tensors) : benchmark ? RunMiniLMBenchmark(tensors) : RunMiniLMInference(tensors);
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

    static ReverseGradTensor<float> CreateRandomInput(int n, int c, int h, int w)
    {
        int total = n * c * h * w;
        var data = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var input = ReverseGradTensor<float>.FromMatrix(data, n, c * h * w);
        input.Reshape(n, c, h, w);
        return input;
    }

    static ReverseGradTensor<float> PreprocessImage(Bitmap img, int size)
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

        var input = ReverseGradTensor<float>.FromMatrix(data, 1, 3 * size * size);
        input.Reshape(1, 3, size, size);
        return input;
    }

    static int RunCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
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
        Console.WriteLine($"  {inputData.Length} floats, mean={inputData.Average():F6}");

        var input = ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        Module<float> model = modelType switch
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

        Console.WriteLine($"Logits stats: min={logits.Min():F6}, max={logits.Max():F6}, mean={logits.Average():F6}");

        PrintTopK(output);

        var logitsPath = Path.Combine("samples", "data", "compare_logits_cs.bin");
        using (var fs = File.Create(logitsPath))
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
        Console.WriteLine($"Saved logits to {logitsPath}");

        return 0;
    }

    static int RunCompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
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
        Console.WriteLine($"  {inputData.Length} floats, mean={inputData.Average():F6}");

        var input = ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        if (modelType == "resnet18")
            RunResNet18Diag(tensors, input, diagDir);
        else if (modelType == "mobilenet_v2")
            RunMobileNetV2Diag(tensors, input, diagDir);
        else
        {
            Console.Error.WriteLine($"Unknown model type: {modelType}");
            return 1;
        }

        Console.WriteLine($"Saved diagnostics to {diagDir}/");
        return 0;
    }

    static void SaveDiag(string diagDir, string name, ReverseGradTensor<float> tensor)
    {
        int total = tensor.Length;
        var data = new float[total];
        tensor.Data.TryGetSpan(out var span);
        if (!span.IsEmpty)
            span.Slice(0, total).CopyTo(data);
        else
            tensor.Data.CopyTo(data, 0);

        string path = Path.Combine(diagDir, $"{name}.bin");
        using var fs = File.Create(path);
        fs.Write(MemoryMarshal.AsBytes(data.AsSpan()));

        double mean = data.Average();
        float min = data.Min(), max = data.Max();
        string shapeStr = string.Join("x", tensor.Shape);
        Console.WriteLine($"  {name}: [{shapeStr}], mean={mean:F6}, min={min:F6}, max={max:F6}");
        Console.Write($"    first9: [");
        for (int i = 0; i < Math.Min(9, total); i++)
        {
            Console.Write($"{data[i]:F6}");
            if (i < Math.Min(9, total) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");
    }

    static void RunResNet18Diag(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        ReverseGradTensor<float> input,
        string diagDir)
    {
        Console.WriteLine("=== ResNet-18 Step-by-Step Diagnostics ===");
        Console.WriteLine();

        var stemConv = new Conv2d<float>(3, 64, 7, stride: 2, padding: 3, bias: false);
        var stemBn = new BatchNorm2d<float>(64);
        var stemPool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);

        ResNet18.LoadConv(stemConv,
            tensors["resnet.embedder.embedder.convolution.weight"].Data,
            tensors["resnet.embedder.embedder.convolution.weight"].Shape);
        ResNet18.LoadBn(stemBn,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.weight", out var sw0) ? sw0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.bias", out var sb0) ? sb0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_mean", out var sm0) ? sm0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_var", out var sv0) ? sv0.Data : null);
        stemBn.Eval();

        Console.WriteLine("--- Step 1: Stem Conv ---");
        var x = stemConv.Forward(input);
        SaveDiag(diagDir, "cs_step1_stem_conv", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 2: Stem BN (eval) ---");
        x = stemBn.Forward(x);
        SaveDiag(diagDir, "cs_step2_stem_bn", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 3: Stem ReLU ---");
        x = ReverseGradOperations.Relu(x);
        SaveDiag(diagDir, "cs_step3_stem_relu", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 4: Stem Pool ---");
        x = stemPool.Forward(x);
        SaveDiag(diagDir, "cs_step4_stem_pool", x);
        Console.WriteLine();

        string[] stagePrefixes = [
            "resnet.encoder.stages.0.layers.0",
            "resnet.encoder.stages.0.layers.1",
            "resnet.encoder.stages.1.layers.0",
            "resnet.encoder.stages.1.layers.1",
            "resnet.encoder.stages.2.layers.0",
            "resnet.encoder.stages.2.layers.1",
            "resnet.encoder.stages.3.layers.0",
            "resnet.encoder.stages.3.layers.1",
        ];
        int[] inChannels = [64, 64, 64, 128, 128, 256, 256, 512];
        int[] outChannels = [64, 64, 128, 128, 256, 256, 512, 512];
        int[] strides = [1, 1, 2, 1, 2, 1, 2, 1];

        for (int i = 0; i < 8; i++)
        {
            bool hasDownsample = inChannels[i] != outChannels[i] || strides[i] != 1;

            var conv1 = new Conv2d<float>(inChannels[i], outChannels[i], 3, stride: strides[i], padding: 1, bias: false);
            var bn1 = new BatchNorm2d<float>(outChannels[i]);
            var conv2 = new Conv2d<float>(outChannels[i], outChannels[i], 3, padding: 1, bias: false);
            var bn2 = new BatchNorm2d<float>(outChannels[i]);

            Conv2d<float>? dsConv = null;
            BatchNorm2d<float>? dsBn = null;
            if (hasDownsample)
            {
                dsConv = new Conv2d<float>(inChannels[i], outChannels[i], 1, stride: strides[i], bias: false);
                dsBn = new BatchNorm2d<float>(outChannels[i]);
            }

            ResNet18.LoadConv(conv1, tensors[$"{stagePrefixes[i]}.layer.0.convolution.weight"].Data,
                tensors[$"{stagePrefixes[i]}.layer.0.convolution.weight"].Shape);
            ResNet18.LoadBn(bn1,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.weight", out var w1) ? w1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.bias", out var b1) ? b1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.running_mean", out var m1) ? m1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.running_var", out var v1) ? v1.Data : null);

            ResNet18.LoadConv(conv2, tensors[$"{stagePrefixes[i]}.layer.1.convolution.weight"].Data,
                tensors[$"{stagePrefixes[i]}.layer.1.convolution.weight"].Shape);
            ResNet18.LoadBn(bn2,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.weight", out var w2) ? w2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.bias", out var b2) ? b2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.running_mean", out var m2) ? m2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.running_var", out var v2) ? v2.Data : null);

            if (hasDownsample && dsConv != null && dsBn != null)
            {
                ResNet18.LoadConv(dsConv, tensors[$"{stagePrefixes[i]}.shortcut.convolution.weight"].Data,
                    tensors[$"{stagePrefixes[i]}.shortcut.convolution.weight"].Shape);
                ResNet18.LoadBn(dsBn,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.weight", out var sw) ? sw.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.bias", out var sb) ? sb.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.running_mean", out var sm) ? sm.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.running_var", out var sv) ? sv.Data : null);
            }

            conv1.Eval(); bn1.Eval(); conv2.Eval(); bn2.Eval();
            dsConv?.Eval(); dsBn?.Eval();

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (conv1) ---");
            var cx = conv1.Forward(x);
            SaveDiag(diagDir, $"cs_stage{i}_conv1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (bn1) ---");
            cx = bn1.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_bn1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (relu1) ---");
            cx = ReverseGradOperations.Relu(cx);
            SaveDiag(diagDir, $"cs_stage{i}_relu1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (conv2) ---");
            cx = conv2.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_conv2", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (bn2) ---");
            cx = bn2.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_bn2", cx);

            var residual = hasDownsample && dsConv != null && dsBn != null
                ? dsBn.Forward(dsConv.Forward(x))
                : x;

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (residual) ---");
            SaveDiag(diagDir, $"cs_stage{i}_residual", residual);

            cx = cx + residual;
            cx = ReverseGradOperations.Relu(cx);
            x = cx;

            Console.WriteLine($"--- After stage {i} ---");
            SaveDiag(diagDir, $"cs_step{5 + i}_stage{i / 2}{'a' + i % 2}", x);
            Console.WriteLine();
        }

        var avgPool = new AdaptiveAvgPool2d<float>(1);
        avgPool.Eval();
        x = avgPool.Forward(x);
        SaveDiag(diagDir, "cs_step9_avgpool", x);

        int n = x.Shape[0], c = x.Shape[1];
        x.Reshape(n, c);
        SaveDiag(diagDir, "cs_step9b_flattened", x);

        var fc = new Linear<float>(512, 1000, bias: true);
        ResNet18.LoadLinear(fc,
            tensors["classifier.1.weight"].Data,
            tensors["classifier.1.weight"].Shape,
            tensors.TryGetValue("classifier.1.bias", out var bias) ? bias.Data : null);
        fc.Eval();
        x = fc.Forward(x);
        SaveDiag(diagDir, "cs_step10_logits", x);
    }

    static void RunMobileNetV2Diag(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        ReverseGradTensor<float> input,
        string diagDir)
    {
        Console.WriteLine("=== MobileNetV2 Step-by-Step Diagnostics ===");
        Console.WriteLine();

        var model = MobileNetV2.LoadWeights(tensors);
        var x = model.Forward(input);
        SaveDiag(diagDir, "cs_final_logits", x);
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

    static int RunMiniLMInference(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Inference ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        string text = "This is a test sentence.";
        var input = MiniLMTokenizer.Tokenize(tokenizer, text, maxLen: 128);

        Console.WriteLine($"Input text: \"{text}\"");
        var inputData = new float[input.Length];
        input.Data.TryGetSpan(out var inSpan);
        if (!inSpan.IsEmpty) inSpan.CopyTo(inputData);
        Console.WriteLine($"Input tokens (first 10): [{string.Join(", ", inputData.Take(10).Select(x => (int)x))}] (seqLen={input.Length})");

        model.Eval();
        var fwdSw = Stopwatch.StartNew();
        var output = model.Forward(input);
        fwdSw.Stop();

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);

        Console.WriteLine($"Forward: {fwdSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{output.Shape[^1]}]");
        Console.WriteLine($"Output stats: min={outputData.Min():F6}, max={outputData.Max():F6}, mean={outputData.Average():F6}");
        Console.Write("Output[:10]: [");
        for (int i = 0; i < Math.Min(10, outputData.Length); i++)
        {
            Console.Write($"{outputData[i]:F6}");
            if (i < Math.Min(10, outputData.Length) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");

        float norm = 0f;
        foreach (var v in outputData) norm += v * v;
        norm = MathF.Sqrt(norm);
        Console.WriteLine($"L2 norm: {norm:F6} (should be ~1.0 for normalized embeddings)");
        Console.WriteLine();

        return 0;
    }

    static int RunMiniLMBenchmark(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        string text = "This is a long test sentence that will be tokenized to demonstrate the performance of the MiniLM model inference across multiple tokens for benchmarking purposes.";
        var input = MiniLMTokenizer.Tokenize(tokenizer, text, maxLen: 128);

        Console.WriteLine($"Input text length: {text.Split(' ').Length} words");
        Console.WriteLine($"Input tokens: {input.Length}");
        Console.WriteLine();

        Console.WriteLine("Warmup (3 passes)...");
        for (int i = 0; i < 3; i++)
            model.Forward(input);

        Console.WriteLine("Benchmarking (10 passes)...");
        var times = new List<long>();
        for (int i = 0; i < 10; i++)
        {
            var sw = Stopwatch.StartNew();
            model.Forward(input);
            sw.Stop();
            times.Add(sw.ElapsedMilliseconds);
        }

        double avg = times.Average();
        double min = times.Min();
        double max = times.Max();
        Console.WriteLine($"  Average: {avg:F1} ms");
        Console.WriteLine($"  Min:     {min} ms");
        Console.WriteLine($"  Max:     {max} ms");
        Console.WriteLine();

        return 0;
    }

    static int RunMiniLMSimilarity(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Cosine Similarity Demo ===");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);

        var sentences = new[]
        {
            "This is a cat.",
            "This is a dog.",
            "I love programming.",
            "The weather is nice today.",
            "I love coding."
        };

        Console.WriteLine($"Sentences ({sentences.Length}):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {sentences[i]}");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        var embeddings = new float[sentences.Length][];
        for (int s = 0; s < sentences.Length; s++)
        {
            var input = MiniLMTokenizer.Tokenize(tokenizer, sentences[s], maxLen: 128);

            var output = model.Forward(input);
            var outputData = new float[output.Length];
            output.Data.TryGetSpan(out var outSpan);
            if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);
            embeddings[s] = outputData;
        }

        Console.WriteLine("Cosine Similarity Matrix:");
        Console.Write("       ");
        for (int i = 0; i < sentences.Length; i++)
            Console.Write($"  [{i}]   ");
        Console.WriteLine();

        for (int i = 0; i < sentences.Length; i++)
        {
            Console.Write($"  [{i}]  ");
            for (int j = 0; j < sentences.Length; j++)
            {
                float sim = CosineSimilarity(embeddings[i], embeddings[j]);
                Console.Write($"{sim,7:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();

        return 0;
    }

    static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0f, normA = 0f, normB = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denom > 1e-12f ? dot / denom : 0f;
    }
}
