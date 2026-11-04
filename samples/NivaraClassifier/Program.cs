using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using Nivara.Onnx;
using Nivara.Samples;
using Nivara.Silk;
using NivaraClassifier;
using System.Globalization;

var options = args.Length == 0
    ? InteractiveWizard.Run()
    : Options.Parse(args);

if (options.Help)
{
    Options.PrintHelp();
    return;
}

if (options.Silk && options.Command is not ("train" or "predict" or "benchmark"))
{
    Console.Error.WriteLine("--silk is supported only with --command train, predict or benchmark.");
    Environment.ExitCode = 2;
    return;
}
if (options.Silk && options.Command == "predict" && !string.IsNullOrWhiteSpace(options.LoadOnnxPath))
{
    Console.Error.WriteLine("--silk cannot be used with --load-onnx (ONNX Runtime has its own providers).");
    Environment.ExitCode = 2;
    return;
}
if (options.Command == "benchmark" && !string.IsNullOrWhiteSpace(options.LoadOnnxPath))
{
    Console.Error.WriteLine("--benchmark does not use --load-onnx; it measures the Nivara and Silk paths.");
    Environment.ExitCode = 2;
    return;
}
if (options.Command == "train")
    RunTraining(options);
else if (options.Command == "predict")
    RunPrediction(options);
else if (options.Command == "benchmark")
    Benchmark.Run(options);
else if (options.Command == "generate")
    RunGenerate(options);
else if (options.Command == "onnx-compare")
    RunOnnxCompare(options);
else
    Options.PrintHelp();

return;

// ── Commands ────────────────────────────────────────────────────────────────

void RunGenerate(Options opt)
{
    string csvPath = opt.DataPath ?? Path.Combine("samples", "data", "sentiment_data.csv");
    string? dir = Path.GetDirectoryName(csvPath);
    if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        Directory.CreateDirectory(dir);
    DataGenerator.SaveCsv(csvPath, opt.NumSamples, opt.Seed);
    Console.WriteLine($"Generated {opt.NumSamples} samples → {csvPath}");
}

void RunTraining(Options opt)
{
    string csvPath = opt.DataPath ?? Path.Combine("samples", "data", "sentiment_data.csv");
    if (!File.Exists(csvPath))
    {
        Console.WriteLine($"Data file not found: {csvPath}");
        Console.WriteLine("Run with --command generate first, or specify --data-path.");
        return;
    }

    var (allTexts, allLabels) = DataGenerator.LoadCsv(csvPath);

    var tokenizer = TextTokenizer.FromDocuments(allTexts, maxVocabSize: opt.MaxVocab);
    Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");
    Console.WriteLine($"Mode: {opt.Mode}");

    int trainCount = (int)(allTexts.Length * 0.8);
    var trainTexts = allTexts.AsSpan(0, trainCount).ToArray();
    var trainLabels = allLabels.AsSpan(0, trainCount).ToArray();
    var testTexts = allTexts.AsSpan(trainCount).ToArray();
    var testLabels = allLabels.AsSpan(trainCount).ToArray();

    var trainTokens = new int[trainCount * opt.MaxSeqLen];
    for (int i = 0; i < trainCount; i++)
    {
        var encoded = tokenizer.Encode(trainTexts[i], fixedLength: opt.MaxSeqLen);
        Array.Copy(encoded, 0, trainTokens, i * opt.MaxSeqLen, opt.MaxSeqLen);
    }

    var testTokens = new int[testLabels.Length * opt.MaxSeqLen];
    for (int i = 0; i < testLabels.Length; i++)
    {
        var encoded = tokenizer.Encode(testTexts[i], fixedLength: opt.MaxSeqLen);
        Array.Copy(encoded, 0, testTokens, i * opt.MaxSeqLen, opt.MaxSeqLen);
    }

    if (opt.Silk && opt.Mode == "conv")
    {
        GpuTextConvClassifier? silkConv = null;
        try
        {
            silkConv = new GpuTextConvClassifier(tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, opt.MaxSeqLen, opt.Seed);
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Silk/OpenCL unavailable ({ex.Message}); using the CPU trainer.");
        }

        if (silkConv is not null)
        {
            Console.WriteLine("Training conv on Silk (device forward, device backward)...");
            var epochs = silkConv.Train(trainTokens, trainLabels, opt.BatchSize, opt.MaxSeqLen, opt.Epochs, (float)opt.LearningRate, opt.Seed);
            foreach (var e in epochs)
                Console.WriteLine($"  epoch {e.Epoch}: loss={e.Loss:F4} ({e.Elapsed.TotalSeconds:F1}s)");

            var layout = silkConv.ExportCpuLayout();
            var cpuModel = new ConvTextClassifierModel<float>(tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, 2, opt.MaxSeqLen);
            cpuModel.SetWeights(tokenizer.VocabSize, opt.EmbeddingDim, opt.MaxSeqLen,
                layout.Embedding, layout.Conv1W, layout.Conv1B, layout.Conv2W, layout.Conv2B,
                layout.Conv3W, layout.Conv3B, layout.Fc1W, layout.Fc1B, layout.Fc2W, layout.Fc2B);

            int convCorrect = 0;
            for (int i = 0; i < testLabels.Length; i++)
            {
                var encoded2 = tokenizer.Encode(testTexts[i], fixedLength: opt.MaxSeqLen);
                var preds2 = cpuModel.Predict(encoded2);
                if (preds2[0] == testLabels[i]) convCorrect++;
            }
            double convAccuracy = (double)convCorrect / testLabels.Length;
            Console.WriteLine($"Silk-trained (conv) test accuracy: {convAccuracy:P1} ({convCorrect}/{testLabels.Length})");

            if (!string.IsNullOrWhiteSpace(opt.SavePath))
            {
                ModelSerializer.Save(cpuModel, opt.SavePath);
                Console.WriteLine($"Saved model: {opt.SavePath}");
            }
            string convVocabPath = Path.Combine("samples", "data", "classifier.vocab.json");
            if (!string.IsNullOrWhiteSpace(opt.SavePath))
                convVocabPath = Path.ChangeExtension(opt.SavePath, ".vocab.json");
            tokenizer.Save(convVocabPath);
            Console.WriteLine($"Saved vocabulary: {convVocabPath}");
            silkConv.Dispose();
            Console.WriteLine("Done.");
            return;
        }
    }

    if (opt.Silk)
    {
        GpuTextLinearClassifier? silkModel = null;
        try
        {
            silkModel = new GpuTextLinearClassifier(tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, opt.Seed);
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Silk/OpenCL unavailable ({ex.Message}); using the CPU trainer.");
        }

        if (silkModel is not null)
        {
            Console.WriteLine("Training on Silk (device backward, device Adam)...");
            var epochs = silkModel.Train(trainTokens, trainLabels, opt.BatchSize, opt.MaxSeqLen, opt.Epochs, (float)opt.LearningRate, opt.Seed);
            foreach (var e in epochs)
                Console.WriteLine($"  epoch {e.Epoch}: loss={e.Loss:F4} ({e.Elapsed.TotalSeconds:F1}s)");

            var layout = silkModel.ExportCpuLayout();
            var cpuModel = new TextClassifierModel<float>(tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, 2, opt.MaxSeqLen);
            cpuModel.SetWeights(layout.Embedding, layout.Fc1W, layout.Fc1B, layout.Fc2W, layout.Fc2B);

            int gpuCorrect = 0;
            for (int i = 0; i < testLabels.Length; i++)
            {
                var encoded = tokenizer.Encode(testTexts[i], fixedLength: opt.MaxSeqLen);
                var preds = cpuModel.Predict(encoded);
                if (preds[0] == testLabels[i]) gpuCorrect++;
            }
            double gpuAccuracyValue = (double)gpuCorrect / testLabels.Length;
            Console.WriteLine($"Silk-trained test accuracy: {gpuAccuracyValue:P1} ({gpuCorrect}/{testLabels.Length})");

            if (!string.IsNullOrWhiteSpace(opt.SavePath))
            {
                ModelSerializer.Save(cpuModel, opt.SavePath);
                Console.WriteLine($"Saved model: {opt.SavePath}");
            }
            string gpuVocabPath = Path.Combine("samples", "data", "classifier.vocab.json");
            if (!string.IsNullOrWhiteSpace(opt.SavePath))
                gpuVocabPath = Path.ChangeExtension(opt.SavePath, ".vocab.json");
            tokenizer.Save(gpuVocabPath);
            Console.WriteLine($"Saved vocabulary: {gpuVocabPath}");
            silkModel.Dispose();
            Console.WriteLine("Done.");
            return;
        }
    }

    Module<float> model;
    Func<int[], int[]> predict;
    if (opt.Mode == "conv")
    {
        var conv = new ConvTextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        model = conv;
        predict = conv.Predict;
    }
    else
    {
        var linear = new TextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        model = linear;
        predict = linear.Predict;
    }

    using var optimizer = new Adam<float>(learningRate: 0.001f);
    var allParams = model.GetParameters().Values;
    optimizer.AddParameterGroup(allParams);

    var lossFn = new CrossEntropyLoss<float>();

    var trainFrame = FrameBuilder.BuildDocumentClassificationFrame(trainTokens, trainLabels, trainCount, opt.MaxSeqLen);
    var featureColumns = Enumerable.Range(0, opt.MaxSeqLen).Select(d => $"tok_{d}").ToArray();
    var trainDataset = new TensorDataset<float>(trainFrame, featureColumns, ["label"]);
    var trainLoader = new DataLoader<float>(trainDataset, opt.BatchSize, shuffle: true, seed: opt.Seed);

    Console.WriteLine($"\nTraining: {trainCount} samples, {testLabels.Length} test");
    Console.WriteLine($"Model: {tokenizer.VocabSize} vocab → {opt.EmbeddingDim} embed → {opt.HiddenDim} hidden → 2 classes ({opt.Mode})\n");

    var trainLoop = new TrainingLoop<float>(
        model, trainLoader,
        (logits, labels) =>
        {
            int batchSize = logits.Length / 2;
            var targets = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
                targets[i] = int.CreateChecked(labels.Data[i]);
            return lossFn.Forward(logits, targets);
        },
        optimizer, opt.Epochs);

    var result = trainLoop.Run();
    result.PrintSummary();

    int correct = 0;
    for (int i = 0; i < testLabels.Length; i++)
    {
        var encoded = tokenizer.Encode(testTexts[i], fixedLength: opt.MaxSeqLen);
        var preds = predict(encoded);
        if (preds[0] == testLabels[i]) correct++;
    }

    double accuracy = (double)correct / testLabels.Length;
    Console.WriteLine($"\nTest accuracy: {accuracy:P1} ({correct}/{testLabels.Length})");

    if (!string.IsNullOrWhiteSpace(opt.SavePath))
    {
        ModelSerializer.Save(model, opt.SavePath);
        Console.WriteLine($"Saved model: {opt.SavePath}");
    }

    string vocabPath = Path.Combine("samples", "data", "classifier.vocab.json");
    if (!string.IsNullOrWhiteSpace(opt.SavePath))
        vocabPath = Path.ChangeExtension(opt.SavePath, ".vocab.json");
    tokenizer.Save(vocabPath);
    Console.WriteLine($"Saved vocabulary: {vocabPath}");

    if (!string.IsNullOrWhiteSpace(opt.ExportOnnxPath))
    {
        model.Eval();
        using var ms = new MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions
        {
            InputShape = new[] { 1, opt.MaxSeqLen },
        });
        var onnxBytes = ms.ToArray();
        File.WriteAllBytes(opt.ExportOnnxPath, onnxBytes);
        Console.WriteLine($"Exported ONNX: {opt.ExportOnnxPath} ({onnxBytes.Length} bytes)");
    }

    Console.WriteLine("Done.");
}

void RunPrediction(Options opt)
{
    if (!string.IsNullOrWhiteSpace(opt.LoadOnnxPath))
    {
        RunPredictionOnnx(opt);
        return;
    }

    string modelPath = opt.LoadPath ?? "classifier.model.json";
    string vocabPath = Path.ChangeExtension(modelPath, ".vocab.json");

    if (!File.Exists(modelPath))
    {
        Console.WriteLine($"Model not found: {modelPath}. Train first with --command train.");
        return;
    }
    if (!File.Exists(vocabPath))
    {
        Console.WriteLine($"Vocabulary not found: {vocabPath}.");
        return;
    }

    var tokenizer = TextTokenizer.Load(vocabPath);
    Func<int[], int[]> predict;
    IDisposable? silkGpu = null;
    Func<int[], int[]>? silkPredict = null;
    if (opt.Silk)
    {
        try
        {
            if (opt.Mode == "conv")
            {
                var conv = new ConvTextClassifierModel<float>(
                    tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
                ModelSerializer.Load(conv, modelPath);
                conv.Eval();

                var ps = conv.GetParameters().Values.ToArray();
                // Module_0 embedding, Module_1/2/3 conv, Module_4 fc1, Module_5 fc2
                var emb = ps[0].Tensor.Data.ToArray();
                var c1w = ps[1].Tensor.Data.ToArray();
                var c1b = ps[2].Tensor.Data.ToArray();
                var c2w = ps[3].Tensor.Data.ToArray();
                var c2b = ps[4].Tensor.Data.ToArray();
                var c3w = ps[5].Tensor.Data.ToArray();
                var c3b = ps[6].Tensor.Data.ToArray();
                var fc1wCpu = ps[7].Tensor.Data.ToArray();
                var fc1b = ps[8].Tensor.Data.ToArray();
                var fc2wCpu = ps[9].Tensor.Data.ToArray();
                var fc2b = ps[10].Tensor.Data.ToArray();

                int embed = opt.EmbeddingDim;
                int hidden = opt.HiddenDim;
                int fcIn = 64 * 3 * opt.MaxSeqLen;
                var gpuFc1 = new float[fcIn * hidden];
                for (int i = 0; i < hidden; i++)
                    for (int j = 0; j < fcIn; j++)
                        gpuFc1[j * hidden + i] = fc1wCpu[i * fcIn + j];
                var gpuFc2 = new float[hidden * 2];
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < hidden; j++)
                        gpuFc2[j * 2 + i] = fc2wCpu[i * hidden + j];

                var gpu = new GpuTextConvClassifier(tokenizer.VocabSize, embed, hidden, opt.MaxSeqLen, opt.Seed);
                silkGpu = gpu;
                gpu.SetWeights(emb, c1w, c1b, c2w, c2b, c3w, c3b, gpuFc1, fc1b, gpuFc2, fc2b);
                silkPredict = encoded => gpu.Predict(encoded, opt.MaxSeqLen);
            }
            else
            {
                var linear = new TextClassifierModel<float>(
                    tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
                ModelSerializer.Load(linear, modelPath);
                linear.Eval();

                var ps = linear.GetParameters().Values.ToArray();
                var emb = ps[0].Tensor.Data.ToArray();
                var fc1wCpu = ps[1].Tensor.Data.ToArray();
                var fc1b = ps[2].Tensor.Data.ToArray();
                var fc2wCpu = ps[3].Tensor.Data.ToArray();
                var fc2b = ps[4].Tensor.Data.ToArray();

                int embed = opt.EmbeddingDim;
                int hidden = opt.HiddenDim;
                var gpuFc1 = new float[embed * hidden];
                for (int e = 0; e < embed; e++)
                    for (int h = 0; h < hidden; h++)
                        gpuFc1[e * hidden + h] = fc1wCpu[h * embed + e];
                var gpuFc2 = new float[hidden * 2];
                for (int h = 0; h < hidden; h++)
                    for (int c = 0; c < 2; c++)
                        gpuFc2[h * 2 + c] = fc2wCpu[c * hidden + h];

                var gpu = new GpuTextLinearClassifier(tokenizer.VocabSize, embed, hidden, opt.Seed);
                silkGpu = gpu;
                gpu.SetWeights(emb, gpuFc1, fc1b, gpuFc2, fc2b);
                silkPredict = encoded => gpu.Predict(encoded, opt.MaxSeqLen);
            }
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Silk/OpenCL unavailable ({ex.Message}); using the CPU model.");
        }
    }

    if (silkPredict is not null)
        predict = silkPredict;
    else if (opt.Mode == "conv")
    {
        var conv = new ConvTextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        ModelSerializer.Load(conv, modelPath);
        conv.Eval();
        predict = conv.Predict;
    }
    else
    {
        var linear = new TextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        ModelSerializer.Load(linear, modelPath);
        linear.Eval();
        predict = linear.Predict;
    }

    Console.WriteLine("Nivara Text Classifier — Interactive Prediction");
    Console.WriteLine("Type a sentence (empty line to quit):\n");

    while (true)
    {
        Console.Write("> ");
        string? line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) break;

        var encoded = tokenizer.Encode(line, fixedLength: opt.MaxSeqLen);
        var preds = predict(encoded);
        string label = preds[0] == 1 ? "POSITIVE" : "NEGATIVE";
        Console.WriteLine($"  → {label}\n");
    }

    silkGpu?.Dispose();
}

void RunOnnxCompare(Options opt)
{
    string modelPath = opt.LoadPath ?? "classifier.model.json";
    string vocabPath = Path.ChangeExtension(modelPath, ".vocab.json");
    if (!File.Exists(modelPath))
    {
        Console.WriteLine($"Model not found: {modelPath}. Train first with --command train.");
        return;
    }
    if (!File.Exists(vocabPath))
    {
        Console.WriteLine($"Vocabulary not found: {vocabPath}.");
        return;
    }

    var tokenizer = TextTokenizer.Load(vocabPath);

    Module<float> model;
    Func<int[], int[]> nivaraPredict;
    if (opt.Mode == "conv")
    {
        var conv = new ConvTextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        ModelSerializer.Load(conv, modelPath);
        conv.Eval();
        model = conv;
        nivaraPredict = conv.Predict;
    }
    else
    {
        var linear = new TextClassifierModel<float>(
            tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
        ModelSerializer.Load(linear, modelPath);
        linear.Eval();
        model = linear;
        nivaraPredict = linear.Predict;
    }

    var sentences = new[]
    {
        "this movie is great and I loved it",
        "terrible waste of time and money",
        "pretty good overall",
        "awful plot and bad acting",
        "not bad at all",
        "I hate this product"
    };

    var nivaraLogitsList = new List<float[]>();
    var nivaraLabels = new List<int>();
    foreach (var s in sentences)
    {
        var encoded = tokenizer.Encode(s, fixedLength: opt.MaxSeqLen);
        var data = new float[encoded.Length];
        for (int i = 0; i < encoded.Length; i++) data[i] = encoded[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, 1, opt.MaxSeqLen, requiresGrad: false);
        var logits = model.Forward(input);
        var arr = new float[logits.Length];
        logits.Data.CopyTo(arr, 0.0f);
        nivaraLogitsList.Add(arr);
        nivaraLabels.Add(nivaraPredict(encoded)[0]);
    }

    string onnxPath = !string.IsNullOrWhiteSpace(opt.ExportOnnxPath)
        ? opt.ExportOnnxPath!
        : Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", "classifier.compare.onnx");
    model.Eval();
    using (var ms = new MemoryStream())
    {
        OnnxExporter.Export(model, ms, new OnnxExportOptions
        {
            InputShape = new[] { 1, opt.MaxSeqLen },
        });
        File.WriteAllBytes(onnxPath, ms.ToArray());
    }
    Console.WriteLine($"Exported ONNX: {onnxPath}");

    using var session = new InferenceSession(onnxPath);
    string inputName = session.InputMetadata.Keys.First();
    string outputName = session.OutputMetadata.Keys.First();

    double globalMaxAbs = 0.0;
    int mismatches = 0;
    for (int i = 0; i < sentences.Length; i++)
    {
        var encoded = tokenizer.Encode(sentences[i], fixedLength: opt.MaxSeqLen);
        var tensor = new DenseTensor<float>(encoded.Select(t => (float)t).ToArray(), new[] { 1, opt.MaxSeqLen });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        });
        var ortLogits = results.First(v => v.Name == outputName).AsTensor<float>().ToArray();
        double maxAbs = 0.0;
        for (int j = 0; j < nivaraLogitsList[i].Length; j++)
            maxAbs = Math.Max(maxAbs, Math.Abs(ortLogits[j] - nivaraLogitsList[i][j]));
        globalMaxAbs = Math.Max(globalMaxAbs, maxAbs);
        int ortLabel = ortLogits[0] > ortLogits[1] ? 0 : 1;
        if (ortLabel != nivaraLabels[i]) mismatches++;
        Console.WriteLine($"  \"{sentences[i]}\"  nivara={LabelName(nivaraLabels[i])}  onnx={LabelName(ortLabel)}  maxAbs={maxAbs:E3}");
    }

    Console.WriteLine($"Global max abs diff: {globalMaxAbs:E3}");
    if (mismatches > 0 || globalMaxAbs > 1e-4)
    {
        Console.WriteLine($"FAIL: {mismatches} label mismatch(es), tolerance 1e-4");
        Environment.ExitCode = 1;
    }
    else
    {
        Console.WriteLine("OK: Nivara / ONNX Runtime parity within 1e-4");
    }
}

static string LabelName(int label) => label == 1 ? "POSITIVE" : "NEGATIVE";

void RunPredictionOnnx(Options opt)
{
    string onnxPath = opt.LoadOnnxPath!;
    string vocabPath = !string.IsNullOrWhiteSpace(opt.LoadPath)
        ? Path.ChangeExtension(opt.LoadPath, ".vocab.json")
        : Path.ChangeExtension(onnxPath, ".vocab.json");

    if (!File.Exists(onnxPath))
    {
        Console.WriteLine($"ONNX model not found: {onnxPath}.");
        return;
    }
    if (!File.Exists(vocabPath))
    {
        Console.WriteLine($"Vocabulary not found: {vocabPath}.");
        return;
    }

    var tokenizer = TextTokenizer.Load(vocabPath);

    using var session = new InferenceSession(onnxPath);
    string inputName = session.InputMetadata.Keys.First();
    string outputName = session.OutputMetadata.Keys.First();

    Console.WriteLine("Nivara Text Classifier (ONNX Runtime) — Interactive Prediction");
    Console.WriteLine("Type a sentence (empty line to quit):\\n");

    while (true)
    {
        Console.Write("> ");
        string? line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) break;

        var encoded = tokenizer.Encode(line, fixedLength: opt.MaxSeqLen);
        var tensor = new DenseTensor<float>(encoded.Select(t => (float)t).ToArray(), new[] { 1, opt.MaxSeqLen });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        });
        var output = results.First(v => v.Name == outputName).AsTensor<float>().ToArray();
        int labelIdx = output[0] > output[1] ? 0 : 1;
        string label = labelIdx == 1 ? "POSITIVE" : "NEGATIVE";
        Console.WriteLine($"  → {label}\\n");
    }
}

// ── Types ───────────────────────────────────────────────────────────────────

sealed class Options
{
    public string Command { get; set; } = "train";
    public string? DataPath { get; set; }
    public string? SavePath { get; set; }
    public string? LoadPath { get; set; }
    public string? ExportOnnxPath { get; set; }
    public string? LoadOnnxPath { get; set; }
    public int Epochs { get; set; } = 10;
    public int BatchSize { get; set; } = 32;
    public int EmbeddingDim { get; set; } = 32;
    public int HiddenDim { get; set; } = 64;
    public int MaxSeqLen { get; set; } = 20;
    public int MaxVocab { get; set; } = 5000;
    public int NumSamples { get; set; } = 1000;
    public double LearningRate { get; set; } = 0.001;
    public int Seed { get; set; } = 42;
    public string Mode { get; set; } = "linear";
    public bool Help { get; set; }
    public bool Silk { get; set; }
    public bool BatchSizeExplicit { get; set; }

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--command":
                case "-c":
                    o.Command = ReadString(args, ref i, args[i]);
                    break;
                case "--data-path":
                    o.DataPath = ReadString(args, ref i, args[i]);
                    break;
                case "--save":
                    o.SavePath = ReadString(args, ref i, args[i]);
                    break;
                case "--load":
                    o.LoadPath = ReadString(args, ref i, args[i]);
                    break;
                case "--export-onnx":
                    o.ExportOnnxPath = ReadString(args, ref i, args[i]);
                    break;
                case "--load-onnx":
                    o.LoadOnnxPath = ReadString(args, ref i, args[i]);
                    break;
                case "--epochs":
                    o.Epochs = ReadInt(args, ref i, args[i]);
                    break;
                case "--batch-size":
                    o.BatchSize = ReadInt(args, ref i, args[i]);
                    o.BatchSizeExplicit = true;
                    break;
                case "--embedding-dim":
                    o.EmbeddingDim = ReadInt(args, ref i, args[i]);
                    break;
                case "--hidden-dim":
                    o.HiddenDim = ReadInt(args, ref i, args[i]);
                    break;
                case "--max-seq-len":
                    o.MaxSeqLen = ReadInt(args, ref i, args[i]);
                    break;
                case "--max-vocab":
                    o.MaxVocab = ReadInt(args, ref i, args[i]);
                    break;
                case "--num-samples":
                    o.NumSamples = ReadInt(args, ref i, args[i]);
                    break;
                case "--lr":
                    o.LearningRate = ReadDouble(args, ref i, args[i]);
                    break;
                case "--seed":
                    o.Seed = ReadInt(args, ref i, args[i]);
                    break;
                case "--mode":
                    o.Mode = ReadString(args, ref i, args[i]);
                    break;
                case "--benchmark":
                    o.Command = "benchmark";
                    break;
                case "--silk":
                    o.Silk = true;
                    break;
                case "--help":
                case "-h":
                    o.Help = true;
                    break;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    o.Help = true;
                    break;
            }
        }
        return o;
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraClassifier — Word-level text sentiment classifier

            Usage:
              NivaraClassifier [options]

            Commands (via --command):
              train         Train model on generated/loaded data (default)
              predict       Interactive prediction with saved model
              generate      Generate synthetic training CSV
              benchmark     Measure CPU vs Silk forward/predict/epoch (Release builds only)
              onnx-compare  Compare Nivara vs ONNX Runtime on a few samples

            Options:
              --command, -c <cmd>     Command: train, predict, generate, benchmark, onnx-compare
              --data-path <path>      CSV data file (default: samples/data/sentiment_data.csv)
              --save <path>           Save trained model
              --load <path>           Load model for prediction
              --export-onnx <path>    Export trained model to ONNX
              --load-onnx <path>      Load ONNX model for prediction (uses ORT)
              --epochs <n>            Training epochs (default: 10)
              --batch-size <n>        Batch size (default: 32; benchmark sweeps 1, 8, 32, 128 unless set)
              --embedding-dim <n>     Embedding dimension (default: 32)
              --hidden-dim <n>        Hidden layer dimension (default: 64)
              --max-seq-len <n>       Max sequence length (default: 20)
              --max-vocab <n>         Max vocabulary size (default: 5000)
              --num-samples <n>       Samples to generate (default: 1000)
              --lr <rate>             Learning rate (default: 0.001)
              --seed <n>              Random seed (default: 42)
              --mode <mode>           Model mode: linear or conv (default: linear)
               --silk                  Train/predict on the Silk/OpenCL device (linear and conv)
              --help, -h              Show this help
            """);
    }

    static string ReadString(string[] args, ref int i, string key)
    {
        if (i + 1 >= args.Length) throw new ArgumentException($"Missing value for {key}");
        return args[++i];
    }

    static int ReadInt(string[] args, ref int i, string key)
    {
        string val = ReadString(args, ref i, key);
        if (!int.TryParse(val, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result))
            throw new ArgumentException($"Invalid integer for {key}: {val}");
        return result;
    }

    static double ReadDouble(string[] args, ref int i, string key)
    {
        string val = ReadString(args, ref i, key);
        if (!double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out double result))
            throw new ArgumentException($"Invalid number for {key}: {val}");
        return result;
    }
}

static class InteractiveWizard
{
    public static Options Run()
    {
        Console.WriteLine("NivaraClassifier — Text Sentiment Classifier\n");
        Console.WriteLine("1) Generate training data");
        Console.WriteLine("2) Train model");
        Console.WriteLine("3) Interactive prediction");
        Console.Write("\nChoice [1]: ");
        string? choice = Console.ReadLine()?.Trim();

        return choice switch
        {
            "2" => AskTrainOptions(),
            "3" => new Options { Command = "predict", LoadPath = AskString("Model path", "classifier.model.json") },
            _ => new Options { Command = "generate" }
        };
    }

    static Options AskTrainOptions()
    {
        int epochs = AskInt("Epochs", 10);
        int batchSize = AskInt("Batch size", 32);
        double lr = AskDouble("Learning rate", 0.001);
        int hiddenDim = AskInt("Hidden dim", 64);
        int numSamples = AskInt("Num samples", 1000);
        string mode = AskString("Model mode (linear/conv)", "linear");

        return new Options
        {
            Command = "train",
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr,
            HiddenDim = hiddenDim,
            NumSamples = numSamples,
            Mode = mode,
            SavePath = AskString("Save model path", "classifier.model.json")
        };
    }

    static string AskString(string prompt, string defaultVal)
    {
        Console.Write($"{prompt} [{defaultVal}]: ");
        string? input = Console.ReadLine()?.Trim();
        return string.IsNullOrEmpty(input) ? defaultVal : input;
    }

    static int AskInt(string prompt, int defaultVal)
    {
        Console.Write($"{prompt} [{defaultVal}]: ");
        string? input = Console.ReadLine()?.Trim();
        return string.IsNullOrEmpty(input) ? defaultVal : int.Parse(input);
    }

    static double AskDouble(string prompt, double defaultVal)
    {
        Console.Write($"{prompt} [{defaultVal}]: ");
        string? input = Console.ReadLine()?.Trim();
        return string.IsNullOrEmpty(input) ? defaultVal : double.Parse(input);
    }
}
