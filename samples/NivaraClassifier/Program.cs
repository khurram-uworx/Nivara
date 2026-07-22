using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
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

if (options.Command == "train")
    RunTraining(options);
else if (options.Command == "predict")
    RunPrediction(options);
else if (options.Command == "generate")
    RunGenerate(options);
else
    Options.PrintHelp();

return;

// ── Commands ────────────────────────────────────────────────────────────────

void RunGenerate(Options opt)
{
    string csvPath = opt.DataPath ?? "sentiment_data.csv";
    DataGenerator.SaveCsv(csvPath, opt.NumSamples, opt.Seed);
    Console.WriteLine($"Generated {opt.NumSamples} samples → {csvPath}");
}

void RunTraining(Options opt)
{
    string csvPath = opt.DataPath ?? "sentiment_data.csv";
    if (!File.Exists(csvPath))
    {
        Console.WriteLine($"Data file not found: {csvPath}");
        Console.WriteLine("Run with --command generate first, or specify --data-path.");
        return;
    }

    var (allTexts, allLabels) = DataGenerator.LoadCsv(csvPath);

    var tokenizer = TextTokenizer.FromDocuments(allTexts, maxVocabSize: opt.MaxVocab);
    Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");

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

    using var model = new TextClassifierModel<double>(
        tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);

    using var optimizer = new Adam<double>(T.CreateChecked(opt.LearningRate));
    var parameters = model.GetParameters();
    optimizer.AddParameterGroup(parameters);

    var lossFn = new CrossEntropyLoss<double>();

    var trainFrame = BuildFrame(trainTokens, trainLabels, trainCount, opt.MaxSeqLen);
    var trainDataset = new TensorDataset<double>(trainFrame, ["tokens"], ["label"]);
    var trainLoader = new DataLoader<double>(trainDataset, opt.BatchSize, shuffle: true, seed: opt.Seed);

    Console.WriteLine($"\nTraining: {trainCount} samples, {testLabels.Length} test");
    Console.WriteLine($"Model: {tokenizer.VocabSize} vocab → {opt.EmbeddingDim} embed → {opt.HiddenDim} hidden → 2 classes\n");

    var trainLoop = new TrainingLoop<double>(
        model, trainLoader,
        (logits, labels) => lossFn.Forward(logits, labels),
        optimizer, opt.Epochs);

    var result = trainLoop.Run();
    result.PrintSummary();

    int correct = 0;
    for (int i = 0; i < testLabels.Length; i++)
    {
        var tokenCol = NivaraColumn<double>.Create(testTokens.AsSpan(i * opt.MaxSeqLen, opt.MaxSeqLen));
        var preds = model.Predict(tokenCol, 1);
        if (preds[0] == testLabels[i]) correct++;
    }

    double accuracy = (double)correct / testLabels.Length;
    Console.WriteLine($"\nTest accuracy: {accuracy:P1} ({correct}/{testLabels.Length})");

    if (!string.IsNullOrWhiteSpace(opt.SavePath))
    {
        ModelSerializer.Save(model, opt.SavePath);
        Console.WriteLine($"Saved model: {opt.SavePath}");
    }

    tokenizer.Save(Path.ChangeExtension(opt.SavePath ?? "classifier", ".vocab.json"));
    Console.WriteLine("Done.");
}

void RunPrediction(Options opt)
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
    using var model = new TextClassifierModel<double>(
        tokenizer.VocabSize, opt.EmbeddingDim, opt.HiddenDim, numClasses: 2, opt.MaxSeqLen);
    ModelSerializer.Load(model, modelPath);

    Console.WriteLine("Nivara Text Classifier — Interactive Prediction");
    Console.WriteLine("Type a sentence (empty line to quit):\n");

    while (true)
    {
        Console.Write("> ");
        string? line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) break;

        var encoded = tokenizer.Encode(line, fixedLength: opt.MaxSeqLen);
        var tokenCol = NivaraColumn<double>.Create(encoded);
        var preds = model.Predict(tokenCol, 1);
        string label = preds[0] == 1 ? "POSITIVE" : "NEGATIVE";
        Console.WriteLine($"  → {label}\n");
    }
}

NivaraFrame BuildFrame(int[] tokens, int[] labels, int count, int seqLen)
{
    var tokenCols = new NivaraColumn<int>[seqLen];
    for (int d = 0; d < seqLen; d++)
    {
        var colData = new int[count];
        for (int i = 0; i < count; i++)
            colData[i] = tokens[i * seqLen + d];
        tokenCols[d] = NivaraColumn<int>.Create(colData);
    }

    var intLabels = NivaraColumn<int>.Create(labels);
    var doubleLabels = new NivaraColumn<double>[1];
    var labelData = new double[count];
    for (int i = 0; i < count; i++)
        labelData[i] = labels[i];
    doubleLabels[0] = NivaraColumn<double>.Create(labelData);

    var columns = new Dictionary<string, NivaraColumnBase>
    {
        ["label"] = doubleLabels[0]
    };
    for (int d = 0; d < seqLen; d++)
        columns[$"tok_{d}"] = tokenCols[d];

    return new NivaraFrame(columns);
}

// ── Types ───────────────────────────────────────────────────────────────────

sealed class Options
{
    public string Command { get; init; } = "train";
    public string? DataPath { get; init; }
    public string? SavePath { get; init; }
    public string? LoadPath { get; init; }
    public int Epochs { get; init; } = 10;
    public int BatchSize { get; init; } = 32;
    public int EmbeddingDim { get; init; } = 32;
    public int HiddenDim { get; init; } = 64;
    public int MaxSeqLen { get; init; } = 20;
    public int MaxVocab { get; init; } = 5000;
    public int NumSamples { get; init; } = 1000;
    public double LearningRate { get; init; } = 0.001;
    public int Seed { get; init; } = 42;
    public bool Help { get; init; }

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i].ToLowerInvariant();
            switch (arg)
            {
                case "--command":
                case "-c":
                    o = o with { Command = ReadString(args, ref i, arg) };
                    break;
                case "--data-path":
                    o = o with { DataPath = ReadString(args, ref i, arg) };
                    break;
                case "--save":
                    o = o with { SavePath = ReadString(args, ref i, arg) };
                    break;
                case "--load":
                    o = o with { LoadPath = ReadString(args, ref i, arg) };
                    break;
                case "--epochs":
                    o = o with { Epochs = ReadInt(args, ref i, arg) };
                    break;
                case "--batch-size":
                    o = o with { BatchSize = ReadInt(args, ref i, arg) };
                    break;
                case "--embedding-dim":
                    o = o with { EmbeddingDim = ReadInt(args, ref i, arg) };
                    break;
                case "--hidden-dim":
                    o = o with { HiddenDim = ReadInt(args, ref i, arg) };
                    break;
                case "--max-seq-len":
                    o = o with { MaxSeqLen = ReadInt(args, ref i, arg) };
                    break;
                case "--max-vocab":
                    o = o with { MaxVocab = ReadInt(args, ref i, arg) };
                    break;
                case "--num-samples":
                    o = o with { NumSamples = ReadInt(args, ref i, arg) };
                    break;
                case "--lr":
                    o = o with { LearningRate = ReadDouble(args, ref i, arg) };
                    break;
                case "--seed":
                    o = o with { Seed = ReadInt(args, ref i, arg) };
                    break;
                case "--help":
                case "-h":
                    o = o with { Help = true };
                    break;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    o = o with { Help = true };
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
              train      Train model on generated/loaded data (default)
              predict    Interactive prediction with saved model
              generate   Generate synthetic training CSV

            Options:
              --command, -c <cmd>     Command: train, predict, generate
              --data-path <path>      CSV data file (default: sentiment_data.csv)
              --save <path>           Save trained model
              --load <path>           Load model for prediction
              --epochs <n>            Training epochs (default: 10)
              --batch-size <n>        Batch size (default: 32)
              --embedding-dim <n>     Embedding dimension (default: 32)
              --hidden-dim <n>        Hidden layer dimension (default: 64)
              --max-seq-len <n>       Max sequence length (default: 20)
              --max-vocab <n>         Max vocabulary size (default: 5000)
              --num-samples <n>       Samples to generate (default: 1000)
              --lr <rate>             Learning rate (default: 0.001)
              --seed <n>              Random seed (default: 42)
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

        return new Options
        {
            Command = "train",
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr,
            HiddenDim = hiddenDim,
            NumSamples = numSamples,
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
