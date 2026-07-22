using Nivara;
using Nivara.AutoDiff;
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

    using var optimizer = new Adam<double>(learningRate: 0.001);
    var allParams = model.GetParameters().Values;
    optimizer.AddParameterGroup(allParams);

    var lossFn = new CrossEntropyLoss<double>();

    var trainFrame = BuildFrame(trainTokens, trainLabels, trainCount, opt.MaxSeqLen);
    var featureColumns = Enumerable.Range(0, opt.MaxSeqLen).Select(d => $"tok_{d}").ToArray();
    var trainDataset = new TensorDataset<double>(trainFrame, featureColumns, ["label"]);
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
        var encoded = tokenizer.Encode(testTexts[i], fixedLength: opt.MaxSeqLen);
        var preds = model.Predict(encoded);
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
        var preds = model.Predict(encoded);
        string label = preds[0] == 1 ? "POSITIVE" : "NEGATIVE";
        Console.WriteLine($"  → {label}\n");
    }
}

NivaraFrame BuildFrame(int[] tokens, int[] labels, int count, int seqLen)
{
    var columns = new List<(string Name, IColumn Column)>();

    for (int d = 0; d < seqLen; d++)
    {
        var colData = new double[count];
        for (int i = 0; i < count; i++)
            colData[i] = tokens[i * seqLen + d];
        columns.Add(($"tok_{d}", NivaraColumn<double>.Create(colData)));
    }

    var labelData = new double[count];
    for (int i = 0; i < count; i++)
        labelData[i] = labels[i];
    columns.Add(("label", NivaraColumn<double>.Create(labelData)));

    return NivaraFrame.Create(columns.ToArray());
}

// ── Types ───────────────────────────────────────────────────────────────────

sealed class Options
{
    public string Command { get; set; } = "train";
    public string? DataPath { get; set; }
    public string? SavePath { get; set; }
    public string? LoadPath { get; set; }
    public int Epochs { get; set; } = 10;
    public int BatchSize { get; set; } = 32;
    public int EmbeddingDim { get; set; } = 32;
    public int HiddenDim { get; set; } = 64;
    public int MaxSeqLen { get; set; } = 20;
    public int MaxVocab { get; set; } = 5000;
    public int NumSamples { get; set; } = 1000;
    public double LearningRate { get; set; } = 0.001;
    public int Seed { get; set; } = 42;
    public bool Help { get; set; }

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
                case "--epochs":
                    o.Epochs = ReadInt(args, ref i, args[i]);
                    break;
                case "--batch-size":
                    o.BatchSize = ReadInt(args, ref i, args[i]);
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
