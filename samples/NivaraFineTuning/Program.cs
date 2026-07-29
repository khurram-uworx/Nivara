using Microsoft.ML.Tokenizers;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;

namespace NivaraFineTuning;

sealed record CliArgs
{
    public string Mode { get; init; } = "train";
    public int Epochs { get; init; } = 3;
    public float Lr { get; init; } = 2e-5f;
    public int BatchSize { get; init; } = 4;
    public int MaxLen { get; init; } = 128;
    public string DataDir { get; init; } = "";
    public string ModelDir { get; init; } = "";
    public string SavePath { get; init; } = "";

    public static CliArgs Parse(string[] args)
    {
        var dataDir = Program.FindDefaultPath("data");
        var modelDir = Program.FindDefaultPath("data/distilbert");
        var savePath = Path.Combine(modelDir, "finetuned_model.json");

        string mode = "train";
        int epochs = 3;
        float lr = 2e-5f;
        int batchSize = 4;
        int maxLen = 128;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--mode" when i + 1 < args.Length:
                    mode = args[++i];
                    break;
                case "--epochs" when i + 1 < args.Length:
                    epochs = int.Parse(args[++i]);
                    break;
                case "--lr" when i + 1 < args.Length:
                    lr = float.Parse(args[++i]);
                    break;
                case "--batch-size" when i + 1 < args.Length:
                    batchSize = int.Parse(args[++i]);
                    break;
                case "--max-len" when i + 1 < args.Length:
                    maxLen = int.Parse(args[++i]);
                    break;
                case "--data-dir" when i + 1 < args.Length:
                    dataDir = args[++i];
                    break;
                case "--model-dir" when i + 1 < args.Length:
                    modelDir = args[++i];
                    savePath = Path.Combine(modelDir, "finetuned_model.json");
                    break;
                case "--save-path" when i + 1 < args.Length:
                    savePath = args[++i];
                    break;
                case "--help":
                case "-h":
                    PrintHelp();
                    Environment.Exit(0);
                    break;
            }
        }

        return new CliArgs
        {
            Mode = mode,
            Epochs = epochs,
            Lr = lr,
            BatchSize = batchSize,
            MaxLen = maxLen,
            DataDir = dataDir,
            ModelDir = modelDir,
            SavePath = savePath
        };
    }

    static void PrintHelp()
    {
        Console.WriteLine("""
NivaraFineTuning - DistilBERT fine-tuning on GLUE SST-2

Usage:
  NivaraFineTuning --mode <train|eval|predict> [options]

Options:
  --mode          train|eval|predict (default: train)
  --epochs        Number of training epochs (default: 3)
  --lr            Learning rate (default: 2e-5)
  --batch-size    Batch size for training/eval (default: 4)
  --max-len       Maximum sequence length (default: 128)
  --data-dir      Path to SST-2 data directory
  --model-dir     Path to DistilBERT model directory
  --save-path     Path to save/load fine-tuned model
  --help          Show this help message

Examples:
  NivaraFineTuning --mode train --epochs 3 --batch-size 4
  NivaraFineTuning --mode eval --save-path ./my_model.json
  NivaraFineTuning --mode predict
""");
    }
}

static class Program
{
    static int Main(string[] args)
    {
        var cli = CliArgs.Parse(args);

        return cli.Mode switch
        {
            "train" => RunTrain(cli),
            "eval" => RunEval(cli),
            "predict" => RunPredict(cli),
            _ => throw new ArgumentException($"Unknown mode '{cli.Mode}'. Use --help for usage.")
        };
    }

    static int RunTrain(CliArgs cli)
    {
        var modelPath = cli.ModelDir;
        var dataPath = cli.DataDir;

        EnsureExists(Path.Combine(modelPath, "config.json"), "DistilBERT config.json");
        EnsureExists(Path.Combine(modelPath, "model.safetensors"), "DistilBERT model.safetensors");
        EnsureExists(Path.Combine(modelPath, "vocab.txt"), "DistilBERT vocab.txt");
        var sst2Dir = Path.Combine(dataPath, "sst2");
        EnsureExists(Path.Combine(sst2Dir, "train-00000-of-00001.parquet"), "SST-2 train.parquet");
        EnsureExists(Path.Combine(sst2Dir, "validation-00000-of-00001.parquet"), "SST-2 dev.parquet");

        Console.WriteLine($"Loading DistilBERT config from {modelPath}...");
        var distilConfig = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelPath, "config.json")));
        var bertConfig = distilConfig.ToBertConfig();
        Console.WriteLine($"  dim={distilConfig.Dim}, layers={distilConfig.NLayers}, heads={distilConfig.NHeads}");

        Console.WriteLine("Loading SafeTensors weights...");
        var tensors = SafeTensorsLoader.Read(Path.Combine(modelPath, "model.safetensors"));
        Console.WriteLine($"  Loaded {tensors.Count} tensors");

        Console.WriteLine($"Building model (numClasses=2)...");
        using var model = new DistilBertForSequenceClassification<float>(bertConfig, numClasses: 2);
        model.LoadWeights(tensors);
        model.Train();
        Console.WriteLine("  Model built, encoder weights loaded, classifier head random-initialized.");

        Console.WriteLine($"Loading tokenizer from {modelPath}/vocab.txt...");
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelPath, "vocab.txt"));

        Console.WriteLine($"Loading SST-2 dataset from {dataPath}...");
        var dataset = Sst2Dataset.LoadFromParquet(dataPath);

        Console.WriteLine("Tokenizing training set...");
        var trainTokenized = Sst2Dataset.Tokenize(tokenizer, dataset.Train, cli.MaxLen);
        Console.WriteLine($"  {trainTokenized.Count} examples, seqLen={cli.MaxLen}");

        Console.WriteLine("Tokenizing dev set...");
        var devTokenized = Sst2Dataset.Tokenize(tokenizer, dataset.Dev, cli.MaxLen);
        Console.WriteLine($"  {devTokenized.Count} examples");

        var lossFn = new CrossEntropyLoss<float>();
        using var optimizer = new AdamW<float>(cli.Lr, beta1: 0.9, beta2: 0.999, eps: 1e-8);

        var allParams = model.GetParameters().Values;
        optimizer.AddParameterGroup(allParams, learningRate: cli.Lr, weightDecay: 0.01f);
        Console.WriteLine($"  Optimizer: AdamW(lr={cli.Lr}, weightDecay=0.01)");

        for (int epoch = 1; epoch <= cli.Epochs; epoch++)
        {
            Console.WriteLine($"\n=== Epoch {epoch}/{cli.Epochs} ===");

            model.Train();
            float epochLoss = 0f;
            int batchCount = 0;

            var trainBatches = Sst2Dataset.CreateBatches(trainTokenized, cli.BatchSize, shuffle: true).ToList();
            int totalTrainBatches = trainBatches.Count;

            foreach (var batch in trainBatches)
            {
                var inputIds = GradientUtils.Constant(batch.TokenIds);
                var attnMask = GradientUtils.Constant(batch.AttentionMask);

                using (GradientUtils.Grad())
                {
                    var logits = model.Forward(inputIds, attnMask, batch.BatchSize, batch.SeqLen);
                    var loss = lossFn.Forward(logits, batch.Labels);
                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();

                    epochLoss += float.CreateChecked(loss.Data[0]);
                }

                batchCount++;
                if (batchCount % 50 == 0 || batchCount == totalTrainBatches)
                    Console.WriteLine($"  Batch {batchCount}/{totalTrainBatches} - loss: {epochLoss / batchCount:F4}");
            }

            float avgTrainLoss = epochLoss / batchCount;
            Console.WriteLine($"Epoch {epoch} training complete. Avg loss: {avgTrainLoss:F4}");

            var (devLoss, devAcc) = Evaluate(model, lossFn, tokenizer, dataset.Dev, cli.MaxLen, cli.BatchSize);
            Console.WriteLine($"Dev - loss: {devLoss:F4}, accuracy: {devAcc:F2}%");
        }

        Console.WriteLine($"\nSaving fine-tuned model to {cli.SavePath}...");
        ModelSerializer.Save(model, cli.SavePath);
        Console.WriteLine("Done.");

        return 0;
    }

    static int RunEval(CliArgs cli)
    {
        var savePath = cli.SavePath;
        var dataPath = cli.DataDir;
        var modelPath = cli.ModelDir;

        EnsureExists(savePath, "Fine-tuned model");
        var sst2Dir = Path.Combine(dataPath, "sst2");
        EnsureExists(Path.Combine(sst2Dir, "validation-00000-of-00001.parquet"), "SST-2 dev.parquet");
        EnsureExists(Path.Combine(modelPath, "vocab.txt"), "DistilBERT vocab.txt");

        Console.WriteLine($"Loading fine-tuned model from {savePath}...");
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelPath, "config.json")));
        using var model = new DistilBertForSequenceClassification<float>(config.ToBertConfig(), numClasses: 2);
        ModelSerializer.Load(model, savePath);
        model.Eval();
        Console.WriteLine("Model loaded.");

        Console.WriteLine($"Loading tokenizer from {modelPath}/vocab.txt...");
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelPath, "vocab.txt"));

        Console.WriteLine($"Loading SST-2 dev set from {dataPath}...");
        var dataset = Sst2Dataset.LoadFromParquet(dataPath);

        var lossFn = new CrossEntropyLoss<float>();
        var (loss, acc) = Evaluate(model, lossFn, tokenizer, dataset.Dev, cli.MaxLen, cli.BatchSize);
        Console.WriteLine($"\nResults - loss: {loss:F4}, accuracy: {acc:F2}%");

        return 0;
    }

    static int RunPredict(CliArgs cli)
    {
        var savePath = cli.SavePath;
        var modelPath = cli.ModelDir;

        EnsureExists(savePath, "Fine-tuned model");
        EnsureExists(Path.Combine(modelPath, "vocab.txt"), "DistilBERT vocab.txt");

        Console.WriteLine($"Loading fine-tuned model from {savePath}...");
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelPath, "config.json")));
        using var model = new DistilBertForSequenceClassification<float>(config.ToBertConfig(), numClasses: 2);
        ModelSerializer.Load(model, savePath);
        model.Eval();

        Console.WriteLine($"Loading tokenizer from {modelPath}/vocab.txt...");
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelPath, "vocab.txt"));

        Console.WriteLine("\n=== Interactive Sentiment Predictor ===");
        Console.WriteLine("Type a sentence and press Enter (or 'quit' to exit).\n");

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;

            var (tokenIds, attnMask, seqLen) = MiniLMTokenizer.Encode(tokenizer, line, cli.MaxLen);

            var inputIds = GradientUtils.Constant(tokenIds);
            var mask = GradientUtils.Constant(attnMask);

            var logits = model.Forward(inputIds, mask, 1, seqLen);
            var posConf = float.CreateChecked(logits.Data[1]);
            var negConf = float.CreateChecked(logits.Data[0]);
            var total = posConf + negConf;
            var sentiment = posConf > negConf ? "POSITIVE" : "NEGATIVE";
            var confidence = Math.Max(posConf, negConf) / total * 100;

            Console.WriteLine($"  Sentiment: {sentiment} ({confidence:F1}%)");
        }

        return 0;
    }

    static (float Loss, float Accuracy) Evaluate(
        DistilBertForSequenceClassification<float> model,
        CrossEntropyLoss<float> lossFn,
        BertTokenizer tokenizer,
        List<Sst2Example> devSet,
        int maxLen,
        int batchSize)
    {
        model.Eval();
        var tokenized = Sst2Dataset.Tokenize(tokenizer, devSet, maxLen);
        var batches = Sst2Dataset.CreateBatches(tokenized, batchSize, shuffle: false).ToList();

        float totalLoss = 0;
        int correct = 0;
        int total = 0;

        foreach (var batch in batches)
        {
            var inputIds = GradientUtils.Constant(batch.TokenIds);
            var attnMask = GradientUtils.Constant(batch.AttentionMask);

            var logits = model.Forward(inputIds, attnMask, batch.BatchSize, batch.SeqLen);
            var loss = lossFn.Forward(logits, batch.Labels);

            totalLoss += float.CreateChecked(loss.Data[0]);

            for (int i = 0; i < batch.BatchSize; i++)
            {
                int predicted = ArgMax(logits, i, model.NumClasses);
                if (predicted == batch.Labels[i])
                    correct++;
            }
            total += batch.BatchSize;
        }

        return (totalLoss / batches.Count, (float)correct / total * 100);
    }

    static int ArgMax(ReverseGradTensor<float> logits, int row, int numClasses)
    {
        int best = 0;
        float bestVal = logits.Data[row * numClasses];
        for (int c = 1; c < numClasses; c++)
        {
            float val = logits.Data[row * numClasses + c];
            if (val > bestVal) { bestVal = val; best = c; }
        }
        return best;
    }

    internal static string FindDefaultPath(string name)
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 5; i++)
        {
            dir = Path.GetDirectoryName(dir);
            if (dir == null) break;
            var candidate = Path.Combine(dir, "samples", name);
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "samples", name);
    }

    static void EnsureExists(string path, string label)
    {
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            Console.Error.WriteLine($"ERROR: {label} not found at '{path}'.");
            Console.Error.WriteLine("Run the Python download scripts first to obtain required data.");
            Environment.Exit(1);
        }
    }
}
