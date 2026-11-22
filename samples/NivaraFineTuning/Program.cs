using Microsoft.ML.Tokenizers;
using Nivara;
using Nivara.AI;
using System.Globalization;
using System.Text.Json;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using Nivara.Metrics;
using Nivara.Samples;
using Nivara.Silk;
using System.Numerics;

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
    public int MaxExamples { get; init; }
    public int TrainableLayers { get; init; } = -1;
    public bool Silk { get; init; }
    public string Precision { get; init; } = "f32";

    // LoRA SFT options
    public string TrainData { get; init; } = "";
    public string ValData { get; init; } = "";
    public string TestData { get; init; } = "";
    public string OutDir { get; init; } = "";
    public string AdapterPath { get; init; } = "";
    public string Template { get; init; } = "qwen";
    public int Rank { get; init; } = 8;
    public float Alpha { get; init; } = 16f;
    public string Targets { get; init; } = "q,v";
    public int Seed { get; init; } = 42;
    public int LogEvery { get; init; } = 10;
    public int Patience { get; init; } = 0;
    public float MinDelta { get; init; } = 0f;
    public int GradAccum { get; init; } = 1;
    public bool Force { get; init; }
    public int MaxNewTokens { get; init; } = 128;
    public double ValRatio { get; init; } = 0.0;
    public string BaseDate { get; init; } = "2026-07-19";

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
        int maxExamples = 0;
        int trainableLayers = -1;
        bool silk = false;
        string precision = "f32";
        string trainData = "";
        string valData = "";
        string testData = "";
        string outDir = "";
        string adapterPath = "";
        string template = "qwen";
        int rank = 8;
        float alpha = 16f;
        string targets = "q,v";
        int seed = 42;
        int logEvery = 10;
        int patience = 0;
        float minDelta = 0f;
        int gradAccum = 1;
        bool force = false;
        int maxNewTokens = 128;
        double valRatio = 0.0;
        string baseDate = "2026-07-19";

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
                case "--max-examples" when i + 1 < args.Length:
                    maxExamples = int.Parse(args[++i]);
                    break;
                case "--trainable-layers" when i + 1 < args.Length:
                    trainableLayers = int.Parse(args[++i]);
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
                case "--silk":
                    silk = true;
                    break;
                case "--precision" when i + 1 < args.Length:
                    precision = args[++i].ToLowerInvariant();
                    break;
                case "--train-data" when i + 1 < args.Length:
                    trainData = args[++i];
                    break;
                case "--val-data" when i + 1 < args.Length:
                    valData = args[++i];
                    break;
                case "--test-data" when i + 1 < args.Length:
                    testData = args[++i];
                    break;
                case "--out-dir" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
                case "--adapter" when i + 1 < args.Length:
                    adapterPath = args[++i];
                    break;
                case "--template" when i + 1 < args.Length:
                    template = args[++i].ToLowerInvariant();
                    break;
                case "--rank" when i + 1 < args.Length:
                    rank = int.Parse(args[++i]);
                    break;
                case "--alpha" when i + 1 < args.Length:
                    alpha = float.Parse(args[++i]);
                    break;
                case "--targets" when i + 1 < args.Length:
                    targets = args[++i].ToLowerInvariant();
                    break;
                case "--seed" when i + 1 < args.Length:
                    seed = int.Parse(args[++i]);
                    break;
                case "--log-every" when i + 1 < args.Length:
                    logEvery = int.Parse(args[++i]);
                    break;
                case "--patience" when i + 1 < args.Length:
                    patience = int.Parse(args[++i]);
                    break;
                case "--min-delta" when i + 1 < args.Length:
                    minDelta = float.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--grad-accum" when i + 1 < args.Length:
                    gradAccum = int.Parse(args[++i]);
                    break;
                case "--force":
                    force = true;
                    break;
                case "--max-new-tokens" when i + 1 < args.Length:
                    maxNewTokens = int.Parse(args[++i]);
                    break;
                case "--val-ratio" when i + 1 < args.Length:
                    valRatio = double.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--base-date" when i + 1 < args.Length:
                    baseDate = args[++i];
                    break;
                case "--help":
                case "-h":
                    PrintHelp();
                    Environment.Exit(0);
                    break;
            }
        }

        if (precision is not ("f32" or "bf16"))
        {
            Console.Error.WriteLine($"--precision expects f32 or bf16; got '{precision}'.");
            Environment.Exit(1);
        }

        return new CliArgs
        {
            Mode = mode,
            Epochs = epochs,
            Lr = lr,
            BatchSize = batchSize,
            MaxLen = maxLen,
            MaxExamples = maxExamples,
            TrainableLayers = trainableLayers,
            DataDir = dataDir,
            ModelDir = modelDir,
            SavePath = savePath,
            Silk = silk,
            Precision = precision,
            TrainData = trainData,
            ValData = valData,
            TestData = testData,
            OutDir = outDir,
            AdapterPath = adapterPath,
            Template = template,
            Rank = rank,
            Alpha = alpha,
            Targets = targets,
            Seed = seed,
            LogEvery = logEvery,
            Patience = patience,
            MinDelta = minDelta,
            GradAccum = gradAccum,
            Force = force,
            MaxNewTokens = maxNewTokens,
            ValRatio = valRatio,
            BaseDate = baseDate
        };
    }

    static void PrintHelp()
    {
        Console.WriteLine("""
NivaraFineTuning - DistilBERT fine-tuning and Llama-family LoRA SFT showcase

Usage:
  NivaraFineTuning --mode <train|eval|predict|ft-lora|ft-eval|ft-generate> [options]

DistilBERT options:
  --mode          train|eval|predict|ft-lora|ft-eval|ft-generate (default: train)
  --epochs        Number of training epochs (default: 3)
  --lr            Learning rate
  --batch-size    Batch size for training/eval (default: 4)
  --max-len       Maximum sequence length (default: 128)
  --max-examples  Limit examples (default: 0 = all)
  --trainable-layers  Top transformer layers to train (-1 all)
  --data-dir      Path to SST-2 data directory
  --model-dir     Path to model directory
  --save-path     Path to save/load fine-tuned model
  --silk          Use OpenCL/Silk path when supported
  --precision     f32 (default) or bf16

LoRA SFT options:
  --train-data    JSONL training data path
  --val-data      JSONL validation data path
  --test-data     JSONL test data path
  --out-dir       Output directory for adapters/reports
  --adapter       Adapter file path
  --template      qwen|smollm (default: qwen)
  --rank          LoRA rank (default: 8)
  --alpha         LoRA alpha (default: 16)
  --targets       q,v,k,o,ffn comma-separated (default: q,v)
  --seed          Random seed (default: 42)
  --log-every     Training log interval (default: 10)
  --patience      Early stopping patience in epochs (default: 0 = disabled)
  --min-delta     Minimum val-loss improvement to reset patience (default: 0)
  --grad-accum    Gradient accumulation steps (default: 1)
  --force         Overwrite outputs
  --max-new-tokens  Max tokens to generate during ft-eval (default: 128)
  --val-ratio     Validation split ratio for ft-generate (default: 0.0)
  --base-date     Base/today date for ft-generate relative dates (YYYY-MM-DD, default: 2026-07-19)

Examples:
  NivaraFineTuning --mode train --epochs 3 --batch-size 4
  NivaraFineTuning --mode ft-generate --out-dir samples/data/worklog --seed 42 --max-examples 350 --force
  NivaraFineTuning --mode ft-lora --model-dir samples/data/qwen2.5-0.5b-instruct --train-data samples/data/worklog/train.jsonl --val-data samples/data/worklog/test.jsonl --template qwen --epochs 1 --max-examples 32 --max-len 256 --out-dir out/worklog-lora
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
            "train" => cli.Silk ? RunTrainSilk(cli) : RunTrain(cli),
            "eval" => RunEval(cli),
            "predict" => RunPredict(cli),
            "ft-lora" => RunFtLora(cli),
            "ft-eval" => RunFtEval(cli),
            "ft-generate" => RunFtGenerate(cli),
            _ => throw new ArgumentException($"Unknown mode '{cli.Mode}'. Use --help for usage.")
        };
    }

    static int RunTrain(CliArgs cli)
        => cli.Precision == "bf16"
            ? RunTrainCore<float>(cli, loadBf16Weights: true)
            : RunTrainCore<float>(cli);

    static int RunTrainCore<T>(CliArgs cli, bool loadBf16Weights = false) where T : struct, System.Numerics.IFloatingPointIeee754<T>
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
        Dictionary<string, (T[] Data, int[] Shape)> tensors;
        if (loadBf16Weights)
        {
            var bf16Tensors = SafeTensorsLoader.Read<BFloat16>(Path.Combine(modelPath, "model.safetensors"));
            tensors = new Dictionary<string, (T[] Data, int[] Shape)>(bf16Tensors.Count);
            foreach (var kv in bf16Tensors)
            {
                var data = new T[kv.Value.Data.Length];
                for (int i = 0; i < data.Length; i++) data[i] = T.CreateChecked(kv.Value.Data[i]);
                tensors[kv.Key] = (data, kv.Value.Shape);
            }
            Console.WriteLine($"  Loaded {bf16Tensors.Count} tensors (BF16, widened to {typeof(T).Name} internally)");
        }
        else
        {
            tensors = SafeTensorsLoader.Read<T>(Path.Combine(modelPath, "model.safetensors"));
            Console.WriteLine($"  Loaded {tensors.Count} tensors");
        }

        Console.WriteLine($"Building model (numClasses=2)...");
        using var model = new DistilBertForSequenceClassification<T>(bertConfig, numClasses: 2);
        model.LoadWeights(tensors);
        model.Train();
        Console.WriteLine("  Model built, encoder weights loaded, classifier head random-initialized.");

        Console.WriteLine($"Loading tokenizer from {modelPath}/vocab.txt...");
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelPath, "vocab.txt"));

        Console.WriteLine($"Loading SST-2 dataset from {dataPath}...");
        var dataset = Sst2Dataset.LoadFromParquet(dataPath);

        Console.WriteLine("Tokenizing training set...");
        var trainData = dataset.Train;
        if (cli.MaxExamples > 0 && cli.MaxExamples < trainData.Count)
        {
            trainData = trainData.Take(cli.MaxExamples).ToList();
            Console.WriteLine($"  (limited to {cli.MaxExamples} examples by --max-examples)");
        }
        var trainTokenized = Sst2Dataset.Tokenize(tokenizer, trainData, cli.MaxLen);
        Console.WriteLine($"  {trainTokenized.Count} examples, seqLen={cli.MaxLen}");

        Console.WriteLine("Tokenizing dev set...");
        var devTokenized = Sst2Dataset.Tokenize(tokenizer, dataset.Dev, cli.MaxLen);
        Console.WriteLine($"  {devTokenized.Count} examples");

        var lossFn = new CrossEntropyLoss<T>();
        using var optimizer = new AdamW<T>(T.CreateChecked(cli.Lr), beta1: 0.9, beta2: 0.999, eps: 1e-8);

        var allParams = model.GetParameters().Values;
        optimizer.AddParameterGroup(allParams, learningRate: T.CreateChecked(cli.Lr), weightDecay: T.CreateChecked(0.01f));
        Console.WriteLine($"  Optimizer: AdamW(lr={cli.Lr}, weightDecay=0.01)");

        for (int epoch = 1; epoch <= cli.Epochs; epoch++)
        {
            Console.WriteLine($"\n=== Epoch {epoch}/{cli.Epochs} ===");

            model.Train();
            float epochLoss = 0f;
            int batchCount = 0;
            var batchSw = System.Diagnostics.Stopwatch.StartNew();
            var epochBatchElapsed = new List<TimeSpan>();

            var trainBatches = Sst2Dataset.CreateBatches(trainTokenized, cli.BatchSize, shuffle: true).ToList();
            int totalTrainBatches = trainBatches.Count;

            foreach (var batch in trainBatches)
            {
                var intIds = Array.ConvertAll(batch.TokenIds, x => (int)x);
                var attnMask = GradientUtils.Constant(Array.ConvertAll(batch.AttentionMask, x => T.CreateChecked(x)));

                using (GradientUtils.Grad())
                {
                    var logits = model.Forward(intIds, attnMask, batch.BatchSize, batch.SeqLen);
                    var loss = lossFn.Forward(logits, batch.Labels);
                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();

                    epochLoss += float.CreateChecked(loss.Data[0]);
                }

                batchCount++;
                epochBatchElapsed.Add(batchSw.Elapsed);
                Console.WriteLine($"  Batch {batchCount}/{totalTrainBatches} [{batchSw.Elapsed.TotalSeconds:F1}s] - loss: {epochLoss / batchCount:F4}");
                batchSw.Restart();
            }

            float avgTrainLoss = epochLoss / batchCount;
            Console.WriteLine($"Epoch {epoch} training complete. Avg loss: {avgTrainLoss:F4}");
            if (epochBatchElapsed.Count > 1)
            {
                double steadySumMs = 0;
                for (int i = 1; i < epochBatchElapsed.Count; i++)
                    steadySumMs += epochBatchElapsed[i].TotalMilliseconds;
                double steadyMs = steadySumMs / (epochBatchElapsed.Count - 1);
                Console.WriteLine($"  Steady-state ms/batch: {steadyMs:F1}");
            }

            var (devLoss, devAcc) = Evaluate<T>(model, lossFn, tokenizer, dataset.Dev, cli.MaxLen, cli.BatchSize);
            Console.WriteLine($"Dev - loss: {devLoss:F4}, accuracy: {devAcc:F2}%");
        }

        Console.WriteLine($"\nSaving fine-tuned model to {cli.SavePath}...");
        ModelSerializer.Save(model, cli.SavePath);
        Console.WriteLine("Done.");

        return 0;
    }

    static int RunTrainSilk(CliArgs cli)
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
        var f32Tensors = SafeTensorsLoader.Read(Path.Combine(modelPath, "model.safetensors"));
        Console.WriteLine($"  Loaded {f32Tensors.Count} tensors");

        Console.WriteLine("Building CPU model (numClasses=2)...");
        using var model = new DistilBertForSequenceClassification<float>(bertConfig, numClasses: 2);
        model.LoadWeights(f32Tensors);
        model.Train();
        Console.WriteLine("  Model built, encoder weights loaded, classifier head random-initialized.");

        Console.WriteLine("Initializing OpenCL device (Silk)...");
        SilkRuntime runtime;
        try
        {
            runtime = new SilkRuntime();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"OpenCL device unavailable: {ex.Message}");
            return 1;
        }
        using (runtime)
        {
            int maxBatch = Math.Min(cli.BatchSize, 8);
            int maxSeqLen = Math.Min(cli.MaxLen, 128);
            if (maxBatch != cli.BatchSize)
                Console.WriteLine($"  (batch size clamped to device cap {maxBatch})");
            if (maxSeqLen != cli.MaxLen)
                Console.WriteLine($"  (seq len clamped to device cap {maxSeqLen})");

            using var trainer = cli.Precision == "bf16"
                ? SilkDistilBertTrainer.CreateTrueBf16(
                    runtime, bertConfig,
                    ToBFloat16Tensors(model.ToSilkTensors()),
                    BertGpuNaming.DistilBert, maxBatch, maxSeqLen,
                    trainableLayerCount: cli.TrainableLayers)
                : new SilkDistilBertTrainer(
                    runtime, bertConfig, model.ToSilkTensors(), BertGpuNaming.DistilBert,
                    maxBatch, maxSeqLen, trainableLayerCount: cli.TrainableLayers);
            using var runner = trainer.CreateInferenceRunner();
            Console.WriteLine($"  Device: {trainer.DeviceName}, classes={trainer.NumClasses}");
            if (cli.TrainableLayers >= 0)
                Console.WriteLine($"  Trainable layers: {Math.Clamp(cli.TrainableLayers, 0, bertConfig.NumHiddenLayers)} (head always trainable)");

            Console.WriteLine($"Loading tokenizer from {modelPath}/vocab.txt...");
            var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelPath, "vocab.txt"));

            Console.WriteLine($"Loading SST-2 dataset from {dataPath}...");
            var dataset = Sst2Dataset.LoadFromParquet(dataPath);

            Console.WriteLine("Tokenizing training set...");
            var trainData = dataset.Train;
            if (cli.MaxExamples > 0 && cli.MaxExamples < trainData.Count)
            {
                trainData = trainData.Take(cli.MaxExamples).ToList();
                Console.WriteLine($"  (limited to {cli.MaxExamples} examples by --max-examples)");
            }
            var trainTokenized = Sst2Dataset.Tokenize(tokenizer, trainData, maxSeqLen);
            Console.WriteLine($"  {trainTokenized.Count} examples, seqLen={maxSeqLen}");

            Console.WriteLine("Tokenizing dev set...");
            var devTokenized = Sst2Dataset.Tokenize(tokenizer, dataset.Dev, maxSeqLen);
            Console.WriteLine($"  {devTokenized.Count} examples");

            int sampleCount = trainTokenized.Count;
            int seqLen = maxSeqLen;
            int[] allTokens = new int[sampleCount * maxSeqLen];
            float[] allMask = new float[sampleCount * maxSeqLen];
            int[] allLabels = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                for (int j = 0; j < maxSeqLen; j++)
                {
                    allTokens[i * maxSeqLen + j] = (int)trainTokenized[i].TokenIds[j];
                    allMask[i * maxSeqLen + j] = trainTokenized[i].AttentionMask[j];
                }
                allLabels[i] = trainTokenized[i].Label;
            }

            int currentEpoch = 0;
            float epochLossSum = 0f;
            int epochBatchCount = 0;
            var epochBatchElapsed = new List<TimeSpan>();

            trainer.Train(allTokens, allMask, allLabels, maxBatch, seqLen, cli.Epochs, cli.Lr, 0.01f, seed: 1,
                onBatchCompleted: (e, b, total, loss, elapsed) =>
                {
                    if (e != currentEpoch)
                    {
                        currentEpoch = e;
                        epochLossSum = 0f;
                        epochBatchCount = 0;
                        epochBatchElapsed.Clear();
                        Console.WriteLine($"\n=== Epoch {e}/{cli.Epochs} ===");
                    }
                    epochLossSum += loss;
                    epochBatchCount++;
                    float runningAvg = epochLossSum / epochBatchCount;
                    epochBatchElapsed.Add(elapsed);
                    Console.WriteLine($"  Batch {b}/{total} [{elapsed.TotalSeconds:F1}s] - loss: {runningAvg:F4}");
                },
                onEpochCompleted: (e, total, avgLoss, elapsed) =>
                {
                    Console.WriteLine($"Epoch {e} training complete. Avg loss: {avgLoss:F4}");
                    if (epochBatchElapsed.Count > 1)
                    {
                        double steadySumMs = 0;
                        for (int i = 1; i < epochBatchElapsed.Count; i++)
                            steadySumMs += epochBatchElapsed[i].TotalMilliseconds;
                        double steadyMs = steadySumMs / (epochBatchElapsed.Count - 1);
                        Console.WriteLine($"  Steady-state ms/batch: {steadyMs:F1}");
                    }

                    // Dev evaluation on the device via the inference runner.
                    var (devLoss, devAcc) = EvaluateSilk(runner, tokenizer, dataset.Dev, maxSeqLen, cli.BatchSize);
                    Console.WriteLine($"Dev - loss: {devLoss:F4}, accuracy: {devAcc:F2}%");
                });

            Console.WriteLine($"\nSaving fine-tuned model to {cli.SavePath}...");
            var finalExported = trainer.ExportCheckpoint();
            model.LoadWeights(finalExported);
            ModelSerializer.Save(model, cli.SavePath);
            Console.WriteLine("Done.");

            return 0;
        }
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

        if (cli.Silk)
        {
            try
            {
                return RunEvalSilk(cli, config, model, tokenizer, dataset);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"OpenCL device unavailable: {ex.Message}");
                return 1;
            }
        }

        var lossFn = new CrossEntropyLoss<float>();
        var (loss, acc) = Evaluate(model, lossFn, tokenizer, dataset.Dev, cli.MaxLen, cli.BatchSize);
        Console.WriteLine($"\nResults - loss: {loss:F4}, accuracy: {acc:F2}%");

        return 0;
    }

    static int RunEvalSilk(CliArgs cli, DistilBertConfig config, DistilBertForSequenceClassification<float> model,
        BertTokenizer tokenizer, Sst2Dataset dataset)
    {
        var silkTensors = model.ToSilkTensors();
        var bertConfig = config.ToBertConfig();

        using var runtime = new SilkRuntime();
        using var runner = new SilkBertEncoderRunner(runtime, bertConfig, silkTensors, BertGpuNaming.DistilBert);

        Console.WriteLine($"Silk device: {runner.DeviceName}");

        var tokenized = Sst2Dataset.Tokenize(tokenizer, dataset.Dev, cli.MaxLen);
        var batches = Sst2Dataset.CreateBatches(tokenized, cli.BatchSize, shuffle: false).ToList();

        float totalLoss = 0f;
        int total = 0;
        var predictedLabels = new List<int>();
        var actualLabels = new List<int>();

        foreach (var batch in batches)
        {
            var intIds = Array.ConvertAll(batch.TokenIds, x => (int)x);
            var result = runner.Forward(intIds, batch.AttentionMask, batch.BatchSize, batch.SeqLen);
            var logits = result.Logits ?? Array.Empty<float>();

            // accuracy
            for (int i = 0; i < batch.BatchSize; i++)
            {
                int predicted = logits[i * 2 + 1] > logits[i * 2] ? 1 : 0;
                predictedLabels.Add(predicted);
                actualLabels.Add(batch.Labels[i]);
            }
            total += batch.BatchSize;

            // mean cross-entropy per batch
            float batchLoss = 0f;
            for (int i = 0; i < batch.BatchSize; i++)
            {
                float l0 = logits[i * 2];
                float l1 = logits[i * 2 + 1];
                float max = MathF.Max(l0, l1);
                float e0 = MathF.Exp(l0 - max);
                float e1 = MathF.Exp(l1 - max);
                float sum = e0 + e1;
                int label = batch.Labels[i];
                float prob = (label == 0 ? e0 : e1) / sum;
                batchLoss += -MathF.Log(prob);
            }
            batchLoss /= batch.BatchSize;
            totalLoss += batchLoss;
        }

        long correct = 0;
        if (predictedLabels.Count > 0)
        {
            var accuracyMatrix = NivaraMetrics.ConfusionMatrixFrame(
                NivaraColumn<int>.Create(predictedLabels.ToArray()),
                NivaraColumn<int>.Create(actualLabels.ToArray()));
            for (int r = 0; r < accuracyMatrix.RowCount; r++)
                correct += accuracyMatrix.GetColumn<long>("Correct")[r];
        }

        float loss = totalLoss / batches.Count;
        float acc = total > 0 ? (float)correct / total * 100f : 0f;
        Console.WriteLine($"\nResults (Silk) - loss: {loss:F4}, accuracy: {acc:F2}%");
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

        if (cli.Silk)
        {
            try
            {
                return RunPredictSilk(cli, config, model, tokenizer);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"OpenCL device unavailable: {ex.Message}");
                return 1;
            }
        }

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

    static int RunPredictSilk(CliArgs cli, DistilBertConfig config, DistilBertForSequenceClassification<float> model,
        BertTokenizer tokenizer)
    {
        var silkTensors = model.ToSilkTensors();
        var bertConfig = config.ToBertConfig();

        using var runtime = new SilkRuntime();
        using var runner = new SilkBertEncoderRunner(runtime, bertConfig, silkTensors, BertGpuNaming.DistilBert);

        Console.WriteLine($"Silk device: {runner.DeviceName}");
        Console.WriteLine("\n=== Interactive Sentiment Predictor (Silk) ===");
        Console.WriteLine("Type a sentence and press Enter (or 'quit' to exit).\n");

        int maxLen = cli.MaxLen;
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;

            var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, line, maxLen);
            var intIds = Array.ConvertAll(tokenIds, x => (int)x);

            var result = runner.Forward(intIds, attnMask, 1, intIds.Length);
            var logits = result.Logits ?? Array.Empty<float>();
            float posConf = logits.Length > 1 ? logits[1] : 0f;
            float negConf = logits.Length > 0 ? logits[0] : 0f;
            float total = posConf + negConf;
            var sentiment = posConf > negConf ? "POSITIVE" : "NEGATIVE";
            var confidence = total > 0 ? Math.Max(posConf, negConf) / total * 100 : 0;

            Console.WriteLine($"  Sentiment: {sentiment} ({confidence:F1}%)");
        }

        return 0;
    }

    static (float Loss, float Accuracy) EvaluateSilk(
        SilkBertEncoderRunner runner,
        BertTokenizer tokenizer,
        List<Sst2Example> devSet,
        int maxLen,
        int batchSize)
    {
        var tokenized = Sst2Dataset.Tokenize(tokenizer, devSet, maxLen);
        var batches = Sst2Dataset.CreateBatches(tokenized, Math.Min(batchSize, 8), shuffle: false).ToList();

        float totalLoss = 0f;
        int total = 0;
        var predictedLabels = new List<int>();
        var actualLabels = new List<int>();

        foreach (var batch in batches)
        {
            var intIds = Array.ConvertAll(batch.TokenIds, x => (int)x);
            var result = runner.Forward(intIds, batch.AttentionMask, batch.BatchSize, batch.SeqLen);
            var logits = result.Logits ?? Array.Empty<float>();

            for (int i = 0; i < batch.BatchSize; i++)
            {
                int predicted = logits[i * 2 + 1] > logits[i * 2] ? 1 : 0;
                predictedLabels.Add(predicted);
                actualLabels.Add(batch.Labels[i]);
            }
            total += batch.BatchSize;

            float batchLoss = 0f;
            for (int i = 0; i < batch.BatchSize; i++)
            {
                float l0 = logits[i * 2];
                float l1 = logits[i * 2 + 1];
                float maxv = MathF.Max(l0, l1);
                float e0 = MathF.Exp(l0 - maxv);
                float e1 = MathF.Exp(l1 - maxv);
                float sum = e0 + e1;
                int label = batch.Labels[i];
                float prob = (label == 0 ? e0 : e1) / sum;
                batchLoss += -MathF.Log(prob);
            }
            batchLoss /= batch.BatchSize;
            totalLoss += batchLoss;
        }

        long correct = 0;
        if (predictedLabels.Count > 0)
        {
            var accuracyMatrix = NivaraMetrics.ConfusionMatrixFrame(
                NivaraColumn<int>.Create(predictedLabels.ToArray()),
                NivaraColumn<int>.Create(actualLabels.ToArray()));
            for (int r = 0; r < accuracyMatrix.RowCount; r++)
                correct += accuracyMatrix.GetColumn<long>("Correct")[r];
        }

        float loss = batches.Count > 0 ? totalLoss / batches.Count : 0f;
        float acc = total > 0 ? (float)correct / total * 100f : 0f;
        return (loss, acc);
    }

    static (float Loss, float Accuracy) Evaluate<T>(
        DistilBertForSequenceClassification<T> model,
        CrossEntropyLoss<T> lossFn,
        BertTokenizer tokenizer,
        List<Sst2Example> devSet,
        int maxLen,
        int batchSize) where T : struct, System.Numerics.IFloatingPointIeee754<T>
    {
        model.Eval();
        var tokenized = Sst2Dataset.Tokenize(tokenizer, devSet, maxLen);
        var batches = Sst2Dataset.CreateBatches(tokenized, batchSize, shuffle: false).ToList();

        float totalLoss = 0;
        int total = 0;
        var predictedLabels = new List<int>();
        var actualLabels = new List<int>();

        foreach (var batch in batches)
        {
            var intIds = Array.ConvertAll(batch.TokenIds, x => (int)x);
            var attnMask = GradientUtils.Constant(Array.ConvertAll(batch.AttentionMask, x => T.CreateChecked(x)));

            var logits = model.Forward(intIds, attnMask, batch.BatchSize, batch.SeqLen);
            var loss = lossFn.Forward(logits, batch.Labels);

            totalLoss += float.CreateChecked(loss.Data[0]);

            for (int i = 0; i < batch.BatchSize; i++)
            {
                int predicted = ArgMax<T>(logits, i, model.NumClasses);
                predictedLabels.Add(predicted);
                actualLabels.Add(batch.Labels[i]);
            }
            total += batch.BatchSize;
        }

        long correct = 0;
        if (predictedLabels.Count > 0)
        {
            var accuracyMatrix = NivaraMetrics.ConfusionMatrixFrame(
                NivaraColumn<int>.Create(predictedLabels.ToArray()),
                NivaraColumn<int>.Create(actualLabels.ToArray()));
            for (int r = 0; r < accuracyMatrix.RowCount; r++)
                correct += accuracyMatrix.GetColumn<long>("Correct")[r];
        }

        return (totalLoss / batches.Count, total > 0 ? (float)correct / total * 100 : 0f);
    }

    static int ArgMax<T>(ReverseGradTensor<T> logits, int row, int numClasses) where T : struct, System.Numerics.IFloatingPointIeee754<T>
    {
        int best = 0;
        float bestVal = float.CreateChecked(logits.Data[row * numClasses]);
        for (int c = 1; c < numClasses; c++)
        {
            float val = float.CreateChecked(logits.Data[row * numClasses + c]);
            if (val > bestVal) { bestVal = val; best = c; }
        }
        return best;
    }

    static int RunFtLora(CliArgs cli)
    {
        var modelDir = ResolveModelDir(cli.ModelDir);
        var trainData = ResolveDataPath(cli.TrainData, cli.DataDir, "train.jsonl");
        var valData = ResolveDataPath(cli.ValData, cli.DataDir, "test.jsonl");
        var outDir = string.IsNullOrWhiteSpace(cli.OutDir)
            ? Path.Combine(FindDefaultPath("data"), "worklog", "lora-output")
            : cli.OutDir;

        Console.WriteLine("=== Nivara LoRA SFT (generic Llama-family) ===");
        Console.WriteLine($"Model: {modelDir}");
        Console.WriteLine($"Train: {trainData}");
        Console.WriteLine($"Val:   {valData}");
        Console.WriteLine($"Template: {cli.Template}, rank={cli.Rank}, alpha={cli.Alpha}, targets={cli.Targets}");
        Console.WriteLine($"Epochs={cli.Epochs}, lr={cli.Lr}, maxLen={cli.MaxLen}, maxExamples={cli.MaxExamples}, seed={cli.Seed}");
        Console.WriteLine($"Silk: {cli.Silk} (D2a parity path; CPU reference implemented, Silk trainer path pending in this branch if unavailable)");

        if (cli.Silk)
        {
            return RunFtLoraSilk(cli, modelDir, trainData, valData, outDir);
        }

        bool qwen = IsQwenModelDir(modelDir, cli.Template);
        var options = new WorklogLoraSftOptions
        {
            ModelDir = modelDir,
            TrainPath = trainData,
            ValPath = valData,
            OutputDir = outDir,
            Template = cli.Template,
            Rank = cli.Rank,
            Alpha = cli.Alpha,
            Targets = ParseTargets(cli.Targets),
            LearningRate = cli.Lr,
            Epochs = cli.Epochs,
            MaxLen = cli.MaxLen,
            MaxExamples = cli.MaxExamples,
            MaxValExamples = cli.MaxExamples > 0 ? Math.Max(1, cli.MaxExamples / 4) : 0,
            Seed = cli.Seed,
            LogEvery = cli.LogEvery,
            Patience = cli.Patience,
            MinDelta = cli.MinDelta,
            GradientAccumulationSteps = cli.GradAccum,
            SaveAdapter = true,
            BaseModelId = qwen ? "qwen2.5-0.5b-instruct" : "smollm-135m-instruct"
        };

        WorklogLoraSftResult result;
        if (qwen)
            result = WorklogLoraSftTrainer.TrainQwenFloatBf16(options);
        else
            result = WorklogLoraSftTrainer.TrainSmolLMFloatBf16(options);

        Console.WriteLine();
        Console.WriteLine("=== LoRA SFT Result ===");
        Console.WriteLine($"Train examples: {result.TrainExamples}, Val examples: {result.ValExamples}");
        Console.WriteLine($"Base params: {result.BaseParams:N0}, Trainable adapter params: {result.TrainableParams:N0}");
        Console.WriteLine($"Initial val loss: {result.InitialValLoss:F4}");
        Console.WriteLine($"Best val loss:    {result.BestValLoss:F4} (epoch {result.BestEpoch})");
        Console.WriteLine($"Elapsed: {result.ElapsedMs / 1000.0:F1}s");
        if (result.AdapterPath is not null)
            Console.WriteLine($"Adapter: {result.AdapterPath}");

        var cpuReport = new LoraSftRunReport
        {
            Mode = "ft-lora",
            Backend = "cpu",
            Device = "cpu",
            Model = qwen ? "qwen2.5-0.5b-instruct" : "smollm",
            Template = cli.Template,
            Rank = cli.Rank,
            Alpha = cli.Alpha,
            Targets = cli.Targets,
            MaxLen = cli.MaxLen,
            MaxExamples = cli.MaxExamples,
            Seed = cli.Seed,
            TrainExamples = result.TrainExamples,
            ValExamples = result.ValExamples,
            BaseParams = result.BaseParams,
            TrainableParams = result.TrainableParams,
            InitialValLoss = result.InitialValLoss,
            BestValLoss = result.BestValLoss,
            BestEpoch = result.BestEpoch,
            ElapsedMs = result.ElapsedMs,
            AdapterPath = result.AdapterPath,
            AdapterFormat = result.AdapterPath is null ? null : "nivara-lora-sft-adapters-v1",
            AdapterSha256 = LoraSftRunReport.Sha256File(result.AdapterPath),
        };
        var cpuReportPath = Path.Combine(outDir, "ft-lora-report.json");
        cpuReport.Save(cpuReportPath);
        Console.WriteLine($"Report: {cpuReportPath}");
        return 0;
    }

    static int RunFtLoraSilk(CliArgs cli, string modelDir, string trainData, string valData, string outDir)
    {
        bool qwen = IsQwenModelDir(modelDir, cli.Template);
        Console.WriteLine($"Silk D2a parity status: IsParityReady=false (initial forward/loss/grad parity tests pass on synthetic fixture; release readiness still requires expanded parity).");

        SilkRuntime runtime;
        try { runtime = new SilkRuntime(); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR: OpenCL/Silk unavailable: {ex.Message}");
            Console.Error.WriteLine("No CPU fallback for --silk. Run without --silk for CPU reference.");
            return 1;
        }

        using (runtime)
        {
            Console.WriteLine($"OpenCL device: {runtime.DeviceName}");
            Directory.CreateDirectory(outDir);
            var (_, tokenizer) = qwen
                ? LlamaLoader.LoadQwenConfigAndTokenizer(modelDir)
                : LlamaLoader.LoadSmolLMConfigAndTokenizer(modelDir);
            var trainRows = WorklogSftDataset.LoadJsonl(trainData);
            var valRows = File.Exists(valData) ? WorklogSftDataset.LoadJsonl(valData) : new List<WorklogSftExample>();

            if (cli.MaxExamples > 0) trainRows = trainRows.Take(cli.MaxExamples).ToList();
            int maxVal = cli.MaxExamples > 0 ? Math.Max(1, cli.MaxExamples / 4) : valRows.Count;
            if (maxVal > 0 && valRows.Count > maxVal) valRows = valRows.Take(maxVal).ToList();

            var trainExamples = WorklogSftDataset.Tokenize(trainRows, tokenizer, cli.Template, cli.MaxLen);
            var valExamples = WorklogSftDataset.Tokenize(valRows, tokenizer, cli.Template, cli.MaxLen);
            Console.WriteLine($"Train examples: {trainExamples.Count}, Val examples: {valExamples.Count}");
            if (trainExamples.Count == 0)
            {
                Console.Error.WriteLine("ERROR: no training examples after tokenization.");
                return 1;
            }

            var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
            if (config.MaxPositionEmbeddings > 0 && cli.MaxLen > config.MaxPositionEmbeddings)
                throw new ArgumentOutOfRangeException(nameof(cli.MaxLen), $"MaxLen {cli.MaxLen} exceeds model MaxPositionEmbeddings {config.MaxPositionEmbeddings}.");
            var tensors = SafeTensorsLoader.Read<float>(Path.Combine(modelDir, "model.safetensors"));
            var targets = ParseSilkTargets(cli.Targets);
            using var trainer = new SilkLlamaLoRATrainer(runtime, new SilkLlamaLoRATrainer.Options(
                VocabSize: config.VocabSize,
                HiddenDim: config.HiddenSize,
                IntermediateDim: config.IntermediateSize,
                NumLayers: config.NumHiddenLayers,
                NumHeads: config.NumAttentionHeads,
                NumKvHeads: config.NumKeyValueHeads,
                MaxSeqLen: cli.MaxLen,
                RmsNormEps: config.RmsNormEps,
                RopeTheta: config.RopeTheta,
                Rank: cli.Rank,
                Alpha: cli.Alpha,
                Targets: targets));
            trainer.UploadBaseWeights(tensors);

            float bestVal = EvaluateSilk(trainer, valExamples);
            Console.WriteLine($"Initial val loss: {bestVal:F4}");
            int bestEpoch = 0;
            string? adapterPath = null;
            var sw = System.Diagnostics.Stopwatch.StartNew();

            for (int epoch = 1; epoch <= cli.Epochs; epoch++)
            {
                var order = Enumerable.Range(0, trainExamples.Count).ToArray();
                Shuffle(order, cli.Seed + epoch);
                float epochLoss = 0f;
                for (int i = 0; i < order.Length; i++)
                {
                    var ex = trainExamples[order[i]];
                    float loss = trainer.TrainStep(ex.InputIds, ex.Labels, cli.Lr);
                    epochLoss += loss;
                    if ((i + 1) % Math.Max(1, cli.LogEvery) == 0)
                        Console.WriteLine($"epoch {epoch}/{cli.Epochs} step {i + 1}/{order.Length} loss={loss:F4}");
                }
                epochLoss /= order.Length;
                float valLoss = EvaluateSilk(trainer, valExamples);
                Console.WriteLine($"epoch {epoch} train_loss={epochLoss:F4} val_loss={valLoss:F4}");
                if (valLoss <= bestVal)
                {
                    bestVal = valLoss;
                    bestEpoch = epoch;
                    adapterPath = Path.Combine(outDir, "lora-adapter.json");
                    trainer.SaveAdapterBundle(adapterPath, qwen ? "qwen2.5-0.5b-instruct" : "smollm-135m-instruct", cli.Template);
                }
            }

            sw.Stop();
            Console.WriteLine();
            Console.WriteLine("=== LoRA SFT Result (Silk) ===");
            Console.WriteLine($"Train examples: {trainExamples.Count}, Val examples: {valExamples.Count}");
            Console.WriteLine($"Best val loss: {bestVal:F4} (epoch {bestEpoch})");
            Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds / 1000.0:F1}s");
            if (adapterPath is not null) Console.WriteLine($"Adapter: {adapterPath}");
            Console.WriteLine("NOTE: Silk D2a path is functional in this branch but not yet release-parity-certified; expanded parity/CI evidence is still required.");

            var silkReport = new LoraSftRunReport
            {
                Mode = "ft-lora",
                Backend = "silk",
                Device = runtime.DeviceName,
                Model = qwen ? "qwen2.5-0.5b-instruct" : "smollm",
                Template = cli.Template,
                Rank = cli.Rank,
                Alpha = cli.Alpha,
                Targets = cli.Targets,
                MaxLen = cli.MaxLen,
                MaxExamples = cli.MaxExamples,
                Seed = cli.Seed,
                TrainExamples = trainExamples.Count,
                ValExamples = valExamples.Count,
                InitialValLoss = bestVal,
                BestValLoss = bestVal,
                BestEpoch = bestEpoch,
                ElapsedMs = sw.ElapsedMilliseconds,
                AdapterPath = adapterPath,
                AdapterFormat = adapterPath is null ? null : "nivara-lora-sft-adapters-v1",
                AdapterSha256 = LoraSftRunReport.Sha256File(adapterPath),
            };
            var silkReportPath = Path.Combine(outDir, "ft-lora-report.json");
            silkReport.Save(silkReportPath);
            Console.WriteLine($"Report: {silkReportPath}");
            return 0;
        }
    }

    static float EvaluateSilk(SilkLlamaLoRATrainer trainer, IReadOnlyList<WorklogTokenizedExample> examples)
    {
        if (examples.Count == 0) return float.PositiveInfinity;
        float sum = 0f;
        for (int i = 0; i < examples.Count; i++)
        {
            var ex = examples[i];
            trainer.Forward(ex.InputIds);
            var (loss, _) = trainer.BackwardPromptMaskedCrossEntropy(ex.Labels, ex.InputIds.Length);
            sum += loss;
        }
        return sum / examples.Count;
    }

    static SilkLlamaLoRATarget ParseSilkTargets(string targets)
    {
        var result = SilkLlamaLoRATarget.None;
        foreach (var part in targets.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            result |= part.ToLowerInvariant() switch
            {
                "q" => SilkLlamaLoRATarget.Q,
                "k" => SilkLlamaLoRATarget.K,
                "v" => SilkLlamaLoRATarget.V,
                "o" => SilkLlamaLoRATarget.O,
                "ffn" => SilkLlamaLoRATarget.FFN,
                _ => throw new ArgumentException($"Unknown LoRA target '{part}'. Expected q,k,v,o,ffn.")
            };
        }
        if (result == SilkLlamaLoRATarget.None) result = SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V;
        return result;
    }

    static void Shuffle(int[] values, int seed)
    {
        var rng = new Random(seed);
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    static int RunFtGenerate(CliArgs cli)
    {
        int total = cli.MaxExamples > 0 ? cli.MaxExamples : 450;
        int seed = cli.Seed;
        double valRatio = cli.ValRatio;
        var baseDate = DateTime.ParseExact(cli.BaseDate, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None);
        string outDir = string.IsNullOrWhiteSpace(cli.OutDir)
            ? Path.Combine(FindDefaultPath("data"), "worklog")
            : cli.OutDir;
        Directory.CreateDirectory(outDir);

        var trainPath = Path.Combine(outDir, "train.jsonl");
        var valPath = Path.Combine(outDir, "val.jsonl");
        var testPath = Path.Combine(outDir, "test.jsonl");
        var metaPath = Path.Combine(outDir, "provenance.json");

        bool willWriteVal = valRatio > 0;
        if (!cli.Force && (File.Exists(trainPath) || File.Exists(testPath) || File.Exists(metaPath) || (willWriteVal && File.Exists(valPath))))
        {
            Console.Error.WriteLine($"ERROR: Output files already exist in '{outDir}'. Use --force to overwrite.");
            return 1;
        }

        Console.WriteLine($"Generating deterministic worklog SFT dataset: total={total}, seed={seed}, baseDate={cli.BaseDate}, valRatio={valRatio.ToString(CultureInfo.InvariantCulture)}");
        var all = WorklogSftDataset.GenerateSynthetic(total, seed, baseDate);
        var split = WorklogSftDataset.SplitDeterministic(all, seed, trainRatio: 0.78, valRatio: valRatio);
        var train = split.Train;
        var val = split.Validation;
        var test = split.Test;
        if (!willWriteVal && val.Count > 0)
        {
            test.AddRange(val);
            val = [];
        }

        WorklogSftDataset.SaveJsonl(trainPath, train, force: cli.Force);
        if (willWriteVal)
            WorklogSftDataset.SaveJsonl(valPath, val, force: cli.Force);
        WorklogSftDataset.SaveJsonl(testPath, test, force: cli.Force);

        var meta = new
        {
            format = "nivara-worklog-sft-v1",
            seed,
            total,
            baseDate = cli.BaseDate,
            trainRatio = 0.78,
            valRatio,
            train = train.Count,
            val = val.Count,
            test = test.Count,
            provenanceHash = split.ProvenanceHash,
            fractionRule = "minutes/60 rounded to 2 decimals (MidpointRounding.AwayFromZero), invariant formatting",
            generator = "NivaraFineTuning.WorklogSftDataset.GenerateSynthetic"
        };
        File.WriteAllText(metaPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"Wrote {trainPath} ({train.Count})");
        if (willWriteVal)
            Console.WriteLine($"Wrote {valPath} ({val.Count})");
        Console.WriteLine($"Wrote {testPath} ({test.Count})");
        Console.WriteLine($"Provenance: {split.ProvenanceHash}");
        return 0;
    }

    static int RunFtEval(CliArgs cli)
    {
        var modelDir = ResolveModelDir(cli.ModelDir);
        var testData = ResolveDataPath(cli.TestData, cli.DataDir, "test.jsonl");
        bool qwen = IsQwenModelDir(modelDir, cli.Template);

        Console.WriteLine("=== Nivara LoRA SFT Eval ===");
        Console.WriteLine($"Model: {modelDir}");
        Console.WriteLine($"Test:  {testData}");
        Console.WriteLine($"Template: {cli.Template}, maxNewTokens={cli.MaxNewTokens}, maxExamples={cli.MaxExamples}");
        if (!string.IsNullOrWhiteSpace(cli.AdapterPath))
            Console.WriteLine($"Adapter: {cli.AdapterPath}");

        var (_, tokenizer) = qwen
            ? LlamaLoader.LoadQwenConfigAndTokenizer(modelDir)
            : LlamaLoader.LoadSmolLMConfigAndTokenizer(modelDir);
        var testRows = WorklogSftDataset.LoadJsonl(testData);
        if (cli.MaxExamples > 0 && testRows.Count > cli.MaxExamples)
            testRows = testRows.Take(cli.MaxExamples).ToList();

        Console.WriteLine($"Loading model...");
        var (model, config, _) = LlamaLoader.LoadModel<float, float>(modelDir, qwen: qwen, requiresGrad: false);
        model.Eval();

        var stopIds = GetStopTokenIds(tokenizer, cli.Template);
        var baseMetrics = EvaluateWorklogSft(model, tokenizer, testRows, cli.Template, cli.MaxNewTokens, stopIds);
        Console.WriteLine(baseMetrics.ToReport("Base"));

        WorklogEvalMetrics? adaptedMetrics = null;
        if (!string.IsNullOrWhiteSpace(cli.AdapterPath))
        {
            var adapters = model.AttachLora(ParseTargets(cli.Targets), cli.Rank, cli.Alpha);
            LoadCpuAdapterBundle(cli.AdapterPath, adapters);
            adaptedMetrics = EvaluateWorklogSft(model, tokenizer, testRows, cli.Template, cli.MaxNewTokens, stopIds);
            Console.WriteLine(adaptedMetrics.ToReport("LoRA-adapted"));
            Console.WriteLine($"Delta valid_structure: {adaptedMetrics.ValidStructureRate - baseMetrics.ValidStructureRate:+0.0%;-0.0%;0%}");
            Console.WriteLine($"Delta ticket:          {adaptedMetrics.TicketAccuracy - baseMetrics.TicketAccuracy:+0.0%;-0.0%;0%}");
            Console.WriteLine($"Delta hours:           {adaptedMetrics.HoursAccuracy - baseMetrics.HoursAccuracy:+0.0%;-0.0%;0%}");
            Console.WriteLine($"Delta date:            {adaptedMetrics.DateAccuracy - baseMetrics.DateAccuracy:+0.0%;-0.0%;0%}");
            Console.WriteLine($"Delta convention:      {adaptedMetrics.ConventionAccuracy - baseMetrics.ConventionAccuracy:+0.0%;-0.0%;0%}");
        }

        var evalReport = new LoraSftRunReport
        {
            Mode = "ft-eval",
            Backend = "cpu",
            Device = "cpu",
            Model = qwen ? "qwen2.5-0.5b-instruct" : "smollm",
            Template = cli.Template,
            Rank = cli.Rank,
            Alpha = cli.Alpha,
            Targets = cli.Targets,
            MaxLen = cli.MaxLen,
            MaxExamples = cli.MaxExamples,
            Seed = cli.Seed,
            TrainExamples = 0,
            ValExamples = testRows.Count,
            AdapterPath = string.IsNullOrWhiteSpace(cli.AdapterPath) ? null : cli.AdapterPath,
            AdapterFormat = string.IsNullOrWhiteSpace(cli.AdapterPath) ? null : "nivara-lora-sft-adapters-v1",
            AdapterSha256 = LoraSftRunReport.Sha256File(string.IsNullOrWhiteSpace(cli.AdapterPath) ? null : cli.AdapterPath),
            BaseMetrics = baseMetrics.ToReport("Base").Trim(),
            AdaptedMetrics = adaptedMetrics?.ToReport("LoRA-adapted").Trim(),
        };
        var evalReportPath = Path.Combine(string.IsNullOrWhiteSpace(cli.OutDir) ? Path.GetDirectoryName(cli.AdapterPath) ?? "." : cli.OutDir, "ft-eval-report.json");
        try { evalReport.Save(evalReportPath); Console.WriteLine($"Report: {evalReportPath}"); }
        catch (Exception ex) { Console.WriteLine($"Report write skipped: {ex.Message}"); }
        return 0;
    }

    static WorklogEvalMetrics EvaluateWorklogSft(
        LlamaForCausalLM<float, float> model,
        Gpt2BpeTokenizer tokenizer,
        IReadOnlyList<WorklogSftExample> examples,
        string template,
        int maxNewTokens,
        int[] stopIds)
    {
        return WorklogSftEvaluator.Evaluate(examples, ex =>
        {
            var promptMessages = ex.Messages.TakeWhile(m => !string.Equals(m.Role, "assistant", StringComparison.OrdinalIgnoreCase)).ToList();
            var promptText = RenderEvalPrompt(promptMessages, template);
            var promptIds = tokenizer.Encode(promptText).ToArray();
            var genIds = GenerateGreedy(model, promptIds, maxNewTokens, stopIds);
            return tokenizer.Decode(genIds);
        });
    }

    static string RenderEvalPrompt(IReadOnlyList<WorklogMessage> messages, string template)
    {
        var chatMessages = messages.Select(m => new Microsoft.Extensions.AI.ChatMessage(
            m.Role.ToLowerInvariant() switch
            {
                "system" => Microsoft.Extensions.AI.ChatRole.System,
                "user" => Microsoft.Extensions.AI.ChatRole.User,
                "assistant" => Microsoft.Extensions.AI.ChatRole.Assistant,
                "tool" => Microsoft.Extensions.AI.ChatRole.Tool,
                _ => Microsoft.Extensions.AI.ChatRole.User
            },
            m.Content ?? string.Empty)).ToList();

        return template.ToLowerInvariant() switch
        {
            "qwen" => QwenChatTemplate.Render(chatMessages, addGenerationPrompt: true),
            "smollm" => SmollmChatTemplate.Render(chatMessages, addGenerationPrompt: true),
            _ => throw new ArgumentException($"Unknown template '{template}'.", nameof(template))
        };
    }

    static int[] GenerateGreedy(LlamaForCausalLM<float, float> model, int[] promptIds, int maxNewTokens, int[] stopIds)
    {
        var stop = new HashSet<int>(stopIds);
        var gen = new List<int>(maxNewTokens);
        var logits = model.Forward(promptIds);
        int vocab = logits.Shape[^1];
        for (int t = 0; t < maxNewTokens; t++)
        {
            int next = ArgMaxLastRow(logits.Data.ToArray(), vocab);
            if (stop.Contains(next)) break;
            gen.Add(next);
            var seq = new int[promptIds.Length + gen.Count];
            Array.Copy(promptIds, seq, promptIds.Length);
            for (int i = 0; i < gen.Count; i++) seq[promptIds.Length + i] = gen[i];
            logits = model.Forward(seq);
        }
        return gen.ToArray();
    }

    static int ArgMaxLastRow(float[] data, int vocab)
    {
        int off = Math.Max(0, data.Length - vocab);
        int best = 0;
        float bestVal = data[off];
        for (int i = 1; i < vocab; i++)
        {
            float v = data[off + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    static int[] GetStopTokenIds(Gpt2BpeTokenizer tokenizer, string template)
    {
        var stops = new List<int>();
        void Add(string token)
        {
            int id = tokenizer.TokenId(token);
            if (id >= 0 && !stops.Contains(id)) stops.Add(id);
        }
        Add("<|im_end|>");
        Add("<|endoftext|>");
        Add("<|eot_id|>");
        return stops.ToArray();
    }

    static void LoadCpuAdapterBundle(string path, IReadOnlyList<LoRALinear<float, float>> adapters)
    {
        var bundle = LoraSftAdapterBundle.Load(path);
        if (bundle.Rank > 0 && adapters.Count > 0 && bundle.Rank != adapters[0].Rank)
            throw new InvalidOperationException($"Adapter rank mismatch: file rank={bundle.Rank}, model rank={adapters[0].Rank}.");
        if (adapters.Count > 0 && bundle.Alpha != 0f && MathF.Abs(bundle.Alpha - (float)adapters[0].Alpha) > 1e-6f)
            throw new InvalidOperationException($"Adapter alpha mismatch: file alpha={bundle.Alpha}, model alpha={adapters[0].Alpha}.");
        if (bundle.Adapters.Length != adapters.Count)
            throw new InvalidOperationException($"Adapter count mismatch: file has {bundle.Adapters.Length}, model has {adapters.Count}.");

        for (int i = 0; i < adapters.Count; i++)
        {
            var entry = bundle.Adapters[i];
            if (entry.Index >= 0 && entry.Index != i)
                throw new InvalidOperationException($"Adapter index mismatch at position {i}: file index={entry.Index}.");

            if (entry.AShape.Length == 2)
            {
                if (entry.AShape[0] != adapters[i].A.Tensor.Shape[0] || entry.AShape[1] != adapters[i].A.Tensor.Shape[1])
                    throw new InvalidOperationException($"Adapter {i} A shape mismatch.");
            }
            if (entry.BShape.Length == 2)
            {
                if (entry.BShape[0] != adapters[i].B.Tensor.Shape[0] || entry.BShape[1] != adapters[i].B.Tensor.Shape[1])
                    throw new InvalidOperationException($"Adapter {i} B shape mismatch.");
            }

            var aBytes = Convert.FromBase64String(entry.A);
            var bBytes = Convert.FromBase64String(entry.B);
            var a = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(aBytes).ToArray();
            var b = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bBytes).ToArray();
            if (a.Length != adapters[i].A.Tensor.Length || b.Length != adapters[i].B.Tensor.Length)
                throw new InvalidOperationException($"Adapter {i} data length mismatch.");
            adapters[i].A.Tensor = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(a, adapters[i].A.Tensor.Shape[0], adapters[i].A.Tensor.Shape[1], requiresGrad: true);
            adapters[i].B.Tensor = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(b, adapters[i].B.Tensor.Shape[0], adapters[i].B.Tensor.Shape[1], requiresGrad: true);
        }
    }

    static string ResolveModelDir(string modelDir)
    {
        if (!string.IsNullOrWhiteSpace(modelDir)) return modelDir;
        var qwen = Path.Combine(FindDefaultPath("data"), "qwen2.5-0.5b-instruct");
        if (Directory.Exists(qwen)) return qwen;
        return FindDefaultPath("data/distilbert");
    }

    static string ResolveDataPath(string explicitPath, string dataDir, string defaultName)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath)) return explicitPath;
        if (!string.IsNullOrWhiteSpace(dataDir))
        {
            var candidate = Path.Combine(dataDir, defaultName);
            if (File.Exists(candidate)) return candidate;
        }
        var worklog = Path.Combine(FindDefaultPath("data"), "worklog", defaultName);
        if (File.Exists(worklog)) return worklog;
        return Path.Combine(FindDefaultPath("data"), defaultName);
    }

    static bool IsQwenModelDir(string modelDir, string template)
    {
        var name = Path.GetFileName(modelDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (name.Contains("qwen", StringComparison.OrdinalIgnoreCase)) return true;
        if (name.Contains("smollm", StringComparison.OrdinalIgnoreCase)) return false;
        return template.Equals("qwen", StringComparison.OrdinalIgnoreCase);
    }

    static LlamaLoRATarget ParseTargets(string targets)
    {
        var result = LlamaLoRATarget.None;
        foreach (var part in targets.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            result |= part.ToLowerInvariant() switch
            {
                "q" => LlamaLoRATarget.Q,
                "k" => LlamaLoRATarget.K,
                "v" => LlamaLoRATarget.V,
                "o" => LlamaLoRATarget.O,
                "ffn" => LlamaLoRATarget.FFN,
                _ => throw new ArgumentException($"Unknown LoRA target '{part}'. Expected q,k,v,o,ffn.")
            };
        }
        if (result == LlamaLoRATarget.None) result = LlamaLoRATarget.Q | LlamaLoRATarget.V;
        return result;
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

    static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16Tensors(
        Dictionary<string, (float[] Data, int[] Shape)> src)
    {
        var result = new Dictionary<string, (BFloat16[] Data, int[] Shape)>(src.Count);
        foreach (var kv in src)
        {
            var data = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < data.Length; i++) data[i] = BFloat16.CreateChecked(kv.Value.Data[i]);
            result[kv.Key] = (data, kv.Value.Shape);
        }
        return result;
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
