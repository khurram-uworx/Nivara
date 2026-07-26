using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NivaraTimeSeries;

var options = Options.Parse(args);

if (options.Help)
{
    Options.PrintHelp();
    return;
}

var dataPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "samples", "data", "timeseries_metrics.csv");
if (options.LoadDataPath != null && File.Exists(options.LoadDataPath))
    dataPath = options.LoadDataPath;

MetricsGenerator dataset;
if (File.Exists(dataPath) && options.Epochs == 0)
{
    dataset = MetricsGenerator.Load(dataPath);
    Console.WriteLine($"Loaded {dataset.Count} windows from {dataPath}");
}
else
{
    dataset = new MetricsGenerator(options.NumSamples, options.WindowSize, options.Seed, options.AnomalyRatio);
    Console.WriteLine($"Generated {dataset.Count} windows ({dataset.NumChannels} channels, {options.WindowSize} timesteps)");
    if (options.SaveDataPath != null)
    {
        dataset.Save(options.SaveDataPath);
        Console.WriteLine($"Saved dataset to {options.SaveDataPath}");
    }
}

if (options.ShowMetrics)
{
    ShowMetrics(dataset, options.WindowSize);
    return;
}

var model = new TimeSeriesModel<float>(
    numChannels: dataset.NumChannels,
    windowSize: options.WindowSize,
    latentDim: options.LatentDim,
    hiddenDim: options.HiddenDim,
    dropout: options.Dropout);

var optimizer = new Adam<float>(options.LearningRate);
optimizer.AddParameterGroup(model.GetParameters().Values);

if (options.LoadPath != null && File.Exists(options.LoadPath))
{
    ModelSerializer.Load(model, options.LoadPath);
    Console.WriteLine($"Loaded model from {options.LoadPath}");
}

if (options.Epochs > 0)
    Train(model, optimizer, dataset, options);

if (options.Detect)
    DetectAnomalies(model, dataset, options);

if (options.SavePath != null && options.Epochs > 0)
{
    ModelSerializer.Save(model, options.SavePath);
    Console.WriteLine($"Saved model to {options.SavePath}");
}

static void Train(TimeSeriesModel<float> model, Adam<float> optimizer, MetricsGenerator dataset, Options options)
{
    model.Train();
    int normalCount = 0;
    for (int i = 0; i < dataset.Count; i++)
        if (!dataset.IsAnomaly(i)) normalCount++;

    var normalIndices = new int[normalCount];
    int idx = 0;
    for (int i = 0; i < dataset.Count; i++)
        if (!dataset.IsAnomaly(i)) normalIndices[idx++] = i;

    var mseLoss = new MSELoss<float>();
    var rng = new Random(options.Seed);
    int elementsPerWindow = dataset.ElementsPerWindow;
    int channels = dataset.NumChannels;
    int windowSize = options.WindowSize;
    float beta = options.Beta;
    var betaTensor = ReverseGradTensor<float>.FromArray(new float[] { beta }, requiresGrad: false);

    Console.WriteLine($"\n--- Training ({options.Epochs} epochs, {normalCount} normal windows, batch size {options.BatchSize}) ---");

    var sw = Stopwatch.StartNew();
    float lastLoss = 0;

    for (int epoch = 1; epoch <= options.Epochs; epoch++)
    {
        Shuffle(normalIndices, rng);
        float epochLoss = 0;
        int batchCount = 0;

        for (int start = 0; start < normalCount; start += options.BatchSize)
        {
            int batchSize = Math.Min(options.BatchSize, normalCount - start);
            var features = BuildBatch(dataset, normalIndices, start, batchSize, channels, windowSize, requiresGrad: true);

            using (GradientUtils.Grad())
            {
                var (mu, logVar) = model.Encode(features);
                var z = model.Reparameterize(mu, logVar);
                var recon = model.Decode(z);

                var reconLoss = mseLoss.Forward(recon, features, reduceToMean: true);
                var kl = ReverseGradOperations.KlDivergence(mu, logVar);
                var klMean = ReverseGradOperations.Divide(kl,
                    ReverseGradTensor<float>.FromArray(new float[] { (float)batchSize }, requiresGrad: false));
                var loss = ReverseGradOperations.Add(reconLoss,
                    ReverseGradOperations.Multiply(klMean, betaTensor));

                loss.Backward();
                GradientUtils.ClipGradNorm(model.Parameters().Values, 1.0);
                optimizer.Step();
                optimizer.ZeroGrad();

                lastLoss = loss[0];
                epochLoss += lastLoss;
                batchCount++;
            }

            features.Dispose();
        }

        float avgLoss = epochLoss / batchCount;
        if (epoch % 1 == 0 || epoch == options.Epochs)
            Console.WriteLine($"  Epoch {epoch,3}/{options.Epochs}  loss={avgLoss:F4}  ({sw.ElapsedMilliseconds}ms)");
    }

    sw.Stop();
    Console.WriteLine($"  Training complete in {sw.ElapsedMilliseconds}ms (final loss: {lastLoss:F4})");
}

static void DetectAnomalies(TimeSeriesModel<float> model, MetricsGenerator dataset, Options options)
{
    model.Eval();
    int channels = dataset.NumChannels;
    int windowSize = options.WindowSize;

    Console.WriteLine("\n--- Anomaly Detection ---");

    var trainErrors = new List<float>();
    for (int i = 0; i < dataset.Count; i++)
    {
        var window = dataset.GetWindowArray(i);
        var input = BuildTensor(window, channels, windowSize, requiresGrad: false);
        var recon = model.Forward(input);
        float error = MseBetween(recon, input);
        trainErrors.Add(error);
        input.Dispose();
    }

    float mean = trainErrors.Average();
    float variance = 0;
    for (int i = 0; i < trainErrors.Count; i++)
    {
        float diff = trainErrors[i] - mean;
        variance += diff * diff;
    }
    float stddev = MathF.Sqrt(variance / trainErrors.Count);
    float threshold = mean + 2 * stddev;

    Console.WriteLine($"  Baseline: mean={mean:F6}, stddev={stddev:F6}, threshold={threshold:F6}");

    int truePositives = 0, falsePositives = 0, falseNegatives = 0, trueNegatives = 0;
    int totalAnomalies = 0;

    for (int i = 0; i < dataset.Count; i++)
    {
        bool actualAnomaly = dataset.IsAnomaly(i);
        if (actualAnomaly) totalAnomalies++;

        float error = trainErrors[i];
        bool detected = error > threshold;

        if (detected && actualAnomaly) truePositives++;
        else if (detected && !actualAnomaly) falsePositives++;
        else if (!detected && actualAnomaly) falseNegatives++;
        else trueNegatives++;
    }

    float precision = truePositives + falsePositives > 0
        ? (float)truePositives / (truePositives + falsePositives) : 0;
    float recall = totalAnomalies > 0
        ? (float)truePositives / totalAnomalies : 0;
    float f1 = precision + recall > 0
        ? 2 * precision * recall / (precision + recall) : 0;

    Console.WriteLine($"\n  Results:");
    Console.WriteLine($"    True positives:  {truePositives}");
    Console.WriteLine($"    False positives: {falsePositives}");
    Console.WriteLine($"    False negatives: {falseNegatives}");
    Console.WriteLine($"    True negatives:  {trueNegatives}");
    Console.WriteLine($"    Precision: {precision:F3}");
    Console.WriteLine($"    Recall:    {recall:F3}");
    Console.WriteLine($"    F1 score:  {f1:F3}");

    Console.WriteLine($"\n  {"Idx",5} {"Actual",10} {"Detected",10} {"Error",10} {"Type"}");
    Console.WriteLine($"  {new string('-', 55)}");
    for (int i = 0; i < Math.Min(50, dataset.Count); i++)
    {
        bool actualAnomaly = dataset.IsAnomaly(i);
        bool detected = trainErrors[i] > threshold;
        string marker = (detected, actualAnomaly) switch
        {
            (true, true) => "TP",
            (true, false) => "FP",
            (false, true) => "FN",
            _ => "TN"
        };
        string type = dataset.GetAnomalyType(i).ToString();
        Console.WriteLine($"  {i,5} {(actualAnomaly ? "ANOMALY" : "normal"),10} {(detected ? "ANOMALY" : "normal"),10} {trainErrors[i],10:F6} {marker} {type}");
    }
}

static float MseBetween(ReverseGradTensor<float> a, ReverseGradTensor<float> b)
{
    float sumSq = 0;
    int len = Math.Min(a.Length, b.Length);
    var aArr = ArrayPool<float>.Shared.Rent(len);
    var bArr = ArrayPool<float>.Shared.Rent(len);
    try
    {
        for (int i = 0; i < len; i++)
        {
            aArr[i] = float.CreateChecked(a[i]);
            bArr[i] = float.CreateChecked(b[i]);
        }
        TensorPrimitives.Subtract(aArr.AsSpan(0, len), bArr.AsSpan(0, len), aArr.AsSpan(0, len));
        sumSq = TensorPrimitives.Dot(aArr.AsSpan(0, len), aArr.AsSpan(0, len));
    }
    finally
    {
        ArrayPool<float>.Shared.Return(aArr);
        ArrayPool<float>.Shared.Return(bArr);
    }
    return sumSq / len;
}

static ReverseGradTensor<float> BuildBatch(MetricsGenerator dataset, int[] indices, int start, int batchSize, int channels, int windowSize, bool requiresGrad)
{
    int elementsPerWindow = channels * windowSize;
    var data = new float[batchSize * elementsPerWindow];
    for (int b = 0; b < batchSize; b++)
    {
        var window = dataset.GetWindow(indices[start + b]);
        window.CopyTo(data.AsSpan(b * elementsPerWindow, elementsPerWindow));
    }
    var tensor = ReverseGradTensor<float>.FromArray(data, requiresGrad: requiresGrad);
    tensor.Reshape(batchSize, channels, windowSize);
    return tensor;
}

static ReverseGradTensor<float> BuildTensor(float[] window, int channels, int windowSize, bool requiresGrad)
{
    var tensor = ReverseGradTensor<float>.FromArray((float[])window.Clone(), requiresGrad: requiresGrad);
    tensor.Reshape(1, channels, windowSize);
    return tensor;
}

static void Shuffle(int[] array, Random rng)
{
    for (int i = array.Length - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (array[i], array[j]) = (array[j], array[i]);
    }
}

static void ShowMetrics(MetricsGenerator dataset, int windowSize)
{
    string[] channelNames = ["CPU", "Mem", "Disk", "Net"];
    string blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588";

    int showSteps = Math.Min(128, windowSize);
    Console.WriteLine($"--- Sample Metrics (first {showSteps} timesteps) ---");
    Console.WriteLine($"     {string.Join("   ", channelNames)}");

    for (int row = 0; row < showSteps; row += 16)
    {
        string line = $"{row,4}: ";
        for (int ch = 0; ch < dataset.NumChannels; ch++)
        {
            var bar = new char[16];
            for (int t = 0; t < 16 && row + t < showSteps; t++)
            {
                float val = dataset.GetWindow(0)[ch * windowSize + row + t];
                int level = Math.Clamp((int)(val * 7 + 0.5f), 0, 7);
                bar[t] = blocks[level];
            }
            line += new string(bar) + " ";
        }
        Console.WriteLine(line);
    }
}

sealed class Options
{
    public int Epochs { get; init; } = 20;
    public int WindowSize { get; init; } = 64;
    public int LatentDim { get; init; } = 16;
    public int HiddenDim { get; init; } = 128;
    public int BatchSize { get; init; } = 64;
    public float LearningRate { get; init; } = 0.001f;
    public float Beta { get; init; } = 0.5f;
    public float Dropout { get; init; } = 0.2f;
    public int NumSamples { get; init; } = 5000;
    public int Seed { get; init; } = 42;
    public float AnomalyRatio { get; init; } = 0.15f;
    public string? SavePath { get; init; }
    public string? LoadPath { get; init; }
    public string? SaveDataPath { get; init; }
    public string? LoadDataPath { get; init; }
    public bool Detect { get; init; }
    public bool ShowMetrics { get; init; }
    public bool Help { get; init; }

    public static Options Parse(string[] args)
    {
        int epochs = 20, windowSize = 64, latentDim = 16, hiddenDim = 128, batchSize = 64;
        int numSamples = 5000, seed = 42;
        float lr = 0.001f, beta = 0.5f, dropout = 0.2f, anomalyRatio = 0.15f;
        string? savePath = null, loadPath = null, saveDataPath = null, loadDataPath = null;
        bool detect = false, showMetrics = false, help = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--epochs" or "-e": epochs = int.Parse(args[++i]); break;
                case "--window-size" or "-w": windowSize = int.Parse(args[++i]); break;
                case "--latent-dim" or "-l": latentDim = int.Parse(args[++i]); break;
                case "--hidden-dim": hiddenDim = int.Parse(args[++i]); break;
                case "--batch-size" or "-b": batchSize = int.Parse(args[++i]); break;
                case "--lr" or "-r": lr = float.Parse(args[++i]); break;
                case "--beta": beta = float.Parse(args[++i]); break;
                case "--dropout": dropout = float.Parse(args[++i]); break;
                case "--num-samples" or "-n": numSamples = int.Parse(args[++i]); break;
                case "--seed" or "-s": seed = int.Parse(args[++i]); break;
                case "--anomaly-ratio": anomalyRatio = float.Parse(args[++i]); break;
                case "--save": savePath = args[++i]; break;
                case "--load": loadPath = args[++i]; break;
                case "--save-data": saveDataPath = args[++i]; break;
                case "--load-data": loadDataPath = args[++i]; break;
                case "--detect": detect = true; break;
                case "--show-metrics": showMetrics = true; break;
                case "--help" or "-h": help = true; break;
            }
        }

        return new Options
        {
            Epochs = epochs, WindowSize = windowSize, LatentDim = latentDim,
            HiddenDim = hiddenDim, BatchSize = batchSize, LearningRate = lr,
            Beta = beta, Dropout = dropout, NumSamples = numSamples,
            Seed = seed, AnomalyRatio = anomalyRatio,
            SavePath = savePath, LoadPath = loadPath,
            SaveDataPath = saveDataPath, LoadDataPath = loadDataPath,
            Detect = detect, ShowMetrics = showMetrics, Help = help
        };
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraTimeSeries — Server Monitoring Anomaly Detection

            USAGE:
              dotnet run --project samples/NivaraTimeSeries -- [OPTIONS]

            OPTIONS:
              --epochs <int>          Training epochs (default: 20)
              --window-size <int>     Timesteps per window (default: 64)
              --latent-dim <int>      Latent space dimension (default: 16)
              --hidden-dim <int>      Decoder hidden layer size (default: 128)
              --batch-size <int>      Batch size (default: 64)
              --lr <float>            Learning rate (default: 0.001)
              --beta <float>          KL divergence weight (default: 0.5)
              --dropout <float>       Dropout probability (default: 0.2)
              --num-samples <int>     Number of synthetic windows (default: 5000)
              --seed <int>            RNG seed (default: 42)
              --anomaly-ratio <float> Fraction of anomalous windows (default: 0.15)
              --save <path>           Save trained model to JSON
              --load <path>           Load model from JSON
              --save-data <path>      Save dataset to CSV
              --load-data <path>      Load dataset from CSV
              --detect                Run anomaly detection after training
              --show-metrics          Display sample metrics as ASCII chart
              --help, -h              Show this help
            """);
    }
}
