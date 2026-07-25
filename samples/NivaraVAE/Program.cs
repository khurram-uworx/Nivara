using System.Buffers;
using System.Numerics;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NivaraVAE;

var options = Options.Parse(args);

if (options.Help)
{
    Options.PrintHelp();
    return;
}

var dataset = options.LoadDataPath != null && File.Exists(options.LoadDataPath)
    ? PatternDataset.Load(options.LoadDataPath)
    : new PatternDataset(options.NumPatterns, options.PatternSize, options.Seed);
Console.WriteLine($"Loaded {dataset.Count} patterns ({options.PatternSize}x{options.PatternSize}, {dataset.NumPixels} pixels)");

if (options.SaveDataPath != null)
{
    dataset.Save(options.SaveDataPath);
    Console.WriteLine($"Saved patterns to {options.SaveDataPath}");
}

if (options.ShowPatterns)
{
    ShowPatterns(dataset, Math.Min(24, dataset.Count));
    return;
}

var model = new VaeModel<float>(dataset.NumPixels, options.HiddenDim, options.LatentDim, options.Dropout);
var optimizer = new Adam<float>(options.LearningRate);
optimizer.AddParameterGroup(model.GetParameters().Values);

if (options.LoadPath != null && File.Exists(options.LoadPath))
{
    ModelSerializer.Load(model, options.LoadPath);
    Console.WriteLine($"Loaded model from {options.LoadPath}");
}

if (options.Epochs > 0)
    Train(model, optimizer, dataset, options);

if (options.GenerateCount is int gc)
    Generate(model, gc, options.LatentDim, options.PatternSize, options.Seed);

if (options.InterpolateCount is int ic)
    Interpolate(model, dataset, ic, options.PatternSize);

if (options.LatentWalk)
    LatentWalk(model, options.LatentDim, options.PatternSize);

if (options.Eval)
    Evaluate(model, dataset, options);

if (options.SavePath != null && (options.Epochs > 0 || options.GenerateCount != null))
{
    ModelSerializer.Save(model, options.SavePath);
    Console.WriteLine($"Saved model to {options.SavePath}");
}

static void Train(VaeModel<float> model, Adam<float> optimizer, PatternDataset dataset, Options opts)
{
    int numPixels = dataset.NumPixels;
    int batchSize = Math.Min(opts.BatchSize, dataset.Count);
    var bceLoss = new BCEWithLogitsLoss<float>();
    var rng = new Random(opts.Seed);
    var indices = new int[dataset.Count];

    Console.WriteLine($"Training: {opts.Epochs} epochs, batch size {batchSize}, lr {opts.LearningRate}");

    for (int epoch = 1; epoch <= opts.Epochs; epoch++)
    {
        model.Train();
        double epochLoss = 0;
        int batchCount = 0;

        for (int i = 0; i < dataset.Count; i++)
            indices[i] = i;
        Shuffle(indices, rng);

        for (int start = 0; start < dataset.Count; start += batchSize)
        {
            int end = Math.Min(start + batchSize, dataset.Count);
            int size = end - start;

            var features = BuildBatch(dataset, indices, start, size, numPixels, requiresGrad: true);
            var targets = BuildBatch(dataset, indices, start, size, numPixels, requiresGrad: false);

            float lossVal;
            using (GradientUtils.Grad())
            {
                var (mu, logVar) = model.Encode(features);
                var z = model.Reparameterize(mu, logVar);
                var reconLogits = model.Decode(z);

                var bce = bceLoss.Forward(reconLogits, targets, reduceToMean: true);
                var kl = ReverseGradOperations.KlDivergence(mu, logVar);
                var klMean = ReverseGradOperations.Divide(kl, new ReverseGradTensor<float>(
                    NivaraColumn<float>.Create(new float[] { (float)size }),
                    requiresGrad: false));
                var loss = ReverseGradOperations.Add(bce, ReverseGradOperations.Multiply(klMean,
                    new ReverseGradTensor<float>(
                        NivaraColumn<float>.Create(new float[] { opts.Beta }),
                        requiresGrad: false)));

                loss.Backward();
                GradientUtils.ClipGradNorm(model.Parameters().Values, 1.0);
                optimizer.Step();
                optimizer.ZeroGrad();

                lossVal = loss[0];
            }

            epochLoss += lossVal;
            batchCount++;
        }

        double avgLoss = epochLoss / batchCount;
        Console.WriteLine($"  Epoch {epoch}/{opts.Epochs}  loss={avgLoss:F4}");
    }
}

static void Generate(VaeModel<float> model, int count, int latentDim, int patternSize, int seed)
{
    model.Eval();
    Console.WriteLine($"\n--- Generated Patterns (n={count}) ---");

    var rng = new Random(seed);
    for (int i = 0; i < count; i++)
    {
        var zData = SampleStandardNormal(latentDim, rng.Next());
        var z = ReverseGradTensor<float>.FromMatrix(zData, 1, latentDim, requiresGrad: false);
        var decoded = model.Decode(z);
        var pixels = ExtractPixels(decoded, patternSize * patternSize);
        Console.WriteLine($"\nSample {i + 1}:");
        Console.WriteLine(PatternDataset.RenderPattern(pixels, patternSize));
    }
}

static void Interpolate(VaeModel<float> model, PatternDataset dataset, int count, int patternSize)
{
    model.Eval();
    int numPixels = dataset.NumPixels;
    Console.WriteLine($"\n--- Interpolations (n={count}) ---");

    var rng = new Random(42);
    for (int i = 0; i < count; i++)
    {
        int idx1 = rng.Next(dataset.Count);
        int idx2 = rng.Next(dataset.Count);

        var p1 = dataset.GetPattern(idx1);
        var p2 = dataset.GetPattern(idx2);

        var t1 = ReverseGradTensor<float>.FromMatrix(p1, 1, p1.Length, requiresGrad: false);
        var t2 = ReverseGradTensor<float>.FromMatrix(p2, 1, p2.Length, requiresGrad: false);

        var (mu1, _) = model.Encode(t1);
        var (mu2, _) = model.Encode(t2);

        Console.WriteLine($"\nInterpolation {i + 1}: sample {idx1} -> sample {idx2}");
        Console.WriteLine("From:");
        Console.WriteLine(PatternDataset.RenderPattern(p1, patternSize));

        var alphas = new float[] { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
        foreach (var alpha in alphas)
        {
            var zData = new float[mu1.Length];
            var mu1Pixels = ExtractPixels(mu1, mu1.Length);
            var mu2Pixels = ExtractPixels(mu2, mu2.Length);
            for (int j = 0; j < zData.Length; j++)
                zData[j] = alpha * mu1Pixels[j] + (1 - alpha) * mu2Pixels[j];

            var z = ReverseGradTensor<float>.FromMatrix(zData, 1, zData.Length, requiresGrad: false);
            var decoded = model.Decode(z);
            var pixels = ExtractPixels(decoded, numPixels);
            Console.WriteLine($"  alpha={alpha:F2}: {PatternDataset.RenderPattern(pixels, patternSize)}");
        }
    }
}

static void LatentWalk(VaeModel<float> model, int latentDim, int patternSize)
{
    model.Eval();
    int numPixels = patternSize * patternSize;
    Console.WriteLine($"\n--- Latent Walk (dim={latentDim}) ---");

    var steps = new float[] { -3f, -2f, -1f, 0f, 1f, 2f, 3f };

    for (int d = 0; d < latentDim; d++)
    {
        Console.WriteLine($"\nDimension {d}:");
        foreach (var val in steps)
        {
            var zData = new float[latentDim];
            zData[d] = val;
            var z = ReverseGradTensor<float>.FromMatrix(zData, 1, latentDim, requiresGrad: false);
            var decoded = model.Decode(z);
            var pixels = ExtractPixels(decoded, numPixels);
            Console.WriteLine($"  {val:+0.0;-0.0}: {PatternDataset.RenderPattern(pixels, patternSize)}");
        }
    }
}

static void Evaluate(VaeModel<float> model, PatternDataset dataset, Options opts)
{
    model.Eval();
    int numPixels = dataset.NumPixels;
    int testSize = Math.Max(1, (int)(dataset.Count * 0.2));
    var bceLoss = new BCEWithLogitsLoss<float>();

    double totalLoss = 0;
    for (int i = 0; i < testSize; i++)
    {
        var pattern = dataset.GetPattern(i);
        var input = ReverseGradTensor<float>.FromMatrix(pattern, 1, numPixels, requiresGrad: false);

        var (mu, logVar) = model.Encode(input);
        var z = model.Reparameterize(mu, logVar, seed: i);
        var reconLogits = model.Decode(z);

        var bce = bceLoss.Forward(reconLogits, input, reduceToMean: true);
        totalLoss += bce[0];
    }

    Console.WriteLine($"\n--- Evaluation ({testSize} samples) ---");
    Console.WriteLine($"  Mean BCE loss: {totalLoss / testSize:F4}");
}

static ReverseGradTensor<float> BuildBatch(PatternDataset dataset, int[] indices, int start, int size, int numPixels, bool requiresGrad)
{
    var data = new float[size * numPixels];
    for (int i = 0; i < size; i++)
    {
        var pattern = dataset.GetPattern(indices[start + i]);
        Array.Copy(pattern, 0, data, i * numPixels, numPixels);
    }
    return ReverseGradTensor<float>.FromMatrix(data, size, numPixels, requiresGrad);
}

static float[] ExtractPixels(ReverseGradTensor<float> tensor, int count)
{
    var pixels = new float[count];
    for (int i = 0; i < count; i++)
        pixels[i] = tensor[i];
    return pixels;
}

static void ShowPatterns(PatternDataset dataset, int count)
{
    int perRow = 4;
    int rows = (count + perRow - 1) / perRow;

    for (int r = 0; r < rows; r++)
    {
        int start = r * perRow;
        int end = Math.Min(start + perRow, count);

        var leftPatterns = new string[end - start];
        for (int i = start; i < end; i++)
            leftPatterns[i - start] = PatternDataset.RenderPattern(dataset.GetPattern(i), dataset.GridSize);

        var leftLines = leftPatterns.Select(p => p.Split('\n')).ToArray();
        int maxLines = leftLines.Max(l => l.Length);

        for (int line = 0; line < maxLines; line++)
        {
            for (int p = 0; p < leftLines.Length; p++)
            {
                if (line < leftLines[p].Length)
                    Console.Write(leftLines[p][line]);
                else
                    Console.Write(new string(' ', dataset.GridSize));
                Console.Write("  ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}

static void Shuffle(int[] array, Random rng)
{
    for (int i = array.Length - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (array[i], array[j]) = (array[j], array[i]);
    }
}

static float[] SampleStandardNormal(int n, int? seed = null)
{
    var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    var result = new float[n];
    for (int i = 0; i < n; i++)
    {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        result[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
    return result;
}

sealed class Options
{
    public int Epochs { get; init; } = 10;
    public int LatentDim { get; init; } = 8;
    public int HiddenDim { get; init; } = 128;
    public int BatchSize { get; init; } = 64;
    public float LearningRate { get; init; } = 0.001f;
    public int PatternSize { get; init; } = 8;
    public int NumPatterns { get; init; } = 5000;
    public int Seed { get; init; } = 42;
    public float Beta { get; init; } = 1.0f;
    public float Dropout { get; init; } = 0.2f;
    public string? SavePath { get; init; }
    public string? LoadPath { get; init; }
    public string? SaveDataPath { get; init; }
    public string? LoadDataPath { get; init; }
    public int? GenerateCount { get; init; }
    public int? InterpolateCount { get; init; }
    public bool LatentWalk { get; init; }
    public bool ShowPatterns { get; init; }
    public bool Eval { get; init; }
    public bool Help { get; init; }

    public static Options Parse(string[] args)
    {
        int epochs = 10, latentDim = 8, hiddenDim = 128, batchSize = 64;
        int patternSize = 8, numPatterns = 5000, seed = 42;
        float lr = 0.001f, beta = 1.0f, dropout = 0.2f;
        string? savePath = null, loadPath = null, saveDataPath = null, loadDataPath = null;
        int? genCount = null, interpCount = null;
        bool latentWalk = false, showPatterns = false, eval = false, help = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--epochs" or "-e": epochs = int.Parse(args[++i]); break;
                case "--latent-dim" or "-l": latentDim = int.Parse(args[++i]); break;
                case "--hidden-dim": hiddenDim = int.Parse(args[++i]); break;
                case "--batch-size" or "-b": batchSize = int.Parse(args[++i]); break;
                case "--lr" or "-r": lr = float.Parse(args[++i]); break;
                case "--pattern-size" or "-p": patternSize = int.Parse(args[++i]); break;
                case "--num-patterns" or "-n": numPatterns = int.Parse(args[++i]); break;
                case "--seed" or "-s": seed = int.Parse(args[++i]); break;
                case "--beta": beta = float.Parse(args[++i]); break;
                case "--dropout": dropout = float.Parse(args[++i]); break;
                case "--save": savePath = args[++i]; break;
                case "--load": loadPath = args[++i]; break;
                case "--save-data": saveDataPath = args[++i]; break;
                case "--load-data": loadDataPath = args[++i]; break;
                case "--generate" or "-g": genCount = int.Parse(args[++i]); break;
                case "--interpolate" or "-i": interpCount = int.Parse(args[++i]); break;
                case "--latent-walk": latentWalk = true; break;
                case "--show-patterns": showPatterns = true; break;
                case "--eval": eval = true; break;
                case "--help" or "-h": help = true; break;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    help = true;
                    break;
            }
        }

        return new Options
        {
            Epochs = epochs, LatentDim = latentDim, HiddenDim = hiddenDim,
            BatchSize = batchSize, LearningRate = lr, PatternSize = patternSize,
            NumPatterns = numPatterns, Seed = seed, Beta = beta, Dropout = dropout,
            SavePath = savePath, LoadPath = loadPath, SaveDataPath = saveDataPath, LoadDataPath = loadDataPath, GenerateCount = genCount,
            InterpolateCount = interpCount, LatentWalk = latentWalk,
            ShowPatterns = showPatterns, Eval = eval, Help = help
        };
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraVAE - Variational Autoencoder for synthetic pattern generation

            Usage: dotnet run [options]

            Training:
              --epochs, -e <n>        Training epochs (default: 10)
              --batch-size, -b <n>    Batch size (default: 64)
              --lr, -r <f>            Learning rate (default: 0.001)
              --beta <f>              KL beta weight (default: 1.0)
              --dropout <f>           Dropout probability (default: 0.2)

            Model:
              --latent-dim, -l <n>    Latent space dimension (default: 8)
              --hidden-dim <n>        Hidden layer dimension (default: 128)

            Data:
              --pattern-size, -p <n>  Grid size per pattern (default: 8)
              --num-patterns, -n <n>  Number of training patterns (default: 5000)
              --seed, -s <n>          Random seed (default: 42)

            Modes:
              --generate, -g <n>      Generate n patterns from latent space
              --interpolate, -i <n>   Interpolate between n random pairs
              --latent-walk           Sweep each latent dimension
              --eval                  Evaluate reconstruction on test set
              --show-patterns         Display training patterns

            I/O:
              --save <path>           Save model weights to file
              --load <path>           Load model weights from file
              --save-data <path>      Save generated patterns to file
              --load-data <path>      Load patterns from file (skip generation)
              --help, -h              Show this help
            """);
    }
}
