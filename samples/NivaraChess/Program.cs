using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using NivaraChess;
using System.Globalization;

var options = Options.Parse(args);
if (options.Help)
{
    Options.PrintHelp();
    return;
}

using ChessEvalModelBase model = options.Phase switch
{
    1 => new ChessEvalModel(options.HiddenDim),
    2 => new NnueChessEvalModel(options.FeatureDim),
    _ => throw new NotSupportedException($"Phase {options.Phase} is not supported. Use 1 or 2.")
};

if (!string.IsNullOrWhiteSpace(options.LoadPath))
{
    ModelSerializer.Load(model, options.LoadPath);
    Console.WriteLine($"Loaded model: {options.LoadPath}");
}

if (options.Uci)
{
    ChessEvalConsole.RunUci(model);
    return;
}

if (!string.IsNullOrWhiteSpace(options.Fen))
{
    EvaluateFen(model, options.Fen);
    return;
}

if (options.Interactive)
{
    ChessEvalConsole.RunInteractive(model);
    return;
}

Train(model, options);

if (!string.IsNullOrWhiteSpace(options.SavePath))
{
    ModelSerializer.Save(model, options.SavePath);
    Console.WriteLine($"Saved model: {options.SavePath}");
}

Console.WriteLine();
Console.WriteLine("Sample evaluations:");
EvaluateFen(model, ChessBoard.StartingFen);

static void Train(ChessEvalModelBase model, Options options)
{
    var phaseLabel = options.Phase switch
    {
        1 => "material evaluator",
        2 => "NNUE halfKP evaluator",
        _ => throw new NotSupportedException($"Phase {options.Phase} is not supported. Use 1 or 2.")
    };

    Console.WriteLine($"NivaraChess Phase {options.Phase}: {phaseLabel}");
    Console.WriteLine($"Generating {options.NumPositions} synthetic positions...");

    var featureNames = ChessDataGenerator.GetFeatureNames(options.Phase);
    var generator = new ChessDataGenerator(options.Seed);
    var frame = generator.GenerateFrame(options.NumPositions, options.Phase);
    var dataset = new TensorDataset<float>(frame, featureNames, "score");
    var loader = new DataLoader<float>(dataset, options.BatchSize, shuffle: true, seed: options.Seed);

    var mse = new MSELoss<float>();
    var optimizer = new AdamW<float>((float)options.LearningRate);
    optimizer.AddParameterGroup(model.GetParameters().Values, (float)options.LearningRate);

    var loop = new TrainingLoop<float>(
        model,
        loader,
        (predictions, targets) => mse.Forward(predictions, targets),
        optimizer,
        options.Epochs);

    var result = loop.Run();
    result.PrintSummary();

    var validation = generator.Generate(256);
    var meanAbsoluteError = validation
        .Select(example => MathF.Abs(model.PredictCentipawns(example.Board) - example.ScoreCentipawns))
        .Average();

    Console.WriteLine();
    Console.WriteLine($"Validation MAE: {meanAbsoluteError:0.0} cp");
}

static void EvaluateFen(ChessEvalModelBase model, string fen)
{
    var resolvedFen = fen.Equals("start", StringComparison.OrdinalIgnoreCase)
        ? ChessBoard.StartingFen
        : fen;
    var board = ChessBoard.ParseFen(resolvedFen);
    model.Eval();
    Console.WriteLine(board.ToAscii());
    ChessEvalConsole.PrintEvaluation(model.PredictCentipawns(board), board.MaterialScoreCentipawns());
}

sealed class Options
{
    public int Epochs { get; private init; } = 20;
    public int BatchSize { get; private init; } = 128;
    public int HiddenDim { get; private init; } = 64;
    public int FeatureDim { get; private init; } = 256;
    public double LearningRate { get; private init; } = 0.001;
    public string? SavePath { get; private init; }
    public string? LoadPath { get; private init; }
    public string? Fen { get; private init; }
    public bool Interactive { get; private init; }
    public bool Uci { get; private init; }
    public int Phase { get; private init; } = 1;
    public int NumPositions { get; private init; } = 10000;
    public int Seed { get; private init; } = 42;
    public bool Help { get; private init; }

    public static Options Parse(string[] args)
    {
        var options = new MutableOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "--generate":
                case "--num-positions":
                    options.NumPositions = ReadInt(args, ref i, arg);
                    break;
                case "--epochs":
                    options.Epochs = ReadInt(args, ref i, arg);
                    break;
                case "--batch-size":
                    options.BatchSize = ReadInt(args, ref i, arg);
                    break;
                case "--hidden-dim":
                    options.HiddenDim = ReadInt(args, ref i, arg);
                    break;
                case "--feature-dim":
                    options.FeatureDim = ReadInt(args, ref i, arg);
                    break;
                case "--lr":
                    options.LearningRate = ReadDouble(args, ref i, arg);
                    break;
                case "--save":
                    options.SavePath = ReadString(args, ref i, arg);
                    break;
                case "--load":
                    options.LoadPath = ReadString(args, ref i, arg);
                    break;
                case "--fen":
                    options.Fen = ReadString(args, ref i, arg);
                    break;
                case "--interactive":
                    options.Interactive = true;
                    break;
                case "--uci":
                    options.Uci = true;
                    break;
                case "--phase":
                    options.Phase = ReadInt(args, ref i, arg);
                    break;
                case "--seed":
                    options.Seed = ReadInt(args, ref i, arg);
                    break;
                case "--help":
                case "-h":
                    options.Help = true;
                    break;
                case "--stockfish":
                    _ = ReadString(args, ref i, arg);
                    break;
                default:
                    throw new ArgumentException($"Unknown option '{arg}'. Use --help for usage.");
            }
        }

        return options.ToOptions();
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
NivaraChess: neural chess position evaluator

Options:
  --generate <int>      Generate N random positions for training
  --num-positions <int> Number of positions to generate (default: 10000)
  --epochs <int>        Training epochs (default: 20)
  --batch-size <int>    Batch size (default: 128)
  --phase <int>         Data generation phase: 1=material, 2=NNUE halfKP (default: 1)
  --hidden-dim <int>    Hidden layer size for phase 1 (default: 64)
  --feature-dim <int>   Feature transformer size for phase 2 (default: 256)
  --lr <float>          Learning rate (default: 0.001)
  --save <path>         Save trained model
  --load <path>         Load trained model; use the same hidden/feature dim used when saving
  --fen <fen-string>    Evaluate a single FEN position
  --interactive         Interactive FEN evaluation REPL
  --uci                 Minimal UCI evaluation mode
  --seed <int>          RNG seed (default: 42)
  --help, -h            Show this help

Examples:
  dotnet run --project samples/NivaraChess -- --phase 1 --epochs 5 --num-positions 2000 --save material.json
  dotnet run --project samples/NivaraChess -- --phase 2 --epochs 10 --feature-dim 128 --save nnue.json
  dotnet run --project samples/NivaraChess -- --load material.json --fen "start"
  dotnet run --project samples/NivaraChess -- --load nnue.json --interactive
""");
    }

    static int ReadInt(string[] args, ref int index, string option)
    {
        return int.Parse(ReadString(args, ref index, option), CultureInfo.InvariantCulture);
    }

    static double ReadDouble(string[] args, ref int index, string option)
    {
        return double.Parse(ReadString(args, ref index, option), CultureInfo.InvariantCulture);
    }

    static string ReadString(string[] args, ref int index, string option)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"{option} requires a value.");

        return args[++index];
    }

    sealed class MutableOptions
    {
        public int Epochs { get; set; } = 20;
        public int BatchSize { get; set; } = 128;
        public int HiddenDim { get; set; } = 64;
        public int FeatureDim { get; set; } = 256;
        public double LearningRate { get; set; } = 0.001;
        public string? SavePath { get; set; }
        public string? LoadPath { get; set; }
        public string? Fen { get; set; }
        public bool Interactive { get; set; }
        public bool Uci { get; set; }
        public int Phase { get; set; } = 1;
        public int NumPositions { get; set; } = 10000;
        public int Seed { get; set; } = 42;
        public bool Help { get; set; }

        public Options ToOptions()
        {
            return new Options
            {
                Epochs = Epochs,
                BatchSize = BatchSize,
                HiddenDim = HiddenDim,
                FeatureDim = FeatureDim,
                LearningRate = LearningRate,
                SavePath = SavePath,
                LoadPath = LoadPath,
                Fen = Fen,
                Interactive = Interactive,
                Uci = Uci,
                Phase = Phase,
                NumPositions = NumPositions,
                Seed = Seed,
                Help = Help
            };
        }
    }
}
