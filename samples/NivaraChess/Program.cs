using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using NivaraChess;
using System.Globalization;
using System.Numerics.Tensors;

var options = Options.Parse(args);
if (options.Help)
{
    Options.PrintHelp();
    return;
}

using ChessEvalModelBase model = options.Phase switch
{
    1 => new ChessEvalModel(options.HiddenDim),
    2 or 3 => new NnueChessEvalModel(options.FeatureDim),
    _ => throw new NotSupportedException($"Phase {options.Phase} is not supported. Use 1, 2, or 3.")
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

if (options.Embed)
{
    RunEmbeddingDemo(model);
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
        3 => "NNUE with Stockfish labels",
        _ => throw new NotSupportedException($"Phase {options.Phase} is not supported. Use 1, 2, or 3.")
    };

    Console.WriteLine($"NivaraChess Phase {options.Phase}: {phaseLabel}");

    StockfishEvaluator? stockfish = null;
    if (options.Phase == 3)
    {
        if (string.IsNullOrWhiteSpace(options.StockfishPath))
            throw new ArgumentException("--stockfish <path> is required for phase 3.");

        Console.WriteLine($"Connecting to Stockfish at: {options.StockfishPath}");
        stockfish = new StockfishEvaluator(options.StockfishPath);
    }

    try
    {
        Console.WriteLine($"Generating {options.NumPositions} synthetic positions...");

        var featureNames = ChessDataGenerator.GetFeatureNames(options.Phase);
        var generator = new ChessDataGenerator(options.Seed);
        var frame = generator.GenerateFrame(options.NumPositions, options.Phase, stockfish);
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

        var validation = generator.Generate(256, options.Phase, stockfish);
        var meanAbsoluteError = validation
            .Select(example => MathF.Abs(model.PredictCentipawns(example.Board) - example.ScoreCentipawns))
            .Average();

        Console.WriteLine();
        Console.WriteLine($"Validation MAE: {meanAbsoluteError:0.0} cp");
    }
    finally
    {
        stockfish?.Dispose();
    }
}

static void EvaluateFen(ChessEvalModelBase model, string fen)
{
    var resolvedFen = fen.Equals("start", StringComparison.OrdinalIgnoreCase)
        ? ChessBoard.StartingFen
        : fen;
    var board = ChessBoard.ParseFen(resolvedFen);
    model.Eval();
    Console.WriteLine(board.ToAscii());
    var target = ChessEvalConsole.TargetValue(board, model.Phase);
    ChessEvalConsole.PrintEvaluation(model.PredictCentipawns(board), target, ChessEvalConsole.TargetLabel(model.Phase));
}

static void RunEmbeddingDemo(ChessEvalModelBase model)
{
    using var generator = new ChessEmbeddingGenerator(model);

    var positions = new[]
    {
        ("Starting position", ChessBoard.ParseFen(ChessBoard.StartingFen)),
        ("Sicilian Defense", ChessBoard.ParseFen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")),
        ("King's Gambit", ChessBoard.ParseFen("rnbqkbnr/pppp1ppp/8/4P3/5p2/8/PPPP1PPP/RNBQKBNR w KQkq f3 0 3")),
        ("Caro-Kann", ChessBoard.ParseFen("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")),
        ("Endgame", ChessBoard.ParseFen("8/5pk1/5p1p/8/8/4B3/5PPP/6K1 w - - 0 1")),
    };

    Console.WriteLine($"NivaraChess embedding demo (Phase {model.Phase}, dim={generator.EmbeddingDimension})");
    Console.WriteLine();

    var boards = positions.Select(p => p.Item2).ToList();
    var embeddings = generator.GenerateAsync(boards).GetAwaiter().GetResult();

    for (int i = 0; i < positions.Length; i++)
    {
        var (name, board) = positions[i];
        var embedding = embeddings[i].Vector;
        Console.WriteLine($"  {name}");
        Console.WriteLine($"    FEN: {board.ToFen()}");
        Console.WriteLine($"    Eval: {model.PredictCentipawns(board):+0.0;-0.0;0.0} cp");
        Console.WriteLine($"    Embedding: [{string.Join(", ", embedding.ToArray().Select(v => v.ToString("0.000")))}]");
        Console.WriteLine();
    }

    // Compute pairwise cosine similarities
    Console.WriteLine("Pairwise cosine similarities:");
    for (int i = 0; i < positions.Length; i++)
    {
        for (int j = i + 1; j < positions.Length; j++)
        {
            float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Vector.Span, embeddings[j].Vector.Span);
            Console.WriteLine($"  {positions[i].Item1} <-> {positions[j].Item1}: {sim:0.000}");
        }
    }
}

sealed class Options
{
    public int Epochs { get; private init; } = 20;
    public int BatchSize { get; private init; } = 128;
    public int HiddenDim { get; private init; } = 64;
    public int FeatureDim { get; private init; } = 256;
    public double LearningRate { get; private init; } = 0.001;
    public string? StockfishPath { get; private init; }
    public string? SavePath { get; private init; }
    public string? LoadPath { get; private init; }
    public string? Fen { get; private init; }
    public bool Interactive { get; private init; }
    public bool Embed { get; private init; }
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
                case "--embed":
                    options.Embed = true;
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
                    options.StockfishPath = ReadString(args, ref i, arg);
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
  --phase <int>         Data generation phase: 1=material, 2=NNUE halfKP, 3=Stockfish (default: 1)
  --stockfish <path>    Path to Stockfish executable (required for phase 3)
  --hidden-dim <int>    Hidden layer size for phase 1 (default: 64)
  --feature-dim <int>   Feature transformer size for phase 2 (default: 256)
  --lr <float>          Learning rate (default: 0.001)
  --save <path>         Save trained model
  --load <path>         Load trained model; use the same hidden/feature dim used when saving
  --fen <fen-string>    Evaluate a single FEN position
  --interactive         Interactive FEN evaluation REPL
  --embed                Demo the IEmbeddingGenerator (compute + compare position embeddings)
  --uci                 Minimal UCI evaluation mode
  --seed <int>          RNG seed (default: 42)
  --help, -h            Show this help

Examples:
  dotnet run --project samples/NivaraChess -- --phase 1 --epochs 5 --num-positions 2000 --save material.json
  dotnet run --project samples/NivaraChess -- --phase 2 --epochs 10 --feature-dim 128 --save nnue.json
  dotnet run --project samples/NivaraChess -- --phase 3 --epochs 5 --stockfish C:\bin\stockfish\stockfish-windows-x86-64-avx2.exe --save stockfish.json
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
        public string? StockfishPath { get; set; }
        public string? SavePath { get; set; }
        public string? LoadPath { get; set; }
        public string? Fen { get; set; }
        public bool Interactive { get; set; }
        public bool Embed { get; set; }
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
                StockfishPath = StockfishPath,
                SavePath = SavePath,
                LoadPath = LoadPath,
                Fen = Fen,
                Interactive = Interactive,
                Embed = Embed,
                Uci = Uci,
                Phase = Phase,
                NumPositions = NumPositions,
                Seed = Seed,
                Help = Help
            };
        }
    }
}
