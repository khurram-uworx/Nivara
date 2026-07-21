using Nivara;

namespace NivaraChess;

public sealed class ChessDataGenerator
{
    readonly Random rng;

    public ChessDataGenerator(int seed)
    {
        rng = new Random(seed);
    }

    public IReadOnlyList<ChessTrainingExample> Generate(int count, int phase = 1, StockfishEvaluator? stockfish = null)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Position count must be positive.");
        if (phase is not 1 and not 2 and not 3)
            throw new ArgumentOutOfRangeException(nameof(phase), "Phases 1, 2, and 3 are supported.");
        if (phase == 3 && stockfish == null)
            throw new ArgumentException("Stockfish evaluator is required for phase 3.", nameof(stockfish));

        var examples = new List<ChessTrainingExample>(count);
        var boards = new List<ChessBoard>(count);
        for (int i = 0; i < count; i++)
        {
            var board = GenerateMaterialPosition();
            boards.Add(board);
            var features = phase == 1
                ? board.ToFeatureVector()
                : board.ToHalfKpFeatureVector();
            examples.Add(new ChessTrainingExample(board, features, 0));
        }

        if (phase == 3)
        {
            Console.WriteLine($"  Evaluating {count} positions with Stockfish (depth {stockfish!.Depth})...");
            var scores = stockfish.EvaluateBatch(boards, (done, total) =>
            {
                if (done % 500 == 0 || done == total)
                    Console.Write($"\r  {done}/{total}");
            });
            Console.WriteLine();

            for (int i = 0; i < count; i++)
                examples[i] = examples[i] with { ScoreCentipawns = scores[i] };
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                int score = phase == 1
                    ? boards[i].MaterialScoreCentipawns()
                    : boards[i].MaterialPlusPieceSquareScoreCentipawns();
                examples[i] = examples[i] with { ScoreCentipawns = score };
            }
        }

        return examples;
    }

    public NivaraFrame GenerateFrame(int count, int phase = 1, StockfishEvaluator? stockfish = null)
    {
        return ToFrame(Generate(count, phase, stockfish), GetFeatureNames(phase));
    }

    public static string[] GetFeatureNames(int phase)
    {
        return phase switch
        {
            1 => ChessFeatures.Names,
            2 or 3 => ChessFeatures.HalfKpNames,
            _ => throw new ArgumentOutOfRangeException(nameof(phase), "Phases 1, 2, and 3 are supported.")
        };
    }

    public static NivaraFrame ToFrame(IReadOnlyList<ChessTrainingExample> examples, string[] featureNames)
    {
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentNullException.ThrowIfNull(featureNames);

        if (examples.Count == 0)
            throw new ArgumentException("At least one example is required.", nameof(examples));

        var columns = new List<(string Name, IColumn Column)>(featureNames.Length + 1);
        for (int feature = 0; feature < featureNames.Length; feature++)
        {
            var values = new float[examples.Count];
            for (int row = 0; row < examples.Count; row++)
                values[row] = examples[row].Features[feature];

            columns.Add((featureNames[feature], NivaraColumn<float>.Create(values)));
        }

        var labels = new float[examples.Count];
        for (int row = 0; row < examples.Count; row++)
            labels[row] = examples[row].ScoreCentipawns;

        columns.Add(("score", NivaraColumn<float>.Create(labels)));
        return new NivaraFrame(columns);
    }

    ChessBoard GenerateMaterialPosition()
    {
        var counts = new int[ChessFeatures.FeatureCount];

        counts[ChessFeatures.IndexOf(new ChessPiece(PieceColor.White, PieceKind.King))] = 1;
        counts[ChessFeatures.IndexOf(new ChessPiece(PieceColor.Black, PieceKind.King))] = 1;

        AddPieceCounts(counts, PieceColor.White);
        AddPieceCounts(counts, PieceColor.Black);

        var side = rng.Next(2) == 0 ? PieceColor.White : PieceColor.Black;
        return ChessBoard.FromPieceCounts(counts, side, rng);
    }

    void AddPieceCounts(int[] counts, PieceColor color)
    {
        Set(counts, color, PieceKind.Pawn, rng.Next(1, 9));
        Set(counts, color, PieceKind.Knight, rng.Next(0, 3));
        Set(counts, color, PieceKind.Bishop, rng.Next(0, 3));
        Set(counts, color, PieceKind.Rook, rng.Next(0, 3));
        Set(counts, color, PieceKind.Queen, rng.Next(0, 2));
    }

    static void Set(int[] counts, PieceColor color, PieceKind kind, int value)
    {
        counts[ChessFeatures.IndexOf(new ChessPiece(color, kind))] = value;
    }
}

public sealed record ChessTrainingExample(
    ChessBoard Board,
    float[] Features,
    float ScoreCentipawns);
