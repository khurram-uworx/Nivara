using Nivara;

namespace NivaraChess;

public sealed class ChessDataGenerator
{
    readonly Random rng;

    public ChessDataGenerator(int seed)
    {
        rng = new Random(seed);
    }

    public IReadOnlyList<ChessTrainingExample> Generate(int count)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Position count must be positive.");

        var examples = new List<ChessTrainingExample>(count);
        for (int i = 0; i < count; i++)
        {
            var board = GenerateMaterialPosition();
            examples.Add(new ChessTrainingExample(
                board,
                board.ToFeatureVector(),
                board.MaterialScoreCentipawns()));
        }

        return examples;
    }

    public NivaraFrame GenerateFrame(int count)
    {
        return ToFrame(Generate(count));
    }

    public static NivaraFrame ToFrame(IReadOnlyList<ChessTrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        if (examples.Count == 0)
            throw new ArgumentException("At least one example is required.", nameof(examples));

        var columns = new List<(string Name, IColumn Column)>(ChessFeatures.FeatureCount + 1);
        for (int feature = 0; feature < ChessFeatures.FeatureCount; feature++)
        {
            var values = new float[examples.Count];
            for (int row = 0; row < examples.Count; row++)
                values[row] = examples[row].Features[feature];

            columns.Add((ChessFeatures.Names[feature], NivaraColumn<float>.Create(values)));
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
        Set(counts, color, PieceKind.Pawn, rng.Next(0, 9));
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
