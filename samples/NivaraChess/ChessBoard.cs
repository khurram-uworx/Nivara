namespace NivaraChess;

public enum PieceColor
{
    White,
    Black
}

public enum PieceKind
{
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King
}

public readonly record struct ChessPiece(PieceColor Color, PieceKind Kind)
{
    public char ToFenChar()
    {
        var c = Kind switch
        {
            PieceKind.Pawn => 'p',
            PieceKind.Knight => 'n',
            PieceKind.Bishop => 'b',
            PieceKind.Rook => 'r',
            PieceKind.Queen => 'q',
            PieceKind.King => 'k',
            _ => throw new InvalidOperationException($"Unsupported piece kind: {Kind}")
        };

        return Color == PieceColor.White ? char.ToUpperInvariant(c) : c;
    }

    public static bool TryParse(char value, out ChessPiece piece)
    {
        var color = char.IsUpper(value) ? PieceColor.White : PieceColor.Black;
        var kind = char.ToLowerInvariant(value) switch
        {
            'p' => PieceKind.Pawn,
            'n' => PieceKind.Knight,
            'b' => PieceKind.Bishop,
            'r' => PieceKind.Rook,
            'q' => PieceKind.Queen,
            'k' => PieceKind.King,
            _ => (PieceKind)(-1)
        };

        if (!Enum.IsDefined(kind))
        {
            piece = default;
            return false;
        }

        piece = new ChessPiece(color, kind);
        return true;
    }
}

public sealed class ChessBoard
{
    public const string StartingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    readonly ChessPiece?[] squares;

    public PieceColor SideToMove { get; }

    public ChessBoard(ChessPiece?[] squares, PieceColor sideToMove)
    {
        if (squares.Length != 64)
            throw new ArgumentException("A chess board must contain exactly 64 squares.", nameof(squares));

        this.squares = (ChessPiece?[])squares.Clone();
        SideToMove = sideToMove;
    }

    public ChessPiece? this[int rank, int file]
    {
        get
        {
            ValidateSquare(rank, file);
            return squares[rank * 8 + file];
        }
    }

    public IEnumerable<(int Square, ChessPiece Piece)> Pieces()
    {
        for (int square = 0; square < squares.Length; square++)
        {
            if (squares[square].HasValue)
                yield return (square, squares[square]!.Value);
        }
    }

    public static ChessBoard ParseFen(string fen)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fen);

        var parts = fen.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2)
            throw new FormatException("FEN must include board placement and side to move.");

        var board = new ChessPiece?[64];
        var ranks = parts[0].Split('/');
        if (ranks.Length != 8)
            throw new FormatException("FEN board placement must contain 8 ranks.");

        for (int fenRank = 0; fenRank < 8; fenRank++)
        {
            int file = 0;
            int rank = 7 - fenRank;

            foreach (var c in ranks[fenRank])
            {
                if (char.IsDigit(c))
                {
                    file += c - '0';
                    continue;
                }

                if (file >= 8 || !ChessPiece.TryParse(c, out var piece))
                    throw new FormatException($"Invalid FEN piece placement character '{c}'.");

                board[rank * 8 + file] = piece;
                file++;
            }

            if (file != 8)
                throw new FormatException("Each FEN rank must contain exactly 8 files.");
        }

        var sideToMove = parts[1] switch
        {
            "w" => PieceColor.White,
            "b" => PieceColor.Black,
            _ => throw new FormatException("FEN side to move must be 'w' or 'b'.")
        };

        return new ChessBoard(board, sideToMove);
    }

    public static ChessBoard FromPieceCounts(int[] counts, PieceColor sideToMove, Random rng)
    {
        ArgumentNullException.ThrowIfNull(counts);
        ArgumentNullException.ThrowIfNull(rng);

        if (counts.Length != ChessFeatures.FeatureCount)
            throw new ArgumentException($"Expected {ChessFeatures.FeatureCount} piece counts.", nameof(counts));

        var board = new ChessPiece?[64];
        var availableSquares = Enumerable.Range(0, 64).ToList();

        for (int feature = 0; feature < counts.Length; feature++)
        {
            var (color, kind) = ChessFeatures.DecodeFeature(feature);
            for (int i = 0; i < counts[feature]; i++)
            {
                if (availableSquares.Count == 0)
                    throw new InvalidOperationException("Cannot place more than 64 pieces.");

                int pick = rng.Next(availableSquares.Count);
                int square = availableSquares[pick];
                availableSquares.RemoveAt(pick);
                board[square] = new ChessPiece(color, kind);
            }
        }

        return new ChessBoard(board, sideToMove);
    }

    public float[] ToFeatureVector()
    {
        var features = new float[ChessFeatures.FeatureCount];
        foreach (var piece in squares)
        {
            if (piece.HasValue)
                features[ChessFeatures.IndexOf(piece.Value)]++;
        }

        return features;
    }

    public float[] ToHalfKpFeatureVector()
    {
        var active = new float[ChessFeatures.MaxActiveFeatures];
        Array.Fill(active, ChessFeatures.PaddingFeatureIndex);

        int offset = 0;
        foreach (var (square, piece) in Pieces())
        {
            if (offset >= active.Length)
                throw new InvalidOperationException(
                    $"HalfKP encoding supports at most {ChessFeatures.MaxActiveFeatures} active pieces.");

            int kingSquare = FindKingSquare(piece.Color);
            active[offset++] = ChessFeatures.HalfKpIndexOf(piece, kingSquare, square);
        }

        return active;
    }

    public int MaterialScoreCentipawns()
    {
        int score = 0;
        foreach (var piece in squares)
        {
            if (!piece.HasValue)
                continue;

            int value = ChessFeatures.PieceValue(piece.Value.Kind);
            score += piece.Value.Color == PieceColor.White ? value : -value;
        }

        return score;
    }

    public int MaterialPlusPieceSquareScoreCentipawns()
    {
        int score = 0;
        foreach (var (square, piece) in Pieces())
        {
            int value = ChessFeatures.PieceValue(piece.Kind);
            value += ChessFeatures.PieceSquareValue(piece, square);
            score += piece.Color == PieceColor.White ? value : -value;
        }

        return score;
    }

    public string ToAscii()
    {
        var lines = new List<string>();
        for (int rank = 7; rank >= 0; rank--)
        {
            var cells = new char[8];
            for (int file = 0; file < 8; file++)
                cells[file] = this[rank, file]?.ToFenChar() ?? '.';

            lines.Add($"{rank + 1}  {string.Join(' ', cells)}");
        }

        lines.Add("");
        lines.Add("   a b c d e f g h");
        lines.Add($"Side to move: {(SideToMove == PieceColor.White ? "white" : "black")}");
        return string.Join(Environment.NewLine, lines);
    }

    static void ValidateSquare(int rank, int file)
    {
        if ((uint)rank >= 8 || (uint)file >= 8)
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank and file must be in the range 0..7.");
    }

    int FindKingSquare(PieceColor color)
    {
        for (int square = 0; square < squares.Length; square++)
        {
            if (squares[square] is { Kind: PieceKind.King } piece && piece.Color == color)
                return square;
        }

        throw new InvalidOperationException($"Board does not contain a {color.ToString().ToLowerInvariant()} king.");
    }
}

public static class ChessFeatures
{
    public const int FeatureCount = 12;
    public const int MaxActiveFeatures = 32;
    public const int PaddingFeatureIndex = -1;
    public const int HalfKpFeatureCount = FeatureCount * 64 * 64;

    public static readonly string[] Names =
    [
        "white_pawns",
        "white_knights",
        "white_bishops",
        "white_rooks",
        "white_queens",
        "white_kings",
        "black_pawns",
        "black_knights",
        "black_bishops",
        "black_rooks",
        "black_queens",
        "black_kings"
    ];

    public static readonly string[] HalfKpNames =
        Enumerable.Range(0, MaxActiveFeatures)
            .Select(i => $"active_feature_{i:00}")
            .ToArray();

    public static int IndexOf(ChessPiece piece)
    {
        int offset = piece.Color == PieceColor.White ? 0 : 6;
        return offset + (int)piece.Kind;
    }

    public static int HalfKpIndexOf(ChessPiece piece, int kingSquare, int pieceSquare)
    {
        if ((uint)kingSquare >= 64)
            throw new ArgumentOutOfRangeException(nameof(kingSquare));
        if ((uint)pieceSquare >= 64)
            throw new ArgumentOutOfRangeException(nameof(pieceSquare));

        return IndexOf(piece) * 64 * 64 + kingSquare * 64 + pieceSquare;
    }

    public static (PieceColor Color, PieceKind Kind) DecodeFeature(int feature)
    {
        if ((uint)feature >= FeatureCount)
            throw new ArgumentOutOfRangeException(nameof(feature));

        return (feature < 6 ? PieceColor.White : PieceColor.Black, (PieceKind)(feature % 6));
    }

    public static int PieceValue(PieceKind kind)
    {
        return kind switch
        {
            PieceKind.Pawn => 100,
            PieceKind.Knight => 320,
            PieceKind.Bishop => 330,
            PieceKind.Rook => 500,
            PieceKind.Queen => 900,
            PieceKind.King => 20000,
            _ => throw new InvalidOperationException($"Unsupported piece kind: {kind}")
        };
    }

    public static int PieceSquareValue(ChessPiece piece, int square)
    {
        if ((uint)square >= 64)
            throw new ArgumentOutOfRangeException(nameof(square));

        int tableSquare = piece.Color == PieceColor.White ? square : MirrorSquare(square);
        return piece.Kind switch
        {
            PieceKind.Pawn => PawnTable[tableSquare],
            PieceKind.Knight => KnightTable[tableSquare],
            PieceKind.Bishop => BishopTable[tableSquare],
            PieceKind.Rook => RookTable[tableSquare],
            PieceKind.Queen => QueenTable[tableSquare],
            PieceKind.King => KingTable[tableSquare],
            _ => throw new InvalidOperationException($"Unsupported piece kind: {piece.Kind}")
        };
    }

    static int MirrorSquare(int square)
    {
        int rank = square / 8;
        int file = square % 8;
        return (7 - rank) * 8 + file;
    }

    static readonly int[] PawnTable =
    [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ];

    static readonly int[] KnightTable =
    [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ];

    static readonly int[] BishopTable =
    [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ];

    static readonly int[] RookTable =
    [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ];

    static readonly int[] QueenTable =
    [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -10, 5, 5, 5, 5, 5, 0, -10,
        0, 0, 5, 5, 5, 5, 0, -5,
        -5, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ];

    static readonly int[] KingTable =
    [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    ];
}
