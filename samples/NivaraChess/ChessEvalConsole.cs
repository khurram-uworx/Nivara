namespace NivaraChess;

public static class ChessEvalConsole
{
    public static void RunInteractive(ChessEvalModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        model.Eval();
        Console.WriteLine("NivaraChess interactive evaluator");
        Console.WriteLine("Type a FEN string, 'start', or 'quit'.");
        Console.WriteLine();

        while (true)
        {
            Console.Write("fen> ");
            var line = Console.ReadLine();
            if (line == null)
                return;

            line = line.Trim();
            if (line.Length == 0)
                continue;
            if (line.Equals("quit", StringComparison.OrdinalIgnoreCase) ||
                line.Equals("exit", StringComparison.OrdinalIgnoreCase))
                return;

            try
            {
                var fen = line.Equals("start", StringComparison.OrdinalIgnoreCase)
                    ? ChessBoard.StartingFen
                    : line;
                var board = ChessBoard.ParseFen(fen);
                Console.WriteLine(board.ToAscii());
                PrintEvaluation(model.PredictCentipawns(board), board.MaterialScoreCentipawns());
            }
            catch (Exception ex) when (ex is FormatException or ArgumentException)
            {
                Console.WriteLine($"Invalid FEN: {ex.Message}");
            }
        }
    }

    public static void RunUci(ChessEvalModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        model.Eval();
        ChessBoard? currentBoard = null;

        while (true)
        {
            var line = Console.ReadLine();
            if (line == null)
                return;

            if (line == "uci")
            {
                Console.WriteLine("id name NivaraChess");
                Console.WriteLine("id author Nivara");
                Console.WriteLine("uciok");
            }
            else if (line == "isready")
            {
                Console.WriteLine("readyok");
            }
            else if (line.StartsWith("position ", StringComparison.Ordinal))
            {
                currentBoard = ParseUciPosition(line);
            }
            else if (line.StartsWith("go", StringComparison.Ordinal))
            {
                var board = currentBoard ?? ChessBoard.ParseFen(ChessBoard.StartingFen);
                int score = (int)MathF.Round(model.PredictCentipawns(board));
                Console.WriteLine($"info score cp {score}");
                Console.WriteLine("bestmove 0000");
            }
            else if (line == "quit")
            {
                return;
            }
        }
    }

    public static void PrintEvaluation(float predicted, float target)
    {
        Console.WriteLine($"Nivara score: {predicted:+0.0;-0.0;0.0} cp");
        Console.WriteLine($"Material score: {target:+0;-0;0} cp");
        Console.WriteLine($"Error: {MathF.Abs(predicted - target):0.0} cp");
        Console.WriteLine();
    }

    static ChessBoard ParseUciPosition(string line)
    {
        const string fenPrefix = "position fen ";
        if (line == "position startpos")
            return ChessBoard.ParseFen(ChessBoard.StartingFen);

        if (!line.StartsWith(fenPrefix, StringComparison.Ordinal))
            throw new FormatException("Only 'position startpos' and 'position fen ...' are supported.");

        var fen = line[fenPrefix.Length..];
        var movesIndex = fen.IndexOf(" moves ", StringComparison.Ordinal);
        if (movesIndex >= 0)
            fen = fen[..movesIndex];

        return ChessBoard.ParseFen(fen);
    }
}
