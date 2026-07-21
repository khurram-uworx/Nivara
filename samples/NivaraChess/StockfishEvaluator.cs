using System.Diagnostics;

namespace NivaraChess;

public sealed class StockfishEvaluator : IDisposable
{
    readonly Process process;
    readonly object sync = new();
    bool disposed;

    public StockfishEvaluator(string stockfishPath, int depth = 16)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stockfishPath);
        if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth));

        Depth = depth;

        process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = stockfishPath,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            }
        };

        process.Start();
        Send("uci");
        WaitFor("uciok");
    }

    public int Depth { get; }

    public int Evaluate(ChessBoard board)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        lock (sync)
        {
            Send($"position fen {board.ToFen()}");
            Send($"go depth {Depth}");

            int score = 0;
            while (true)
            {
                var line = process.StandardOutput.ReadLine();
                if (line == null)
                    throw new InvalidOperationException("Stockfish process ended unexpectedly.");

                if (line.StartsWith("info") && line.Contains("score cp"))
                    score = ParseInfoScoreCp(line);

                if (line.StartsWith("bestmove"))
                    break;
            }

            return score;
        }
    }

    public int[] EvaluateBatch(IReadOnlyList<ChessBoard> boards, Action<int, int>? progress = null)
    {
        var scores = new int[boards.Count];
        for (int i = 0; i < boards.Count; i++)
        {
            scores[i] = Evaluate(boards[i]);
            progress?.Invoke(i + 1, boards.Count);
        }
        return scores;
    }

    void Send(string command)
    {
        process.StandardInput.WriteLine(command);
        process.StandardInput.Flush();
    }

    string WaitFor(string expected)
    {
        while (true)
        {
            var line = process.StandardOutput.ReadLine();
            if (line == null)
                throw new InvalidOperationException("Stockfish process ended unexpectedly.");
            if (line.StartsWith(expected))
                return line;
        }
    }

    static int ParseInfoScoreCp(string infoLine)
    {
        // info score cp -35 depth 16 ...
        var parts = infoLine.Split(' ');
        for (int i = 0; i < parts.Length - 1; i++)
        {
            if (parts[i] == "score" && parts[i + 1] == "cp")
            {
                if (int.TryParse(parts[i + 2], out int cp))
                    return cp;
            }
        }
        return 0;
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;

        try
        {
            Send("quit");
            process.WaitForExit(2000);
        }
        catch { }

        process.Dispose();
    }
}
