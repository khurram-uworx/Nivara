using System.Diagnostics;
using System.Text.RegularExpressions;

namespace NivaraChess;

public sealed class StockfishEvaluator : IDisposable
{
    readonly string stockfishPath;
    readonly object sync = new();
    Process? process;
    bool disposed;

    public StockfishEvaluator(string stockfishPath, int depth = 16)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stockfishPath);
        if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth));

        this.stockfishPath = stockfishPath;
        Depth = depth;

        StartStockfish();
    }

    public int Depth { get; }

    void StartStockfish()
    {
        process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = stockfishPath,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            }
        };

        process.ErrorDataReceived += (_, _) => { };
        process.Start();
        process.BeginErrorReadLine();

        Send("uci");
        WaitFor("uciok");
        Send("setoption name Threads value 1");
        Send("setoption name MultiPV value 1");
        Send("isready");
        WaitFor("readyok");
    }

    void StopStockfish()
    {
        try
        {
            if (process is { HasExited: false })
            {
                Send("quit");
                process.WaitForExit(2000);
            }
        }
        catch { }

        try { process?.CancelErrorRead(); } catch { }
        process?.Dispose();
        process = null;
    }

    public int Evaluate(ChessBoard board)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        lock (sync)
        {
            var fen = board.ToFen();
            const int maxRetries = 3;

            for (int attempt = 0; attempt < maxRetries; attempt++)
            {
                try
                {
                    if (process == null || process.HasExited)
                    {
                        Console.WriteLine($"  [INFO] Starting Stockfish (attempt {attempt + 1})...");
                        StopStockfish();
                        StartStockfish();
                    }

                    Send("ucinewgame");
                    Send("isready");
                    WaitFor("readyok");

                    Send($"position fen {fen}");
                    Send("eval");

                    int score = ParseEvalOutput();
                    return score;
                }
                catch (InvalidOperationException ex) when (attempt < maxRetries - 1)
                {
                    Console.WriteLine($"  [WARN] Stockfish issue on attempt {attempt + 1}: {ex.Message} — restarting...");
                    StopStockfish();
                }
            }

            Console.WriteLine($"  [ERROR] Stockfish failed after {maxRetries} attempts. FEN={fen}. Returning 0.");
            return 0;
        }
    }

    int ParseEvalOutput()
    {
        int score = 0;
        int linesRead = 0;
        const int maxLines = 80;

        while (linesRead < maxLines)
        {
            var line = process!.StandardOutput.ReadLine();
            if (line == null)
                break;

            linesRead++;

            if (line.Contains("Final evaluation"))
            {
                var match = Regex.Match(line, @"([+-]?\d+\.\d+)");
                if (match.Success && double.TryParse(match.Groups[1].Value,
                    System.Globalization.CultureInfo.InvariantCulture, out double val))
                {
                    score = (int)(val * 100);
                }
                break;
            }

            if (line.Contains("NNUE evaluation") && !line.Contains("NNUE derived") && !line.Contains("NNUE network"))
            {
                var match = Regex.Match(line, @"([+-]?\d+\.\d+)");
                if (match.Success && double.TryParse(match.Groups[1].Value,
                    System.Globalization.CultureInfo.InvariantCulture, out double val))
                {
                    score = (int)(val * 100);
                }
            }
        }

        return score;
    }

    public int[] EvaluateBatch(IReadOnlyList<ChessBoard> boards, Action<int, int>? progress = null)
    {
        var scores = new int[boards.Count];
        for (int i = 0; i < boards.Count; i++)
        {
            try
            {
                scores[i] = Evaluate(boards[i]);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [ERROR] Position {i + 1}/{boards.Count} failed: {ex.Message}. Score=0.");
                scores[i] = 0;
            }

            progress?.Invoke(i + 1, boards.Count);
        }
        return scores;
    }

    void Send(string command)
    {
        if (process == null || process.HasExited)
            throw new InvalidOperationException("Stockfish process is not running.");

        process.StandardInput.WriteLine(command);
        process.StandardInput.Flush();
    }

    string WaitFor(string expected)
    {
        while (true)
        {
            var line = process!.StandardOutput.ReadLine();
            if (line == null)
                throw new InvalidOperationException($"Stockfish ended while waiting for '{expected}'.");
            if (line.StartsWith(expected))
                return line;
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;

        StopStockfish();
    }
}
