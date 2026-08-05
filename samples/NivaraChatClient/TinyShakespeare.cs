using System.Net.Http;

namespace NivaraChatClient;

/// <summary>
/// Loads the TinyShakespeare corpus (karpathy/char-rnn), downloading it to
/// <c>samples/data/tinyshakespeare.txt</c> on first use when not already present.
/// </summary>
internal static class TinyShakespeare
{
    const string CorpusFileName = "tinyshakespeare.txt";
    const string DownloadUrl = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

    public static string Load(string? explicitPath = null)
    {
        string path = ResolvePath(explicitPath);
        if (File.Exists(path))
            return path;

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        Console.WriteLine($"Downloading TinyShakespeare to {path} ...");

        using var client = new HttpClient();
        client.DefaultRequestHeaders.UserAgent.ParseAdd("NivaraChatClient/1.0");
        string text = client.GetStringAsync(DownloadUrl).GetAwaiter().GetResult();
        File.WriteAllText(path, text);
        return path;
    }

    /// <summary>
    /// Splits the corpus into line documents suitable for
    /// <c>TextTokenizer.FromDocuments</c>.
    /// </summary>
    public static List<string> ReadDocuments(string corpusPath)
    {
        string text = File.ReadAllText(corpusPath);
        var docs = new List<string>();
        foreach (var line in text.Split('\n'))
        {
            string trimmed = line.Trim();
            if (trimmed.Length > 0)
                docs.Add(trimmed);
        }
        return docs;
    }

    static string ResolvePath(string? explicitPath)
    {
        if (!string.IsNullOrEmpty(explicitPath))
            return Path.GetFullPath(explicitPath);

        // Prefer the repo data folder (when run from repo root).
        string repoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        var candidates = new[]
        {
            Path.Combine(repoRoot, "samples", "data", CorpusFileName),
            Path.Combine(Environment.CurrentDirectory, "samples", "data", CorpusFileName),
            Path.Combine(Environment.CurrentDirectory, CorpusFileName),
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
                return candidate;
        }

        return candidates[0];
    }
}
