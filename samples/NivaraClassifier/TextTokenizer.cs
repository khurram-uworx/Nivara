using System.Text;
using System.Text.Json;

namespace NivaraClassifier;

public sealed class TextTokenizer
{
    public int VocabSize { get; }
    public int PadToken { get; }
    public int UnkToken { get; }
    public int BosToken { get; }
    public int EosToken { get; }

    readonly Dictionary<string, int> stoi;
    readonly Dictionary<int, string> itos;

    const string PadStr = "<PAD>";
    const string UnkStr = "<UNK>";
    const string BosStr = "<BOS>";
    const string EosStr = "<EOS>";

    TextTokenizer(Dictionary<string, int> stoi, Dictionary<int, string> itos)
    {
        this.stoi = stoi;
        this.itos = itos;
        VocabSize = stoi.Count;
        PadToken = stoi[PadStr];
        UnkToken = stoi[UnkStr];
        BosToken = stoi[BosStr];
        EosToken = stoi[EosStr];
    }

    public static TextTokenizer FromDocuments(
        IEnumerable<string> documents,
        int maxVocabSize = 10000,
        int minFreq = 1)
    {
        var freq = new Dictionary<string, int>();
        foreach (var doc in documents)
        {
            foreach (var token in Tokenize(doc))
            {
                if (!freq.TryAdd(token, 1))
                    freq[token]++;
            }
        }

        var ordered = freq
            .Where(kv => kv.Value >= minFreq)
            .OrderByDescending(kv => kv.Value)
            .Take(maxVocabSize)
            .Select(kv => kv.Key);

        var stoi = new Dictionary<string, int>();
        var itos = new Dictionary<int, string>();

        int idx = 0;
        foreach (var s in new[] { PadStr, UnkStr, BosStr, EosStr })
        {
            stoi[s] = idx;
            itos[idx] = s;
            idx++;
        }

        foreach (var word in ordered)
        {
            if (!stoi.ContainsKey(word))
            {
                stoi[word] = idx;
                itos[idx] = word;
                idx++;
            }
        }

        return new TextTokenizer(stoi, itos);
    }

    public int[] Encode(string text, int? fixedLength = null, bool addBosEos = true)
    {
        var tokens = new List<int>();

        if (addBosEos)
            tokens.Add(BosToken);

        foreach (var token in Tokenize(text))
        {
            if (stoi.TryGetValue(token, out int id))
                tokens.Add(id);
            else
                tokens.Add(UnkToken);
        }

        if (addBosEos)
            tokens.Add(EosToken);

        int[] result;
        if (fixedLength.HasValue)
        {
            int len = fixedLength.Value;
            result = new int[len];
            for (int i = 0; i < len; i++)
                result[i] = i < tokens.Count ? tokens[i] : PadToken;
        }
        else
        {
            result = tokens.ToArray();
        }

        return result;
    }

    public string Decode(ReadOnlySpan<int> tokens)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < tokens.Length; i++)
        {
            int id = tokens[i];
            if (id == PadToken) continue;
            if (id == BosToken) continue;
            if (id == EosToken) break;

            if (i > 0 && sb.Length > 0)
                sb.Append(' ');

            if (itos.TryGetValue(id, out var word))
                sb.Append(word);
            else
                sb.Append($"<{id}>");
        }
        return sb.ToString();
    }

    public void Save(string path)
    {
        var data = itos.ToDictionary(kv => kv.Key.ToString(), kv => kv.Value);
        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    public static TextTokenizer Load(string path)
    {
        var json = File.ReadAllText(path);
        var data = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
            ?? throw new InvalidOperationException("Failed to deserialize tokenizer.");

        var stoi = new Dictionary<string, int>();
        var itos = new Dictionary<int, string>();

        foreach (var kv in data)
        {
            int id = int.Parse(kv.Key);
            stoi[kv.Value] = id;
            itos[id] = kv.Value;
        }

        return new TextTokenizer(stoi, itos);
    }

    internal static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var current = new StringBuilder();

        foreach (char c in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                current.Append(c);
            }
            else
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }
        }

        if (current.Length > 0)
            tokens.Add(current.ToString());

        return tokens;
    }
}
