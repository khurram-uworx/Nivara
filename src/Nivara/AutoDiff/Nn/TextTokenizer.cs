using System.Text;
using System.Text.Json;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Simple word-level tokenizer built from a corpus. Maps tokens to integer IDs with reserved
/// padding, unknown, begin-of-sequence, and end-of-sequence tokens; supports encode/decode,
/// fixed-length padding, and JSON save/load.
/// </summary>
public sealed class TextTokenizer
{
    /// <summary>Gets the number of tokens in the vocabulary (including special tokens).</summary>
    public int VocabSize { get; }
    /// <summary>Gets the reserved padding token ID.</summary>
    public int PadToken { get; }
    /// <summary>Gets the reserved unknown-token ID.</summary>
    public int UnkToken { get; }
    /// <summary>Gets the reserved begin-of-sequence token ID.</summary>
    public int BosToken { get; }
    /// <summary>Gets the reserved end-of-sequence token ID.</summary>
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

    /// <summary>
    /// Builds a tokenizer from a corpus, keeping the most frequent tokens up to
    /// <paramref name="maxVocabSize"/>, filtered by a minimum frequency.
    /// </summary>
    /// <param name="documents">The documents to learn the vocabulary from</param>
    /// <param name="maxVocabSize">Maximum vocabulary size (excluding the four special tokens)</param>
    /// <param name="minFreq">Minimum token frequency to be kept</param>
    /// <returns>The trained tokenizer</returns>
    public static TextTokenizer FromDocuments(
        IEnumerable<string> documents,
        int maxVocabSize = 10000,
        int minFreq = 1)
    {
        ArgumentNullException.ThrowIfNull(documents);

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

    /// <summary>
    /// Encodes text into token IDs, optionally wrapping with BOS/EOS and padding to a fixed length.
    /// </summary>
    /// <param name="text">The text to encode</param>
    /// <param name="fixedLength">When set, pads (or truncates) the result to this length</param>
    /// <param name="addBosEos">Whether to prepend BOS and append EOS</param>
    /// <returns>The token ID sequence</returns>
    public int[] Encode(string text, int? fixedLength = null, bool addBosEos = true)
    {
        ArgumentNullException.ThrowIfNull(text);

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

    /// <summary>
    /// Decodes token IDs back into text, skipping padding and BOS tokens and stopping at EOS.
    /// Unknown IDs are rendered as <c>&lt;id&gt;</c>.
    /// </summary>
    /// <param name="tokens">The token ID sequence</param>
    /// <returns>The decoded text</returns>
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

    /// <summary>
    /// Saves the tokenizer vocabulary to a JSON file.
    /// </summary>
    /// <param name="path">The destination file path</param>
    public void Save(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        var data = itos.ToDictionary(kv => kv.Key.ToString(), kv => kv.Value);
        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Loads a tokenizer previously saved with <see cref="Save"/>.
    /// </summary>
    /// <param name="path">The JSON file path</param>
    /// <returns>The loaded tokenizer</returns>
    public static TextTokenizer Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

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

    /// <summary>
    /// Splits lower-cased text into word tokens on any non-letter/non-digit character.
    /// </summary>
    /// <param name="text">The text to tokenize</param>
    /// <returns>The word tokens</returns>
    public static List<string> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

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
