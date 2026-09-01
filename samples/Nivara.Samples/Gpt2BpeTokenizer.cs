using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Nivara.Samples;

/// <summary>
/// GPT-2 byte-level BPE tokenizer (as used by SmolLM's <c>GPT2Tokenizer</c>, with
/// <c>add_prefix_space:false</c>). Loads <c>vocab.json</c> + <c>merges.txt</c>, maps raw
/// bytes to printable unicode (HF <c>bytes_to_unicode</c>), pretokenizes with the GPT-2
/// regex, and applies BPE merges. This reproduces HuggingFace token IDs exactly, which the
/// Microsoft <c>BpeTokenizer</c> cannot (it has no byte-level normalizer/pretokenizer).
/// </summary>
public sealed class Gpt2BpeTokenizer
{
    static readonly Regex BytePretoken = new(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    readonly Dictionary<string, int> vocab;
    readonly Dictionary<int, string> idToToken;
    readonly Dictionary<(string, string), int> mergeRanks;
    readonly Dictionary<char, string> byteToChar;
    readonly Dictionary<string, char> charToByte;
    readonly int unkTokenId;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => vocab.Count;

    /// <summary>Gets the unknown-token id.</summary>
    public int UnknownTokenId => unkTokenId;

    /// <summary>
    /// Returns the token id for an exact vocabulary token (used for special tokens such as
    /// <c>&lt;|im_start|&gt;</c> that byte-level BPE would otherwise split). Returns -1 when the
    /// token is not in the vocabulary.
    /// </summary>
    public int TokenId(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        return vocab.TryGetValue(token, out var id) ? id : -1;
    }

    /// <summary>
    /// Loads the tokenizer from <c>vocab.json</c> and <c>merges.txt</c>.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json</param>
    /// <param name="mergesPath">Path to merges.txt</param>
    /// <param name="unkToken">Unknown-token string (must be a vocab key); defaults to &lt;|endoftext|&gt;</param>
    public Gpt2BpeTokenizer(string vocabPath, string mergesPath, string unkToken = "<|endoftext|>")
    {
        vocab = JsonVocab(File.ReadAllText(vocabPath));
        idToToken = new Dictionary<int, string>(vocab.Count);
        foreach (var (token, id) in vocab)
            idToToken[id] = token;

        if (!vocab.TryGetValue(unkToken, out unkTokenId))
            throw new InvalidOperationException($"Unknown token '{unkToken}' not present in vocab. Vocab has {vocab.Count} tokens.");

        (byteToChar, charToByte) = BuildByteMap();

        var mergesList = new List<(string, string)>();
        var lines = File.ReadAllLines(mergesPath);
        foreach (var line in lines)
        {
            if (line.Length == 0 || line[0] == '#')
                continue;
            int sp = line.IndexOf(' ');
            if (sp <= 0 || sp == line.Length - 1)
                continue;
            mergesList.Add((line.Substring(0, sp), line.Substring(sp + 1)));
        }

        mergeRanks = new Dictionary<(string, string), int>(mergesList.Count);
        for (int i = 0; i < mergesList.Count; i++)
            mergeRanks[mergesList[i]] = i;
    }

    /// <summary>Encodes text into token ids (no special-token wrapping).</summary>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var mapped = MapBytesToChars(text);
        var ids = new List<int>();
        var matches = BytePretoken.Matches(mapped);
        foreach (Match m in matches)
        {
            var word = m.Value;
            if (vocab.TryGetValue(word, out var directId))
            {
                ids.Add(directId);
                continue;
            }

            foreach (var piece in Bpe(word))
                ids.Add(vocab.TryGetValue(piece, out var id) ? id : unkTokenId);
        }
        return ids;
    }

    /// <summary>Encodes text as byte-level pieces (no special-token wrapping), for diagnostics.</summary>
    public IReadOnlyList<string> EncodePieces(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var mapped = MapBytesToChars(text);
        var result = new List<string>();
        foreach (Match m in BytePretoken.Matches(mapped))
            result.AddRange(Bpe(m.Value));
        return result;
    }

    /// <summary>Decodes token ids back to text.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (!idToToken.TryGetValue(id, out var token))
            {
                sb.Append(' ');
                continue;
            }

            foreach (var c in token)
                sb.Append(charToByte.TryGetValue(c.ToString(), out var b) ? (char)b : c);
        }
        return sb.ToString();
    }

    static Dictionary<string, int> JsonVocab(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var dict = new Dictionary<string, int>(doc.RootElement.EnumerateObject().Count());
        foreach (var prop in doc.RootElement.EnumerateObject())
            dict[prop.Name] = prop.Value.GetInt32();
        return dict;
    }

    static (Dictionary<char, string>, Dictionary<string, char>) BuildByteMap()
    {
        var bs = new List<int>();   // byte values that keep their own codepoint
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

        var bsSet = new HashSet<int>(bs);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bsSet.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var byteToChar = new Dictionary<char, string>();
        var charToByte = new Dictionary<string, char>();
        for (int i = 0; i < bs.Count; i++)
        {
            byteToChar[(char)bs[i]] = ((char)cs[i]).ToString();
            charToByte[((char)cs[i]).ToString()] = (char)bs[i];
        }
        return (byteToChar, charToByte);
    }

    /// <summary>
    /// Applies the ranked BPE merge to a single pretoken (canonical GPT-2 algorithm).
    /// </summary>
    IReadOnlyList<string> Bpe(string token)
    {
        if (token.Length == 0)
            return [];

        var word = new List<string>();
        foreach (var c in token)
            word.Add(c.ToString());

        while (word.Count > 1)
        {
            var (bestRank, bestPair) = LowestRankPair(word);
            if (bestPair == null)
                break;

            var (first, second) = bestPair.Value;
            var merged = new List<string>(word.Count);
            int i = 0;
            while (i < word.Count)
            {
                // Find next occurrence of 'first' at or after i.
                int j = IndexOf(word, first, i);
                if (j < 0)
                {
                    for (int k = i; k < word.Count; k++)
                        merged.Add(word[k]);
                    break;
                }

                for (int k = i; k < j; k++)
                    merged.Add(word[k]);

                if (j < word.Count - 1 && word[j + 1] == second)
                {
                    merged.Add(first + second);
                    i = j + 2;
                }
                else
                {
                    merged.Add(word[j]);
                    i = j + 1;
                }
            }
            word = merged;

            if (word.Count == 1)
                break;
        }

        return word;
    }

    (int Rank, (string, string)? Pair) LowestRankPair(IReadOnlyList<string> word)
    {
        int bestRank = int.MaxValue;
        (string, string)? best = null;
        for (int i = 0; i < word.Count - 1; i++)
        {
            var pair = (word[i], word[i + 1]);
            if (mergeRanks.TryGetValue(pair, out var rank) && rank < bestRank)
            {
                bestRank = rank;
                best = pair;
            }
        }
        return (bestRank, best);
    }

    static int IndexOf(IReadOnlyList<string> list, string value, int start)
    {
        for (int i = start; i < list.Count; i++)
            if (list[i] == value)
                return i;
        return -1;
    }

    string MapBytesToChars(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var sb = new StringBuilder(bytes.Length);
        foreach (var b in bytes)
            sb.Append(byteToChar.TryGetValue((char)b, out var mapped) ? mapped : ((char)b).ToString());
        return sb.ToString();
    }
}
