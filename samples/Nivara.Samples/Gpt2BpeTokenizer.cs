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
    readonly Dictionary<string, int> addedTokens;
    readonly HashSet<int> addedTokenIds;
    readonly int unkTokenId;

    /// <summary>
    /// When the tokenizer.json declares a <c>Split</c> pretokenizer with a regex (Qwen-style
    /// pipeline), that regex is applied to the RAW normalized text first and each resulting
    /// chunk is byte-mapped afterwards — matching HuggingFace's Split + ByteLevel(use_regex:false)
    /// composition. Null keeps the legacy GPT-2 path (regex applied to the mapped text), which
    /// is what SmolLM's <c>Gpt2BpeTokenizer(vocab, merges)</c> construction relies on.
    /// </summary>
    readonly Regex? declaredSplitRegex;

    /// <summary>Gets the total number of tokens known to this tokenizer (base vocab plus added tokens).</summary>
    public int VocabSize => vocab.Count + addedTokens.Count;

    /// <summary>Gets the unknown-token id.</summary>
    public int UnknownTokenId => unkTokenId;

    /// <summary>
    /// Returns the token id for an exact token — either a base-vocab entry or an added token
    /// (special tokens such as <c>&lt;|im_start|&gt;</c> that byte-level BPE would otherwise split).
    /// Returns -1 when the token is not known.
    /// </summary>
    public int TokenId(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        if (vocab.TryGetValue(token, out var id))
            return id;
        return addedTokens.TryGetValue(token, out id) ? id : -1;
    }

    /// <summary>
    /// Loads the tokenizer from <c>vocab.json</c> and <c>merges.txt</c>, optionally merging the
    /// added tokens declared in a <c>tokenizer.json</c> (either the Qwen-style <c>added_tokens</c>
    /// array or the HF <c>added_tokens_decoder</c> dict).
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json</param>
    /// <param name="mergesPath">Path to merges.txt</param>
    /// <param name="unkToken">Unknown-token string (must be a known token); defaults to &lt;|endoftext|&gt;</param>
    /// <param name="tokenizerJsonPath">Optional path to tokenizer.json whose added tokens are merged in</param>
    public Gpt2BpeTokenizer(
        string vocabPath,
        string mergesPath,
        string unkToken = "<|endoftext|>",
        string? tokenizerJsonPath = null)
    {
        vocab = JsonVocab(File.ReadAllText(vocabPath));
        idToToken = new Dictionary<int, string>(vocab.Count);
        foreach (var (token, id) in vocab)
            idToToken[id] = token;

        addedTokens = new Dictionary<string, int>();
        addedTokenIds = new HashSet<int>();
        if (tokenizerJsonPath != null)
        {
            var json = File.ReadAllText(tokenizerJsonPath);
            MergeAddedTokens(json);
            declaredSplitRegex = LoadSplitRegex(json);
        }

        if (!vocab.TryGetValue(unkToken, out unkTokenId) && !addedTokens.TryGetValue(unkToken, out unkTokenId))
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

    /// <summary>Encodes text into token ids (no special-token wrapping). Added tokens declared in
    /// the tokenizer are emitted as their single ids wherever they appear; everything else is
    /// byte-level BPE encoded.</summary>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var ids = new List<int>();
        int pos = 0;
        int segmentStart = 0;
        while (pos < text.Length)
        {
            if (TryMatchAddedToken(text, pos, out var token, out var id))
            {
                if (pos > segmentStart)
                    ids.AddRange(EncodeBpeSegment(text[segmentStart..pos]));
                ids.Add(id);
                pos += token.Length;
                segmentStart = pos;
            }
            else
            {
                pos++;
            }
        }
        if (segmentStart < text.Length)
            ids.AddRange(EncodeBpeSegment(text[segmentStart..]));
        return ids;
    }

    /// <summary>Encodes text as byte-level pieces (no special-token wrapping), for diagnostics.
    /// Added tokens are kept as single whole-token pieces.</summary>
    public IReadOnlyList<string> EncodePieces(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var result = new List<string>();
        int pos = 0;
        int segmentStart = 0;
        while (pos < text.Length)
        {
            if (TryMatchAddedToken(text, pos, out var token, out _))
            {
                if (pos > segmentStart)
                    result.AddRange(BpePieces(text[segmentStart..pos]));
                result.Add(token);
                pos += token.Length;
                segmentStart = pos;
            }
            else
            {
                pos++;
            }
        }
        if (segmentStart < text.Length)
            result.AddRange(BpePieces(text[segmentStart..]));
        return result;
    }

    /// <summary>Runs the byte-level BPE pipeline over a text chunk that contains no added tokens.</summary>
    IReadOnlyList<int> EncodeBpeSegment(string text)
    {
        var ids = new List<int>();

        if (declaredSplitRegex != null)
        {
            // Qwen pipeline: split the RAW normalized text with the declared regex (Isolated),
            //  byte-map each chunk, then BPE per chunk — merges never cross chunk boundaries.
            var normalized = text.Normalize(NormalizationForm.FormC);
            foreach (var chunk in PretokenizeSplit(normalized, declaredSplitRegex))
                AppendChunkIds(ids, chunk);
            return ids;
        }

        var mapped = MapBytesToChars(text);
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

    /// <summary>Runs the byte-level BPE pipeline over a text chunk that contains no added tokens,
    /// returning the BPE pieces (not ids).</summary>
    IReadOnlyList<string> BpePieces(string text)
    {
        var result = new List<string>();

        if (declaredSplitRegex != null)
        {
            var normalized = text.Normalize(NormalizationForm.FormC);
            foreach (var chunk in PretokenizeSplit(normalized, declaredSplitRegex))
                foreach (var piece in Bpe(MapBytesToChars(chunk)))
                    result.Add(piece);
            return result;
        }

        var mapped = MapBytesToChars(text);
        foreach (Match m in BytePretoken.Matches(mapped))
            result.AddRange(Bpe(m.Value));
        return result;
    }

    /// <summary>
    /// Encodes a single pretoken chunk: maps its bytes to the byte-level alphabet, emits the
    /// whole chunk when it is a known vocab entry, otherwise applies ranked BPE merges.
    /// </summary>
    void AppendChunkIds(List<int> ids, string chunk)
    {
        var mapped = MapBytesToChars(chunk);
        if (vocab.TryGetValue(mapped, out var directId))
        {
            ids.Add(directId);
            return;
        }

        foreach (var piece in Bpe(mapped))
            ids.Add(vocab.TryGetValue(piece, out var id) ? id : unkTokenId);
    }

    /// <summary>Splits raw text on a declared regex with HuggingFace "Isolated" semantics:
    /// the regex matches (and every gap between consecutive matches) each becomes a chunk.</summary>
    static IEnumerable<string> PretokenizeSplit(string text, Regex regex)
    {
        int last = 0;
        foreach (Match m in regex.Matches(text))
        {
            if (m.Index > last)
                yield return text[last..m.Index];
            if (m.Length > 0)
                yield return m.Value;
            last = m.Index + m.Length;
        }
        if (last < text.Length)
            yield return text[last..];
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

            if (addedTokenIds.Contains(id))
            {
                sb.Append(token);
                continue;
            }

            foreach (var c in token)
                sb.Append(charToByte.TryGetValue(c.ToString(), out var b) ? (char)b : c);
        }
        return sb.ToString();
    }

    /// <summary>Merges the added tokens declared in a tokenizer.json into the token↔id maps so
    /// special tokens resolve as single atomic tokens. Accepts both the Qwen <c>added_tokens</c>
    /// array form and the HF <c>added_tokens_decoder</c> dict form.</summary>
    void MergeAddedTokens(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("added_tokens", out var addedTokensEl)
            && addedTokensEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in addedTokensEl.EnumerateArray())
            {
                if (!item.TryGetProperty("id", out var idEl) || !item.TryGetProperty("content", out var contentEl))
                    continue;
                int id = idEl.GetInt32();
                string content = contentEl.GetString()!;
                AddAddedToken(content, id);
            }
            return;
        }

        if (root.TryGetProperty("added_tokens_decoder", out var decoderEl)
            && decoderEl.ValueKind == JsonValueKind.Object)
        {
            foreach (var prop in decoderEl.EnumerateObject())
            {
                if (!int.TryParse(prop.Name, out int id) || !prop.Value.TryGetProperty("content", out var contentEl))
                    continue;
                AddAddedToken(contentEl.GetString()!, id);
            }
        }
    }

    /// <summary>
    /// Extracts the <c>Split</c> pretokenizer regex from a tokenizer.json (Qwen-style
    /// <c>Sequence</c> of Split + ByteLevel), or null when none is declared (e.g. SmolLM's
    /// Digits + ByteLevel-UseRegex pipeline, which the legacy GPT-2 path already matches).
    /// </summary>
    static Regex? LoadSplitRegex(string json)
    {
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("pre_tokenizer", out var pre)
            || pre.ValueKind != JsonValueKind.Object)
            return null;

        string? pattern = null;
        if (pre.TryGetProperty("type", out var type) && type.GetString() == "Split")
            pattern = TryReadRegexPattern(pre);

        if (pattern == null && pre.TryGetProperty("pretokenizers", out var list)
            && list.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in list.EnumerateArray())
            {
                if (item.TryGetProperty("type", out var itemType) && itemType.GetString() == "Split")
                {
                    pattern = TryReadRegexPattern(item);
                    if (pattern != null)
                        break;
                }
            }
        }

        return pattern == null
            ? null
            : new Regex(pattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
    }

    static string? TryReadRegexPattern(JsonElement split)
        => split.TryGetProperty("pattern", out var pat) && pat.TryGetProperty("Regex", out var regex)
            ? regex.GetString()
            : null;

    void AddAddedToken(string content, int id)
    {
        addedTokens[content] = id;
        idToToken[id] = content;
        addedTokenIds.Add(id);
    }

    /// <summary>Finds the longest added token starting at <paramref name="pos"/>, so overlapping
    /// prefixes (e.g. <c>&lt;|im_|&gt;</c> vs <c>&lt;|im_start|&gt;</c>) resolve to the full token.</summary>
    bool TryMatchAddedToken(string text, int pos, out string token, out int id)
    {
        token = "";
        id = -1;
        int bestLength = -1;
        foreach (var (content, contentId) in addedTokens)
        {
            if (content.Length <= bestLength)
                continue;
            if (pos + content.Length <= text.Length
                && string.CompareOrdinal(text, pos, content, 0, content.Length) == 0)
            {
                bestLength = content.Length;
                token = content;
                id = contentId;
            }
        }
        return id >= 0;
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
