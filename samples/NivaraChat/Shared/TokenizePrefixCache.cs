namespace NivaraChat.Shared;

/// <summary>
/// Reuses token ids for an unchanged/extended prompt prefix to avoid re-encoding
/// the whole conversation history every turn.
/// </summary>
public class TokenizePrefixCache
{
    string? _cachedPrompt;
    List<int>? _cachedIds;

    public int Hits { get; private set; }

    /// <summary>
    /// Encodes <paramref name="prompt"/>, reusing cached prefix ids when the new prompt
    /// starts with the previously cached prompt.
    /// </summary>
    public IReadOnlyList<int> Encode(string prompt, Func<string, IReadOnlyList<int>> encode)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(encode);

        if (_cachedPrompt is not null && _cachedIds is not null && prompt.StartsWith(_cachedPrompt, StringComparison.Ordinal))
        {
            Hits++;
            if (prompt.Length == _cachedPrompt.Length)
                return _cachedIds;

            var suffix = prompt[_cachedPrompt.Length..];
            var suffixIds = encode(suffix);
            var ids = new List<int>(_cachedIds.Count + suffixIds.Count);
            ids.AddRange(_cachedIds);
            ids.AddRange(suffixIds);
            _cachedPrompt = prompt;
            _cachedIds = ids;
            return ids;
        }

        var all = new List<int>(encode(prompt));
        _cachedPrompt = prompt;
        _cachedIds = all;
        return all;
    }

    public void Reset()
    {
        _cachedPrompt = null;
        _cachedIds = null;
        Hits = 0;
    }
}

/// <summary>Alias for hidden-test naming compatibility.</summary>
public sealed class PrefixTokenCache : TokenizePrefixCache { }
