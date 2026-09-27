using System.Text;
using System.Text.Json;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Pins the two on-disk encodings of <c>model.merges</c> in a serialized <c>tokenizer.json</c>:
/// the space-separated string (<c>"a b"</c>, shipped by ModernBERT and SmolLM) and the
/// two-element array (<c>["a","b"]</c>, shipped by Laya and current <c>tokenizers</c> builds).
/// </summary>
/// <remarks>
/// Accepting only the string form does not fail loudly. A chunk that happens to be a whole
/// vocabulary entry is looked up directly and never reaches the BPE loop, so a tokenizer with zero
/// loaded merges still returns correct ids for every familiar word and silently falls back to
/// character splitting for everything else. That is precisely how the Laya tokenizer shipped
/// 50,009 merges and applied none of them.
///
/// The probe text is therefore deliberately <b>not</b> a vocabulary entry, so the merge is the only
/// route to the expected ids. <see cref="Encode_WithoutMergesSplitsTheProbeIntoCharacters"/> is the
/// control that makes the other three discriminating rather than vacuous.
/// </remarks>
[TestFixture]
public class Gpt2BpeTokenizerMergeFormatTests
{
    const string Unk = "<|endoftext|>";

    static readonly Dictionary<string, int> Vocab = new()
    {
        ["a"] = 0,
        ["b"] = 1,
        ["c"] = 2,
        ["ab"] = 3,
        ["abc"] = 4,
        [Unk] = 5
    };

    /// <summary>
    /// Writes a tokenizer over <see cref="Vocab"/> with the given <c>merges</c> array spliced in
    /// verbatim, so the caller controls the on-disk encoding rather than a serializer choosing it.
    /// </summary>
    /// <param name="mergesArrayJson">
    /// A JSON array literal, e.g. <c>["a b"]</c> or <c>[["a","b"]]</c>.
    /// </param>
    static Gpt2BpeTokenizer LoadWithRawMerges(string mergesArrayJson)
    {
        string json = "{\"model\":{\"vocab\":" + JsonSerializer.Serialize(Vocab)
            + ",\"merges\":" + mergesArrayJson
            + "},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}}";

        string dir = Path.Combine(Path.GetTempPath(), "nivara-laya-tests");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "merge-format-tokenizer.json");
        File.WriteAllText(path, json);
        return Gpt2BpeTokenizer.LoadFromTokenizerJson(path, Unk, normalizeNfc: false);
    }

    [Test]
    public void Encode_AppliesMergesWrittenAsSpaceSeparatedStrings()
    {
        var tokenizer = LoadWithRawMerges("""["a b"]""");

        // "aab" is not in the vocabulary, so the only way to reach id 3 is to merge the trailing
        // "a"+"b". With no merges loaded the ids would be [0, 0, 1].
        Assert.That(tokenizer.Encode("aab"), Is.EqualTo(new[] { 0, 3 }));
    }

    [Test]
    public void Encode_AppliesMergesWrittenAsTwoElementArrays()
    {
        var tokenizer = LoadWithRawMerges("""[["a","b"]]""");

        Assert.That(tokenizer.Encode("aab"), Is.EqualTo(new[] { 0, 3 }),
            "the array form must load identically to the string form");
    }

    [Test]
    public void Encode_WithoutMergesSplitsTheProbeIntoCharacters()
    {
        var tokenizer = LoadWithRawMerges("[]");

        Assert.That(tokenizer.Encode("aab"), Is.EqualTo(new[] { 0, 0, 1 }),
            "the control: this is what a dropped merge list actually produces");
    }

    [Test]
    public void Encode_HonoursMergeRankAndAMultiCharacterLeftToken()
    {
        var tokenizer = LoadWithRawMerges("""["a b","ab c"]""");

        // Ranks force "a"+"b" before "ab"+"c", and the second merge's left token is two characters,
        // so the space-separated form has to split on the first space only.
        Assert.That(tokenizer.Encode("cabc"), Is.EqualTo(new[] { 2, 4 }));
    }

    [Test]
    public void Encode_AppliesTheSameChainWrittenInTheArrayForm()
    {
        var tokenizer = LoadWithRawMerges("""[["a","b"],["ab","c"]]""");

        Assert.That(tokenizer.Encode("cabc"), Is.EqualTo(new[] { 2, 4 }));
    }

    [Test]
    public void Encode_LooksUpAWholeVocabularyEntryWithoutConsultingMerges()
    {
        var tokenizer = LoadWithRawMerges("[]");

        // The shortcut that makes a dropped merge list invisible: familiar words are still right.
        Assert.That(tokenizer.Encode("ab"), Is.EqualTo(new[] { 3 }));
    }
}

