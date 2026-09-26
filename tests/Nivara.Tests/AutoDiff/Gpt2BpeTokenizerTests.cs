using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Verifies the GPT-2 byte-level BPE reader reproduces HuggingFace SmolLM token IDs.
/// The reference ids are produced by the real <c>AutoTokenizer</c> (tokenizer_class:
/// GPT2Tokenizer, add_prefix_space:false) for the fixed prompt used by the generation
/// fixture. Loads the locally-downloaded SmolLM vocab/merges; skipped if absent (CI/clean).
/// </summary>
[TestFixture]
public class Gpt2BpeTokenizerTests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "smollm-135m");

    static Gpt2BpeTokenizer? cachedTokenizer;

    static Gpt2BpeTokenizer Tokenizer
    {
        get
        {
            if (cachedTokenizer != null)
                return cachedTokenizer;

            var vocab = Path.Combine(ModelDir, "vocab.json");
            var merges = Path.Combine(ModelDir, "merges.txt");
            if (!File.Exists(vocab) || !File.Exists(merges))
                Assert.Ignore("SmolLM tokenizer files absent; skipping byte-level BPE verification.");

            cachedTokenizer = new Gpt2BpeTokenizer(vocab, merges);
            return cachedTokenizer;
        }
    }

    [Test]
    public void Prompt_MatchesHuggingFaceReferenceTokenIds()
    {
        var ids = Tokenizer.Encode("The capital of France is");

        // Reference from the real AutoTokenizer (GPT-2 byte-level BPE, add_prefix_space:false).
        Assert.That(ids, Is.EqualTo(new[] { 504, 3575, 282, 4649, 314 }));
    }

    [Test]
    public void EncodeDecode_RoundTripsPrompt()
    {
        const string prompt = "The capital of France is";
        var decoded = Tokenizer.Decode(Tokenizer.Encode(prompt));
        Assert.That(decoded, Is.EqualTo(prompt));
    }

    [Test]
    public void SingleToken_MatchesKnownVocabId()
    {
        // "The" (no leading space) is a single vocab entry at id 504.
        Assert.That(Tokenizer.Encode("The"), Is.EqualTo(new[] { 504 }));
    }

    [Test]
    public void VocabSize_MatchesConfig()
    {
        Assert.That(Tokenizer.VocabSize, Is.EqualTo(49152));
    }

    //  ── pre-tokenizer order (regression) ──────────────────────────────────────
    //  The GPT-2 pre-tokenizer pattern must be matched against the RAW text, and only then
    //  byte-mapped. Byte level maps a space (0x20) to U+0120, which \p{L} classifies as a
    //  letter, so matching on the mapped string makes the space join the following word:
    //  "a - b" byte-maps to "aĠ-Ġb" and then chunks as ["aĠ", "-Ġ", "b"], which can never
    //  produce the real " -" token (id 731) that HuggingFace emits. The pre-existing tests above
    //  all use letter-only text, which hides the bug, because "Ġ" followed by letters is still
    //  one all-letter chunk.

    [Test]
    public void Encode_SpaceBeforePunctuation_PretokenizesBeforeByteMapping()
    {
        // Reference from the real AutoTokenizer: ['a', ' -', ' b'].
        Assert.That(Tokenizer.Encode("a - b"), Is.EqualTo(new[] { 81, 731, 278 }));
    }

    [Test]
    public void Encode_SpaceBeforeDigitRun_MatchesHuggingFace()
    {
        // SmolLM's vocabulary has no 'Ġ2' token, so HuggingFace falls back to a bare 'Ġ'
        // followed by single digits: ['in', ' ', '2', '0', '2', '6']. ModernBERT *does* have
        // 'Ġ20', which is why the digit case shows up as ids 1384/3349 in the fixture test below.
        Assert.That(Tokenizer.Encode("in 2026"), Is.EqualTo(new[] { 254, 216, 34, 32, 34, 38 }));
    }

    [Test]
    public void Encode_MixedDigitsAndPunctuation_MatchesHuggingFace()
    {
        Assert.That(Tokenizer.Encode("in 2026, a - b (c)"),
            Is.EqualTo(new[] { 254, 216, 34, 32, 34, 38, 28, 253, 731, 278, 365, 83, 25 }));
    }

    [Test]
    public void Encode_Decode_RoundTripsTextWithDigitsAndPunctuation()
    {
        const string text = "in 2026, a - b (c)";
        Assert.That(Tokenizer.Decode(Tokenizer.Encode(text)), Is.EqualTo(text));
    }

    //  ── tokenizer.json-only load ───────────────────────────────────────────────

    static string ModernBertDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "modernbert");

    static Gpt2BpeTokenizer? cachedModernBert;

    static Gpt2BpeTokenizer ModernBertTokenizer
    {
        get
        {
            if (cachedModernBert != null)
                return cachedModernBert;

            var path = Path.Combine(ModernBertDir, "tokenizer.json");
            if (!File.Exists(path))
                Assert.Ignore("ModernBERT tokenizer absent; skipping tokenizer.json load verification.");

            cachedModernBert = Gpt2BpeTokenizer.LoadFromTokenizerJson(path, unkToken: "[UNK]");
            return cachedModernBert;
        }
    }

    [Test]
    public void TokenizerJson_LoadsVocabAndMergesInline()
    {
        // 50280 inline vocab entries plus the 116 added tokens that MergeAddedTokens folds in.
        Assert.That(ModernBertTokenizer.VocabSize, Is.EqualTo(50396));
    }

    [Test]
    public void TokenizerJson_MatchesHuggingFaceForTheParityFixtureSentence()
    {
        var ids = ModernBertTokenizer.EncodeWithSpecialTokens(
            "Nivara runs ModernBERT-large on CPU in 2026, with 28 layers and sliding-window attention.");

        // Reference from the real AutoTokenizer, and the same 26 ids the `modernbert compare`
        // mode prints: ids 1384 and 3349 are the digit-run splits that the pre-tokenizer-order
        // bug produced wrongly.
        Assert.That(ids, Is.EqualTo(new[]
        {
            50281, 47, 400, 4595, 6613, 16349, 35, 6366, 14, 16374, 327, 12874, 275, 1384, 1731,
            13, 342, 3349, 8090, 285, 20661, 14, 13686, 4116, 15, 50282,
        }));
    }

    [Test]
    public void EncodeWithSpecialTokens_WrapsWithClsAndSep()
    {
        var ids = ModernBertTokenizer.EncodeWithSpecialTokens("Nivara");

        Assert.That(ids[0], Is.EqualTo(50281));
        Assert.That(ids[^1], Is.EqualTo(50282));
        Assert.That(ids.Count, Is.EqualTo(ModernBertTokenizer.Encode("Nivara").Count + 2));
    }

    [Test]
    public void EncodeWithSpecialTokens_UnknownSpecialToken_Throws()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => ModernBertTokenizer.EncodeWithSpecialTokens("Nivara", "<NOT_A_TOKEN>"));
        Assert.That(ex!.Message, Does.Contain("not in the vocabulary"));
    }

    [Test]
    public void LoadFromTokenizerJson_WithoutModelObject_Throws()
    {
        string path = Path.Combine(Path.GetTempPath(), $"nivara-bpe-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, """{ "added_tokens": [] }""");
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(
                () => Gpt2BpeTokenizer.LoadFromTokenizerJson(path));
            Assert.That(ex!.Message, Does.Contain("top-level 'model'"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public void LoadFromTokenizerJson_NfcIsOptOut()
    {
        // "e" + combining acute is two code points pre-normalization and one after. NFD text
        // encodes to more ids when NFC is off.
        const string decomposed = "cafe\u0301";
        string path = Path.Combine(ModernBertDir, "tokenizer.json");
        if (!File.Exists(path))
            Assert.Ignore("ModernBERT tokenizer absent; skipping NFC verification.");

        var withNfc = Gpt2BpeTokenizer.LoadFromTokenizerJson(path, unkToken: "[UNK]", normalizeNfc: true);
        var withoutNfc = Gpt2BpeTokenizer.LoadFromTokenizerJson(path, unkToken: "[UNK]", normalizeNfc: false);

        Assert.That(withNfc.Encode(decomposed).Count, Is.LessThan(withoutNfc.Encode(decomposed).Count));
    }

    //  ── added tokens ──────────────────────────────────────────────────────────
    //  ModernBERT declares 23 whitespace-run added tokens (ids 50254-50276, runs of 24 down to
    //  2 spaces) plus the |||EMAIL_ADDRESS||| family and [unusedN]. HuggingFace matches these over
    //  the RAW text, leftmost-longest, before the pre-tokenizer runs: 25 spaces comes out as the
    //  24-space token followed by " b", which is only reachable by raw-text matching. Matching
    //  them per pre-tokenized piece would never find them, because the GPT-2 pattern never emits a
    //  whitespace-only piece that ends mid-run.

    [Test]
    public void Encode_WhitespaceRunAddedToken_MatchesHuggingFace()
    {
        // Reference from the real AutoTokenizer: ['a', '  ', 'b'] with 50276 = two spaces.
        Assert.That(ModernBertTokenizer.Encode("a  b"), Is.EqualTo(new[] { 66, 50276, 67 }));
    }

    [Test]
    public void Encode_WhitespaceRunLongerThanTheLongestToken_TakesTheLongestThenContinues()
    {
        // 25 spaces: 24 of them are the longest added token, the 25th starts a " b" chunk.
        var ids = ModernBertTokenizer.Encode("a" + new string(' ', 25) + "b");
        Assert.That(ids, Is.EqualTo(new[] { 66, 50254, 270 }));
    }

    [Test]
    public void Encode_PipeDelimitedAddedToken_MatchesHuggingFace()
    {
        Assert.That(ModernBertTokenizer.Encode("mail |||EMAIL_ADDRESS||| here"),
            Is.EqualTo(new[] { 5719, 209, 50277, 1060 }));
    }

    [Test]
    public void Encode_UnusedToken_MatchesHuggingFace()
    {
        Assert.That(ModernBertTokenizer.Encode("[unused1]"), Is.EqualTo(new[] { 50286 }));
    }
}
