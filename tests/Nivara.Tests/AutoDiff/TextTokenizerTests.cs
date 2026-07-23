using Nivara.AutoDiff.Nn;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class TextTokenizerTests
{
    private static readonly string TestDir = Path.Combine(Path.GetTempPath(), "nivara_tokenizer_tests");

    [OneTimeSetUp]
    public void Setup() => Directory.CreateDirectory(TestDir);

    [OneTimeTearDown]
    public void Cleanup()
    {
        if (Directory.Exists(TestDir))
            Directory.Delete(TestDir, recursive: true);
    }

    [Test]
    public void FromDocuments_NullDocuments_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => TextTokenizer.FromDocuments(null!));
    }

    [Test]
    public void FromDocuments_SpecialTokensAreFirstFour()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        Assert.That(tokenizer.PadToken, Is.EqualTo(0));
        Assert.That(tokenizer.UnkToken, Is.EqualTo(1));
        Assert.That(tokenizer.BosToken, Is.EqualTo(2));
        Assert.That(tokenizer.EosToken, Is.EqualTo(3));
    }

    [Test]
    public void FromDocuments_VocabSize_IncludesSpecialTokensAndWords()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world", "hello there" });

        // 4 special tokens + hello, world, there = 7
        Assert.That(tokenizer.VocabSize, Is.EqualTo(7));
    }

    [Test]
    public void FromDocuments_FrequencyOrdering_MostFrequentFirst()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] {
            "a a a b b c",
            "a a d"
        });

        // 4 special + a(5), b(2), c(1), d(1) = 8
        Assert.That(tokenizer.VocabSize, Is.EqualTo(8));

        // a should be at index 4 (after 4 special tokens) since it's most frequent
        var encoded = tokenizer.Encode("a", addBosEos: false);
        Assert.That(encoded[0], Is.EqualTo(4));
    }

    [Test]
    public void FromDocuments_MinFreq_FiltersRareWords()
    {
        var tokenizer = TextTokenizer.FromDocuments(
            new[] { "a a a b" },
            minFreq: 2);

        // 4 special + a(3) = 5. b has freq 1, filtered out.
        Assert.That(tokenizer.VocabSize, Is.EqualTo(5));
    }

    [Test]
    public void FromDocuments_MaxVocabSize_LimitsVocab()
    {
        var tokenizer = TextTokenizer.FromDocuments(
            new[] { "a b c d e f g h" },
            maxVocabSize: 6);

        // 4 special + 6 words = 10 (maxVocabSize=6 limits words to 6)
        Assert.That(tokenizer.VocabSize, Is.EqualTo(10));
    }

    [Test]
    public void Encode_NullText_ThrowsArgumentNullException()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello" });

        Assert.Throws<ArgumentNullException>(() => tokenizer.Encode(null!));
    }

    [Test]
    public void Encode_WithBosEos_AddsSpecialTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("hello world", addBosEos: true);

        Assert.That(encoded[0], Is.EqualTo(tokenizer.BosToken));
        Assert.That(encoded[^1], Is.EqualTo(tokenizer.EosToken));
    }

    [Test]
    public void Encode_WithoutBosEos_NoSpecialTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("hello world", addBosEos: false);

        for (int i = 0; i < encoded.Length; i++)
        {
            Assert.That(encoded[i], Is.Not.EqualTo(tokenizer.BosToken));
            Assert.That(encoded[i], Is.Not.EqualTo(tokenizer.EosToken));
        }
    }

    [Test]
    public void Encode_FixedLength_PadsWithPadToken()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("hello", fixedLength: 10, addBosEos: false);

        Assert.That(encoded.Length, Is.EqualTo(10));
        // First tokens should be the word, rest should be PAD
        bool foundPad = false;
        for (int i = 0; i < encoded.Length; i++)
        {
            if (encoded[i] == tokenizer.PadToken)
                foundPad = true;
            if (foundPad)
                Assert.That(encoded[i], Is.EqualTo(tokenizer.PadToken),
                    "After first PAD, all remaining should be PAD");
        }
        Assert.That(foundPad, Is.True, "Should have at least one PAD token");
    }

    [Test]
    public void Encode_UnknownWord_MapsToUnkToken()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("xyz_unknown", addBosEos: false);

        Assert.That(encoded.Length, Is.GreaterThan(0));
        Assert.That(encoded[0], Is.EqualTo(tokenizer.UnkToken));
    }

    [Test]
    public void Encode_EmptyString_ReturnsOnlySpecialTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello" });

        var encoded = tokenizer.Encode("", addBosEos: true);

        Assert.That(encoded.Length, Is.EqualTo(2));
        Assert.That(encoded[0], Is.EqualTo(tokenizer.BosToken));
        Assert.That(encoded[1], Is.EqualTo(tokenizer.EosToken));
    }

    [Test]
    public void Decode_RoundTrip_WithSpecialTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("hello world", addBosEos: true);
        var decoded = tokenizer.Decode(encoded);

        Assert.That(decoded, Is.EqualTo("hello world"));
    }

    [Test]
    public void Decode_RoundTrip_WithoutSpecialTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello world" });

        var encoded = tokenizer.Encode("hello world", addBosEos: false);
        var decoded = tokenizer.Decode(encoded);

        Assert.That(decoded, Is.EqualTo("hello world"));
    }

    [Test]
    public void Decode_SkipsPadTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello" });

        var encoded = tokenizer.Encode("hello", fixedLength: 10, addBosEos: false);
        var decoded = tokenizer.Decode(encoded);

        Assert.That(decoded, Is.EqualTo("hello"));
    }

    [Test]
    public void Decode_StopsAtEosToken()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello", "world" });

        // Manually construct: BOS hello EOS world
        var tokens = new int[] { tokenizer.BosToken, 4, tokenizer.EosToken, 5 };
        var decoded = tokenizer.Decode(tokens);

        Assert.That(decoded, Is.EqualTo("hello"));
    }

    [Test]
    public void Save_NullPath_ThrowsArgumentNullException()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello" });

        Assert.Throws<ArgumentNullException>(() => tokenizer.Save(null!));
    }

    [Test]
    public void Load_NullPath_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => TextTokenizer.Load(null!));
    }

    [Test]
    public void SaveLoad_RoundTrip()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] {
            "hello world",
            "foo bar baz",
            "hello foo"
        });
        var path = Path.Combine(TestDir, $"tokenizer_{Guid.NewGuid()}.json");

        try
        {
            tokenizer.Save(path);
            var loaded = TextTokenizer.Load(path);

            Assert.That(loaded.VocabSize, Is.EqualTo(tokenizer.VocabSize));
            Assert.That(loaded.PadToken, Is.EqualTo(tokenizer.PadToken));
            Assert.That(loaded.UnkToken, Is.EqualTo(tokenizer.UnkToken));
            Assert.That(loaded.BosToken, Is.EqualTo(tokenizer.BosToken));
            Assert.That(loaded.EosToken, Is.EqualTo(tokenizer.EosToken));

            var text = "hello world foo";
            var original = tokenizer.Encode(text, addBosEos: false);
            var restored = loaded.Encode(text, addBosEos: false);

            Assert.That(restored, Is.EqualTo(original));
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Test]
    public void Tokenize_NullText_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => TextTokenizer.Tokenize(null!));
    }

    [Test]
    public void Tokenize_Lowercases()
    {
        var tokens = TextTokenizer.Tokenize("Hello WORLD");

        Assert.That(tokens, Has.Count.EqualTo(2));
        Assert.That(tokens[0], Is.EqualTo("hello"));
        Assert.That(tokens[1], Is.EqualTo("world"));
    }

    [Test]
    public void Tokenize_SplitsOnNonAlphanumeric()
    {
        var tokens = TextTokenizer.Tokenize("hello, world! how-are you?");

        Assert.That(tokens, Is.EqualTo(new[] { "hello", "world", "how", "are", "you" }));
    }

    [Test]
    public void Tokenize_EmptyString_ReturnsEmpty()
    {
        var tokens = TextTokenizer.Tokenize("");

        Assert.That(tokens, Is.Empty);
    }

    [Test]
    public void Tokenize_OnlySpecialCharacters_ReturnsEmpty()
    {
        var tokens = TextTokenizer.Tokenize("!!! ??? ---");

        Assert.That(tokens, Is.Empty);
    }

    [Test]
    public void Tokenize_Numbers_ArePreserved()
    {
        var tokens = TextTokenizer.Tokenize("test 123 value");

        Assert.That(tokens, Is.EqualTo(new[] { "test", "123", "value" }));
    }

    [Test]
    public void Encode_MultipleWords_AllPresent()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] {
            "the cat sat on the mat"
        });

        var encoded = tokenizer.Encode("the cat sat on the mat", addBosEos: false);

        Assert.That(encoded.Length, Is.EqualTo(6));
    }

    [Test]
    public void Decode_IdentifiesUnknownTokens()
    {
        var tokenizer = TextTokenizer.FromDocuments(new[] { "hello" });

        // Token ID 999 is not in vocab
        var tokens = new int[] { tokenizer.BosToken, 4, 999, tokenizer.EosToken };
        var decoded = tokenizer.Decode(tokens);

        Assert.That(decoded, Does.Contain("<999>"));
    }
}
