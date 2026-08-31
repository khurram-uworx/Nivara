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
}
