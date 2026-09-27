using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Golden prompt fixtures: the exact token ids and marker positions produced by
/// <c>build_sequence</c> in <c>laya/common.py</c> (PyPI <c>laya</c> 0.3.20) for the Laya tokenizer,
/// against a fixed state and four questions covering each question type.
/// </summary>
/// <remarks>
/// The expectations were produced by the wheel itself, not by this port - <c>build_sequence</c> was
/// imported by path from the downloaded <c>laya-0.3.20-py3-none-any.whl</c> and run against
/// <c>samples/data/laya/tokenizer</c> via <c>AutoTokenizer</c>. That is the whole reason the parity
/// reference is the wheel rather than a transcription: a transcription would agree with this code
/// for the wrong reason.
/// <para>
/// Skipped when the model directory is absent, following the same convention as
/// <c>Gpt2BpeTokenizerTests</c> (the 822 MB checkpoint is gitignored, so this runs locally and on a
/// provisioned machine, not in a clean CI checkout). The synthetic-tokenizer fixtures in
/// <c>LayaPromptBuilderTests</c> always run and cover the same layout and budget logic.
/// </para>
/// </remarks>
[TestFixture]
public class LayaGoldenPromptTests
{
    const string State = "User: how do I reset my password?\nAgent: open settings, then security.";

    static Gpt2BpeTokenizer? cachedTokenizer;

    static Gpt2BpeTokenizer Tokenizer
    {
        get
        {
            if (cachedTokenizer != null)
                return cachedTokenizer;

            string path = Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..",
                "..", "..", "samples", "data", "laya", "tokenizer", "tokenizer.json");
            if (!File.Exists(path))
                Assert.Ignore("Laya tokenizer absent; skipping golden prompt verification.");

            cachedTokenizer = Gpt2BpeTokenizer.LoadFromTokenizerJson(path, "[UNK]", normalizeNfc: true);
            return cachedTokenizer;
        }
    }

    static LayaQuestion Choice2() => new()
    {
        Id = "choice2",
        Type = LayaQuestionType.Choice,
        Instructions = "Is this a login question?",
        Options = [new("reset", null), new("other", "not a password question")]
    };

    static LayaQuestion Choice13()
    {
        var options = new List<LayaChoiceOption>();
        for (int i = 0; i < 13; i++)
            options.Add(new($"opt{i}", $"topic number {i}"));
        return new LayaQuestion
        {
            Id = "choice13",
            Type = LayaQuestionType.Choice,
            Instructions = "Which topic?",
            Options = options
        };
    }

    static LayaQuestion Score4() => new()
    {
        Id = "score4",
        Type = LayaQuestionType.Score,
        Instructions = "How urgent?",
        ScoreCriteria = ["low", "medium", "high", "critical"]
    };

    static LayaQuestion Noul() => new()
    {
        Id = "noul",
        Type = LayaQuestionType.Noul,
        Instructions = "Is it resolved?",
        NoulFalseLabel = " no ",
        NoulTrueLabel = "yes",
        NoulFalseCriterion = "still broken",
        NoulTrueCriterion = "fixed"
    };

    [Test]
    public void RenderedOptions_MatchTheWheel()
    {
        var tokenizer = Tokenizer;
        Assert.That(tokenizer.VocabSize, Is.GreaterThan(0), "sanity: the tokenizer loaded");

        Assert.Multiple(() =>
        {
            Assert.That(LayaPromptBuilder.RenderOptions(Choice2()),
                Is.EqualTo(new[] { "reset", "other: not a password question" }),
                "a None description renders as the bare key");
            Assert.That(LayaPromptBuilder.RenderOptions(Score4()),
                Is.EqualTo(new[] { "level 0: low", "level 1: medium", "level 2: high", "level 3: critical" }));
            Assert.That(LayaPromptBuilder.RenderOptions(Noul()),
                Is.EqualTo(new[] { "no: still broken", "yes: fixed" }),
                "label whitespace is stripped and false/true order is fixed");
        });
    }

    [Test]
    public void ChoiceWithTwoOptions_MatchesTheWheelByteForByte()
    {
        var seq = LayaPromptBuilder.BuildSequence(Tokenizer, State, Choice2());

        Assert.Multiple(() =>
        {
            Assert.That(seq.Markers, Is.EqualTo(new[] { 11, 13 }));
            Assert.That(seq.Ids, Is.EqualTo(new[]
            {
                50281, 22122, 1953, 27, 1680, 436, 247, 16164, 1953, 32, 50282,
                50284, 14932,
                50284, 643, 27, 417, 247, 9868, 1953,
                50282,
                6989, 27, 849, 513, 309, 14932, 619, 9868, 32, 187, 28172, 27, 1527, 7533, 13, 840, 3988, 15,
                50282
            }));
        });
    }

    [Test]
    public void ChoiceWithThirteenOptions_MatchesTheWheelByteForByte()
    {
        // 13 options is the high-cardinality case: it selects the choice:11+ temperature bucket,
        // whose shipped value is out of range and gets clamped.
        var seq = LayaPromptBuilder.BuildSequence(Tokenizer, State, Choice13());

        Assert.Multiple(() =>
        {
            Assert.That(seq.Markers,
                Is.EqualTo(new[] { 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92 }));
            Assert.That(seq.Length, Is.EqualTo(119));
            Assert.That(seq.Ids, Is.EqualTo(new[]
            {
                50281, 22122, 1953, 27, 6758, 9400, 32, 50282,
                50284, 1478, 17, 27, 9400, 1180, 470,
                50284, 1478, 18, 27, 9400, 1180, 337,
                50284, 1478, 19, 27, 9400, 1180, 374,
                50284, 1478, 20, 27, 9400, 1180, 495,
                50284, 1478, 21, 27, 9400, 1180, 577,
                50284, 1478, 22, 27, 9400, 1180, 608,
                50284, 1478, 23, 27, 9400, 1180, 721,
                50284, 1478, 24, 27, 9400, 1180, 818,
                50284, 1478, 25, 27, 9400, 1180, 854,
                50284, 1478, 26, 27, 9400, 1180, 898,
                50284, 1478, 740, 27, 9400, 1180, 884,
                50284, 1478, 883, 27, 9400, 1180, 1903,
                50284, 1478, 805, 27, 9400, 1180, 1249,
                50282,
                6989, 27, 849, 513, 309, 14932, 619, 9868, 32, 187, 28172, 27, 1527, 7533, 13, 840, 3988, 15,
                50282
            }));
        });
    }

    [Test]
    public void ScoreWithFourLevels_MatchesTheWheelByteForByte()
    {
        var seq = LayaPromptBuilder.BuildSequence(Tokenizer, State, Score4());

        Assert.Multiple(() =>
        {
            Assert.That(seq.Markers, Is.EqualTo(new[] { 8, 13, 18, 23 }));
            Assert.That(seq.Ids, Is.EqualTo(new[]
            {
                50281, 18891, 1953, 27, 1359, 21007, 32, 50282,
                50284, 1268, 470, 27, 1698,
                50284, 1268, 337, 27, 4646,
                50284, 1268, 374, 27, 1029,
                50284, 1268, 495, 27, 4619,
                50282,
                6989, 27, 849, 513, 309, 14932, 619, 9868, 32, 187, 28172, 27, 1527, 7533, 13, 840, 3988, 15,
                50282
            }));
        });
    }

    [Test]
    public void NoulQuestion_MatchesTheWheelByteForByte()
    {
        var seq = LayaPromptBuilder.BuildSequence(Tokenizer, State, Noul());

        Assert.Multiple(() =>
        {
            Assert.That(seq.Markers, Is.EqualTo(new[] { 10, 15 }));
            Assert.That(seq.Ids, Is.EqualTo(new[]
            {
                50281, 79, 3941, 1953, 27, 1680, 352, 11512, 32, 50282,
                50284, 642, 27, 1335, 7154,
                50284, 4754, 27, 4229,
                50282,
                6989, 27, 849, 513, 309, 14932, 619, 9868, 32, 187, 28172, 27, 1527, 7533, 13, 840, 3988, 15,
                50282
            }));
        });
    }

    [Test]
    public void StateIsNotCappedAtModelMaxLength()
    {
        // HF warns above model_max_length (8192) but does not truncate, so a state far longer than
        // 8192 tokens must still tokenize in full and be cut only by max_len. A port that imposed
        // the tokenizer's cap would diverge here.
        string longState = string.Concat(Enumerable.Repeat("word ", 20000));
        var stateIds = Tokenizer.Encode(longState).ToArray();
        Assert.That(stateIds.Length, Is.GreaterThan(8192), "sanity: the state really is over the cap");

        var seq = LayaPromptBuilder.BuildSequence(
            Tokenizer, longState, Choice2(), stateIds: stateIds);

        Assert.That(seq.Length, Is.LessThanOrEqualTo(512), "max_len is the only bound that applies");
    }
}
