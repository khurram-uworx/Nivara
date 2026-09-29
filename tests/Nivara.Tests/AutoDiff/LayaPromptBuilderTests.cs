using Nivara.Samples;
using NUnit.Framework;
using System.Text;
using System.Text.Json;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// A deliberately trivial byte-level tokenizer for the prompt-builder tests: the full 256-character
/// byte-level alphabet with <b>no merges</b>, so <c>Encode</c> yields exactly one id per input byte
/// and the id is that byte's value. Token counts and id values are then hand-computable, which is
/// what lets the budget, marker and truncation tests assert literal arrays instead of re-deriving
/// the algorithm they are meant to pin.
/// </summary>
static class ByteTokenizerFixture
{
    internal const int ClsId = 256;
    internal const int SepId = 257;
    internal const int MaskId = 258;

    static Gpt2BpeTokenizer? cached;

    /// <summary>The ids a piece of ASCII text encodes to, i.e. its bytes.</summary>
    internal static int[] Bytes(string text) => [.. Encoding.UTF8.GetBytes(text)];

    internal static Gpt2BpeTokenizer Tokenizer
    {
        get
        {
            if (cached != null)
                return cached;

            var vocab = new Dictionary<string, int>();
            for (int b = 0; b < 256; b++)
                vocab[ByteToChar(b)] = b;
            vocab[LayaPromptBuilder.ClsToken] = ClsId;
            vocab[LayaPromptBuilder.SepToken] = SepId;
            vocab[LayaPromptBuilder.MaskToken] = MaskId;
            vocab[LayaPromptBuilder.PadToken] = 259;
            const string unk = "<|endoftext|>";
            vocab[unk] = 260;

            var added = new[]
            {
                new { id = ClsId, content = LayaPromptBuilder.ClsToken, special = true },
                new { id = SepId, content = LayaPromptBuilder.SepToken, special = true },
                new { id = MaskId, content = LayaPromptBuilder.MaskToken, special = true },
                new { id = 259, content = LayaPromptBuilder.PadToken, special = true }
            };

            var document = new
            {
                added_tokens = added,
                model = new { vocab, merges = Array.Empty<string>() },
                pre_tokenizer = new { type = "ByteLevel", add_prefix_space = false, use_regex = false }
            };

            string dir = Path.Combine(Path.GetTempPath(), "nivara-laya-tests");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "byte-tokenizer.json");
            File.WriteAllText(path, JsonSerializer.Serialize(document));

            cached = Gpt2BpeTokenizer.LoadFromTokenizerJson(path, unk);
            return cached;
        }
    }

    /// <summary>The GPT-2 byte-to-unicode table, so the vocab covers every encodable byte.</summary>
    static string ByteToChar(int b)
    {
        var bytes = new List<int>();
        var codePoints = new List<int>();
        void AddRange(int lo, int hi)
        {
            for (int i = lo; i <= hi; i++)
            {
                bytes.Add(i);
                codePoints.Add(i);
            }
        }

        AddRange('!', '~');
        AddRange(0xA1, 0xAC);
        AddRange(0xAE, 0xFF);

        int next = 0;
        for (int i = 0; i < 256; i++)
        {
            if (bytes.Contains(i))
                continue;
            bytes.Add(i);
            codePoints.Add(256 + next++);
        }

        int index = bytes.IndexOf(b);
        return index < 0 ? ((char)b).ToString() : ((char)codePoints[index]).ToString();
    }
}

/// <summary>
/// Pins the prompt surface - option rendering and sequence construction - against the reference in
/// <c>laya/common.py</c> (PyPI <c>laya</c> 0.3.20).
/// </summary>
[TestFixture]
public class LayaPromptBuilderTests
{
    static LayaQuestion ChoiceQuestion(string id, params (string Key, string? Description)[] options)
        => new()
        {
            Id = id,
            Type = LayaQuestionType.Choice,
            Instructions = "pick",
            Options = [.. options.Select(o => new LayaChoiceOption(o.Key, o.Description))]
        };

    [Test]
    public void RenderOptions_Choice_EmitsKeyColonDescription()
    {
        var q = ChoiceQuestion("q", ("a", "first"), ("b", "second"));
        Assert.That(LayaPromptBuilder.RenderOptions(q),
            Is.EqualTo(new[] { "a: first", "b: second" }));
    }

    [Test]
    public void RenderOptions_Choice_OmitsOnlyNullAndEmptyDescriptions()
    {
        // "0" and "false" are real criterion values. The stale reference copy dropped them, which
        // silently changes the prompt for a legitimately-described option.
        var q = ChoiceQuestion("q", ("a", null), ("b", ""), ("c", "0"), ("d", "false"));
        Assert.That(LayaPromptBuilder.RenderOptions(q),
            Is.EqualTo(new[] { "a", "b", "c: 0", "d: false" }));
    }

    [Test]
    public void RenderOptions_Choice_PreservesInsertionOrder()
    {
        var q = ChoiceQuestion("q", ("z", null), ("a", null), ("m", null));
        Assert.That(LayaPromptBuilder.RenderOptions(q), Is.EqualTo(new[] { "z", "a", "m" }));
    }

    [Test]
    public void RenderOptions_Score_NumbersLevelsFromZero()
    {
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Score,
            Instructions = "how urgent",
            ScoreCriteria = ["low", "medium", "high"]
        };
        Assert.That(LayaPromptBuilder.RenderOptions(q),
            Is.EqualTo(new[] { "level 0: low", "level 1: medium", "level 2: high" }));
    }

    [Test]
    public void RenderOptions_Noul_UsesTheFallbackCriterionStrings()
    {
        // These two strings are prompt content, not decoration: dropping them changes the prompt
        // for any noul question that supplies no criterion.
        var q = new LayaQuestion { Id = "q", Type = LayaQuestionType.Noul, Instructions = "resolved" };
        Assert.That(LayaPromptBuilder.RenderOptions(q), Is.EqualTo(new[]
        {
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        }));
    }

    [Test]
    public void RenderOptions_Noul_StripsLabelWhitespaceAndKeepsFalseTrueOrder()
    {
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Noul,
            Instructions = "resolved",
            NoulFalseLabel = "  no  ",
            NoulTrueLabel = "yes",
            NoulFalseCriterion = "still broken",
            NoulTrueCriterion = "fixed"
        };
        Assert.That(LayaPromptBuilder.RenderOptions(q),
            Is.EqualTo(new[] { "no: still broken", "yes: fixed" }));
    }

    [Test]
    public void ResolveNoulLabels_RejectsLabelsThatAreNotDistinctAndNonEmpty()
    {
        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentException>(
                () => LayaPromptBuilder.ResolveNoulLabels("same", "same"), "equal labels");
            Assert.Throws<ArgumentException>(
                () => LayaPromptBuilder.ResolveNoulLabels("  ", "yes"), "blank label");
            Assert.Throws<ArgumentException>(
                () => LayaPromptBuilder.ResolveNoulLabels(null, "   "), "blank true label");
        });
    }

    [Test]
    public void RenderOptions_RejectsLabelsOnANonNoulQuestion()
    {
        var q = ChoiceQuestion("q", ("a", null)) with { NoulTrueLabel = "yes" };
        Assert.Throws<ArgumentException>(() => LayaPromptBuilder.RenderOptions(q));
    }

    [Test]
    public void BuildSequence_LaysOutTheReferenceFormat()
    {
        var q = ChoiceQuestion("q", ("a", null), ("b", "two"));
        var seq = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, "STATE", q, maxLen: 512, headMaxLen: 192);

        var head = ByteTokenizerFixture.Bytes("choice question: pick");
        int firstMarker = 1 + head.Length + 1;
        int secondMarker = firstMarker + 1 + ByteTokenizerFixture.Bytes(" a").Length;
        int afterOptions = secondMarker + 1 + ByteTokenizerFixture.Bytes(" b: two").Length;

        Assert.Multiple(() =>
        {
            Assert.That(seq.Ids.Take(head.Length + 2),
                Is.EqualTo([ByteTokenizerFixture.ClsId, .. head, ByteTokenizerFixture.SepId]),
                "[CLS] <type> question: <ins> [SEP]");
            Assert.That(seq.Markers, Is.EqualTo(new[] { firstMarker, secondMarker }));
            Assert.That(seq.Ids[firstMarker], Is.EqualTo(ByteTokenizerFixture.MaskId), "option 0 mask");
            Assert.That(seq.Ids[(firstMarker + 1)..secondMarker],
                Is.EqualTo(ByteTokenizerFixture.Bytes(" a")));
            Assert.That(seq.Ids[secondMarker], Is.EqualTo(ByteTokenizerFixture.MaskId), "option 1 mask");
            Assert.That(seq.Ids[(secondMarker + 1)..afterOptions],
                Is.EqualTo(ByteTokenizerFixture.Bytes(" b: two")));
            Assert.That(seq.Ids[afterOptions], Is.EqualTo(ByteTokenizerFixture.SepId),
                "[SEP] closes the option block");
            Assert.That(seq.Ids[(afterOptions + 1)..^1], Is.EqualTo(ByteTokenizerFixture.Bytes("STATE")));
            Assert.That(seq.Ids[^1], Is.EqualTo(ByteTokenizerFixture.SepId), "closing [SEP]");
        });
    }

    [Test]
    public void BuildSequence_StripsMaskTokensFromInstructionsAndOptions()
    {
        // A [MASK] surviving into the prompt would give the head an extra candidate position to
        // gather from, so the reference scrubs it out of both the stem and each option.
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Choice,
            Instructions = "a[MASK]b",
            Options = [new LayaChoiceOption("k", "x[MASK]y")]
        };
        var seq = LayaPromptBuilder.BuildSequence(ByteTokenizerFixture.Tokenizer, "S", q);
        var stem = ByteTokenizerFixture.Bytes("choice question: a b");
        var group = ByteTokenizerFixture.Bytes(" k: x y");
        int marker = 1 + stem.Length + 1;

        Assert.Multiple(() =>
        {
            Assert.That(seq.Ids[1..(1 + stem.Length)], Is.EqualTo(stem),
                "the literal [MASK] in the stem becomes a space");
            Assert.That(seq.Ids[marker..(marker + 1 + group.Length)],
                Is.EqualTo([ByteTokenizerFixture.MaskId, .. group]),
                "the literal [MASK] in the option becomes a space, leaving one mask per option");
            Assert.That(seq.Ids.Count(i => i == ByteTokenizerFixture.MaskId),
                Is.EqualTo(seq.Markers.Length));
        });
    }

    [Test]
    public void BuildSequence_ShrinksEveryOptionWhenTheyLeaveTooLittleHeadRoom()
    {
        // headMaxLen 64, four options of 17 tokens each = 68, so the budget goes negative and the
        // per-option shrink fires: per = max(4, (64 - 16) / 4) = 12.
        const int headMaxLen = 64;
        const int per = 12;
        var wide = new (string Key, string? Description)[4];
        for (int i = 0; i < wide.Length; i++)
            wide[i] = ($"opt{i}", "0123456789");
        var q = ChoiceQuestion("q", wide);
        var seq = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, "S", q, maxLen: 512, headMaxLen: headMaxLen);

        // The budget is recomputed from the shrunk groups, so the head keeps max(8, 64 - 48) = 16.
        int headKept = Math.Min(
            ByteTokenizerFixture.Bytes("choice question: pick").Length,
            Math.Max(8, headMaxLen - 4 * per));
        int firstMarker = 1 + headKept + 1;

        Assert.Multiple(() =>
        {
            Assert.That(seq.Markers,
                Is.EqualTo(Enumerable.Range(0, 4).Select(i => firstMarker + i * per).ToArray()),
                "options are packed back to back at a uniform width");
            foreach (int m in seq.Markers)
                Assert.That(seq.Ids[m], Is.EqualTo(ByteTokenizerFixture.MaskId), $"mask at {m}");
            Assert.That(firstMarker - headKept, Is.EqualTo(2), "[CLS] then head then [SEP]");
        });
    }

    [Test]
    public void BuildSequence_TakesTheFirstStateTokensByDefaultAndTheLastWhenTruncateLeft()
    {
        // The reference defaults truncate_left to False, so the FIRST room tokens are kept; a list
        // state (a conversation) is the exception that sets it. Copying "keep the last" as the
        // default keeps the wrong end of a long state and still runs.
        const int maxLen = 60;
        string state = new([.. Enumerable.Range(0, 100).Select(i => (char)('A' + i % 26))]);
        var q = ChoiceQuestion("q", ("a", null));

        var first = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, state, q, maxLen: maxLen);
        var last = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, state, q, maxLen: maxLen, truncateLeft: true);

        int stateStart = first.Markers[0] + 1 + ByteTokenizerFixture.Bytes(" a").Length + 1;
        int room = maxLen - stateStart - 1;
        var all = ByteTokenizerFixture.Bytes(state);

        Assert.Multiple(() =>
        {
            Assert.That(room, Is.EqualTo(32), "sanity: the state is genuinely over budget");
            Assert.That(first.Ids[stateStart..^1], Is.EqualTo(all[..room]),
                "default keeps the first room tokens");
            Assert.That(last.Ids[stateStart..^1], Is.EqualTo(all[^room..]),
                "truncateLeft keeps the last room tokens");
            Assert.That(last.Ids, Is.Not.EqualTo(first.Ids), "the two ends differ");
        });
    }

    [Test]
    public void BuildSequence_DropsAllStateTokensWhenThereIsNoRoom()
    {
        // room = max(0, maxLen - len(ids) - 1) reaches 0 once the prompt itself fills the budget.
        // Python's state[-0:] would be the whole state here, which is why the slice is written the
        // long way round.
        string state = new('a', 100);
        var q = ChoiceQuestion("q", ("a", null));
        const int maxLenUsed = 28;
        var seq = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, state, q, maxLen: maxLenUsed, headMaxLen: 192);

        // [CLS] + stem + [SEP] + [MASK] + " a" + [SEP]
        int promptLength = 1 + ByteTokenizerFixture.Bytes("choice question: pick").Length + 1
            + 1 + ByteTokenizerFixture.Bytes(" a").Length + 1;

        Assert.Multiple(() =>
        {
            Assert.That(promptLength, Is.EqualTo(maxLenUsed - 1),
                "sanity: the prompt alone already fills max_len, leaving room == 0");
            Assert.That(seq.Ids[promptLength..^1], Is.Empty, "no state token leaked in");
            Assert.That(seq.Ids[^1], Is.EqualTo(ByteTokenizerFixture.SepId), "only the closing [SEP]");
            Assert.That(seq.Length, Is.EqualTo(promptLength + 1));
        });
    }

    [Test]
    public void BuildSequence_DropsMarkersBeyondTheTruncationPoint()
    {
        // A marker past maxLen has no [MASK] left to gather. Filtering it is what makes
        // len(markers) != len(options) reachable - which is exactly the condition the reference's
        // agent rejects a question for.
        var q = ChoiceQuestion("q", ("a", null), ("b", null), ("c", null));
        var seq = LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, "S", q, maxLen: 12, headMaxLen: 192);

        Assert.Multiple(() =>
        {
            Assert.That(seq.Length, Is.EqualTo(12), "ids are capped at maxLen");
            Assert.That(seq.Markers, Is.Empty);
            Assert.That(LayaPromptBuilder.RenderOptions(q), Has.Count.EqualTo(3),
                "three options were rendered, so the counts genuinely disagree");
        });
    }

    [Test]
    public void BuildSequence_ReusesPreTokenizedStateIds()
    {
        // The reference lets a caller tokenize a shared state once and reuse it per question.
        var q = ChoiceQuestion("q", ("a", null));
        var tokenizer = ByteTokenizerFixture.Tokenizer;
        var stateIds = tokenizer.Encode("SHARED").ToArray();

        var seq = LayaPromptBuilder.BuildSequence(
            tokenizer, "ignored", q, stateIds: stateIds);

        int stateStart = seq.Markers[0] + 1 + ByteTokenizerFixture.Bytes(" a").Length + 1;
        Assert.That(seq.Ids[stateStart..], Is.EqualTo(
            [.. stateIds, ByteTokenizerFixture.SepId]),
            "the supplied ids are used, the state text is not re-tokenized");
    }

    [Test]
    public void BuildSequence_RejectsAnOptionOrderOfTheWrongLength()
    {
        var q = ChoiceQuestion("q", ("a", null), ("b", null));
        Assert.Throws<ArgumentException>(() => LayaPromptBuilder.BuildSequence(
            ByteTokenizerFixture.Tokenizer, "S", q, optionOrder: [0]));
    }
}
