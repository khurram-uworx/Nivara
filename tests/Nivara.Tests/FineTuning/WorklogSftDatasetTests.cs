using NivaraFineTuning;
using NUnit.Framework;
using System.Globalization;
using System.Text.Json;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class WorklogSftDatasetTests
{
    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "nivara-worklog-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, recursive: true);
    }

    [Test]
    public void GenerateSynthetic_SameSeedAndInputs_IsDeterministic()
    {
        var a = WorklogSftDataset.GenerateSynthetic(40, seed: 123, baseDate: new DateTime(2026, 7, 19));
        var b = WorklogSftDataset.GenerateSynthetic(40, seed: 123, baseDate: new DateTime(2026, 7, 19));

        Assert.That(JsonSerializer.Serialize(a), Is.EqualTo(JsonSerializer.Serialize(b)));
    }

    [Test]
    public void GenerateSynthetic_DifferentSeed_ProducesDifferentExamples()
    {
        var a = WorklogSftDataset.GenerateSynthetic(20, seed: 1, baseDate: new DateTime(2026, 7, 19));
        var b = WorklogSftDataset.GenerateSynthetic(20, seed: 2, baseDate: new DateTime(2026, 7, 19));

        Assert.That(JsonSerializer.Serialize(a), Is.Not.EqualTo(JsonSerializer.Serialize(b)));
    }

    [Test]
    public void GenerateSynthetic_CoversRequiredCategories()
    {
        var examples = WorklogSftDataset.GenerateSynthetic(200, seed: 42, baseDate: new DateTime(2026, 7, 19));

        bool hasAllDay = examples.Any(e => UserText(e).Contains("all day", StringComparison.OrdinalIgnoreCase) || UserText(e).Contains("full day", StringComparison.OrdinalIgnoreCase));
        bool hasHalfDay = examples.Any(e => UserText(e).Contains("half day", StringComparison.OrdinalIgnoreCase) || UserText(e).Contains("half-day", StringComparison.OrdinalIgnoreCase));
        bool hasMultiTicket = examples.Any(e => WorklogSftEvaluator.ParseLogTimeCalls(AssistantText(e)).Count > 1);
        bool hasClarification = examples.Any(e => WorklogSftEvaluator.IsClarification(AssistantText(e)));
        bool hasRelativeDate = examples.Any(e => UserText(e).StartsWith("Today", StringComparison.OrdinalIgnoreCase)
            || UserText(e).StartsWith("Yesterday", StringComparison.OrdinalIgnoreCase)
            || UserText(e).StartsWith("Tomorrow", StringComparison.OrdinalIgnoreCase));
        bool hasExplicitDecimal = examples.Any(e => UserText(e).Contains("minutes", StringComparison.OrdinalIgnoreCase) || UserText(e).Contains("2.5 hours", StringComparison.OrdinalIgnoreCase) || UserText(e).Contains("1.5 hours", StringComparison.OrdinalIgnoreCase));

        Assert.That(hasAllDay, Is.True, "expected all-day/full-day examples");
        Assert.That(hasHalfDay, Is.True, "expected half-day examples");
        Assert.That(hasMultiTicket, Is.True, "expected multi-ticket examples");
        Assert.That(hasClarification, Is.True, "expected clarification examples");
        Assert.That(hasRelativeDate, Is.True, "expected relative-date examples");
        Assert.That(hasExplicitDecimal, Is.True, "expected explicit decimal/minute examples");
    }

    [Test]
    public void GenerateSynthetic_ProducesValidSchemaAndResolvedIsoDates()
    {
        var baseDate = new DateTime(2026, 7, 19);
        var examples = WorklogSftDataset.GenerateSynthetic(120, seed: 7, baseDate: baseDate);

        foreach (var ex in examples)
        {
            Assert.That(ex.Messages, Is.Not.Null);
            Assert.That(ex.Messages.Select(m => m.Role), Is.EqualTo(new[] { "system", "user", "assistant" }));
            Assert.That(ex.Messages.All(m => !string.IsNullOrWhiteSpace(m.Content)), Is.True);

            var assistant = AssistantText(ex);
            var calls = WorklogSftEvaluator.ParseLogTimeCalls(assistant);
            if (calls.Count == 0)
            {
                Assert.That(WorklogSftEvaluator.IsClarification(assistant), Is.True, "Non-tool-call assistant message must be clarification");
                continue;
            }

            foreach (var call in calls)
            {
                Assert.That(call.Ticket, Is.Not.Empty);
                Assert.That(call.Hours, Is.GreaterThan(0m));
                Assert.That(DateTime.TryParseExact(call.Date, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var parsed), Is.True, $"Invalid ISO date: {call.Date}");
                Assert.That(parsed.Year, Is.EqualTo(2026));
            }
        }
    }

    [Test]
    public void GenerateSynthetic_RelativeDates_ResolveFromBaseDate()
    {
        var baseDate = new DateTime(2026, 7, 19);
        var examples = WorklogSftDataset.GenerateSynthetic(200, seed: 99, baseDate: baseDate);
        var relative = examples.Where(e => UserText(e).StartsWith("Today", StringComparison.OrdinalIgnoreCase)
            || UserText(e).StartsWith("Yesterday", StringComparison.OrdinalIgnoreCase)
            || UserText(e).StartsWith("Tomorrow", StringComparison.OrdinalIgnoreCase)).ToList();

        Assert.That(relative, Is.Not.Empty);
        foreach (var ex in relative)
        {
            var user = UserText(ex);
            var calls = WorklogSftEvaluator.ParseLogTimeCalls(AssistantText(ex));
            Assert.That(calls, Is.Not.Empty);
            var expected = user.StartsWith("Today", StringComparison.OrdinalIgnoreCase) ? baseDate
                : user.StartsWith("Yesterday", StringComparison.OrdinalIgnoreCase) ? baseDate.AddDays(-1)
                : baseDate.AddDays(1);
            Assert.That(calls.All(c => c.Date == expected.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)), Is.True, user);
        }
    }

    [Test]
    public void MinutesToDecimalHours_UsesDeterministicFractionRule()
    {
        Assert.That(WorklogSftDataset.MinutesToDecimalHours(20), Is.EqualTo(0.33m));
        Assert.That(WorklogSftDataset.MinutesToDecimalHours(30), Is.EqualTo(0.5m));
        Assert.That(WorklogSftDataset.MinutesToDecimalHours(45), Is.EqualTo(0.75m));
    }

    [Test]
    public void SplitDeterministic_SameInputs_ProducesStableSplitAndHash()
    {
        var examples = WorklogSftDataset.GenerateSynthetic(60, seed: 5, baseDate: new DateTime(2026, 7, 19));

        var a = WorklogSftDataset.SplitDeterministic(examples, seed: 42, trainRatio: 0.7, valRatio: 0.15);
        var b = WorklogSftDataset.SplitDeterministic(examples, seed: 42, trainRatio: 0.7, valRatio: 0.15);

        Assert.That(a.Train.Count, Is.EqualTo(b.Train.Count));
        Assert.That(a.Validation.Count, Is.EqualTo(b.Validation.Count));
        Assert.That(a.Test.Count, Is.EqualTo(b.Test.Count));
        Assert.That(a.ProvenanceHash, Is.EqualTo(b.ProvenanceHash));
        Assert.That(JsonSerializer.Serialize(a.Train.Select(AssistantText).ToList()), Is.EqualTo(JsonSerializer.Serialize(b.Train.Select(AssistantText).ToList())));
    }

    [Test]
    public void SplitDeterministic_ValRatioZero_ProducesNoValidation()
    {
        var examples = WorklogSftDataset.GenerateSynthetic(30, seed: 9);
        var split = WorklogSftDataset.SplitDeterministic(examples, seed: 1, trainRatio: 0.8, valRatio: 0.0);

        Assert.That(split.Validation, Is.Empty);
        Assert.That(split.Train.Count + split.Test.Count, Is.EqualTo(examples.Count));
    }

    [Test]
    public void SplitDeterministic_PartitionsWithoutOverlap()
    {
        var examples = WorklogSftDataset.GenerateSynthetic(55, seed: 11, baseDate: new DateTime(2026, 7, 19));
        var split = WorklogSftDataset.SplitDeterministic(examples, seed: 3, trainRatio: 0.8, valRatio: 0.1);

        Assert.That(split.Train.Count + split.Validation.Count + split.Test.Count, Is.EqualTo(examples.Count));

        var all = split.Train.Concat(split.Validation).Concat(split.Test).Select(ex => JsonSerializer.Serialize(ex)).ToList();
        var source = examples.Select(ex => JsonSerializer.Serialize(ex)).ToList();
        Assert.That(all.Count, Is.EqualTo(source.Count));
        var sourceCounts = source.GroupBy(x => x, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
        var allCounts = all.GroupBy(x => x, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
        Assert.That(allCounts.Count, Is.EqualTo(sourceCounts.Count));
        foreach (var kv in sourceCounts)
        {
            Assert.That(allCounts.TryGetValue(kv.Key, out var count), Is.True);
            Assert.That(count, Is.EqualTo(kv.Value));
        }
    }

    [Test]
    public void SaveJsonl_ExistingFileWithoutForce_Throws()
    {
        var path = Path.Combine(tempDir, "examples.jsonl");
        var examples = WorklogSftDataset.GenerateSynthetic(2, seed: 1);
        WorklogSftDataset.SaveJsonl(path, examples, force: false);

        var ex = Assert.Throws<InvalidOperationException>(() => WorklogSftDataset.SaveJsonl(path, examples, force: false));
        Assert.That(ex!.Message, Does.Contain("--force"));
    }

    [Test]
    public void SaveJsonl_WithForce_OverwritesAndRoundTrips()
    {
        var path = Path.Combine(tempDir, "examples.jsonl");
        var first = WorklogSftDataset.GenerateSynthetic(2, seed: 1);
        var second = WorklogSftDataset.GenerateSynthetic(3, seed: 2);
        WorklogSftDataset.SaveJsonl(path, first, force: false);
        WorklogSftDataset.SaveJsonl(path, second, force: true);

        var loaded = WorklogSftDataset.LoadJsonl(path);
        Assert.That(loaded.Count, Is.EqualTo(3));
        Assert.That(JsonSerializer.Serialize(loaded), Is.EqualTo(JsonSerializer.Serialize(second)));
    }

    [Test]
    public void TokenizeOne_MasksPromptAndKeepsAnswerLabels()
    {
        var tokenizer = CreateByteLevelTokenizer();
        var example = WorklogSftDataset.GenerateSynthetic(1, seed: 42).Single();
        var tokenized = WorklogSftDataset.TokenizeOne(example, tokenizer, "qwen", maxLen: 512);

        Assert.That(tokenized, Is.Not.Null);
        Assert.That(tokenized!.PromptTokenCount, Is.GreaterThan(0));
        Assert.That(tokenized.AnswerTokenCount, Is.GreaterThan(0));
        Assert.That(tokenized.InputIds.Length, Is.EqualTo(tokenized.Labels.Length));
        Assert.That(tokenized.Labels.Take(tokenized.PromptTokenCount), Is.All.EqualTo(-100));
        Assert.That(tokenized.Labels.Skip(tokenized.PromptTokenCount).Any(l => l != -100), Is.True);
    }

    static string UserText(WorklogSftExample ex) => ex.Messages.Last(m => m.Role == "user").Content;
    static string AssistantText(WorklogSftExample ex) => ex.Messages.Last(m => m.Role == "assistant").Content;

    Nivara.AI.Gpt2BpeTokenizer CreateByteLevelTokenizer()
    {
        var vocab = BuildByteLevelBaseVocab();
        // Add common tokens/specials used by Qwen template rendering.
        string[] specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "user", "assistant", "tool"];
        foreach (var token in specials)
        {
            if (!vocab.ContainsKey(token))
                vocab[token] = vocab.Count;
        }

        string json = "{\"model\":{\"vocab\":" + JsonSerializer.Serialize(vocab)
            + ",\"merges\":[]},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false},\"added_tokens\":[]}";
        var path = Path.Combine(tempDir, "tokenizer.json");
        File.WriteAllText(path, json);
        return Nivara.AI.Gpt2BpeTokenizer.LoadFromTokenizerJson(path, "<|endoftext|>", normalizeNfc: false);
    }

    static Dictionary<string, int> BuildByteLevelBaseVocab()
    {
        var bs = new List<int>();
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

        var vocab = new Dictionary<string, int>(512);
        for (int i = 0; i < bs.Count; i++)
            vocab[((char)cs[i]).ToString()] = i;
        return vocab;
    }
}
