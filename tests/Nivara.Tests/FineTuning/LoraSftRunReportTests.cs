using NivaraFineTuning;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class LoraSftRunReportTests
{
    [Test]
    public void Save_RoundTripsAllFields()
    {
        var report = new LoraSftRunReport
        {
            Mode = "ft-lora",
            Backend = "cpu",
            Device = "cpu",
            Model = "qwen2.5-0.5b-instruct",
            Template = "qwen",
            Rank = 8,
            Alpha = 16f,
            Targets = "q,v",
            MaxLen = 256,
            MaxExamples = 32,
            Seed = 42,
            TrainExamples = 24,
            ValExamples = 8,
            BaseParams = 1000,
            AdapterParams = 200,
            TrainableParams = 200,
            TotalParams = 1200,
            InitialValLoss = 2.5f,
            BestValLoss = 2.0f,
            BestEpoch = 1,
            ElapsedMs = 1234,
            AdapterPath = "out/lora_adapters.json",
            AdapterFormat = "nivara-lora-sft-adapters-v1",
            AdapterSha256 = "abc123",
            MemoryBudgetBytes = 1073741824,
            MemoryBudgetMode = "advisory",
            EvalRan = true,
            EvalSplit = "val",
            EvalExamples = 8,
            BaseMetrics = "Base\ntotal=8",
            AdaptedMetrics = "LoRA-adapted\ntotal=8",
            EvalDeltas = "ticket: +10.0%",
            BaseWorklogMetrics = new WorklogEvalMetrics(8, 6, 5, 5, 5, 1, 2, 1, 1, 1, 1),
            AdaptedWorklogMetrics = new WorklogEvalMetrics(8, 8, 7, 7, 7, 2, 2, 1, 1, 1, 1),
            EvalElapsedMs = 456,
            LossCurve =
            {
                new LoraSftEpochLoss(1, 2.4f, 2.1f),
            },
        };

        var path = Path.Combine(Path.GetTempPath(), "lora-report-test.json");
        report.Save(path);
        var json = File.ReadAllText(path);
        var loaded = JsonSerializer.Deserialize<LoraSftRunReport>(json, LoraSftRunReport.SerializerOptions);

        Assert.That(loaded, Is.Not.Null);
        Assert.That(loaded!.Mode, Is.EqualTo("ft-lora"));
        Assert.That(loaded.Backend, Is.EqualTo("cpu"));
        Assert.That(loaded.BaseParams, Is.EqualTo(1000));
        Assert.That(loaded.AdapterParams, Is.EqualTo(200));
        Assert.That(loaded.TrainableParams, Is.EqualTo(200));
        Assert.That(loaded.TotalParams, Is.EqualTo(1200));
        Assert.That(loaded.MemoryBudgetBytes, Is.EqualTo(1073741824));
        Assert.That(loaded.MemoryBudgetMode, Is.EqualTo("advisory"));
        Assert.That(loaded.EvalRan, Is.True);
        Assert.That(loaded.EvalExamples, Is.EqualTo(8));
        Assert.That(loaded.LossCurve, Has.Count.EqualTo(1));
        Assert.That(loaded.LossCurve[0].TrainLoss, Is.EqualTo(2.4f));
        Assert.That(loaded.LossCurve[0].ValLoss, Is.EqualTo(2.1f));
        Assert.That(loaded.EvalDeltas, Does.Contain("ticket"));
        Assert.That(loaded.BaseWorklogMetrics, Is.Not.Null);
        Assert.That(loaded.BaseWorklogMetrics!.Total, Is.EqualTo(8));
        Assert.That(loaded.BaseWorklogMetrics.ValidStructure, Is.EqualTo(6));
        Assert.That(loaded.AdaptedWorklogMetrics, Is.Not.Null);
        Assert.That(loaded.AdaptedWorklogMetrics!.TicketCorrect, Is.EqualTo(7));
        Assert.That(loaded.EvalElapsedMs, Is.EqualTo(456));

        File.Delete(path);
    }

    [Test]
    public void SaveMarkdown_ContainsKeySections()
    {
        var report = new LoraSftRunReport
        {
            Mode = "ft-eval",
            Backend = "cpu",
            Device = "cpu",
            Model = "smollm",
            Template = "smollm",
            Rank = 4,
            Alpha = 8f,
            Targets = "q,v",
            MaxLen = 128,
            MaxExamples = 8,
            Seed = 7,
            TrainExamples = 0,
            ValExamples = 4,
            BaseParams = 500,
            AdapterParams = 50,
            TrainableParams = 50,
            TotalParams = 550,
            InitialValLoss = 3f,
            BestValLoss = 2.8f,
            BestEpoch = 1,
            ElapsedMs = 100,
            MemoryBudgetBytes = null,
            MemoryBudgetMode = "unspecified",
            EvalRan = true,
            EvalSplit = "test",
            EvalExamples = 4,
            BaseMetrics = "Base\ntotal=4",
            AdaptedMetrics = "LoRA-adapted\ntotal=4",
        };

        var path = Path.Combine(Path.GetTempPath(), "lora-report-test.md");
        report.SaveMarkdown(path);
        var md = File.ReadAllText(path);

        Assert.That(md, Does.Contain("# LoRA SFT Run Report"));
        Assert.That(md, Does.Contain("## Parameters"));
        Assert.That(md, Does.Contain("Base params:"));
        Assert.That(md, Does.Contain("Adapter params:"));
        Assert.That(md, Does.Contain("## Memory budget"));
        Assert.That(md, Does.Contain("unspecified"));
        Assert.That(md, Does.Contain("## Eval"));
        Assert.That(md, Does.Contain("## Parity status"));
        Assert.That(md, Does.Contain("CPU/Silk parity confirmed"));

        File.Delete(path);
    }

    [Test]
    public void Sha256File_MissingPath_ReturnsNull()
    {
        Assert.That(LoraSftRunReport.Sha256File(null), Is.Null);
        Assert.That(LoraSftRunReport.Sha256File("does-not-exist.json"), Is.Null);
    }

    [Test]
    public void Save_SilkFtEvalReport_RoundTripsBackendDeviceAndMetrics()
    {
        var report = new LoraSftRunReport
        {
            Mode = "ft-eval",
            Backend = "silk",
            Device = "Intel(R) Graphics",
            Model = "qwen2.5-0.5b-instruct",
            Template = "qwen",
            Rank = 4,
            Alpha = 8f,
            Targets = "q,v",
            MaxLen = 128,
            MaxExamples = 4,
            Seed = 42,
            TrainExamples = 0,
            ValExamples = 4,
            EvalRan = true,
            EvalSplit = "test",
            EvalExamples = 4,
            BaseMetrics = "Base\ntotal=4",
            AdaptedMetrics = "LoRA-adapted\ntotal=4",
            EvalDeltas = "ticket: +25.0%",
            BaseWorklogMetrics = new WorklogEvalMetrics(4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0),
            AdaptedWorklogMetrics = new WorklogEvalMetrics(4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0),
            AdapterPath = "out/lora-adapter.json",
            AdapterFormat = "nivara-lora-sft-adapters-v1",
            AdapterSha256 = "deadbeef",
        };

        var path = Path.Combine(Path.GetTempPath(), $"lora-silk-eval-{Guid.NewGuid():N}.json");
        try
        {
            report.Save(path);
            var loaded = JsonSerializer.Deserialize<LoraSftRunReport>(File.ReadAllText(path), LoraSftRunReport.SerializerOptions);

            Assert.That(loaded, Is.Not.Null);
            Assert.That(loaded!.Mode, Is.EqualTo("ft-eval"));
            Assert.That(loaded.Backend, Is.EqualTo("silk"));
            Assert.That(loaded.Device, Is.EqualTo("Intel(R) Graphics"));
            Assert.That(loaded.EvalRan, Is.True);
            Assert.That(loaded.EvalSplit, Is.EqualTo("test"));
            Assert.That(loaded.BaseWorklogMetrics, Is.Not.Null);
            Assert.That(loaded.AdaptedWorklogMetrics, Is.Not.Null);
            Assert.That(loaded.EvalDeltas, Does.Contain("ticket"));
            Assert.That(loaded.AdapterFormat, Is.EqualTo("nivara-lora-sft-adapters-v1"));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void SaveMarkdown_SilkFtEvalReport_ContainsBackendDeviceAndEval()
    {
        var report = new LoraSftRunReport
        {
            Mode = "ft-eval",
            Backend = "silk",
            Device = "Intel(R) Graphics",
            Model = "qwen2.5-0.5b-instruct",
            Template = "qwen",
            EvalRan = true,
            EvalSplit = "test",
            EvalExamples = 4,
            BaseMetrics = "Base\ntotal=4",
            AdaptedMetrics = "LoRA-adapted\ntotal=4",
        };

        var md = report.ToMarkdown();

        Assert.That(md, Does.Contain("**Backend:** silk"));
        Assert.That(md, Does.Contain("**Device:** Intel(R) Graphics"));
        Assert.That(md, Does.Contain("## Eval"));
        Assert.That(md, Does.Contain("Base metrics"));
        Assert.That(md, Does.Contain("Adapted metrics"));
    }
}
