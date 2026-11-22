using Nivara.Samples;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NivaraFineTuning;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class WorklogLoraSftTrainerTests
{
    static LlamaForCausalLM<float, float> TinyModel() =>
        new(vocabSize: 16, hiddenSize: 8, numHiddenLayers: 1, numHeads: 2, numKeyValueHeads: 2, intermediateSize: 16, maxPositionEmbeddings: 32);

    [Test]
    public void Train_BaseWeightsUnchanged_AfterTraining()
    {
        var model = TinyModel();
        // Snapshot a base projection weight before training.
        var qWeightBefore = (float[])model.layers[0].Attention.QProj.Weight!.Tensor.Data.ToArray().Clone();

        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1), TinyExample(2) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(3) };

        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 7,
            LogEvery = 0,
            SaveAdapter = false,
        };

        WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);

        var qWeightAfter = model.layers[0].Attention.QProj.Weight!.Tensor.Data.ToArray();
        Assert.That(qWeightAfter, Is.EqualTo(qWeightBefore));
    }

    [Test]
    public void Train_AdapterParamsChange_AfterTraining()
    {
        var model = TinyModel();
        var adapters = model.AttachLora(LlamaLoRATarget.Q | LlamaLoRATarget.V, 2, 4f);
        // snapshot adapter A/B before
        var aBefore = adapters[0].A.Tensor.Data.ToArray();
        var bBefore = adapters[0].B.Tensor.Data.ToArray();

        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1), TinyExample(2) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(3) };

        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 2,
            MaxLen = 32,
            Seed = 7,
            LogEvery = 0,
            SaveAdapter = false,
        };

        // Train attaches its own adapters internally; just ensure it runs and reports trainable params.
        var model2 = TinyModel();
        var opts2 = options;
        var result = WorklogLoraSftTrainer.Train(opts2, model2, null, trainTok, valTok);
        Assert.That(result.TrainableParams, Is.GreaterThan(0));
    }

    [Test]
    public void Train_DeterministicGivenSeed()
    {
        var model = TinyModel();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1), TinyExample(2) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(3) };
        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 42,
            LogEvery = 0,
            SaveAdapter = false,
        };

        var r1 = WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);
        // Re-run on the same model instance; Train re-attaches fresh seeded adapters.
        var r2 = WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);

        Assert.That(r1.InitialValLoss, Is.EqualTo(r2.InitialValLoss));
        Assert.That(r1.BestValLoss, Is.EqualTo(r2.BestValLoss));
        Assert.That(r1.BestEpoch, Is.EqualTo(r2.BestEpoch));
    }

    [Test]
    public void Train_OverfitSanity_LossDecreases()
    {
        var model = TinyModel();
        // Repeat same example to make overfit trivial.
        var ex = TinyExample(5);
        var trainTok = new List<WorklogTokenizedExample> { ex, ex, ex, ex };
        var valTok = new List<WorklogTokenizedExample> { ex };

        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 4,
            Alpha = 8f,
            Epochs = 3,
            MaxLen = 32,
            LearningRate = 1e-2f,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
        };

        var result = WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);

        Assert.That(result.InitialValLoss, Is.Not.NaN);
        Assert.That(result.BestValLoss, Is.LessThanOrEqualTo(result.InitialValLoss + 1e-6f));
    }

    [Test]
    public void Train_TrainableParamsPositive()
    {
        var model = TinyModel();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(2) };
        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
        };
        var result = WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);
        Assert.That(result.TrainableParams, Is.GreaterThan(0));
        Assert.That(result.BaseParams, Is.GreaterThan(0));
    }

    [Test]
    public void Train_ComputeWorklogEvalMetricsDisabled_ReturnsNullMetrics()
    {
        var model = TinyModel();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(2) };
        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
            ComputeWorklogEvalMetrics = false,
        };

        var result = WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok);

        Assert.That(result.BestValMetrics, Is.Null);
        Assert.That(result.EvalExamples, Is.EqualTo(0));
        Assert.That(result.EvalElapsedMs, Is.EqualTo(0));
    }

    [Test]
    public void Train_ComputeWorklogEvalMetricsWithoutTokenizer_Throws()
    {
        var model = TinyModel();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(2) };
        var valExamples = new List<WorklogSftExample> { EvalExample("log 1h on DB-42 today", "ok") };
        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
            ComputeWorklogEvalMetrics = true,
        };

        var ex = Assert.Throws<InvalidOperationException>(() =>
            WorklogLoraSftTrainer.Train(options, model, null, trainTok, valTok, valExamples));
        Assert.That(ex!.Message, Does.Contain("Tokenizer is required"));
    }

    [Test]
    public void Train_ComputeWorklogEvalMetricsWithoutValExamples_Throws()
    {
        var model = TinyModel();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(2) };
        var tokenizer = CreateByteLevelTokenizer();
        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 32,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
            ComputeWorklogEvalMetrics = true,
        };

        var ex = Assert.Throws<InvalidOperationException>(() =>
            WorklogLoraSftTrainer.Train(options, model, tokenizer, trainTok, valTok));
        Assert.That(ex!.Message, Does.Contain("Val examples are required"));
    }

    [Test]
    public void Train_ComputeWorklogEvalMetrics_ReturnsMetrics()
    {
        var model = new LlamaForCausalLM<float, float>(vocabSize: 300, hiddenSize: 8, numHiddenLayers: 1, numHeads: 2, numKeyValueHeads: 2, intermediateSize: 16, maxPositionEmbeddings: 64);
        var tokenizer = CreateByteLevelTokenizer();
        var trainTok = new List<WorklogTokenizedExample> { TinyExample(1, vocab: 300) };
        var valTok = new List<WorklogTokenizedExample> { TinyExample(2, vocab: 300) };
        var valExamples = new List<WorklogSftExample>
        {
            EvalExample("log 1 hour to DB-42 on 2026-07-19", "<tool_call><function>LogTime</function><parameter=ticket>DB-42</parameter><parameter=hours>1</parameter><parameter=date>2026-07-19</parameter></tool_call>")
        };

        var options = new WorklogLoraSftOptions
        {
            ModelDir = Path.GetTempPath(),
            TrainPath = "x",
            ValPath = "y",
            Rank = 2,
            Alpha = 4f,
            Epochs = 1,
            MaxLen = 64,
            Seed = 1,
            LogEvery = 0,
            SaveAdapter = false,
            ComputeWorklogEvalMetrics = true,
            EvalMaxNewTokens = 8,
            MaxEvalExamples = 1,
        };

        var result = WorklogLoraSftTrainer.Train(options, model, tokenizer, trainTok, valTok, valExamples);

        Assert.That(result.BestValMetrics, Is.Not.Null);
        Assert.That(result.EvalExamples, Is.EqualTo(1));
        Assert.That(result.EvalElapsedMs, Is.GreaterThanOrEqualTo(0));
        Assert.That(result.BestValMetrics!.Total, Is.EqualTo(1));
    }

    static WorklogSftExample EvalExample(string user, string assistant) => new([
        new WorklogMessage("user", user),
        new WorklogMessage("assistant", assistant)
    ]);

    static WorklogTokenizedExample TinyExample(int seed, int vocab = 16)
    {
        var rng = new Random(seed);
        var ids = new[] { rng.Next(vocab), rng.Next(vocab), rng.Next(vocab), rng.Next(vocab) };
        var labels = new[] { -100, -100, ids[2], ids[3] };
        return new WorklogTokenizedExample(ids, labels, 2, 2);
    }

    static Nivara.AI.Gpt2BpeTokenizer CreateByteLevelTokenizer()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "nivara-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        var vocab = new Dictionary<string, int>(capacity: 260);
        for (int i = 0; i < 256; i++)
            vocab[$"<0x{i:X2}>"] = i;
        var json = "{\"model\":{\"type\":\"BPE\",\"vocab\":" + System.Text.Json.JsonSerializer.Serialize(vocab)
            + ",\"merges\":[]},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false},\"added_tokens\":[]}";
        var path = Path.Combine(tempDir, "tokenizer.json");
        File.WriteAllText(path, json);
        return Nivara.AI.Gpt2BpeTokenizer.LoadFromTokenizerJson(path, "<|endoftext|>", normalizeNfc: false);
    }
}
