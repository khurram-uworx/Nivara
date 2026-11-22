using System.Diagnostics;
using System.Numerics;
using System.Text.Json;
using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.Samples;

namespace NivaraFineTuning;

public sealed record WorklogLoraSftOptions
{
    public required string ModelDir { get; init; }
    public required string TrainPath { get; init; }
    public required string ValPath { get; init; }
    public string? OutputDir { get; init; }
    public string Template { get; init; } = "qwen";
    public int Rank { get; init; } = 8;
    public float Alpha { get; init; } = 16f;
    public LlamaLoRATarget Targets { get; init; } = LlamaLoRATarget.Q | LlamaLoRATarget.V;
    public float LearningRate { get; init; } = 2e-4f;
    public int Epochs { get; init; } = 3;
    public int MaxLen { get; init; } = 256;
    public int MaxExamples { get; init; } = 0;
    public int MaxValExamples { get; init; } = 0;
    public int Patience { get; init; } = 0;
    public float MinDelta { get; init; } = 0f;
    public int GradientAccumulationSteps { get; init; } = 1;
    public int Seed { get; init; } = 42;
    public int LogEvery { get; init; } = 10;
    public bool SaveAdapter { get; init; } = true;
    public string BaseModelId { get; init; } = "";
}

public sealed record WorklogLoraSftResult(
    float InitialValLoss,
    float BestValLoss,
    int BestEpoch,
    int TrainExamples,
    int ValExamples,
    int TrainableParams,
    int BaseParams,
    long ElapsedMs,
    string? AdapterPath);

public static class WorklogLoraSftTrainer
{
    public static WorklogLoraSftResult TrainQwenFloatBf16(WorklogLoraSftOptions options)
        => Train<float, BFloat16>(options);

    public static WorklogLoraSftResult TrainSmolLMFloatBf16(WorklogLoraSftOptions options)
        => Train<float, BFloat16>(options);

    public static WorklogLoraSftResult Train<TModel, TWeight>(
        WorklogLoraSftOptions options,
        LlamaForCausalLM<TModel, TWeight>? model = null,
        Gpt2BpeTokenizer? tokenizer = null,
        List<WorklogTokenizedExample>? trainTok = null,
        List<WorklogTokenizedExample>? valTok = null)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.MaxLen < 2 || options.MaxLen > 2048)
            throw new ArgumentOutOfRangeException(nameof(options.MaxLen), "MaxLen must be between 2 and 2048.");
        if (options.GradientAccumulationSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(options.GradientAccumulationSteps), "GradientAccumulationSteps must be >= 1.");
        if (options.Patience < 0)
            throw new ArgumentOutOfRangeException(nameof(options.Patience), "Patience must be >= 0.");

        bool ownsModel = model is null;
        if (ownsModel)
        {
            if (!Directory.Exists(options.ModelDir))
                throw new DirectoryNotFoundException($"Model dir not found: {options.ModelDir}");
            Console.WriteLine($"Loading model from {options.ModelDir} ({typeof(TModel).Name}/{typeof(TWeight).Name})...");
            var loaded = LlamaLoader.LoadModel<TModel, TWeight>(options.ModelDir, requiresGrad: false);
            model = loaded.Model;
            tokenizer ??= loaded.Tokenizer;
        }
        model!.Eval();

        Console.WriteLine("Attaching LoRA adapters...");
        var adapters = model.AttachLora(options.Targets, options.Rank, TModel.CreateChecked(options.Alpha));
        int trainable = adapters.Sum(a => a.A.Tensor.Length + a.B.Tensor.Length);
        int baseParams = model.GetParameters().Values.Sum(p => p.Tensor.Length);
        Console.WriteLine($"  Base params: {baseParams:N0}");
        Console.WriteLine($"  Adapters: {adapters.Count}, trainable params: {trainable:N0}");

        if (trainTok is null || valTok is null)
        {
            if (tokenizer is null)
                throw new InvalidOperationException("Tokenizer is required when tokenized examples are not provided.");
            var trainExamples = WorklogSftDataset.LoadJsonl(options.TrainPath);
            var valExamples = WorklogSftDataset.LoadJsonl(options.ValPath);
            if (options.MaxExamples > 0 && trainExamples.Count > options.MaxExamples)
                trainExamples = trainExamples.Take(options.MaxExamples).ToList();
            if (options.MaxValExamples > 0 && valExamples.Count > options.MaxValExamples)
                valExamples = valExamples.Take(options.MaxValExamples).ToList();

            Console.WriteLine($"Tokenizing train={trainExamples.Count}, val={valExamples.Count}, maxLen={options.MaxLen}, template={options.Template}...");
            trainTok = WorklogSftDataset.Tokenize(trainExamples, tokenizer, options.Template, options.MaxLen);
            valTok = WorklogSftDataset.Tokenize(valExamples, tokenizer, options.Template, options.MaxLen);
        }
        if (trainTok.Count == 0) throw new InvalidOperationException("No train examples after tokenization.");
        if (valTok.Count == 0) throw new InvalidOperationException("No val examples after tokenization.");
        Console.WriteLine($"  Tokenized train={trainTok.Count}, val={valTok.Count}");
        Console.WriteLine("  Running initial validation...");

        var optimizer = new AdamW<TModel>(TModel.CreateChecked(options.LearningRate));
        foreach (var adapter in adapters)
        {
            optimizer.AddParameterGroup(new Parameter<TModel>[] { adapter.A, adapter.B });
        }

        var lossFn = new CrossEntropyLoss<TModel>();
        var rng = new Random(options.Seed);

        float initialValLoss = EvaluateLoss(model, lossFn, valTok);
        Console.WriteLine($"Initial val loss: {initialValLoss:F4}");

        float bestValLoss = initialValLoss;
        int bestEpoch = 0;
        int epochsWithoutImprovement = 0;
        List<(string Name, TModel[] Data, int[] Shape)>? bestSnapshot = null;

        var sw = Stopwatch.StartNew();
        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            Shuffle(trainTok, rng);
            model.Train();
            float epochLoss = 0f;
            int steps = 0;

            int accumSteps = Math.Max(1, options.GradientAccumulationSteps);
            for (int i = 0; i < trainTok.Count; i++)
            {
                var ex = trainTok[i];
                if (options.LogEvery > 0 && (i + 1) % options.LogEvery == 0)
                    Console.WriteLine($"  epoch {epoch}/{options.Epochs} step {i + 1}/{trainTok.Count}: forward...");
                using (Nivara.AutoDiff.Utilities.GradientUtils.Grad())
                {
                    var logits = model.Forward(ex.InputIds); // [L, vocab]
                    var loss = lossFn.Forward(logits, ex.Labels, -100);
                    if (options.LogEvery > 0 && (i + 1) % options.LogEvery == 0)
                        Console.WriteLine($"  epoch {epoch}/{options.Epochs} step {i + 1}/{trainTok.Count}: backward (loss={float.CreateChecked(loss.Data[0]):F4})...");
                    loss.Backward();
                    epochLoss += float.CreateChecked(loss.Data[0]);
                    steps++;
                }

                bool stepNow = ((i + 1) % accumSteps == 0) || (i == trainTok.Count - 1);
                if (stepNow)
                {
                    optimizer.Step();
                    optimizer.ZeroGrad();
                }

                if (options.LogEvery > 0 && (i + 1) % options.LogEvery == 0)
                    Console.WriteLine($"  epoch {epoch}/{options.Epochs} step {i + 1}/{trainTok.Count} avgLoss={epochLoss / steps:F4}");
            }

            float trainAvg = steps > 0 ? epochLoss / steps : float.NaN;
            float valLoss = EvaluateLoss(model, lossFn, valTok);
            Console.WriteLine($"Epoch {epoch}: trainLoss={trainAvg:F4} valLoss={valLoss:F4}");

            bool improved = valLoss < bestValLoss - options.MinDelta;
            if (improved)
            {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                bestSnapshot = SnapshotAdapters(adapters);
                epochsWithoutImprovement = 0;
            }
            else
            {
                epochsWithoutImprovement++;
            }

            if (options.Patience > 0 && epochsWithoutImprovement >= options.Patience)
            {
                Console.WriteLine($"Early stopping at epoch {epoch} (patience={options.Patience}).");
                break;
            }
        }
        sw.Stop();

        if (bestSnapshot is not null)
            RestoreAdapters(adapters, bestSnapshot);

        string? adapterPath = null;
        if (options.SaveAdapter && !string.IsNullOrWhiteSpace(options.OutputDir))
        {
            Directory.CreateDirectory(options.OutputDir!);
            adapterPath = Path.Combine(options.OutputDir!, "lora_adapters.json");
            SaveAdapters(adapters, adapterPath, options);
            Console.WriteLine($"Saved adapters to {adapterPath}");
        }

        if (ownsModel) model.Dispose();
        return new WorklogLoraSftResult(initialValLoss, bestValLoss, bestEpoch, trainTok.Count, valTok.Count, trainable, baseParams, sw.ElapsedMilliseconds, adapterPath);
    }

    static float EvaluateLoss<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model,
        CrossEntropyLoss<TModel> lossFn,
        List<WorklogTokenizedExample> examples)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        model.Eval();
        float total = 0f;
        int count = 0;
        foreach (var ex in examples)
        {
            var logits = model.Forward(ex.InputIds);
            var loss = lossFn.Forward(logits, ex.Labels, -100);
            total += float.CreateChecked(loss.Data[0]);
            count++;
        }
        return count > 0 ? total / count : float.NaN;
    }

    static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    static List<(string Name, TModel[] Data, int[] Shape)> SnapshotAdapters<TModel, TWeight>(
        IReadOnlyList<LoRALinear<TModel, TWeight>> adapters)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var snap = new List<(string, TModel[], int[])>(adapters.Count * 2);
        for (int i = 0; i < adapters.Count; i++)
        {
            snap.Add(($"adapters.{i}.A", Copy(adapters[i].A.Tensor), adapters[i].A.Tensor.Shape.ToArray()));
            snap.Add(($"adapters.{i}.B", Copy(adapters[i].B.Tensor), adapters[i].B.Tensor.Shape.ToArray()));
        }
        return snap;
    }

    static void RestoreAdapters<TModel, TWeight>(
        IReadOnlyList<LoRALinear<TModel, TWeight>> adapters,
        List<(string Name, TModel[] Data, int[] Shape)> snapshot)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        int idx = 0;
        for (int i = 0; i < adapters.Count; i++)
        {
            var (_, aData, aShape) = snapshot[idx++];
            var (_, bData, bShape) = snapshot[idx++];
            adapters[i].A.Tensor = ReverseGradTensor<TModel>.FromMatrix(aData, aShape[0], aShape[1], requiresGrad: true);
            adapters[i].B.Tensor = ReverseGradTensor<TModel>.FromMatrix(bData, bShape[0], bShape[1], requiresGrad: true);
        }
    }

    static T[] Copy<T>(ReverseGradTensor<T> tensor) where T : struct, IFloatingPointIeee754<T>
    {
        var data = new T[tensor.Length];
        tensor.Data.CopyTo(data, default);
        return data;
    }

    static void SaveAdapters<TModel, TWeight>(
        IReadOnlyList<LoRALinear<TModel, TWeight>> adapters,
        string path,
        WorklogLoraSftOptions options)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var payload = new
        {
            format = "nivara-lora-sft-adapters-v1",
            baseModelId = options.BaseModelId,
            template = options.Template,
            rank = options.Rank,
            alpha = options.Alpha,
            targets = options.Targets.ToString(),
            trainableParams = adapters.Sum(a => a.A.Tensor.Length + a.B.Tensor.Length),
            adapters = adapters.Select((a, i) => new
            {
                index = i,
                aShape = a.A.Tensor.Shape.ToArray(),
                bShape = a.B.Tensor.Shape.ToArray(),
                a = Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(Copy(a.A.Tensor).AsSpan())),
                b = Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(Copy(a.B.Tensor).AsSpan())),
            }).ToArray()
        };
        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }
}
