using Nivara.AutoDiff;
using Nivara.Samples;
using System.Diagnostics;
using System.Numerics;
using System.Text.Json;

namespace NivaraInference;

/// <summary>
/// Laya decision-head inference: a ModernBERT-large encoder plus the typed decision head from
/// <c>laya.common.DecisionModel</c> (PyPI <c>laya</c> 0.3.20).
/// </summary>
/// <remarks>
/// <para>
/// The default mode scores the fixture's questions and prints the typed answer, the two
/// confidences, the per-option probabilities, the temperature actually applied, and the act head's
/// reading. The benchmark mode times the same questions at the config's <c>max_len</c>.
/// </para>
/// <para>
/// <b>The CPU path is a reference, not a deployment claim.</b> The encoder is ~95% of the runtime
/// and GEMM is ~99.9% of the arithmetic; the head is a few percent on top. The numbers that justify
/// that split live in <c>docs/LAYA.md</c> rather than here, because they are probe measurements and
/// a sample's console output is the wrong place for them.
/// </para>
/// </remarks>
public static class Laya
{
    const string ModelTypeName = "Laya";

    const string FixtureState = "User: how do I reset my password?\nAgent: open settings, then security.";

    /// <summary>
    /// The fixture questions, one per question type plus the high-cardinality choice case. The
    /// thirteen-option question is what selects the <c>choice:11+</c> temperature bucket, whose
    /// shipped value is out of range and gets clamped to 0.5.
    /// </summary>
    static readonly LayaQuestion[] FixtureQuestions =
    [
        new()
        {
            Id = "choice2",
            Type = LayaQuestionType.Choice,
            Instructions = "Is this a login question?",
            Options = [new("reset", null), new("other", "not a password question")]
        },
        new()
        {
            Id = "choice13",
            Type = LayaQuestionType.Choice,
            Instructions = "Which topic?",
            Options = Enumerable.Range(0, 13).Select(i => new LayaChoiceOption($"opt{i}", $"topic number {i}")).ToArray()
        },
        new()
        {
            Id = "score4",
            Type = LayaQuestionType.Score,
            Instructions = "How urgent?",
            ScoreCriteria = ["low", "medium", "high", "critical"]
        },
        new()
        {
            Id = "noul",
            Type = LayaQuestionType.Noul,
            Instructions = "Is it resolved?",
            NoulFalseLabel = " no ",
            NoulTrueLabel = "yes",
            NoulFalseCriterion = "still broken",
            NoulTrueCriterion = "fixed"
        }
    ];

    public static int Run<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string modelDir,
        string mode,
        string precisionLabel)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        if (mode == "benchmark")
            return RunBenchmark<TModel, TWeight>(tensors, modelDir, precisionLabel);
        if (mode.Length > 0)
        {
            Console.Error.WriteLine($"laya does not support mode '{mode}' (supported: default, benchmark).");
            return 1;
        }
        return RunInference<TModel, TWeight>(tensors, modelDir, precisionLabel);
    }

    /// <summary>The agent config's subset that the sample actually consumes.</summary>
    sealed record AgentConfig(
        int MaxLen,
        int HeadMaxLen,
        int HeadLayers,
        double[] Temperature,
        IReadOnlyDictionary<string, double> TemperatureByOptions);

    static AgentConfig LoadAgentConfig(string modelDir)
    {
        string path = Path.Combine(modelDir, "rl_agent_config.json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Agent config not found: {path}. The Laya checkpoint ships it alongside the weights.", path);

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;

        var temperature = root.GetProperty("temperature").EnumerateArray()
            .Select(e => e.GetDouble()).ToArray();
        var byOptions = root.GetProperty("temperature_by_options").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetDouble());

        return new AgentConfig(
            root.GetProperty("max_len").GetInt32(),
            root.GetProperty("head_max_len").GetInt32(),
            root.GetProperty("head_layers").GetInt32(),
            temperature,
            byOptions);
    }

    static int RunInference<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string modelDir,
        string precisionLabel)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        Console.WriteLine($"=== {ModelTypeName} Inference ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version}), precision {precisionLabel}");
        Console.WriteLine("The CPU path is a reference implementation; the encoder dominates and it is a GEMM problem.");
        Console.WriteLine();

        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);
        PrintSetup(config, agent);

        var (encoder, head, buildMs) = LoadModels<TModel, TWeight>(tensors, config, agent);
        Console.WriteLine($"Load weights: {buildMs} ms");
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);

        // One state, tokenized once and reused: every question sees the same state ids.
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        Console.WriteLine($"State ({stateIds.Count} tokens): {Truncate(FixtureState, 72)}");
        Console.WriteLine();

        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);

            var sw = Stopwatch.StartNew();
            var hidden = encoder.Forward(sequence.Ids, sequence.Length);
            var output = head.Forward(hidden, question.Type, sequence.Markers, sequence.Length);
            sw.Stop();

            var decision = calibration.Decode(question, output.Logits, output.ActionProbability);
            PrintDecision(question, sequence, decision, sw.ElapsedMilliseconds);
        }

        Console.WriteLine();
        Console.WriteLine("The act_head reading above is reproduced faithfully and is not a signal: it reads");
        Console.WriteLine("~1.0 on the shipped checkpoint regardless of input (upstream NandhaKishorM/laya#185,");
        Console.WriteLine("documented on the model card). Do not gate anything on it.");
        return 0;
    }

    static int RunBenchmark<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string modelDir,
        string precisionLabel)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        Console.WriteLine($"=== {ModelTypeName} Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version}), precision {precisionLabel}");
        Console.WriteLine();

        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);
        PrintSetup(config, agent);

        var (encoder, head, buildMs) = LoadModels<TModel, TWeight>(tensors, config, agent);
        Console.WriteLine($"Load weights: {buildMs} ms");
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        // Timed at the config's max_len, padded, so the number is comparable across questions rather
        // than reflecting how short each prompt happens to be. The head's key-padding mask makes the
        // padding a no-op numerically, which is also what makes the timing honest.
        Console.WriteLine($"1 warmup + 3 timed passes at max_len {agent.MaxLen}, median reported.");
        Console.WriteLine();
        Console.WriteLine($"  {"question",-10} {"tok",5} {"markers",7} {"median",10} {"min",8} {"max",8}");

        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);
            var (ids, validLength) = PadTo(sequence.Ids, agent.MaxLen, config.PadTokenId);

            encoder.Forward(ids, validLength);

            const int iterations = 3;
            var samples = new long[iterations];
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                var hidden = encoder.Forward(ids, validLength);
                head.Forward(hidden, question.Type, sequence.Markers, validLength);
                sw.Stop();
                samples[i] = sw.ElapsedMilliseconds;
            }

            Array.Sort(samples);
            Console.WriteLine(
                $"  {question.Id,-10} {validLength,5} {sequence.Markers.Length,7} {samples[iterations / 2],8} ms {samples[0],6} ms {samples[^1],6} ms");
        }

        Console.WriteLine();
        Console.WriteLine("The encoder is the cost; the head is a few percent on top of it. See docs/LAYA.md");
        Console.WriteLine("for the probe numbers behind that split.");
        return 0;
    }

    static (ModernBertEncoder<TModel> Encoder, LayaDecisionHead<TModel> Head, long Milliseconds)
        LoadModels<TModel, TWeight>(
            Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
            ModernBertConfig config,
            AgentConfig agent)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var sw = Stopwatch.StartNew();
        // Laya nests the backbone under "encoder"; the stock checkpoint uses "model".
        var encoder = ModernBertEncoder<TModel>.LoadWeights<TModel, TWeight>(tensors, config, prefix: "encoder");
        var head = LayaDecisionHead<TModel>.LoadWeights<TModel, TWeight>(tensors, agent.HeadLayers);
        sw.Stop();
        return (encoder, head, sw.ElapsedMilliseconds);
    }

    static ModernBertConfig LoadEncoderConfig(string modelDir)
    {
        // The encoder config lives in a subdirectory, not at the model root: Laya ships the
        // backbone's config alongside the backbone rather than flattening it into one file.
        string path = Path.Combine(modelDir, "encoder", "config.json");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Encoder config not found: {path}", path);
        return ModernBertConfig.FromJson(File.ReadAllText(path));
    }

    static Gpt2BpeTokenizer LoadTokenizer(string modelDir)
    {
        // The tokenizer lives in its own subdirectory. Loading the model root's tokenizer.json
        // would fail, and failing here rather than emitting garbage ids is the point.
        string path = Path.Combine(modelDir, "tokenizer", "tokenizer.json");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Tokenizer not found: {path}", path);
        return Gpt2BpeTokenizer.LoadFromTokenizerJson(path, "[UNK]", normalizeNfc: true);
    }

    static (int[] Ids, int ValidLength) PadTo(int[] ids, int maxLength, int padTokenId)
    {
        if (ids.Length >= maxLength)
            return (ids, ids.Length);
        var padded = new int[maxLength];
        Array.Copy(ids, padded, ids.Length);
        Array.Fill(padded, padTokenId, ids.Length, maxLength - ids.Length);
        return (padded, ids.Length);
    }

    static void PrintSetup(ModernBertConfig config, AgentConfig agent)
    {
        Console.WriteLine($"Encoder: ModernBERT-large, {config.NumHiddenLayers} layers, d={config.HiddenSize}, " +
                          $"{config.NumAttentionHeads} heads, intermediate {config.IntermediateSize}");
        Console.WriteLine($"Head: {agent.HeadLayers} layers, {Math.Max(1, config.HiddenSize / LayaDecisionHead<float>.HeadDimTarget)} heads, " +
                          $"max_len {agent.MaxLen}, head_max_len {agent.HeadMaxLen}");
        Console.WriteLine($"Temperature per type: {string.Join(", ", agent.Temperature.Select(t => t.ToString("F4")))}");
        Console.WriteLine();
    }

    static void PrintDecision(LayaQuestion question, LayaSequence sequence, LayaDecision decision, long milliseconds)
    {
        string answer = decision.Type switch
        {
            LayaQuestionType.Choice => decision.Choice ?? "(none)",
            LayaQuestionType.Score => decision.Score?.ToString("F4") ?? "(none)",
            LayaQuestionType.Noul => decision.NoulProbability?.ToString("F4") ?? "(none)",
            _ => "(none)"
        };

        Console.WriteLine($"[{question.Id}] {LayaPromptBuilder.TypeName(question.Type)}  " +
                          $"{sequence.Length} tok, {sequence.Markers.Length} markers  {milliseconds} ms");
        Console.WriteLine($"  answer: {answer}");
        Console.WriteLine($"  answer_confidence: {decision.AnswerConfidence:F4}   confidence: {decision.Confidence:F4}");

        if (decision.Probabilities.Count > 0)
            Console.WriteLine($"  probabilities: {string.Join("  ", decision.Probabilities.Select(p => p.ToString("F4")))}");

        var temperature = decision.Temperature;
        string provenance = temperature.FromBucketTable
            ? $"bucket {temperature.Bucket}"
            : $"per-type fallback ({temperature.Bucket} has no entry)";
        string clamped = temperature.WasClamped
            ? $"  clamped from {temperature.RawValue:F4}"
            : "";
        Console.WriteLine($"  temperature: {temperature.Scale:F4}  ({provenance}{clamped})");

        // Printed per question so a reader sees the reading next to the answer it does not inform,
        // and again as a footer so it cannot be mistaken for one of the confidences.
        Console.WriteLine($"  act_head: {decision.ActionProbability:F4}  (not a signal; reads ~1.0 regardless of input)");
        Console.WriteLine();
    }

    static string Truncate(string text, int width)
        => text.Length <= width ? text : string.Concat(text.AsSpan(0, width - 1), "…");
}
