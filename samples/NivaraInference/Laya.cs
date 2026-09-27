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

    /// <summary>Parity bound: |csharp - reference| &lt;= GateRelTol * (1 + |reference|).</summary>
    const double GateRelTol = 1e-3;

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
        Console.WriteLine($"  {"question",-10} {"valid",5} {"markers",7} {"median",10} {"min",8} {"max",8}");

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

    /// <summary>
    /// The wheel parity gate. F32-only: the reference fixtures are float32, so a narrow-precision run
    /// would report its own weight rounding rather than a porting defect.
    /// </summary>
    /// <remarks>
    /// Prompt parity runs first and fails the gate on its own. A silently different render still
    /// produces logits, and comparing those would report a numeric disagreement whose real cause was
    /// a different question. The fixtures come from <c>Python/laya_compare.py</c>, which loads
    /// <c>laya/common.py</c> out of the <c>laya==0.3.20</c> wheel rather than transcribing it.
    /// </remarks>
    public static int Compare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        string metaPath = Path.Combine(modelDir, "laya_meta.json");
        string promptsPath = Path.Combine(modelDir, "prompts_py.bin");
        string logitsPath = Path.Combine(modelDir, "logits_py.bin");
        string actPath = Path.Combine(modelDir, "act_py.bin");
        string answersPath = Path.Combine(modelDir, "answers_py.json");
        foreach (string required in new[] { metaPath, promptsPath, logitsPath, actPath, answersPath })
        {
            if (File.Exists(required)) continue;
            Console.Error.WriteLine($"Reference file not found: {required}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/laya_compare.py");
            return 1;
        }

        Console.WriteLine($"=== {ModelTypeName} Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        using var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
        var meta = metaDoc.RootElement;
        string wheel = meta.GetProperty("wheel").GetString() ?? "";
        if (wheel != "laya==0.3.20")
        {
            Console.Error.WriteLine($"Fixtures were generated from '{wheel}', expected laya==0.3.20. Regenerate them.");
            return 1;
        }
        if (meta.GetProperty("state").GetString() != FixtureState)
        {
            Console.Error.WriteLine("Fixture state does not match the sample. Regenerate the fixtures.");
            return 1;
        }

        var questionIds = meta.GetProperty("question_ids").EnumerateArray()
            .Select(e => e.GetString()).ToArray();
        if (!questionIds.SequenceEqual(FixtureQuestions.Select(q => q.Id)))
        {
            Console.Error.WriteLine(
                "Fixture question order does not match the sample " +
                $"({string.Join(", ", questionIds)}). Regenerate the fixtures.");
            return 1;
        }

        var agent = LoadAgentConfig(modelDir);
        var tokenizer = LoadTokenizer(modelDir);
        var prompts = ReadPrompts(promptsPath);
        if (prompts.Length != FixtureQuestions.Length)
        {
            Console.Error.WriteLine(
                $"prompts_py.bin has {prompts.Length} questions, the sample has {FixtureQuestions.Length}.");
            return 1;
        }

        Console.WriteLine($"Reference: {wheel}, attn_implementation=" +
                          $"{meta.GetProperty("attn_implementation").GetString()}");
        Console.WriteLine();
        Console.WriteLine("Prompt parity (byte-exact, before any numeric comparison):");

        bool promptsOk = true;
        for (int q = 0; q < FixtureQuestions.Length; q++)
        {
            var question = FixtureQuestions[q];
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen);
            var (refIds, refMarkers) = prompts[q];

            int idMismatches = CountMismatches(sequence.Ids, refIds);
            int markerMismatches = CountMismatches(sequence.Markers, refMarkers);
            bool ok = idMismatches == 0 && markerMismatches == 0 && sequence.Ids.Length == refIds.Length
                      && sequence.Markers.Length == refMarkers.Length;
            promptsOk &= ok;
            Console.WriteLine(
                $"  {question.Id,-10} {sequence.Length,4} tok  {sequence.Markers.Length,2} markers  " +
                $"{(ok ? "match" : $"MISMATCH ids={idMismatches} markers={markerMismatches}")}");
            if (ok) continue;

            int shown = Math.Min(8, Math.Max(sequence.Ids.Length, refIds.Length));
            Console.WriteLine($"    csharp ids[:{shown}]: {string.Join(", ", sequence.Ids.Take(shown))}");
            Console.WriteLine($"    python ids[:{shown}]: {string.Join(", ", refIds.Take(shown))}");
        }
        Console.WriteLine();

        if (!promptsOk)
        {
            Console.WriteLine("Prompt parity: FAIL");
            Console.WriteLine("Not comparing logits: a different prompt makes a numeric diff meaningless.");
            return 1;
        }
        Console.WriteLine("Prompt parity: PASS");
        Console.WriteLine();

        var config = LoadEncoderConfig(modelDir);
        var (encoder, head, buildMs) = LoadModels<float, float>(tensors, config, agent);
        Console.WriteLine($"Load weights: {buildMs} ms");
        Console.WriteLine();

        float[] refLogits = ReadFloats(logitsPath);
        float[] refAct = ReadFloats(actPath);
        using var answersDoc = JsonDocument.Parse(File.ReadAllText(answersPath));
        var answers = answersDoc.RootElement;
        var calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);

        bool logitsOk = true;
        bool decisionsOk = true;
        int logitCursor = 0;
        for (int q = 0; q < FixtureQuestions.Length; q++)
        {
            var question = FixtureQuestions[q];
            var (refIds, refMarkers) = prompts[q];

            var hidden = encoder.Forward(refIds, refIds.Length);
            var output = head.Forward(hidden, question.Type, refMarkers, refIds.Length);

            int k = refMarkers.Length;
            if (logitCursor + k > refLogits.Length || output.Logits.Length != k)
            {
                Console.Error.WriteLine(
                    $"{question.Id}: logit shape mismatch, csharp {output.Logits.Length}, reference {k}.");
                return 1;
            }

            var reference = refLogits.AsSpan(logitCursor, k);
            logitCursor += k;
            logitsOk &= ReportLogits(question.Id, output.Logits, reference);

            if (q * 2 + 1 >= refAct.Length)
            {
                Console.Error.WriteLine("act_py.bin is shorter than the question list.");
                return 1;
            }
            double refActProbability = SoftmaxFirst(refAct[q * 2], refAct[q * 2 + 1]);
            double actDiff = Math.Abs(output.ActionProbability - refActProbability);
            Console.WriteLine(
                $"  act probability: csharp {output.ActionProbability:F6}  python {refActProbability:F6}  " +
                $"diff {actDiff:F8}  (weak signal: the shipped head saturates near 1.0)");

            var decision = calibration.Decode(question, output.Logits, output.ActionProbability);
            decisionsOk &= ReportDecision(question, decision, answers.GetProperty(question.Id));
            Console.WriteLine();
        }

        if (logitCursor != refLogits.Length)
        {
            Console.Error.WriteLine(
                $"logits_py.bin has {refLogits.Length} values, the prompts account for {logitCursor}.");
            return 1;
        }

        Console.WriteLine($"Prompt parity:  PASS");
        Console.WriteLine($"Logit parity:   {(logitsOk ? "PASS" : $"FAIL (beyond {GateRelTol:G} relative)")}");
        Console.WriteLine($"Decision parity: {(decisionsOk ? "PASS" : "FAIL")}");

        if (!logitsOk || !decisionsOk) return 1;

        Console.WriteLine();
        Console.WriteLine("Gate passed. Prompts, marker logits, and the typed decision match the laya 0.3.20 wheel.");
        return 0;
    }

    static bool ReportLogits(string questionId, float[] actual, ReadOnlySpan<float> reference)
    {
        double maxAbs = 0.0;
        int beyond = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - reference[i]);
            if (diff > maxAbs) maxAbs = diff;
            double allowed = GateRelTol * (1.0 + Math.Abs(reference[i]));
            if (diff > allowed) beyond++;
        }

        Console.WriteLine(
            $"[{questionId}] logits ({actual.Length}): max|diff|={maxAbs:F6}  " +
            $"{(beyond == 0 ? "match" : $"{beyond} beyond {GateRelTol:G} relative")}");
        Console.WriteLine($"  csharp: {string.Join("  ", actual.Select(v => v.ToString("F4")))}");
        Console.WriteLine($"  python: {string.Join("  ", reference.ToArray().Select(v => v.ToString("F4")))}");
        return beyond == 0;
    }

    static bool ReportDecision(LayaQuestion question, LayaDecision decision, JsonElement expected)
    {
        var mismatches = new List<string>();

        void Check(string name, string actual, string? reference)
        {
            if (actual == reference) return;
            mismatches.Add($"{name}: csharp '{actual}' python '{reference}'");
        }

        void CheckNumber(string name, double actual, double reference)
        {
            // Both sides round to 4 decimal places for the reported figures. Half a display unit
            // is the disagreement worth failing on; anything smaller is the same printed number.
            if (Math.Abs(actual - reference) <= 5e-5) return;
            mismatches.Add($"{name}: csharp {actual:F4} python {reference:F4}");
        }

        Check("type", LayaPromptBuilder.TypeName(question.Type), expected.GetProperty("type").GetString());
        Check("bucket", decision.Temperature.Bucket, expected.GetProperty("temperature_bucket").GetString());

        bool fromBucket = expected.GetProperty("temperature_from_bucket").GetBoolean();
        bool wasClamped = expected.GetProperty("temperature_was_clamped").GetBoolean();
        if (decision.Temperature.FromBucketTable != fromBucket)
            mismatches.Add($"from_bucket: csharp {decision.Temperature.FromBucketTable} python {fromBucket}");
        if (decision.Temperature.WasClamped != wasClamped)
            mismatches.Add($"was_clamped: csharp {decision.Temperature.WasClamped} python {wasClamped}");
        if (Math.Abs(decision.Temperature.Scale - expected.GetProperty("temperature_scale").GetDouble()) > 1e-9)
            mismatches.Add(
                $"temperature: csharp {decision.Temperature.Scale:R} python {expected.GetProperty("temperature_scale").GetDouble():R}");

        switch (question.Type)
        {
            case LayaQuestionType.Choice:
                Check("choice", decision.Choice ?? "", expected.GetProperty("choice").GetString());
                CheckRoundedList("probabilities", decision.Probabilities, expected.GetProperty("probabilities"), mismatches);
                CheckNumber("confidence", decision.Confidence, expected.GetProperty("confidence").GetDouble());
                break;
            case LayaQuestionType.Score:
                CheckNumber("score", decision.Score ?? double.NaN, expected.GetProperty("score").GetDouble());
                CheckRoundedList("probabilities", decision.Probabilities, expected.GetProperty("probabilities"), mismatches);
                CheckNumber("confidence", decision.Confidence, expected.GetProperty("confidence").GetDouble());
                break;
            case LayaQuestionType.Noul:
                CheckNumber("noul", decision.NoulProbability ?? double.NaN, expected.GetProperty("noul").GetDouble());
                CheckNumber("confidence", decision.Confidence, expected.GetProperty("confidence").GetDouble());
                break;
        }

        CheckNumber("answer_confidence", decision.AnswerConfidence, expected.GetProperty("answer_confidence").GetDouble());
        CheckNumber("act_probability", decision.ActionProbability, expected.GetProperty("act_probability").GetDouble());

        if (mismatches.Count == 0)
        {
            Console.WriteLine($"  decision: match ({decision.Temperature.Bucket}" +
                              $"{(decision.Temperature.WasClamped ? ", clamped" : "")})");
            return true;
        }

        Console.WriteLine($"  decision: {mismatches.Count} MISMATCH(ES)");
        foreach (string mismatch in mismatches)
            Console.WriteLine($"    {mismatch}");
        return false;
    }

    static void CheckRoundedList(
        string name, IReadOnlyList<double> actual, JsonElement expected, List<string> mismatches)
    {
        var reference = expected.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        if (actual.Count != reference.Length)
        {
            mismatches.Add($"{name}: csharp {actual.Count} values, python {reference.Length}");
            return;
        }
        for (int i = 0; i < actual.Count; i++)
        {
            if (Math.Abs(actual[i] - reference[i]) <= 5e-5) continue;
            mismatches.Add($"{name}[{i}]: csharp {actual[i]:F4} python {reference[i]:F4}");
        }
    }

    static int CountMismatches(int[] actual, int[] reference)
    {
        int n = Math.Min(actual.Length, reference.Length);
        int mismatches = Math.Abs(actual.Length - reference.Length);
        for (int i = 0; i < n; i++)
            if (actual[i] != reference[i]) mismatches++;
        return mismatches;
    }

    static (int[] Ids, int[] Markers)[] ReadPrompts(string path)
    {
        int[] words = ReadInts(path);
        if (words.Length == 0)
            throw new InvalidDataException($"{path} is empty.");

        int cursor = 0;
        int n = words[cursor++];
        var prompts = new (int[] Ids, int[] Markers)[n];
        for (int q = 0; q < n; q++)
        {
            int nIds = words[cursor++];
            var ids = new int[nIds];
            Array.Copy(words, cursor, ids, 0, nIds);
            cursor += nIds;
            int nMarkers = words[cursor++];
            var markers = new int[nMarkers];
            Array.Copy(words, cursor, markers, 0, nMarkers);
            cursor += nMarkers;
            prompts[q] = (ids, markers);
        }
        if (cursor != words.Length)
            throw new InvalidDataException($"{path} has {words.Length - cursor} trailing words.");
        return prompts;
    }

    static double SoftmaxFirst(float a, float b)
    {
        double max = Math.Max(a, b);
        double ea = Math.Exp(a - max);
        double eb = Math.Exp(b - max);
        return ea / (ea + eb);
    }

    static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    static int[] ReadInts(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new int[bytes.Length / sizeof(int)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    static string Truncate(string text, int width)
        => text.Length <= width ? text : string.Concat(text.AsSpan(0, width - 1), "…");
}
