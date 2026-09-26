using Nivara.AutoDiff;
using Nivara.Samples;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;

namespace NivaraInference;

/// <summary>
/// ModernBERT-large encoder inference. Unlike the causal and classifier models, this mode has no
/// task head: it reports the encoder's <c>last_hidden_state</c>, which is what a masked-LM or
/// embedding model consumes. The <c>compare</c> mode is the parity gate against HuggingFace and
/// diffs only the valid (non-padding) positions, because a padding row far enough from the valid
/// region has every key suppressed and is non-finite on both sides.
/// </summary>
public static class ModernBert
{
    /// <summary>Parity bound: |csharp - reference| &lt;= GateRelTol * (1 + |reference|).</summary>
    const double GateRelTol = 1e-3;

    const string ModelTypeName = "ModernBERT-large";

    /// <summary>
    /// Sentences used by the default and benchmark modes. Deliberately single-spaced: ModernBERT
    /// declares 23 whitespace-run tokens in its added vocabulary, which our tokenizer matches over
    /// the raw text rather than per pre-tokenized piece, so a run of two or more spaces is a known
    /// divergence from HuggingFace. See the tokenizer notes in the README.
    /// </summary>
    static readonly string[] SampleSentences =
    [
        "Nivara runs ModernBERT-large on CPU in 2026, with 28 layers and sliding-window attention.",
        "ModernBERT alternates local and global attention layers so long contexts stay affordable.",
        "Bidirectional encoders mask padding on both sides, which causal decoders never have to do.",
        "Rotary embeddings give ModernBERT positions without a learned position table.",
        "Gated linear units keep the feed-forward block expressive without doubling the parameters.",
        "A 129-wide sliding window means a query cannot see more than 64 tokens in either direction.",
        "HuggingFace pads on the right by default, so the real tokens are a prefix of the input.",
        "Pre-norm layers keep gradients stable through twenty-eight residual blocks.",
        "Nivara's AutoDiff engine runs the whole encoder inside a single forward pass.",
        "The parity gate compares the last hidden state, not a classification head.",
    ];

    const int BenchmarkMaxLength = 128;

    public static int Run<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string modelDir,
        string mode,
        string precisionLabel)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        if (mode == "compare")
        {
            Console.Error.WriteLine(
                "modernbert compare is F32-only: the reference fixture is float32, so a bf16/fp16 run " +
                "would differ by its own weight-rounding error rather than by a porting defect. " +
                "Drop --precision bf16|fp16 and rerun.");
            return 1;
        }

        return mode == "benchmark"
            ? RunBenchmark<TModel, TWeight>(tensors, modelDir, precisionLabel)
            : RunInference<TModel, TWeight>(tensors, modelDir, precisionLabel);
    }

    static ModernBertConfig LoadConfig(string modelDir)
    {
        string configPath = Path.Combine(modelDir, "config.json");
        if (!File.Exists(configPath))
        {
            Console.Error.WriteLine($"Config file not found: {configPath}");
            throw new FileNotFoundException(configPath);
        }
        return ModernBertConfig.FromJson(File.ReadAllText(configPath));
    }

    static void PrintConfig(ModernBertConfig config)
    {
        int full = config.LayerTypes.Count(t => t == "full_attention");
        Console.WriteLine(
            $"Config: dim={config.HiddenSize}, layers={config.NumHiddenLayers}, heads={config.NumAttentionHeads}, " +
            $"headDim={config.HeadDim}, intermediate={config.IntermediateSize}, vocab={config.VocabSize}");
        Console.WriteLine(
            $"Attention: {full} full + {config.NumHiddenLayers - full} sliding, " +
            $"local_attention={config.LocalAttention} (inclusive band {config.SlidingWindow}), " +
            $"rope theta {config.RopeThetaFull:F0} full / {config.RopeThetaSliding:F0} sliding");
        Console.WriteLine($"Norm: bias-free LayerNorm, eps={config.NormEps}, activation={config.HiddenActivation}");
        Console.WriteLine();
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
        Console.WriteLine();

        var config = LoadConfig(modelDir);
        PrintConfig(config);

        var buildSw = Stopwatch.StartNew();
        var encoder = ModernBertEncoder<TModel>.LoadWeights<TModel, TWeight>(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Load weights: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Parameters: {tensors.Values.Sum(t => t.Data.Length):N0} " +
                          $"({tensors.Values.Sum(t => t.Data.Length) * 4.0 / (1024.0 * 1024.0):F1} MB as F32)");
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir, config);

        Console.WriteLine($"Sentences ({SampleSentences.Length}), last_hidden_state stats per run:");
        Console.WriteLine();
        for (int i = 0; i < SampleSentences.Length; i++)
        {
            var ids = Encode(tokenizer, SampleSentences[i]);
            int validLength = ids.Length;

            var sw = Stopwatch.StartNew();
            var hidden = encoder.Forward(ids, validLength);
            sw.Stop();

            var stats = Stats(hidden, validLength);
            Console.WriteLine(
                $"  [{i}] {stats.Min,9:F3} {stats.Max,9:F3} {stats.Mean,8:F4} {stats.StdDev,8:F4}  " +
                $"{sw.ElapsedMilliseconds,5} ms  {validLength,3} tok  {Truncate(SampleSentences[i], 58)}");
        }

        Console.WriteLine();
        Console.WriteLine("The encoder emits no logits — it is the trunk a masked-LM or embedding model " +
                          "sits on. Run 'modernbert compare' for the HuggingFace parity gate.");
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

        var config = LoadConfig(modelDir);
        PrintConfig(config);

        var buildSw = Stopwatch.StartNew();
        var encoder = ModernBertEncoder<TModel>.LoadWeights<TModel, TWeight>(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Load weights: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir, config);
        string text = SampleSentences[0];

        foreach (int maxLength in new[] { 128, 256 })
        {
            var (ids, validLength) = PadTo(tokenizer, text, maxLength, config.PadTokenId);

            // One untimed pass so JIT, the rope cache, and first-touch page faults are not in the samples.
            encoder.Forward(ids, validLength);

            const int iterations = 3;
            var timings = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                var hidden = encoder.Forward(ids, validLength);
                sw.Stop();
                timings[i] = sw.Elapsed.TotalMilliseconds;
                _ = hidden.Length;
            }

            Array.Sort(timings);
            double median = timings[iterations / 2];
            double tokensPerSecond = validLength / (median / 1000.0);

            Console.WriteLine(
                $"seq={maxLength,4} (valid {validLength,3})  median {median,8:F1} ms  " +
                $"min {timings[0],8:F1} ms  {tokensPerSecond,9:F1} tok/s  " +
                $"~{median / config.NumHiddenLayers:F2} ms/layer");
        }

        Console.WriteLine();
        Console.WriteLine("Each pass encodes the full padded sequence — this mode has no KV cache, " +
                          "since an encoder re-reads the whole sequence every time.");
        return 0;
    }

    /// <summary>
    /// Walks the encoder one stage at a time and diffs each stage against HuggingFace's
    /// <c>output_hidden_states</c>, reporting the first stage that diverges. This is the first rung
    /// of the parity debug ladder: a finite-but-wrong value is an arithmetic or ordering bug, while
    /// a non-finite value points at a masking or index bug instead. Because every stage is compared,
    /// the first divergent layer localizes the fault to that layer's own arithmetic rather than to
    /// anything downstream of it.
    /// </summary>
    public static int CompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        string stagesPath = Path.Combine(modelDir, "hidden_states_py.bin");
        if (!File.Exists(stagesPath))
        {
            Console.Error.WriteLine($"Reference file not found: {stagesPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/modernbert_compare.py");
            return 1;
        }

        Console.WriteLine($"=== {ModelTypeName} Compare Diag ===");
        Console.WriteLine();

        var config = LoadConfig(modelDir);
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);
        var tokenizer = LoadTokenizer(modelDir, config);
        var (ids, validLength) = PadTo(tokenizer, SampleSentences[0], 128, config.PadTokenId);

        var fullMask = ModernBertMasks.Build<float>(ids.Length, -1, validLength);
        var slidingMask = ModernBertMasks.Build<float>(ids.Length, config.SlidingWindow, validLength);

        // HuggingFace returns num_layers + 1 states: the post-embedding-norm state, then the output
        // of every layer except the last, then the final-norm state. The last layer's raw output is
        // never exposed (the residual stream reaches ~2.6e4 there and final_norm rescales it to ~28),
        // so that one stage is checked only through the final norm, which covers it end to end.
        float[] stages = ReadFloats(stagesPath);
        int stageStride = ids.Length * config.HiddenSize;
        int stageCount = stages.Length / stageStride;
        if (stageCount != config.NumHiddenLayers + 1)
        {
            Console.Error.WriteLine(
                $"Reference has {stageCount} stages, expected {config.NumHiddenLayers + 1} " +
                $"(embeddings + layers - 1 + final norm). Regenerate the fixture.");
            return 1;
        }

        int validValues = validLength * config.HiddenSize;
        int lastLayer = config.NumHiddenLayers - 1;
        Console.WriteLine($"Comparing {stageCount} stages over {validLength} valid rows " +
                          $"({validValues:N0} values each).");
        Console.WriteLine();

        int stage = 0;
        var hidden = encoder.embedNorm.Forward(encoder.tokenEmbedding.Forward(ids));
        Report(stage++, "embeddings + norm", hidden, stages, stageStride, validValues);

        for (int i = 0; i < encoder.layers.Length; i++)
        {
            var mask = config.IsFullAttention(i) ? fullMask : slidingMask;
            hidden = encoder.layers[i].Forward(hidden, mask);
            string kind = config.IsFullAttention(i) ? "full" : "sliding";
            int referenceStage = i < lastLayer ? stage : -1;
            Report(stage, $"layer {i,2} ({kind})", hidden, stages, stageStride, validValues, referenceStage);
            stage++;
        }

        // The reference's last entry is the final-norm state, i.e. last_hidden_state.
        Report(stage, "final norm", encoder.finalNorm.Forward(hidden), stages, stageStride,
            validValues, referenceStage: stage - 1);
        return 0;
    }

    /// <summary>
    /// Diffs one stage against the reference. <paramref name="referenceStage"/> indexes into the
    /// reference array; a negative value means the reference does not expose this stage, so only the
    /// local magnitude is reported.
    /// </summary>
    static void Report(
        int stage,
        string label,
        ReverseGradTensor<float> tensor,
        float[] reference,
        int stageStride,
        int validValues,
        int referenceStage = -1)
    {
        tensor.Data.TryGetSpan(out var span);

        int nonFinite = 0;
        float maxMagnitude = 0.0f;
        for (int i = 0; i < validValues; i++)
        {
            float value = span[i];
            if (!float.IsFinite(value))
            {
                nonFinite++;
                continue;
            }
            float magnitude = Math.Abs(value);
            if (magnitude > maxMagnitude) maxMagnitude = magnitude;
        }

        if (referenceStage < 0)
        {
            Console.WriteLine(
                $"  stage {stage,2} {label,-22} max|.|={maxMagnitude,12:F6}  " +
                $"non-finite={nonFinite,6}  (no reference; covered by the final norm)");
            return;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < validValues; i++)
        {
            float value = span[i];
            if (!float.IsFinite(value)) continue;
            float diff = Math.Abs(value - reference[referenceStage * stageStride + i]);
            if (diff > maxAbs) maxAbs = diff;
        }

        double sumSquaredActual = 0.0;
        double sumSquaredReference = 0.0;
        double dot = 0.0;
        for (int i = 0; i < validValues; i++)
        {
            double actual = span[i];
            double expected = reference[referenceStage * stageStride + i];
            sumSquaredActual += actual * actual;
            sumSquaredReference += expected * expected;
            dot += actual * expected;
        }
        double cosine = dot / (Math.Sqrt(sumSquaredActual) * Math.Sqrt(sumSquaredReference));

        string verdict = nonFinite > 0
            ? "NON-FINITE"
            : maxAbs < 1e-3 ? "match" : maxAbs < 1e-1 ? "close" : "DIVERGED";

        Console.WriteLine(
            $"  stage {stage,2} {label,-22} max|diff|={maxAbs,12:F6}  cosine={cosine,10:F6}  " +
            $"non-finite={nonFinite,6}  {verdict}");
    }

    /// <summary>
    /// The HuggingFace parity gate. F32-only by construction: the reference fixture is float32, so a
    /// narrow-precision run would report its own weight-rounding error rather than a porting defect.
    /// </summary>
    public static int Compare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        string metaPath = Path.Combine(modelDir, "compare_meta.json");
        string refPath = Path.Combine(modelDir, "last_hidden_state_py.bin");
        string refIdsPath = Path.Combine(modelDir, "input_ids_py.bin");
        foreach (string required in new[] { metaPath, refPath, refIdsPath })
        {
            if (!File.Exists(required))
            {
                Console.Error.WriteLine($"Reference file not found: {required}");
                Console.Error.WriteLine("Run: python samples/NivaraInference/Python/modernbert_compare.py");
                return 1;
            }
        }

        Console.WriteLine($"=== {ModelTypeName} Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        using var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
        var meta = metaDoc.RootElement;
        string text = meta.GetProperty("text").GetString()!;
        int maxLength = meta.GetProperty("max_length").GetInt32();
        int validLength = meta.GetProperty("valid_len").GetInt32();
        int refRows = meta.GetProperty("hidden_shape")[0].GetInt32();
        int refCols = meta.GetProperty("hidden_shape")[1].GetInt32();
        string attnImpl = meta.TryGetProperty("attn_implementation", out var impl) ? impl.GetString()! : "unknown";

        var config = LoadConfig(modelDir);
        PrintConfig(config);
        Console.WriteLine($"Reference: HuggingFace transformers, attn_implementation={attnImpl}, " +
                          $"output shape [{refRows}, {refCols}]");
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir, config);

        // Tokenizer parity first: a single wrong id invalidates the whole numeric comparison.
        var (ids, _) = PadTo(tokenizer, text, maxLength, config.PadTokenId);
        int[] refIds = ReadInts(refIdsPath);
        Console.WriteLine($"Tokenizer parity ({ids.Length} positions, {validLength} valid):");
        int idMismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < Math.Min(ids.Length, refIds.Length); i++)
        {
            if (ids[i] == refIds[i]) continue;
            if (idMismatches == 0) firstMismatch = i;
            idMismatches++;
        }
        if (idMismatches == 0)
        {
            Console.WriteLine("  all ids match");
        }
        else
        {
            Console.WriteLine($"  {idMismatches} MISMATCH(ES), first at position {firstMismatch}:");
            int from = Math.Max(0, firstMismatch - 2);
            int to = Math.Min(ids.Length, firstMismatch + 3);
            for (int i = from; i < to; i++)
                Console.WriteLine($"    [{i}] csharp={ids[i]} python={refIds[i]}");
        }
        Console.WriteLine();
        Console.WriteLine($"  csharp ids: {string.Join(", ", ids.Take(validLength))}");
        Console.WriteLine($"  python ids: {string.Join(", ", refIds.Take(validLength))}");
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Load weights: {buildSw.ElapsedMilliseconds} ms");

        var forwardSw = Stopwatch.StartNew();
        var output = encoder.Forward(ids, validLength);
        forwardSw.Stop();
        Console.WriteLine($"Forward pass ({maxLength} positions): {forwardSw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        var actual = ToFloats(output);
        float[] reference = ReadFloats(refPath);

        if (actual.Length != reference.Length)
        {
            Console.Error.WriteLine(
                $"Shape mismatch: got {actual.Length} values ({output.Shape[0]}x{output.Shape[1]}), " +
                $"reference has {reference.Length}. Check max_length and hidden_size.");
            return 1;
        }

        // Only the valid prefix is comparable. A padding row beyond the sliding window has every key
        // suppressed, so its softmax denominator is zero: we produce NaN (Nivara's row softmax has no
        // safe-softmax clamp) while PyTorch's sdpa produces zeros. Neither value is meaningful and
        // neither can reach a valid position, because a valid query never reads a padding key.
        int validValues = validLength * refCols;
        var actualValid = actual.AsSpan(0, validValues);
        var referenceValid = reference.AsSpan(0, validValues);

        int nonFiniteActual = CountNonFinite(actualValid);
        int nonFiniteReference = CountNonFinite(referenceValid);
        int nonFiniteActualPad = CountNonFinite(actual.AsSpan(validValues));
        int nonFiniteReferencePad = CountNonFinite(reference.AsSpan(validValues));

        var diffs = new float[validValues];
        TensorPrimitives.Subtract(actualValid, referenceValid, diffs);
        var absDiffs = new float[validValues];
        TensorPrimitives.Abs(diffs, absDiffs);

        float maxAbs = TensorPrimitives.Max(absDiffs);
        double meanAbs = TensorPrimitives.Sum(absDiffs) / validValues;
        double cosine = TensorPrimitives.CosineSimilarity(actualValid, referenceValid);

        // maxRel over the positions where the reference magnitude is meaningful; dividing by a
        // near-zero reference would report meaningless ratios, so those are counted separately.
        const double significantFloor = 1e-2;
        double maxRel = 0.0;
        int significant = 0;
        for (int i = 0; i < validValues; i++)
        {
            double magnitude = Math.Abs(referenceValid[i]);
            if (magnitude < significantFloor) continue;
            significant++;
            maxRel = Math.Max(maxRel, absDiffs[i] / magnitude);
        }
        int insignificant = validValues - significant;

        int beyondGate = 0;
        for (int i = 0; i < validValues; i++)
        {
            double allowed = GateRelTol * (1.0 + Math.Abs(referenceValid[i]));
            if (absDiffs[i] > allowed) beyondGate++;
        }

        Console.WriteLine($"Valid-region parity ({validLength} rows x {refCols} cols = {validValues:N0} values):");
        Console.WriteLine($"  csharp stats: min={TensorPrimitives.Min(actualValid):F6}, max={TensorPrimitives.Max(actualValid):F6}, " +
                          $"mean={TensorPrimitives.Average(actualValid):F6}");
        Console.WriteLine($"  python stats: min={TensorPrimitives.Min(referenceValid):F6}, max={TensorPrimitives.Max(referenceValid):F6}, " +
                          $"mean={TensorPrimitives.Average(referenceValid):F6}");
        Console.WriteLine($"  max abs diff: {maxAbs:F8}");
        Console.WriteLine($"  mean abs diff: {meanAbs:F10}");
        Console.WriteLine($"  max rel diff (|ref| >= {significantFloor}): {maxRel:F8}  " +
                          $"[{significant:N0} positions; {insignificant:N0} below the floor excluded]");
        Console.WriteLine($"  cosine similarity: {cosine:F10}");
        Console.WriteLine($"  non-finite in valid region: csharp {nonFiniteActual}, python {nonFiniteReference}");
        Console.WriteLine($"  non-finite in padding region: csharp {nonFiniteActualPad}, python {nonFiniteReferencePad}" +
                          $"  (undefined rows: every key masked, band edge past the valid region)");
        Console.WriteLine();

        Console.Write("C#  [:10]: [");
        PrintSlice(actual, 10);
        Console.WriteLine();
        Console.Write("Py  [:10]: [");
        PrintSlice(reference, 10);
        Console.WriteLine();
        Console.WriteLine();

        bool tokenizerOk = idMismatches == 0;
        bool valuesOk = beyondGate == 0 && nonFiniteActual == 0;
        Console.WriteLine($"Tokenizer parity: {(tokenizerOk ? "PASS" : $"FAIL ({idMismatches} mismatched ids)")}");
        Console.WriteLine($"Value parity:     {(valuesOk ? "PASS" : $"FAIL ({beyondGate} values beyond {GateRelTol:F0} relative)")}");

        if (!tokenizerOk || !valuesOk) return 1;

        Console.WriteLine();
        Console.WriteLine("Gate passed. ModernBERT-large last_hidden_state matches HuggingFace.");
        return 0;
    }

    static Gpt2BpeTokenizer LoadTokenizer(string modelDir, ModernBertConfig config)
    {
        string tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            Console.Error.WriteLine($"Tokenizer file not found: {tokenizerPath}");
            throw new FileNotFoundException(tokenizerPath);
        }

        // The declared unk_token is [UNK]; the checkpoint's model.merges and model.vocab live inline
        // in tokenizer.json, so there is no vocab.json / merges.txt pair to read.
        return Gpt2BpeTokenizer.LoadFromTokenizerJson(tokenizerPath, "[UNK]", normalizeNfc: true);
    }

    static int[] Encode(Gpt2BpeTokenizer tokenizer, string text)
        => [.. tokenizer.EncodeWithSpecialTokens(text, "[CLS]", "[SEP]")];

    static (int[] Ids, int ValidLength) PadTo(Gpt2BpeTokenizer tokenizer, string text, int maxLength, int padTokenId)
    {
        var ids = Encode(tokenizer, text);
        if (ids.Length > maxLength)
        {
            // Match HuggingFace's truncation=True: keep the leading tokens and always close with [SEP].
            var truncated = new int[maxLength];
            Array.Copy(ids, truncated, maxLength - 1);
            truncated[maxLength - 1] = ids[^1];
            return (truncated, maxLength);
        }

        if (ids.Length == maxLength) return (ids, maxLength);

        var padded = new int[maxLength];
        Array.Copy(ids, padded, ids.Length);
        Array.Fill(padded, padTokenId, ids.Length, maxLength - ids.Length);
        return (padded, ids.Length);
    }

    static float[] ToFloats(ReverseGradTensor<float> tensor)
    {
        var data = new float[tensor.Length];
        tensor.Data.TryGetSpan(out var span);
        if (!span.IsEmpty) span.CopyTo(data);
        return data;
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

    static int CountNonFinite(ReadOnlySpan<float> values)
    {
        int count = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i])) count++;
        }
        return count;
    }

    static (float Min, float Max, float Mean, float StdDev) Stats<T>(ReverseGradTensor<T> tensor, int validRows)
        where T : struct, IFloatingPointIeee754<T>
    {
        tensor.Data.TryGetSpan(out var span);

        int rows = Math.Min(validRows, tensor.Shape[0]);
        int cols = tensor.Shape[1];
        int count = rows * cols;

        // Accumulate in double so a Half or BFloat16 run reports F32-comparable statistics.
        double min = double.MaxValue;
        double max = double.MinValue;
        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            double value = double.CreateChecked(span[i]);
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
        }

        double mean = sum / count;
        double sumSquares = 0.0;
        for (int i = 0; i < count; i++)
        {
            double delta = double.CreateChecked(span[i]) - mean;
            sumSquares += delta * delta;
        }

        return ((float)min, (float)max, (float)mean, (float)Math.Sqrt(sumSquares / count));
    }

    static void PrintSlice(float[] values, int count)
    {
        for (int i = 0; i < Math.Min(count, values.Length); i++)
        {
            Console.Write($"{values[i]:F6}");
            if (i < Math.Min(count, values.Length) - 1) Console.Write(", ");
        }
    }

    static string Truncate(string text, int max)
        => text.Length <= max ? text : text[..(max - 3)] + "...";
}
