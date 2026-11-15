using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.Tests.Inference;

/// <summary>
/// Checkpoint-level gate for the native SmolLM path: BF16 weights stay exactly as shipped and
/// every kernel widens them in-register, so the run holds one copy of the snapshot payload.
/// Everything here reads the real checkpoint, so the fixture skips (with the reason) when the
/// locally-downloaded files are absent — CI has neither the weights nor the PyTorch fixtures.
/// </summary>
/// <remarks>
/// <para>
/// <b>Coverage.</b> Three properties, each pinned against the file rather than against a number
/// this code produced: (1) every tensor the loader returns is byte-identical to the safetensors
/// payload, so no dtype translation happens at load; (2) <see cref="LlamaLoader"/> puts those
/// same arrays into the model by reference, so no widening copy sits between the file and the
/// parameters — the failure this catches is "the bytes are right but the model doubled them";
/// (3) a 32-token greedy run matches the BF16 PyTorch fixture and its final-position logits
/// (computed over the fixture's own prefix, so both sides see identical input) stay within the
/// documented cosine bound.
/// </para>
/// <para>
/// <b>Not covered.</b> The resident-bytes arithmetic below proves the loader and the model; it
/// does not observe the process working set, so a third copy made somewhere else would escape it.
/// Numeric parity is one fixture, one prompt.
/// </para>
/// <para>
/// <b>Duration.</b> A full 30-layer 135M-parameter build plus a 32-token decode is ~10 s in
/// Release, over the 2-second rule, so the fixture carries <c>[Category("Performance")]</c> and
/// stays out of the filtered suite. Run it deliberately:
/// <c>dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SmolLMNativeCheckpointTests"</c>.
/// </para>
/// </remarks>
[TestFixture]
[Category("Performance")]
public class SmolLMNativeCheckpointTests
{
    static CheckpointFile? checkpoint;
    static Dictionary<string, (BFloat16[] Data, int[] Shape)>? tensors;
    static LlamaConfig? config;
    static LlamaForCausalLM<float, BFloat16>? model;

    static CheckpointFile Checkpoint
    {
        get
        {
            if (checkpoint != null)
                return checkpoint;

            SmolLMCheckpoint.RequireCheckpoint();
            checkpoint = CheckpointFile.Open(SmolLMCheckpoint.ModelPath);
            return checkpoint;
        }
    }

    /// <summary>The loader's own arrays, shared with <see cref="Model"/> so the model build is paid once.</summary>
    static Dictionary<string, (BFloat16[] Data, int[] Shape)> Tensors
    {
        get
        {
            if (tensors != null)
                return tensors;

            _ = Checkpoint;
            tensors = SafeTensorsLoader.Read<BFloat16>(SmolLMCheckpoint.ModelPath);
            return tensors;
        }
    }

    static LlamaConfig Config
    {
        get
        {
            if (config != null)
                return config;

            SmolLMCheckpoint.RequireCheckpoint();
            config = LlamaConfig.FromJson(File.ReadAllText(SmolLMCheckpoint.ConfigPath));
            return config;
        }
    }

    static LlamaForCausalLM<float, BFloat16> Model
    {
        get
        {
            if (model != null)
                return model;

            model = LlamaLoader.Load<float, BFloat16>(Config, Tensors);
            return model;
        }
    }

    [OneTimeTearDown]
    public void ReleaseModel()
    {
        model?.Dispose();
        model = null;
        tensors = null;
        config = null;
        checkpoint?.Dispose();
        checkpoint = null;
    }

    /// <summary>
    /// The premise of the whole path: the checkpoint is BF16 on disk, and reading it as
    /// <see cref="BFloat16"/> returns the payload bit for bit. A re-upload in F32 would silently
    /// make the native run a wide run, so the dtype is asserted rather than assumed.
    /// </summary>
    [Test]
    public void LoadedTensors_AreTheCheckpointPayloadBitForBit()
    {
        var file = Checkpoint;
        var loaded = Tensors;

        Assert.That(loaded.Count, Is.EqualTo(file.Entries.Length),
            "the loader must return exactly one array per checkpoint tensor");
        Assert.That(file.Entries.Sum(e => e.ElementCount), Is.EqualTo(134_515_008),
            "SmolLM-135M parameter count; a different checkpoint would invalidate every fixture below");

        foreach (var entry in file.Entries)
        {
            Assert.That(entry.Dtype, Is.EqualTo("BF16"),
                $"{entry.Name} is {entry.Dtype} on disk; the native path is defined against the BF16 snapshot");

            Assert.That(loaded.ContainsKey(entry.Name), Is.True, $"{entry.Name} must be loaded");
            var (data, shape) = loaded[entry.Name];
            Assert.That(shape, Is.EqualTo(entry.Shape), $"{entry.Name} shape");
            Assert.That(data.Length, Is.EqualTo(entry.ElementCount), $"{entry.Name} element count");

            var raw = file.ReadPayload(entry);
            Assert.That(MemoryMarshal.AsBytes(data.AsSpan()).SequenceEqual(raw), Is.True,
                $"{entry.Name} is not byte-identical to the checkpoint payload");
        }
    }

    /// <summary>
    /// Widening happens in the dot kernels, never at load or in the module: every projection and
    /// the embedding must be the loader's array instance, and the only widened parameters are the
    /// 61 RMSNorm gammas (140 KB), not the 134 M projection weights (538 MB as F32).
    /// </summary>
    [Test]
    public void NativeModel_KeepsTheCheckpointArrays_AndOnlyTheNormsAreWidened()
    {
        var file = Checkpoint;
        var loaded = Tensors;
        var nativeModel = Model;

        Assert.That(LlamaFusedKernels.DecoderBlockFused, Is.True,
            "the native generation gate below measures the fused decode/prefill path, which is where the mixed-dtype matmul lives");

        long storedBytes = 0;
        int storedCount = 0;

        void AssertStoredWeight(Parameter<BFloat16> parameter, string checkpointKey)
        {
            storedBytes += AssertSameWeight(parameter, loaded[checkpointKey].Data, checkpointKey);
            storedCount++;
        }

        AssertStoredWeight(nativeModel.Embed.Weight!, "model.embed_tokens.weight");

        for (int i = 0; i < nativeModel.layers.Length; i++)
        {
            var layer = nativeModel.layers[i];
            var prefix = $"model.layers.{i}";
            AssertStoredWeight(layer.Attention.QProj.Weight!, $"{prefix}.self_attn.q_proj.weight");
            AssertStoredWeight(layer.Attention.KProj.Weight!, $"{prefix}.self_attn.k_proj.weight");
            AssertStoredWeight(layer.Attention.VProj.Weight!, $"{prefix}.self_attn.v_proj.weight");
            AssertStoredWeight(layer.Attention.OProj.Weight!, $"{prefix}.self_attn.o_proj.weight");
            AssertStoredWeight(layer.GateProj.Weight!, $"{prefix}.mlp.gate_proj.weight");
            AssertStoredWeight(layer.UpProj.Weight!, $"{prefix}.mlp.up_proj.weight");
            AssertStoredWeight(layer.DownProj.Weight!, $"{prefix}.mlp.down_proj.weight");
        }

        long normElements = nativeModel.finalNorm.Weight!.Tensor.Length;
        for (int i = 0; i < nativeModel.layers.Length; i++)
        {
            normElements += nativeModel.layers[i].InputNorm.Weight!.Tensor.Length;
            normElements += nativeModel.layers[i].PostNorm.Weight!.Tensor.Length;
        }

        long payloadBytes = file.Entries.Sum(e => e.Length);
        long checkpointNormElements = file.Entries
            .Where(e => IsNormWeight(e.Name))
            .Sum(e => e.ElementCount);

        Assert.That(storedCount + file.Entries.Count(e => IsNormWeight(e.Name)), Is.EqualTo(file.Entries.Length),
            "every checkpoint tensor must be either a stored projection/embedding or an RMSNorm gamma");
        Assert.That(checkpointNormElements, Is.EqualTo(normElements),
            "one gamma per RMSNorm, at the compute dtype");
        Assert.That(normElements, Is.EqualTo((long)(2 * Config.NumHiddenLayers + 1) * Config.HiddenSize),
            "2 norms per layer plus the final norm, each hidden-wide");
        Assert.That(storedBytes, Is.EqualTo(payloadBytes - checkpointNormElements * 2),
            "the projection/embedding store must be exactly the checkpoint payload minus the BF16 gammas");

        TestContext.Out.WriteLine(
            $"checkpoint payload {payloadBytes:N0} B; model parameter storage " +
            $"{storedBytes + normElements * 4:N0} B (BF16 weights {storedBytes:N0} B + F32 norms {normElements * 4:N0} B)");
    }

    /// <summary>
    /// End-to-end native run: batched prefill plus one fused cached-decode step per generated
    /// token, compared with the PyTorch BF16 reference. The token stream is a coarse gate (a single
    /// near-tie flips every later token), so the numeric claim is the cosine on final-position
    /// logits computed over the reference's own prefix — identical input on both sides.
    /// </summary>
    [Test]
    public void NativeGeneration_MatchesTheBfloat16Reference()
    {
        SmolLMCheckpoint.RequireFixtures("bf16");

        var (promptLength, referenceIds) = SmolLMCheckpoint.ReadTokenFixture("compare_smollm_py_bf16.bin");
        var referenceLogits = SmolLMCheckpoint.ReadFloat32Fixture("compare_smollm_logits_py_bf16.bin");
        Assert.That(referenceIds.Length, Is.EqualTo(promptLength + SmolLMCheckpoint.MaxNewTokens),
            "the BF16 fixture must hold the prompt plus a full 32-token continuation");

        var promptIds = SmolLMCheckpoint.PromptIds();
        Assert.That(promptIds, Is.EqualTo(referenceIds[..promptLength]),
            "the fixture was generated from this prompt; a tokenizer change must fail here, not downstream");

        var generated = SmolLMCheckpoint.Greedy(Model, Config, promptIds, SmolLMCheckpoint.MaxNewTokens);
        int matches = generated
            .Select((id, i) => id == referenceIds[promptLength + i])
            .Count(match => match);

        var logits = SmolLMCheckpoint.LastPositionLogits(Model, referenceIds[..^1], Config.VocabSize);
        double cosine = SmolLMCheckpoint.Cosine(logits, referenceLogits);

        TestContext.Out.WriteLine($"generated ids: {string.Join(", ", generated)}");
        TestContext.Out.WriteLine($"token match {matches}/{generated.Length}; cosine {cosine:F8}; " +
                                  $"final-position argmax {SmolLMCheckpoint.ArgMax(logits)} vs reference {SmolLMCheckpoint.ArgMax(referenceLogits)}");

        // Observed 25/32 at the time of writing. The bound is the documented parity floor shared
        // with the narrow `--precision bf16` run; the cosine carries the numeric claim.
        Assert.That(matches, Is.GreaterThanOrEqualTo(22),
            "native generation must stay on the reference trajectory");
        Assert.That(cosine, Is.GreaterThanOrEqualTo(0.999),
            "native final-position logits must match the BF16 reference");
    }

    static bool IsNormWeight(string name)
        => name.EndsWith(".input_layernorm.weight", StringComparison.Ordinal)
        || name.EndsWith(".post_attention_layernorm.weight", StringComparison.Ordinal)
        || name == "model.norm.weight";

    static long AssertSameWeight(Parameter<BFloat16> parameter, BFloat16[] expected, string name)
    {
        Assert.That(parameter.Tensor.Data.TryGetSpan(out var span), Is.True, $"{name} must be null-free");
        Assert.That(span.Length, Is.EqualTo(expected.Length), $"{name} element count");
        Assert.That(
            Unsafe.AreSame(ref MemoryMarshal.GetReference(span), ref MemoryMarshal.GetReference(expected.AsSpan())),
            Is.True,
            $"{name} must be the loader's own array; a copy is exactly the widening this path exists to avoid");
        return expected.Length * 2L;
    }
}

/// <summary>
/// The other half of the default switch: <c>--precision f32</c> must still be the widened run it
/// always was, byte-for-byte on the token stream against the F32 PyTorch fixture. Kept in its own
/// fixture so the native model is released before the 538 MB F32 copy is loaded.
/// </summary>
[TestFixture]
[Category("Performance")]
public class SmolLMF32CheckpointTests
{
    static LlamaForCausalLM<float, float>? model;

    static LlamaForCausalLM<float, float> Model
    {
        get
        {
            if (model != null)
                return model;

            SmolLMCheckpoint.RequireCheckpoint();
            var config = LlamaConfig.FromJson(File.ReadAllText(SmolLMCheckpoint.ConfigPath));
            model = LlamaLoader.Load<float, float>(config, SafeTensorsLoader.Read<float>(SmolLMCheckpoint.ModelPath));
            return model;
        }
    }

    [OneTimeTearDown]
    public void ReleaseModel()
    {
        model?.Dispose();
        model = null;
    }

    /// <summary>
    /// F32 is same-dtype end to end, so the gate is exact: the whole 32-token greedy stream must
    /// match the F32 reference, and the logits are held to a bound far wider than the observed
    /// 1.7e-5 difference yet far narrower than any structural defect.
    /// </summary>
    [Test]
    public void F32Generation_MatchesTheFloat32ReferenceExactly()
    {
        SmolLMCheckpoint.RequireFixtures("f32");

        var config = LlamaConfig.FromJson(File.ReadAllText(SmolLMCheckpoint.ConfigPath));
        var (promptLength, referenceIds) = SmolLMCheckpoint.ReadTokenFixture("compare_smollm_py_f32.bin");
        var referenceLogits = SmolLMCheckpoint.ReadFloat32Fixture("compare_smollm_logits_py_f32.bin");
        Assert.That(referenceIds.Length, Is.EqualTo(promptLength + SmolLMCheckpoint.MaxNewTokens),
            "the F32 fixture must hold the prompt plus a full 32-token continuation");

        var promptIds = SmolLMCheckpoint.PromptIds();
        Assert.That(promptIds, Is.EqualTo(referenceIds[..promptLength]),
            "the fixture was generated from this prompt; a tokenizer change must fail here, not downstream");

        var generated = SmolLMCheckpoint.Greedy(Model, config, promptIds, SmolLMCheckpoint.MaxNewTokens);
        Assert.That(generated, Is.EqualTo(referenceIds[promptLength..]),
            "the F32 run must reproduce the PyTorch F32 token stream exactly");

        var logits = SmolLMCheckpoint.LastPositionLogits(Model, referenceIds[..^1], config.VocabSize);
        double cosine = SmolLMCheckpoint.Cosine(logits, referenceLogits);
        float maxAbsDiff = logits
            .Select((value, i) => Math.Abs(value - referenceLogits[i]))
            .Max();

        TestContext.Out.WriteLine($"token match {generated.Length}/{generated.Length}; max abs diff {maxAbsDiff:E3}; cosine {cosine:F8}");

        Assert.That(maxAbsDiff, Is.LessThan(1e-3f),
            "F32 final-position logits must match the F32 reference to accumulation-order noise");
        Assert.That(cosine, Is.GreaterThanOrEqualTo(0.9999),
            "F32 final-position logits must match the F32 reference");
    }
}

/// <summary>Shared paths, fixtures and the greedy loop for the checkpoint-dependent SmolLM gates.</summary>
static class SmolLMCheckpoint
{
    public const string Prompt = "The capital of France is";
    public const int MaxNewTokens = 32;

    public static string RepoRoot
        => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", ".."));

    public static string ModelDir => Path.Combine(RepoRoot, "samples", "data", "smollm-135m");

    public static string ModelPath => Path.Combine(ModelDir, "model.safetensors");

    public static string ConfigPath => Path.Combine(ModelDir, "config.json");

    public static string VocabPath => Path.Combine(ModelDir, "vocab.json");

    public static string MergesPath => Path.Combine(ModelDir, "merges.txt");

    /// <summary>Skips the calling test unless the checkpoint and tokenizer are present.</summary>
    public static void RequireCheckpoint()
    {
        string[] missing = new[] { ModelPath, ConfigPath, VocabPath, MergesPath }
            .Where(path => !File.Exists(path))
            .ToArray();
        if (missing.Length > 0)
            Assert.Ignore($"SmolLM checkpoint files absent; skipping native-checkpoint verification. Missing: {string.Join(", ", missing)}");
    }

    /// <summary>Skips the calling test unless the token/logits fixtures for <paramref name="tag"/> are present.</summary>
    public static void RequireFixtures(string tag)
    {
        string[] missing = new[] { FixturePath($"compare_smollm_py_{tag}.bin"), FixturePath($"compare_smollm_logits_py_{tag}.bin") }
            .Where(path => !File.Exists(path))
            .ToArray();
        if (missing.Length > 0)
            Assert.Ignore($"SmolLM PyTorch fixtures absent; skipping {tag} parity verification. " +
                          "Regenerate with samples/NivaraInference/Python/smollm_generate_reference.py " +
                          $"--dtype {(tag == "f32" ? "float32" : "bfloat16")}. Missing: {string.Join(", ", missing)}");
    }

    public static string FixturePath(string name) => Path.Combine(RepoRoot, "samples", "data", name);

    /// <summary>Encodes the fixed prompt with the checkpoint's own tokenizer.</summary>
    public static int[] PromptIds()
    {
        RequireCheckpoint();
        var tokenizer = new Gpt2BpeTokenizer(VocabPath, MergesPath);
        return tokenizer.Encode(Prompt).ToArray();
    }

    /// <summary>Reads a little-endian int32 binary fixture, whose first value is the prompt length.</summary>
    public static (int PromptLength, int[] Ids) ReadTokenFixture(string name)
    {
        var raw = ReadInt32Fixture(name);
        Assert.That(raw.Length, Is.GreaterThan(1), $"{name} must hold at least a prompt length and one token");
        return (raw[0], raw[1..]);
    }

    /// <summary>Reads a little-endian int32 binary fixture.</summary>
    public static int[] ReadInt32Fixture(string name)
    {
        var bytes = File.ReadAllBytes(FixturePath(name));
        var result = new int[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    /// <summary>Reads a little-endian float32 binary fixture.</summary>
    public static float[] ReadFloat32Fixture(string name)
    {
        var bytes = File.ReadAllBytes(FixturePath(name));
        var result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    /// <summary>
    /// Greedy decode through the same prefill + cached-decode loop the CLI uses, so the gate
    /// measures the fused inference path rather than a bespoke one.
    /// </summary>
    public static int[] Greedy<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model, LlamaConfig config, int[] promptIds, int maxNewTokens)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<TModel>(
            config.NumHiddenLayers, kvWidth, initialCapacity: promptIds.Length + maxNewTokens);

        var logits = model.ForwardPrefill(promptIds, cache);
        var generated = new int[maxNewTokens];
        int position = promptIds.Length;
        for (int i = 0; i < maxNewTokens; i++)
        {
            generated[i] = ArgMaxLastRow(logits, config.VocabSize);
            logits = model.ForwardCached(generated[i], position++, cache);
        }
        return generated;
    }

    /// <summary>Final-position logits for <paramref name="prefix"/>, as the CLI computes them.</summary>
    public static float[] LastPositionLogits<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model, int[] prefix, int vocab)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var logits = model.Forward(prefix);
        Assert.That(logits.Data.TryGetSpan(out var span), Is.True, "logits must be null-free");
        int offset = (logits.Shape[0] - 1) * vocab;

        var result = new float[vocab];
        for (int i = 0; i < vocab; i++)
            result[i] = (float)double.CreateChecked(span[offset + i]);
        return result;
    }

    public static int ArgMaxLastRow<T>(ReverseGradTensor<T> logits, int vocab)
        where T : struct, IFloatingPointIeee754<T>
    {
        Assert.That(logits.Data.TryGetSpan(out var span), Is.True, "logits must be null-free");
        int offset = (logits.Shape[0] - 1) * vocab;
        int best = 0;
        double bestValue = double.CreateChecked(span[offset]);
        for (int i = 1; i < vocab; i++)
        {
            double value = double.CreateChecked(span[offset + i]);
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }
        return best;
    }

    public static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }

    /// <summary>Cosine similarity accumulated in double so the gate is not itself a float rounding.</summary>
    public static double Cosine(float[] left, float[] right)
    {
        Assert.That(left.Length, Is.EqualTo(right.Length), "cosine needs equal-length vectors");
        double dot = 0, leftNorm = 0, rightNorm = 0;
        for (int i = 0; i < left.Length; i++)
        {
            dot += (double)left[i] * right[i];
            leftNorm += (double)left[i] * left[i];
            rightNorm += (double)right[i] * right[i];
        }
        return dot / Math.Sqrt(leftNorm * rightNorm);
    }
}

/// <summary>
/// Minimal read-only safetensors reader: enough of the format to hand the gate the file's own
/// bytes per tensor, independent of <see cref="SafeTensorsLoader"/>. Parsing the header here is
/// deliberate — comparing the loader against itself would prove nothing.
/// </summary>
sealed class CheckpointFile : IDisposable
{
    readonly FileStream stream;
    readonly long dataOffset;

    CheckpointFile(FileStream stream, long dataOffset, Entry[] entries)
    {
        this.stream = stream;
        this.dataOffset = dataOffset;
        Entries = entries;
    }

    public Entry[] Entries { get; }

    public static CheckpointFile Open(string path)
    {
        var stream = File.OpenRead(path);
        try
        {
            Span<byte> length = stackalloc byte[8];
            stream.ReadExactly(length);
            int headerSize = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(length));

            var header = new byte[headerSize];
            stream.ReadExactly(header);

            return new CheckpointFile(stream, 8 + headerSize, Parse(header));
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>Reads one tensor's raw payload exactly as it sits in the file.</summary>
    public byte[] ReadPayload(Entry entry)
    {
        var buffer = new byte[entry.Length];
        stream.Position = dataOffset + entry.Offset;
        stream.ReadExactly(buffer);
        return buffer;
    }

    public void Dispose() => stream.Dispose();

    static Entry[] Parse(byte[] header)
    {
        using var document = JsonDocument.Parse(header);
        var entries = new List<Entry>();

        foreach (var property in document.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__")
                continue;

            var tensor = property.Value;
            var shapeElement = tensor.GetProperty("shape");
            var shape = new int[shapeElement.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++)
                shape[i] = shapeElement[i].GetInt32();

            var offsets = tensor.GetProperty("data_offsets");
            long begin = offsets[0].GetInt64();
            long end = offsets[1].GetInt64();

            entries.Add(new Entry(
                property.Name,
                tensor.GetProperty("dtype").GetString()!,
                shape,
                begin,
                end - begin));
        }

        return entries.ToArray();
    }

    public sealed record Entry(string Name, string Dtype, int[] Shape, long Offset, long Length)
    {
        public long ElementCount => Shape.Aggregate(1L, (count, dimension) => count * dimension);
    }
}
