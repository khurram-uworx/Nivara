using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using NivaraInference;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.Tests.Inference;

/// <summary>
/// Checkpoint-level gate for the native Qwen path: BF16 weights stay exactly as shipped and
/// every kernel widens them in-register, so the run holds one copy of the snapshot payload.
/// Everything here reads the real checkpoint, so the fixture skips (with the reason) when the
/// locally-downloaded files are absent — CI has neither the weights nor the PyTorch fixtures.
/// </summary>
[TestFixture]
[Category("Performance")]
public class QwenNativeCheckpointTests
{
    static QwenCheckpointFile? checkpoint;
    static Dictionary<string, (BFloat16[] Data, int[] Shape)>? tensors;
    static LlamaConfig? config;
    static LlamaForCausalLM<float, BFloat16>? model;

    static QwenCheckpointFile Checkpoint
    {
        get
        {
            if (checkpoint != null) return checkpoint;
            QwenCheckpoint.RequireCheckpoint();
            checkpoint = QwenCheckpointFile.Open(QwenCheckpoint.ModelPath);
            return checkpoint;
        }
    }

    static Dictionary<string, (BFloat16[] Data, int[] Shape)> Tensors
    {
        get
        {
            if (tensors != null) return tensors;
            _ = Checkpoint;
            tensors = SafeTensorsLoader.Read<BFloat16>(QwenCheckpoint.ModelPath);
            return tensors;
        }
    }

    static LlamaConfig Config
    {
        get
        {
            if (config != null) return config;
            QwenCheckpoint.RequireCheckpoint();
            config = LlamaConfig.FromJson(File.ReadAllText(QwenCheckpoint.ConfigPath));
            return config;
        }
    }

    static LlamaForCausalLM<float, BFloat16> Model
    {
        get
        {
            if (model != null) return model;
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

    [Test]
    public void LoadedTensors_AreTheCheckpointPayloadBitForBit()
    {
        var file = Checkpoint;
        var loaded = Tensors;

        Assert.That(loaded.Count, Is.EqualTo(file.Entries.Length),
            "the loader must return exactly one array per checkpoint tensor");
        Assert.That(file.Entries.Sum(e => e.ElementCount), Is.EqualTo(494_032_768L),
            "Qwen2.5-0.5B parameter count; a different checkpoint would invalidate every fixture below");

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

    [Test]
    public void NativeModel_KeepsTheCheckpointArrays_AndOnlyNormsAndBiasesAreWidened()
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

        // Widened (TModel = float) parameters: RMSNorm gammas (49) + QKV biases (72).
        long normElements = nativeModel.finalNorm.Weight!.Tensor.Length;
        long biasElements = 0;
        for (int i = 0; i < nativeModel.layers.Length; i++)
        {
            normElements += nativeModel.layers[i].InputNorm.Weight!.Tensor.Length;
            normElements += nativeModel.layers[i].PostNorm.Weight!.Tensor.Length;
            var layer = nativeModel.layers[i];
            if (layer.Attention.QProj.Bias is { } qb) biasElements += qb.Tensor.Length;
            if (layer.Attention.KProj.Bias is { } kb) biasElements += kb.Tensor.Length;
            if (layer.Attention.VProj.Bias is { } vb) biasElements += vb.Tensor.Length;
        }

        long widenedElements = normElements + biasElements;
        int widenedTensorCount = file.Entries.Count(e => IsWidened(e.Name));

        Assert.That(storedCount + widenedTensorCount, Is.EqualTo(file.Entries.Length),
            "every checkpoint tensor must be either a stored projection/embedding or a widened norm/bias");
        Assert.That(widenedTensorCount, Is.EqualTo(121),
            "49 RMSNorm gammas + 72 QKV biases are widened to F32");
        Assert.That(widenedElements, Is.EqualTo(49L * 896 + 24L * (896 + 128 + 128)),
            "widened element count (norms + biases)");

        long payloadBytes = file.Entries.Sum(e => e.Length);
        long widenedBf16Bytes = file.Entries.Where(e => IsWidened(e.Name)).Sum(e => e.Length);
        Assert.That(storedBytes, Is.EqualTo(payloadBytes - widenedBf16Bytes),
            "the projection/embedding store must be exactly the checkpoint payload minus the BF16 widened tensors");

        TestContext.Out.WriteLine(
            $"checkpoint payload {payloadBytes:N0} B; model parameter storage " +
            $"{storedBytes + widenedElements * 4:N0} B (BF16 weights {storedBytes:N0} B + F32 widened {widenedElements * 4:N0} B)");
    }

    [Test]
    public void NativeGeneration_MatchesWidenedF32Path()
    {
        // Build the widened (F32) model from the same checkpoint, once.
        var f32Tensors = SafeTensorsLoader.Read<float>(QwenCheckpoint.ModelPath);
        var f32Model = LlamaLoader.Load<float, float>(Config, f32Tensors);

        try
        {
            var promptIds = new List<int>(new Gpt2BpeTokenizer(
                QwenCheckpoint.VocabPath, QwenCheckpoint.MergesPath,
                tokenizerJsonPath: QwenCheckpoint.TokenizerJsonPath).Encode("What is the capital of France?"));

            const int maxNew = 16;
            var nativeIds = NivaraInference.Qwen.Generate(Model, Config, promptIds, maxNew, useKvCache: true);
            var f32Ids = NivaraInference.Qwen.Generate(f32Model, Config, promptIds, maxNew, useKvCache: true);

            TestContext.Out.WriteLine($"native: {string.Join(",", nativeIds)}");
            TestContext.Out.WriteLine($"f32:    {string.Join(",", f32Ids)}");

            Assert.That(nativeIds.Count, Is.EqualTo(f32Ids.Count),
                "native and widened F32 must agree on stop position");
            for (int i = 0; i < nativeIds.Count; i++)
                Assert.That(nativeIds[i], Is.EqualTo(f32Ids[i]),
                    $"generated token[{i}] differs (native {nativeIds[i]} vs f32 {f32Ids[i]})");
        }
        finally
        {
            f32Model.Dispose();
        }
    }

    static bool IsWidened(string name)
        => name.EndsWith(".input_layernorm.weight", StringComparison.Ordinal)
        || name.EndsWith(".post_attention_layernorm.weight", StringComparison.Ordinal)
        || name == "model.norm.weight"
        || name.EndsWith(".self_attn.q_proj.bias", StringComparison.Ordinal)
        || name.EndsWith(".self_attn.k_proj.bias", StringComparison.Ordinal)
        || name.EndsWith(".self_attn.v_proj.bias", StringComparison.Ordinal);

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

static class QwenCheckpoint
{
    public const string Prompt = "What is the capital of France?";
    public const int MaxNewTokens = 16;

    public static string RepoRoot
        => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", ".."));

    public static string ModelDir => Path.Combine(RepoRoot, "samples", "data", "qwen2.5-0.5b-instruct");
    public static string ModelPath => Path.Combine(ModelDir, "model.safetensors");
    public static string ConfigPath => Path.Combine(ModelDir, "config.json");
    public static string VocabPath => Path.Combine(ModelDir, "vocab.json");
    public static string MergesPath => Path.Combine(ModelDir, "merges.txt");
    public static string TokenizerJsonPath => Path.Combine(ModelDir, "tokenizer.json");

    public static void RequireCheckpoint()
    {
        string[] missing = new[] { ModelPath, ConfigPath, VocabPath, MergesPath, TokenizerJsonPath }
            .Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
            Assert.Ignore($"Qwen checkpoint files absent; skipping native-checkpoint verification. Missing: {string.Join(", ", missing)}");
    }
}

sealed class QwenCheckpointFile : IDisposable
{
    readonly FileStream stream;
    readonly long dataOffset;
    public QwenCheckpointFile(FileStream stream, long dataOffset, QwenCheckpointEntry[] entries)
    {
        this.stream = stream;
        this.dataOffset = dataOffset;
        Entries = entries;
    }
    public QwenCheckpointEntry[] Entries { get; }
    public static QwenCheckpointFile Open(string path)
    {
        var stream = File.OpenRead(path);
        try
        {
            Span<byte> length = stackalloc byte[8];
            stream.ReadExactly(length);
            int headerSize = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(length));
            var header = new byte[headerSize];
            stream.ReadExactly(header);
            return new QwenCheckpointFile(stream, 8 + headerSize, Parse(header));
        }
        catch { stream.Dispose(); throw; }
    }
    public byte[] ReadPayload(QwenCheckpointEntry entry)
    {
        var buffer = new byte[entry.Length];
        stream.Position = dataOffset + entry.Offset;
        stream.ReadExactly(buffer);
        return buffer;
    }
    public void Dispose() => stream.Dispose();
    static QwenCheckpointEntry[] Parse(byte[] header)
    {
        using var document = JsonDocument.Parse(header);
        var entries = new List<QwenCheckpointEntry>();
        foreach (var property in document.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue;
            var tensor = property.Value;
            var shapeElement = tensor.GetProperty("shape");
            var shape = new int[shapeElement.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++) shape[i] = shapeElement[i].GetInt32();
            var offsets = tensor.GetProperty("data_offsets");
            long begin = offsets[0].GetInt64();
            long end = offsets[1].GetInt64();
            entries.Add(new QwenCheckpointEntry(property.Name, tensor.GetProperty("dtype").GetString()!, shape, begin, end - begin));
        }
        return entries.ToArray();
    }
}

sealed record QwenCheckpointEntry(string Name, string Dtype, int[] Shape, long Offset, long Length)
{
    public long ElementCount => Shape.Aggregate(1L, (count, dimension) => count * dimension);
}
