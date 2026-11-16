using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Tests.Inference;

[TestFixture]
[Category("Performance")]
public class LayaNativeCheckpointTests
{
    static Dictionary<string, (Half[] Data, int[] Shape)>? tensorsHalf;
    static Dictionary<string, (float[] Data, int[] Shape)>? tensorsF32;
    static ModernBertConfig? config;
    static int headLayers = 2;
    static (ModernBertEncoder<float, Half> Encoder, LayaDecisionHead<float, Half> Head)? nativeModel;
    static (ModernBertEncoder<float, float> Encoder, LayaDecisionHead<float, float> Head)? f32Model;

    static Dictionary<string, (Half[] Data, int[] Shape)> TensorsHalf
    {
        get
        {
            if (tensorsHalf != null) return tensorsHalf;
            LayaCheckpoint.RequireCheckpoint();
            tensorsHalf = SafeTensorsLoader.Read<Half>(LayaCheckpoint.ModelPath);
            return tensorsHalf;
        }
    }

    static Dictionary<string, (float[] Data, int[] Shape)> TensorsF32
    {
        get
        {
            if (tensorsF32 != null) return tensorsF32;
            LayaCheckpoint.RequireCheckpoint();
            tensorsF32 = SafeTensorsLoader.Read(LayaCheckpoint.ModelPath);
            return tensorsF32;
        }
    }

    static ModernBertConfig Config
    {
        get
        {
            if (config != null) return config;
            LayaCheckpoint.RequireCheckpoint();
            config = ModernBertConfig.FromJson(File.ReadAllText(LayaCheckpoint.EncoderConfigPath));
            return config;
        }
    }

    static int HeadLayers
    {
        get
        {
            if (headLayers != 2) return headLayers;
            LayaCheckpoint.RequireCheckpoint();
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(LayaCheckpoint.ModelDir, "rl_agent_config.json")));
            headLayers = doc.RootElement.GetProperty("head_layers").GetInt32();
            return headLayers;
        }
    }

    static (ModernBertEncoder<float, Half> Encoder, LayaDecisionHead<float, Half> Head) NativeModel
    {
        get
        {
            if (nativeModel != null) return nativeModel.Value;
            var enc = ModernBertEncoder<float, Half>.LoadWeights(TensorsHalf, Config, "encoder");
            var head = LayaDecisionHead<float, Half>.LoadWeights(TensorsHalf, HeadLayers);
            nativeModel = (enc, head);
            return nativeModel.Value;
        }
    }

    static (ModernBertEncoder<float, float> Encoder, LayaDecisionHead<float, float> Head) F32Model
    {
        get
        {
            if (f32Model != null) return f32Model.Value;
            var enc = ModernBertEncoder<float, float>.LoadWeights(TensorsF32, Config, "encoder");
            var head = LayaDecisionHead<float, float>.LoadWeights(TensorsF32, HeadLayers);
            f32Model = (enc, head);
            return f32Model.Value;
        }
    }

    [OneTimeTearDown]
    public void Release()
    {
        nativeModel = null;
        f32Model = null;
        tensorsHalf = null;
        tensorsF32 = null;
        config = null;
    }

    [Test]
    public void LoadedTensors_AreHalfBitForBit()
    {
        LayaCheckpoint.RequireCheckpoint();
        var ts = TensorsHalf;
        Assert.That(ts.Count, Is.EqualTo(206));

        long totalParams = ts.Values.Sum(t => (long)t.Data.Length);
        Assert.That(totalParams, Is.EqualTo(421293830));

        int f16 = 0, f32 = 0;
        foreach (var kv in ts)
        {
            if (kv.Key == "temperature") { f32++; Assert.That(kv.Value.Data.Length, Is.EqualTo(3)); }
            else f16++;
        }
        Assert.That(f16, Is.EqualTo(205));
        Assert.That(f32, Is.EqualTo(1));
    }

    [Test]
    public void NativeModel_KeepsCheckpointArrays_ByReference()
    {
        LayaCheckpoint.RequireCheckpoint();
        var ts = TensorsHalf;
        var (enc, head) = NativeModel;

        var typeCol = head.typeEmb.Weight!.Tensor.Data;
        Assert.That(typeCol.TryGetSpan(out var typeSpan), Is.True);
        Assert.That(typeSpan.Length, Is.EqualTo(3 * Config.HiddenSize));
        var typeLoader = ts["type_emb.weight"].Data;
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(typeSpan), ref MemoryMarshal.GetReference(typeLoader.AsSpan())), Is.True);

        var embCol = enc.tokenEmbedding.Weight!.Tensor.Data;
        Assert.That(embCol.TryGetSpan(out var embSpan), Is.True);
        var embLoader = ts["encoder.embeddings.tok_embeddings.weight"].Data;
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(embSpan), ref MemoryMarshal.GetReference(embLoader.AsSpan())), Is.True);

        var layer0 = enc.layers[0];
        var qCol = layer0.attn.qProj.Weight!.Tensor.Data;
        Assert.That(qCol.TryGetSpan(out var qSpan), Is.True);
        var wqkvLoader = ts["encoder.layers.0.attn.Wqkv.weight"].Data;
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(qSpan), ref MemoryMarshal.GetReference(wqkvLoader.AsSpan())), Is.True);

        var oCol = layer0.attn.oProj.Weight!.Tensor.Data;
        Assert.That(oCol.TryGetSpan(out var oSpan), Is.True);
        var woLoader = ts["encoder.layers.0.attn.Wo.weight"].Data;
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(oSpan), ref MemoryMarshal.GetReference(woLoader.AsSpan())), Is.True);
    }

    [Test]
    public void NativeForward_MatchesF32Path()
    {
        LayaCheckpoint.RequireCheckpoint();
        var ids = new[] { 1, 2, 3, 4, 5 };
        var n = NativeModel.Encoder.Forward(ids, ids.Length);
        var f = F32Model.Encoder.Forward(ids, ids.Length);
        n.Data.TryGetSpan(out var nSpan);
        f.Data.TryGetSpan(out var fSpan);
        float maxDiff = 0;
        for (int i = 0; i < nSpan.Length; i++)
        {
            float d = Math.Abs(nSpan[i] - fSpan[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Assert.That(maxDiff, Is.LessThanOrEqualTo(1e-4f * (1 + 1f)));
    }

    [Test]
    public void Forward_AllocationBound()
    {
        LayaCheckpoint.RequireCheckpoint();
        var ids = new[] { 1, 2, 3, 4, 5 };
        var enc = NativeModel.Encoder;
        _ = enc.Forward(ids, ids.Length);
        long before = GC.GetAllocatedBytesForCurrentThread();
        _ = enc.Forward(ids, ids.Length);
        long alloc = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.That(alloc, Is.LessThan(10 * 1024 * 1024),
            $"Forward allocated {alloc} bytes; a weight widen would be >= 1.37 GB.");
    }
    [Test]
    public void LoadWeights_AllocatesLessThanOneWeightCopy()
    {
        LayaCheckpoint.RequireCheckpoint();
        var ts = TensorsHalf;
        var cfg = Config;
        var hl = HeadLayers;

        long before = GC.GetAllocatedBytesForCurrentThread();
        var enc = ModernBertEncoder<float, Half>.LoadWeights(ts, cfg, "encoder");
        var head = LayaDecisionHead<float, Half>.LoadWeights(ts, hl);
        long alloc = GC.GetAllocatedBytesForCurrentThread() - before;
        enc.Dispose();
        head.Dispose();

        Assert.That(alloc, Is.LessThan(16 * 1024 * 1024),
            $"LoadWeights allocated {alloc} bytes; expect < 16 MB once ctor garbage is removed.");
    }
}

static class LayaCheckpoint
{
    public static string RepoRoot => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", ".."));
    public static string ModelDir => Path.Combine(RepoRoot, "samples", "data", "laya");
    public static string ModelPath => Path.Combine(ModelDir, "model.safetensors");
    public static string EncoderConfigPath => Path.Combine(ModelDir, "encoder", "config.json");

    public static void RequireCheckpoint()
    {
        string[] missing = new[] { ModelPath, EncoderConfigPath }
            .Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
            Assert.Ignore($"Laya checkpoint absent. Missing: {string.Join(", ", missing)}");
    }
}
