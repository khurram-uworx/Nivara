using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Helpers;
using NUnit.Framework;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Tests.NivaraChat;

[TestFixture]
public class LoraAdapterLoaderTests
{
    static LlamaForCausalLM<float, float> TinyModel() =>
        new(vocabSize: 16, hiddenSize: 8, numHiddenLayers: 1, numHeads: 2, numKeyValueHeads: 2, intermediateSize: 16, maxPositionEmbeddings: 32);

    static string SaveTinyBundle(string path, int rank = 2, float alpha = 4f, LlamaLoRATarget targets = LlamaLoRATarget.Q | LlamaLoRATarget.V)
    {
        using var model = TinyModel();
        var adapters = model.AttachLora(targets, rank, alpha);

        // Give adapters non-zero data so load is observable.
        for (int i = 0; i < adapters.Count; i++)
        {
            var aData = new float[adapters[i].A.Tensor.Length];
            var bData = new float[adapters[i].B.Tensor.Length];
            for (int j = 0; j < aData.Length; j++) aData[j] = 0.1f * (i + 1) * (j + 1);
            for (int j = 0; j < bData.Length; j++) bData[j] = 0.2f * (i + 1) * (j + 1);
            adapters[i].A.Tensor = ReverseGradTensor<float>.FromMatrix(aData, adapters[i].A.Tensor.Shape[0], adapters[i].A.Tensor.Shape[1], requiresGrad: true);
            adapters[i].B.Tensor = ReverseGradTensor<float>.FromMatrix(bData, adapters[i].B.Tensor.Shape[0], adapters[i].B.Tensor.Shape[1], requiresGrad: true);
        }

        var entries = adapters.Select((a, i) => new LoraSftAdapterEntry
        {
            Index = i,
            Layer = -1,
            Target = "",
            AShape = a.A.Tensor.Shape.ToArray(),
            BShape = a.B.Tensor.Shape.ToArray(),
            A = Convert.ToBase64String(MemoryMarshal.AsBytes(a.A.Tensor.Data.ToArray().AsSpan())),
            B = Convert.ToBase64String(MemoryMarshal.AsBytes(a.B.Tensor.Data.ToArray().AsSpan())),
        }).ToArray();

        var bundle = new LoraSftAdapterBundle
        {
            BaseModelId = "test-model",
            Template = "qwen",
            Rank = rank,
            Alpha = alpha,
            Targets = targets.ToString(),
            TrainableParams = adapters.Sum(a => a.A.Tensor.Length + a.B.Tensor.Length),
            Adapters = entries,
        };
        LoraSftAdapterBundle.Save(path, bundle);
        return path;
    }

    [Test]
    public void Apply_ValidBundle_LoadsAdapter_IntoFreshModel()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora-test-{Guid.NewGuid()}.json");
        try
        {
            SaveTinyBundle(path);

            using var model = TinyModel();
            LoraAdapterLoader.Apply(model, path);

            // Adapters should be attached.
            Assert.That(model.layers[0].Attention.QAdapter, Is.Not.Null);
            Assert.That(model.layers[0].Attention.VAdapter, Is.Not.Null);
            Assert.That(model.layers[0].HasAdapter, Is.True);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void Apply_ValidBundle_ForwardProducesFiniteLogits()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora-test-{Guid.NewGuid()}.json");
        try
        {
            SaveTinyBundle(path);

            using var model = TinyModel();
            LoraAdapterLoader.Apply(model, path);

            int[] tokens = [1, 2, 3, 4, 5];
            var logits = model.Forward(tokens);
            Assert.That(logits.Length, Is.EqualTo(tokens.Length * 16));
            for (int i = 0; i < logits.Length; i++)
                Assert.That(float.IsFinite(logits[i]), Is.True);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void Apply_RankMismatch_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora-test-{Guid.NewGuid()}.json");
        try
        {
            SaveTinyBundle(path, rank: 2);

            using var model = TinyModel();
            // Attach with different rank first is not possible via Apply (it attaches itself).
            // Instead, corrupt the bundle rank.
            var bundle = LoraSftAdapterBundle.Load(path);
            bundle.Rank = 99;
            LoraSftAdapterBundle.Save(path, bundle);

            Assert.Throws<InvalidOperationException>(() => LoraAdapterLoader.Apply(model, path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void Apply_CountMismatch_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora-test-{Guid.NewGuid()}.json");
        try
        {
            SaveTinyBundle(path);

            // Remove an adapter entry.
            var bundle = LoraSftAdapterBundle.Load(path);
            bundle.Adapters = bundle.Adapters.Take(bundle.Adapters.Length - 1).ToArray();
            LoraSftAdapterBundle.Save(path, bundle);

            using var model = TinyModel();
            Assert.Throws<InvalidOperationException>(() => LoraAdapterLoader.Apply(model, path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void Apply_ShapeMismatch_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora-test-{Guid.NewGuid()}.json");
        try
        {
            SaveTinyBundle(path);

            var bundle = LoraSftAdapterBundle.Load(path);
            bundle.Adapters[0].AShape = new[] { 999, 999 };
            LoraSftAdapterBundle.Save(path, bundle);

            using var model = TinyModel();
            Assert.Throws<InvalidOperationException>(() => LoraAdapterLoader.Apply(model, path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void Apply_MissingFile_Throws()
    {
        using var model = TinyModel();
        Assert.Throws<FileNotFoundException>(() => LoraAdapterLoader.Apply(model, Path.Combine(Path.GetTempPath(), "nonexistent-lora.json")));
    }
}
