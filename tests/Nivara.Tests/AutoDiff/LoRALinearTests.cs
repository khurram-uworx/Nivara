using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;
using System.Numerics;
using System.Text.Json;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LoRALinearTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static ReverseGradTensor<float> MakeInput(int batch, int inFeatures)
    {
        var data = new float[batch * inFeatures];
        for (int i = 0; i < data.Length; i++) data[i] = (i + 1) * 0.1f;
        var t = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        t.Reshape(batch, inFeatures);
        return t;
    }

    [Test]
    public void LoRALinear_Step0_AdaptedEqualsBase_Exact()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2);
        var x = MakeInput(2, 4);

        var baseOut = lora.Base.Forward(x);
        var adaptedOut = lora.Forward(x);

        Assert.That(adaptedOut.Length, Is.EqualTo(baseOut.Length));
        for (int i = 0; i < baseOut.Length; i++)
            Assert.That(adaptedOut[i], Is.EqualTo(baseOut[i]));
    }

    [Test]
    public void LoRALinear_FrozenBase_HasNoGradients_AfterBackward()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2);
        var x = MakeInput(2, 4);

        var y = lora.Forward(x);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
        grad.Reshape(y.Shape[0], y.Shape[1]);
        y.Backward(grad);

        foreach (var (name, param) in lora.Base.GetParameters())
            Assert.That(param.Tensor.Grad, Is.Null,
                $"Frozen base parameter '{name}' should have no gradient");

        Assert.That(lora.A.Tensor.Grad, Is.Not.Null);
        Assert.That(lora.B.Tensor.Grad, Is.Not.Null);
    }

    [Test]
    public void LoRALinear_AdapterGradients_MatchFiniteDifferences()
    {
        const int inF = 3, outF = 2, r = 2, batch = 2;
        using var lora = new LoRALinear<float>(inF, outF, r, alpha: 4f);
        var x = MakeInput(batch, inF);

        // Autodiff gradients
        float[] gradA, gradB;
        using (var gs = GradientUtils.Grad())
        {
            var y = lora.Forward(x);
            var ones = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
            ones.Reshape(y.Shape[0], y.Shape[1]);
            y.Backward(ones);

            gradA = new float[lora.A.Tensor.Length];
            for (int i = 0; i < gradA.Length; i++) gradA[i] = lora.A.Tensor.Grad![i];
            gradB = new float[lora.B.Tensor.Length];
            for (int i = 0; i < gradB.Length; i++) gradB[i] = lora.B.Tensor.Grad![i];
        }

        float Loss()
        {
            var y = lora.Forward(x);
            float s = 0f;
            for (int i = 0; i < y.Length; i++) s += y[i];
            return s;
        }

        void SetParam(Parameter<float> p, int idx, float val)
        {
            var t = p.Tensor;
            var data = new float[t.Length];
            for (int i = 0; i < data.Length; i++) data[i] = t[i];
            data[idx] = val;
            var col = NivaraColumn<float>.Create(data);
            var ten = new ReverseGradTensor<float>(col, requiresGrad: true);
            ten.Reshape(t.Shape[0], t.Shape[1]);
            p.Tensor = ten;
        }

        const float eps = 1e-3f;

        // A
        for (int idx = 0; idx < lora.A.Tensor.Length; idx++)
        {
            var t = lora.A.Tensor;
            float orig = t[idx];
            SetParam(lora.A, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.A, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.A, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradA[idx], Is.EqualTo(fd).Within(1e-2f),
                $"A grad mismatch at {idx}: autodiff={gradA[idx]} fd={fd}");
        }

        // B
        for (int idx = 0; idx < lora.B.Tensor.Length; idx++)
        {
            var t = lora.B.Tensor;
            float orig = t[idx];
            SetParam(lora.B, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.B, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.B, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradB[idx], Is.EqualTo(fd).Within(1e-2f),
                $"B grad mismatch at {idx}: autodiff={gradB[idx]} fd={fd}");
        }
    }

    [Test]
    public void LoRALinear_Serialization_RoundTrip()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var path = Path.Combine(Path.GetTempPath(), $"lora_test_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(lora, path);

            using var loaded = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
            ModelSerializer.Load(loaded, path);

            var x = MakeInput(2, 4);
            var a = lora.Forward(x);
            var b = loaded.Forward(x);
            for (int i = 0; i < a.Length; i++)
                Assert.That(b[i], Is.EqualTo(a[i]).Within(1e-6f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_RoundTrip_IdenticalLogitsOnFreshBase()
    {
        const int inF = 4, outF = 3, r = 2;
        float[] weightData = new float[outF * inF];
        for (int i = 0; i < weightData.Length; i++) weightData[i] = (i + 1) * 0.25f;

        using var source = MakeLoRALinear(weightData, inF, outF, r, alpha: 4f);
        SetAdapterValues(source.A, source.B, inF, outF, r);

        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            LoraAdapterSerializer.Save(source, path, "qwen2.5-0.5b-instruct", new[] { "q_proj" });

            using var fresh = MakeLoRALinear(weightData, inF, outF, r, alpha: 4f);
            LoraAdapterSerializer.Load(fresh, path, "qwen2.5-0.5b-instruct");

            var x = MakeInput(2, inF);
            var sourceOut = source.Forward(x);
            var freshOut = fresh.Forward(x);
            Assert.That(freshOut.Length, Is.EqualTo(sourceOut.Length));
            for (int i = 0; i < sourceOut.Length; i++)
                Assert.That(freshOut[i], Is.EqualTo(sourceOut[i]).Within(1e-6f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_Mixed_RoundTrip_IdenticalLogitsOnFreshBase()
    {
        const int inF = 4, outF = 3, r = 2;
        var sourceBase = Linear<float, BFloat16>.Uninitialized(inF, outF);
        var sourceWeights = new BFloat16[outF * inF];
        for (int i = 0; i < sourceWeights.Length; i++)
            sourceWeights[i] = BFloat16.CreateChecked((float)(0.05 * (i + 1)));
        sourceBase.BindWeight(ReverseGradTensor<BFloat16>.FromMatrix(sourceWeights, outF, inF, requiresGrad: true));

        using var source = new LoRALinear<float, BFloat16>(sourceBase, r, alpha: 4f);
        SetAdapterValues(source.A, source.B, inF, outF, r);

        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            LoraAdapterSerializer.Save(source, path, "qwen2.5-0.5b-instruct", new[] { "q_proj" });

            var freshBase = Linear<float, BFloat16>.Uninitialized(inF, outF);
            var freshWeights = new BFloat16[outF * inF];
            for (int i = 0; i < freshWeights.Length; i++)
                freshWeights[i] = BFloat16.CreateChecked((float)(0.05 * (i + 1)));
            freshBase.BindWeight(ReverseGradTensor<BFloat16>.FromMatrix(freshWeights, outF, inF, requiresGrad: true));

            using var fresh = new LoRALinear<float, BFloat16>(freshBase, r, alpha: 4f);
            LoraAdapterSerializer.Load(fresh, path, "qwen2.5-0.5b-instruct");

            var x = MakeInput(2, inF);
            var sourceOut = source.Forward(x);
            var freshOut = fresh.Forward(x);
            Assert.That(freshOut.Length, Is.EqualTo(sourceOut.Length));
            for (int i = 0; i < sourceOut.Length; i++)
                Assert.That(freshOut[i], Is.EqualTo(sourceOut[i]).Within(1e-6f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_File_ContainsAdapterOnly_TensorsAndMetadata()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            LoraAdapterSerializer.Save(lora, path, "qwen2.5-0.5b-instruct", new[] { "q_proj" });

            var json = File.ReadAllText(path);
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;

            Assert.That(root.TryGetProperty("Format", out var format), Is.True);
            Assert.That(format.GetString(), Is.EqualTo(LoraAdapterSerializer.AdapterFormat));
            Assert.That(root.TryGetProperty("BaseModelId", out var baseModelId), Is.True);
            Assert.That(baseModelId.GetString(), Is.EqualTo("qwen2.5-0.5b-instruct"));
            Assert.That(root.TryGetProperty("Rank", out var rank), Is.True);
            Assert.That(rank.GetInt32(), Is.EqualTo(lora.Rank));
            Assert.That(root.TryGetProperty("Version", out var version), Is.True);
            Assert.That(version.GetInt32(), Is.EqualTo(1));

            var parameters = root.GetProperty("Parameters");
            Assert.That(parameters.TryGetProperty("A", out var a), Is.True);
            Assert.That(parameters.TryGetProperty("B", out var b), Is.True);
            Assert.That(parameters.TryGetProperty("Weight", out _), Is.False);
            Assert.That(parameters.TryGetProperty("Bias", out _), Is.False);

            Assert.That(root.TryGetProperty("Targets", out var targets), Is.True);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_Load_RejectsMismatchedRank()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            LoraAdapterSerializer.Save(lora, path, "qwen2.5-0.5b-instruct");

            using var mismatched = new LoRALinear<float>(4, 3, r: 4, alpha: 4f);
            Assert.That(() => LoraAdapterSerializer.Load(mismatched, path, "qwen2.5-0.5b-instruct"),
                Throws.InvalidOperationException);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_Load_RejectsMismatchedBaseModel()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            LoraAdapterSerializer.Save(lora, path, "qwen2.5-0.5b-instruct");

            using var otherBase = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
            Assert.That(() => LoraAdapterSerializer.Load(otherBase, path, "smollm3b-causal"),
                Throws.InvalidOperationException);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_Load_RejectsUnsupportedFormat()
    {
        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        try
        {
            File.WriteAllText(path, "{\"Format\":\"nivara-lora-adapter-v2\",\"Type\":\"Float\",\"Version\":1,\"BaseModelId\":\"qwen2.5-0.5b-instruct\",\"Rank\":2,\"Alpha\":4.0,\"Targets\":[],\"Parameters\":{\"A\":{\"Shape\":[4,2],\"Values\":\"\"},\"B\":{\"Shape\":[2,3],\"Values\":\"\"}}}");

            using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
            Assert.That(() => LoraAdapterSerializer.Load(lora, path, "qwen2.5-0.5b-instruct"),
                Throws.InvalidOperationException);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void LoRALinear_AdapterSerializer_Save_RejectsMergedAdapter()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        lora.Merge();

        var path = Path.Combine(Path.GetTempPath(), $"lora_adapter_{Guid.NewGuid()}.json");
        Assert.That(() => LoraAdapterSerializer.Save(lora, path, "qwen2.5-0.5b-instruct"),
            Throws.InvalidOperationException);
    }

    [Test]
    public void LoRALinear_Merge_EquivalentToUnmerged()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var x = MakeInput(2, 4);

        var before = new float[lora.Forward(x).Length];
        var outBefore = lora.Forward(x);
        for (int i = 0; i < before.Length; i++) before[i] = outBefore[i];

        lora.Merge();

        var after = lora.Forward(x);
        for (int i = 0; i < before.Length; i++)
            Assert.That(after[i], Is.EqualTo(before[i]).Within(1e-5f));
    }

    [Test]
    public void LoRALinear_Mixed_Step0_AdaptedEqualsBase_Exact()
    {
        using var lora = new LoRALinear<float, BFloat16>(4, 3, r: 2, alpha: 4f);
        var x = MakeInput(2, 4);

        var baseOut = lora.Base.Forward(x);
        var adaptedOut = lora.Forward(x);

        Assert.That(adaptedOut.Length, Is.EqualTo(baseOut.Length));
        for (int i = 0; i < baseOut.Length; i++)
            Assert.That(adaptedOut[i], Is.EqualTo(baseOut[i]));
    }

    [Test]
    public void LoRALinear_Mixed_FrozenBase_HasNoGradients_AfterBackward()
    {
        using var lora = new LoRALinear<float, BFloat16>(4, 3, r: 2);
        var x = MakeInput(2, 4);

        var y = lora.Forward(x);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
        grad.Reshape(y.Shape[0], y.Shape[1]);
        y.Backward(grad);

        Assert.That(lora.Base.Weight!.Tensor.Grad, Is.Null,
            "Frozen base weight should have no gradient");
        Assert.That(lora.A.Tensor.Grad, Is.Not.Null);
        Assert.That(lora.B.Tensor.Grad, Is.Not.Null);
    }

    [Test]
    public void LoRALinear_Mixed_AdapterGradients_MatchFiniteDifferences()
    {
        const int inF = 3, outF = 2, r = 2, batch = 2;
        using var lora = new LoRALinear<float, BFloat16>(inF, outF, r, alpha: 4f);
        var x = MakeInput(batch, inF);

        float[] gradA, gradB;
        using (var gs = GradientUtils.Grad())
        {
            var y = lora.Forward(x);
            var ones = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
            ones.Reshape(y.Shape[0], y.Shape[1]);
            y.Backward(ones);

            gradA = new float[lora.A.Tensor.Length];
            for (int i = 0; i < gradA.Length; i++) gradA[i] = lora.A.Tensor.Grad![i];
            gradB = new float[lora.B.Tensor.Length];
            for (int i = 0; i < gradB.Length; i++) gradB[i] = lora.B.Tensor.Grad![i];
        }

        float Loss()
        {
            var y = lora.Forward(x);
            float s = 0f;
            for (int i = 0; i < y.Length; i++) s += y[i];
            return s;
        }

        void SetParam(Parameter<float> p, int idx, float val)
        {
            var t = p.Tensor;
            var data = new float[t.Length];
            for (int i = 0; i < data.Length; i++) data[i] = t[i];
            data[idx] = val;
            var col = NivaraColumn<float>.Create(data);
            var ten = new ReverseGradTensor<float>(col, requiresGrad: true);
            ten.Reshape(t.Shape[0], t.Shape[1]);
            p.Tensor = ten;
        }

        const float eps = 1e-3f;

        for (int idx = 0; idx < lora.A.Tensor.Length; idx++)
        {
            var t = lora.A.Tensor;
            float orig = t[idx];
            SetParam(lora.A, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.A, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.A, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradA[idx], Is.EqualTo(fd).Within(1e-2f),
                $"A grad mismatch at {idx}: autodiff={gradA[idx]} fd={fd}");
        }

        for (int idx = 0; idx < lora.B.Tensor.Length; idx++)
        {
            var t = lora.B.Tensor;
            float orig = t[idx];
            SetParam(lora.B, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.B, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.B, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradB[idx], Is.EqualTo(fd).Within(1e-2f),
                $"B grad mismatch at {idx}: autodiff={gradB[idx]} fd={fd}");
        }
    }

    [Test]
    public void LoRALinear_Mixed_Merge_EquivalentToUnmerged()
    {
        using var lora = new LoRALinear<float, BFloat16>(4, 3, r: 2, alpha: 4f);
        var x = MakeInput(2, 4);

        var before = new float[lora.Forward(x).Length];
        var outBefore = lora.Forward(x);
        for (int i = 0; i < before.Length; i++) before[i] = outBefore[i];

        lora.Merge();

        var after = lora.Forward(x);
        for (int i = 0; i < before.Length; i++)
            Assert.That(after[i], Is.EqualTo(before[i]).Within(1e-5f));
    }

    [Test]
    public void LoRALinear_Mixed_LlamaFrozenBase_AdapterGradients_AreAdapterOnly()
    {
        var config = new LlamaConfig
        {
            HiddenSize = 4,
            NumHiddenLayers = 1,
            NumAttentionHeads = 2,
            NumKeyValueHeads = 1,
            IntermediateSize = 4,
            VocabSize = 8,
            AttentionBias = false,
        };
        var tensors = NivaraInference.Qwen.SynthesizeTensors<System.Numerics.BFloat16>(config);
        using var model = LlamaLoader.Load<float, BFloat16>(config, tensors, requiresGrad: false);
        var attn = model.layers[0].Attention;
        var qLora = new LoRALinear<float, BFloat16>(attn.QProj, r: 2, alpha: 4f);
        var vLora = new LoRALinear<float, BFloat16>(attn.VProj, r: 2, alpha: 4f);
        attn.QAdapter = qLora;
        attn.VAdapter = vLora;

        var logits = model.Forward(new[] { 0, 1, 2, 3 });
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, logits.Length).ToArray()), requiresGrad: false);
        grad.Reshape(logits.Shape[0], logits.Shape[1]);
        logits.Backward(grad);

        Assert.That(qLora.A.Tensor.Grad, Is.Not.Null);
        Assert.That(qLora.B.Tensor.Grad, Is.Not.Null);
        Assert.That(vLora.A.Tensor.Grad, Is.Not.Null);
        Assert.That(vLora.B.Tensor.Grad, Is.Not.Null);
        Assert.That(attn.QProj.Weight!.Tensor.Grad, Is.Null,
            "Frozen mixed base weight should have no gradient");
    }

    static LoRALinear<float> MakeLoRALinear(float[] weightData, int inFeatures, int outFeatures, int r, float alpha)
    {
        var baseLinear = Linear<float>.Uninitialized(inFeatures, outFeatures);
        baseLinear.BindWeight(ReverseGradTensor<float>.FromMatrix(weightData, outFeatures, inFeatures, requiresGrad: true));
        return new LoRALinear<float>(baseLinear, r, alpha);
    }

    static void SetAdapterValues(Parameter<float> a, Parameter<float> b, int inFeatures, int outFeatures, int r)
    {
        var aData = new float[inFeatures * r];
        for (int i = 0; i < aData.Length; i++)
            aData[i] = (float)((i % 7) + 1) * 0.25f - 0.5f;
        a.Tensor = ReverseGradTensor<float>.FromMatrix(aData, inFeatures, r, requiresGrad: true);

        var bData = new float[r * outFeatures];
        for (int i = 0; i < bData.Length; i++)
            bData[i] = (float)((i % 5) + 1) * 0.5f - 1f;
        b.Tensor = ReverseGradTensor<float>.FromMatrix(bData, r, outFeatures, requiresGrad: true);
    }
}
