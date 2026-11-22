using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;
using System.Collections;
using System.Reflection;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public sealed class SilkLlamaLoRATrainerParityTests
{
    [Test]
    public void Forward_SyntheticWeights_LoraQv_MatchesCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, config) = Nivara.Tests.Qwen.QwenTestWeights.Synthesize(seed: 1234);
        int[] prompt = [1, 2, 3, 4];
        const int rank = 4;
        const float alpha = 8f;

        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);
        cpuModel.AttachLora(LlamaLoRATarget.Q | LlamaLoRATarget.V, rank, alpha);

        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);
        CopyCpuLoraToSilk(cpuModel, silk, config.NumHiddenLayers);

        var cpuLogits = cpuModel.Forward(prompt).Data.ToArray();
        var silkLogits = silk.Forward(prompt);

        int seqLen = prompt.Length;
        Assert.That(silkLogits.Length, Is.EqualTo(seqLen * config.VocabSize));

        Assert.That(ArgMaxLast(silkLogits, seqLen, config.VocabSize),
            Is.EqualTo(ArgMaxLast(cpuLogits, seqLen, config.VocabSize)), "Last-token argmax mismatch");

        float maxAbs = MaxAbsDiff(cpuLogits, silkLogits);
        float scale = MaxAbs(cpuLogits) + 1f;
        Assert.That(maxAbs / scale, Is.LessThanOrEqualTo(1.5e-3f),
            $"Max abs diff {maxAbs}, scale {scale}, rel {maxAbs / scale}");
    }

    [Test]
    public void BackwardPromptMaskedCE_LoraQv_GradsMatchCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, config) = Nivara.Tests.Qwen.QwenTestWeights.Synthesize(seed: 1234);
        int[] input = [1, 2, 3, 4, 5, 6];
        int[] labels = [-100, -100, -100, 4, 5, 6];
        const int rank = 4;
        const float alpha = 8f;

        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);
        cpuModel.AttachLora(LlamaLoRATarget.Q | LlamaLoRATarget.V, rank, alpha);

        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);
        CopyCpuLoraToSilk(cpuModel, silk, config.NumHiddenLayers);
        ZeroCpuLoraGrads(cpuModel);
        silk.ZeroLoraGradients();

        float cpuLoss;
        var lossFn = new CrossEntropyLoss<float>();
        using (GradientUtils.Grad())
        {
            var logits = cpuModel.Forward(input);
            var logits2d = logits;
            if (logits2d.Shape.Length == 3) logits2d.Reshape(input.Length, config.VocabSize);
            var loss = lossFn.Forward(logits2d, labels, ignoreIndex: -100);
            loss.Backward();
            cpuLoss = loss.Data.ToArray()[0];
        }

        silk.Forward(input);
        var (silkLoss, valid) = silk.BackwardPromptMaskedCrossEntropy(labels, input.Length);
        silk.BackwardTransformerBlocks(input.Length);

        Assert.That(valid, Is.EqualTo(3));
        Assert.That(silkLoss, Is.EqualTo(cpuLoss).Within(1.5e-3f * (MathF.Abs(cpuLoss) + 1f)),
            $"Loss mismatch: cpu={cpuLoss}, silk={silkLoss}");

        AssertLoraGradsMatch(cpuModel, silk, config.NumHiddenLayers);
    }

    static SilkLlamaLoRATrainer CreateSilkTrainer(SilkRuntime runtime, LlamaConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors, int rank, float alpha, SilkLlamaLoRATarget targets)
    {
        var silk = new SilkLlamaLoRATrainer(runtime, new SilkLlamaLoRATrainer.Options(
            VocabSize: config.VocabSize,
            HiddenDim: config.HiddenSize,
            IntermediateDim: config.IntermediateSize,
            NumLayers: config.NumHiddenLayers,
            NumHeads: config.NumAttentionHeads,
            NumKvHeads: config.NumKeyValueHeads,
            MaxSeqLen: 16,
            RmsNormEps: config.RmsNormEps,
            RopeTheta: config.RopeTheta,
            Rank: rank,
            Alpha: alpha,
            Targets: targets));
        silk.UploadBaseWeights(tensors);
        return silk;
    }

    static void AssertLoraGradsMatch(LlamaForCausalLM<float, float> cpu, SilkLlamaLoRATrainer silk, int layers, float relTol = 3e-2f)
    {
        var cpuLayers = GetCpuLayers(cpu);
        int li = 0;
        foreach (var cpuLayer in cpuLayers)
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            CompareAdapterGrad(attn, "QAdapter", silk, li, SilkLlamaLoRATarget.Q, relTol);
            CompareAdapterGrad(attn, "KAdapter", silk, li, SilkLlamaLoRATarget.K, relTol);
            CompareAdapterGrad(attn, "VAdapter", silk, li, SilkLlamaLoRATarget.V, relTol);
            CompareAdapterGrad(attn, "OAdapter", silk, li, SilkLlamaLoRATarget.O, relTol);
            CompareAdapterGrad(cpuLayer, "GateAdapter", silk, li, SilkLlamaLoRATarget.Gate, relTol);
            CompareAdapterGrad(cpuLayer, "UpAdapter", silk, li, SilkLlamaLoRATarget.Up, relTol);
            CompareAdapterGrad(cpuLayer, "DownAdapter", silk, li, SilkLlamaLoRATarget.Down, relTol);
            li++;
        }
        Assert.That(li, Is.EqualTo(layers));
    }

    static void CompareAdapterGrad(object owner, string prop, SilkLlamaLoRATrainer silk, int layer, SilkLlamaLoRATarget target, float relTol)
    {
        var adapter = owner.GetType().GetProperty(prop)!.GetValue(owner);
        if (adapter is null) return;
        var (aGrad, bGrad) = GetAdapterGrads(adapter);
        var (silkGA, silkGB) = silk.DownloadLoraGrads(layer, target);
        float aRel = MaxAbsDiff(aGrad, silkGA) / (MaxAbs(aGrad) + 1e-6f);
        float bRel = MaxAbsDiff(bGrad, silkGB) / (MaxAbs(bGrad) + 1e-6f);
        Assert.That(aRel, Is.LessThanOrEqualTo(relTol), $"Layer {layer} {target} GradA rel diff {aRel} (maxAbs={MaxAbsDiff(aGrad, silkGA)})");
        Assert.That(bRel, Is.LessThanOrEqualTo(relTol), $"Layer {layer} {target} GradB rel diff {bRel} (maxAbs={MaxAbsDiff(bGrad, silkGB)})");
    }

    static void CopyCpuLoraToSilk(LlamaForCausalLM<float, float> cpu, SilkLlamaLoRATrainer silk, int layers)
    {
        var cpuLayers = GetCpuLayers(cpu);
        int li = 0;
        foreach (var cpuLayer in cpuLayers)
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            CopyAdapter(attn, "QAdapter", silk, li, SilkLlamaLoRATarget.Q);
            CopyAdapter(attn, "KAdapter", silk, li, SilkLlamaLoRATarget.K);
            CopyAdapter(attn, "VAdapter", silk, li, SilkLlamaLoRATarget.V);
            CopyAdapter(attn, "OAdapter", silk, li, SilkLlamaLoRATarget.O);
            CopyAdapter(cpuLayer, "GateAdapter", silk, li, SilkLlamaLoRATarget.Gate);
            CopyAdapter(cpuLayer, "UpAdapter", silk, li, SilkLlamaLoRATarget.Up);
            CopyAdapter(cpuLayer, "DownAdapter", silk, li, SilkLlamaLoRATarget.Down);
            li++;
        }
        Assert.That(li, Is.EqualTo(layers));
    }

    static void CopyAdapter(object owner, string prop, SilkLlamaLoRATrainer silk, int layer, SilkLlamaLoRATarget target)
    {
        var adapter = owner.GetType().GetProperty(prop)!.GetValue(owner);
        if (adapter is null) return;
        var (aData, bData) = GetAdapterData(adapter);
        silk.SetLoraWeights(layer, target, aData, bData);
    }

    static void ZeroCpuLoraGrads(LlamaForCausalLM<float, float> cpu)
    {
        foreach (var cpuLayer in GetCpuLayers(cpu))
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            ZeroAdapterGrad(attn, "QAdapter");
            ZeroAdapterGrad(attn, "KAdapter");
            ZeroAdapterGrad(attn, "VAdapter");
            ZeroAdapterGrad(attn, "OAdapter");
            ZeroAdapterGrad(cpuLayer, "GateAdapter");
            ZeroAdapterGrad(cpuLayer, "UpAdapter");
            ZeroAdapterGrad(cpuLayer, "DownAdapter");
        }
    }

    static void ZeroAdapterGrad(object owner, string prop)
    {
        var adapter = owner.GetType().GetProperty(prop)!.GetValue(owner);
        if (adapter is null) return;
        var aTensor = GetParamTensor(adapter, "A");
        var bTensor = GetParamTensor(adapter, "B");
        aTensor.GetType().GetMethod("ZeroGrad")!.Invoke(aTensor, null);
        bTensor.GetType().GetMethod("ZeroGrad")!.Invoke(bTensor, null);
    }

    static (float[] A, float[] B) GetAdapterData(object adapter)
    {
        var aTensor = GetParamTensor(adapter, "A");
        var bTensor = GetParamTensor(adapter, "B");
        return (GetTensorData(aTensor), GetTensorData(bTensor));
    }

    static (float[] A, float[] B) GetAdapterGrads(object adapter)
    {
        var aTensor = GetParamTensor(adapter, "A");
        var bTensor = GetParamTensor(adapter, "B");
        return (GetTensorGrad(aTensor), GetTensorGrad(bTensor));
    }

    static object GetParamTensor(object adapter, string paramName)
    {
        var param = adapter.GetType().GetProperty(paramName)!.GetValue(adapter)!;
        return param.GetType().GetProperty("Tensor")!.GetValue(param)!;
    }

    static float[] GetTensorData(object tensor) => ((IEnumerable<float>)tensor.GetType().GetProperty("Data")!.GetValue(tensor)!).ToArray();
    static float[] GetTensorGrad(object tensor)
    {
        var grad = tensor.GetType().GetProperty("Grad")?.GetValue(tensor);
        if (grad is float[] fa) return fa;
        if (grad is IEnumerable<float> ef) return ef.ToArray();
        throw new InvalidOperationException("Could not read tensor grad as float array.");
    }

    static IEnumerable GetCpuLayers(LlamaForCausalLM<float, float> cpu)
    {
        var field = cpu.GetType().GetField("layers", BindingFlags.NonPublic | BindingFlags.Instance);
        if (field?.GetValue(cpu) is IEnumerable e) return e;
        throw new InvalidOperationException("Could not access CPU model layers.");
    }

    static int ArgMaxLast(float[] data, int seqLen, int vocab)
    {
        int off = (seqLen - 1) * vocab;
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (data[off + i] > data[off + best]) best = i;
        return best;
    }

    static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    static float MaxAbs(float[] a)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i]));
        return m;
    }

    [Test]
    public void TrainStep_LoraQv_AdamStepMatchesCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, config) = Nivara.Tests.Qwen.QwenTestWeights.Synthesize(seed: 1234);
        int[] input = [1, 2, 3, 4, 5, 6];
        int[] labels = [-100, -100, -100, 4, 5, 6];
        const int rank = 4;
        const float alpha = 8f;
        const float lr = 1e-3f;

        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);
        var cpuAdapters = cpuModel.AttachLora(LlamaLoRATarget.Q | LlamaLoRATarget.V, rank, alpha);
        var opt = new Nivara.AutoDiff.Optimizer.AdamW<float>(lr);
        foreach (var a in cpuAdapters) opt.AddParameterGroup(new[] { a.A, a.B });

        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);
        CopyCpuLoraToSilk(cpuModel, silk, config.NumHiddenLayers);

        // CPU one step
        float cpuLoss;
        var lossFn = new CrossEntropyLoss<float>();
        using (GradientUtils.Grad())
        {
            var logits = cpuModel.Forward(input);
            var logits2d = logits;
            if (logits2d.Shape.Length == 3) logits2d.Reshape(input.Length, config.VocabSize);
            var loss = lossFn.Forward(logits2d, labels, ignoreIndex: -100);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            cpuLoss = loss.Data.ToArray()[0];
        }

        float silkLoss = silk.TrainStep(input, labels, lr);

        Assert.That(silkLoss, Is.EqualTo(cpuLoss).Within(1.5e-3f * (MathF.Abs(cpuLoss) + 1f)),
            $"Loss mismatch: cpu={cpuLoss}, silk={silkLoss}");

        // Compare updated LoRA weights
        var cpuLayers = GetCpuLayers(cpuModel);
        int li = 0;
        foreach (var cpuLayer in cpuLayers)
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            CompareAdapterData(attn, "QAdapter", silk, li, SilkLlamaLoRATarget.Q);
            CompareAdapterData(attn, "VAdapter", silk, li, SilkLlamaLoRATarget.V);
            li++;
        }
    }

    static void CompareAdapterData(object attn, string prop, SilkLlamaLoRATrainer silk, int layer, SilkLlamaLoRATarget target)
    {
        var adapter = attn.GetType().GetProperty(prop)!.GetValue(attn);
        if (adapter is null) return;
        var (aData, bData) = GetAdapterData(adapter);
        var (silkA, silkB) = silk.DownloadLoraWeights(layer, target);
        float aRel = MaxAbsDiff(aData, silkA) / (MaxAbs(aData) + 1e-6f);
        float bRel = MaxAbsDiff(bData, silkB) / (MaxAbs(bData) + 1e-6f);
        Assert.That(aRel, Is.LessThanOrEqualTo(3e-2f), $"Layer {layer} {target} A rel diff {aRel}");
        Assert.That(bRel, Is.LessThanOrEqualTo(3e-2f), $"Layer {layer} {target} B rel diff {bRel}");
    }

    [Test]
    public void AdapterBundle_RoundTrip_PreservesWeights()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, config) = Nivara.Tests.Qwen.QwenTestWeights.Synthesize(seed: 1234);
        const int rank = 4;
        const float alpha = 8f;
        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);
        using var silk2 = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);

        int inF = config.HiddenSize;
        int outF = config.HiddenSize;
        var a = Enumerable.Range(0, inF * rank).Select(i => (i % 13 - 6) * 0.01f).ToArray();
        var b = Enumerable.Range(0, rank * outF).Select(i => (i % 7 - 3) * 0.02f).ToArray();
        silk.SetLoraWeights(0, SilkLlamaLoRATarget.Q, a, b);

        var path = Path.Combine(Path.GetTempPath(), $"nivara-lora-bundle-{Guid.NewGuid():N}.json");
        try
        {
            silk.SaveAdapterBundle(path, "qwen2.5-0.5b-instruct", "qwen");
            silk2.LoadAdapterBundle(path);
            var (a2, b2) = silk2.DownloadLoraWeights(0, SilkLlamaLoRATarget.Q);
            Assert.That(MaxAbsDiff(a, a2), Is.EqualTo(0f));
            Assert.That(MaxAbsDiff(b, b2), Is.EqualTo(0f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void AdapterBundle_LoadsIntoCpuModel_PreservesWeights()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, config) = Nivara.Tests.Qwen.QwenTestWeights.Synthesize(seed: 1234);
        const int rank = 4;
        const float alpha = 8f;
        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);

        int inF = config.HiddenSize;
        int outF = config.HiddenSize;
        var a = Enumerable.Range(0, inF * rank).Select(i => (i % 13 - 6) * 0.01f).ToArray();
        var b = Enumerable.Range(0, rank * outF).Select(i => (i % 7 - 3) * 0.02f).ToArray();
        silk.SetLoraWeights(0, SilkLlamaLoRATarget.Q, a, b);

        var path = Path.Combine(Path.GetTempPath(), $"nivara-lora-bundle-{Guid.NewGuid():N}.json");
        try
        {
            silk.SaveAdapterBundle(path, "qwen2.5-0.5b-instruct", "qwen");

            // Load into a CPU model via the shared bundle DTO.
            var cpuModel = LlamaLoader.Load<float, float>(config, tensors);
            var cpuAdapters = cpuModel.AttachLora(LlamaLoRATarget.Q | LlamaLoRATarget.V, rank, alpha);

            var bundle = Nivara.Silk.LoraSftAdapterBundle.Load(path);
            Assert.That(bundle.Adapters.Length, Is.EqualTo(cpuAdapters.Count));

            for (int i = 0; i < cpuAdapters.Count; i++)
            {
                var entry = bundle.Adapters[i];
                var aBytes = Convert.FromBase64String(entry.A);
                var bBytes = Convert.FromBase64String(entry.B);
                var aData = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(aBytes).ToArray();
                var bData = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bBytes).ToArray();
                cpuAdapters[i].A.Tensor = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(aData, cpuAdapters[i].A.Tensor.Shape[0], cpuAdapters[i].A.Tensor.Shape[1], requiresGrad: true);
                cpuAdapters[i].B.Tensor = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(bData, cpuAdapters[i].B.Tensor.Shape[0], cpuAdapters[i].B.Tensor.Shape[1], requiresGrad: true);
            }

            // Verify the CPU adapter weights match the original Silk weights for layer 0 Q.
            var (cpuA, cpuB) = (GetTensorData(cpuAdapters[0].A.Tensor), GetTensorData(cpuAdapters[0].B.Tensor));
            Assert.That(MaxAbsDiff(cpuA, a), Is.EqualTo(0f));
            Assert.That(MaxAbsDiff(cpuB, b), Is.EqualTo(0f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Test]
    public void ForwardAndGrads_SmolLMShape_AllTargets_MatchesCpu()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Llama LoRA parity verification."); }

        var (tensors, smolConfig) = Nivara.Tests.SmolLM.SmolLMTestWeights.Synthesize(seed: 1234);
        var config = new LlamaConfig
        {
            HiddenSize = smolConfig.HiddenSize,
            NumHiddenLayers = smolConfig.NumHiddenLayers,
            NumAttentionHeads = smolConfig.NumAttentionHeads,
            NumKeyValueHeads = smolConfig.NumKeyValueHeads,
            IntermediateSize = smolConfig.IntermediateSize,
            VocabSize = smolConfig.VocabSize,
            MaxPositionEmbeddings = smolConfig.MaxPositionEmbeddings,
            RmsNormEps = smolConfig.RmsNormEps,
            RopeTheta = smolConfig.RopeTheta,
            TieWordEmbeddings = true,
            HiddenAct = "silu",
            AttentionBias = smolConfig.AttentionBias,
        };

        int[] input = [1, 2, 3, 4, 5, 6];
        int[] labels = [-100, -100, -100, 4, 5, 6];
        const int rank = 4;
        const float alpha = 8f;
        var cpuTargets = LlamaLoRATarget.Q | LlamaLoRATarget.K | LlamaLoRATarget.V | LlamaLoRATarget.O | LlamaLoRATarget.FFN;
        var silkTargets = SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.K | SilkLlamaLoRATarget.V | SilkLlamaLoRATarget.O | SilkLlamaLoRATarget.FFN;

        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);
        cpuModel.AttachLora(cpuTargets, rank, alpha);

        using var runtime = new SilkRuntime();
        using var silk = CreateSilkTrainer(runtime, config, tensors, rank, alpha, silkTargets);
        CopyCpuLoraToSilk(cpuModel, silk, config.NumHiddenLayers);
        ZeroCpuLoraGrads(cpuModel);
        silk.ZeroLoraGradients();

        var cpuLogits = cpuModel.Forward(input).Data.ToArray();
        var silkLogits = silk.Forward(input);
        Assert.That(ArgMaxLast(silkLogits, input.Length, config.VocabSize),
            Is.EqualTo(ArgMaxLast(cpuLogits, input.Length, config.VocabSize)));
        Assert.That(MaxAbsDiff(cpuLogits, silkLogits) / (MaxAbs(cpuLogits) + 1f), Is.LessThanOrEqualTo(2e-3f));

        float cpuLoss;
        var lossFn = new CrossEntropyLoss<float>();
        using (GradientUtils.Grad())
        {
            var logits = cpuModel.Forward(input);
            var logits2d = logits;
            if (logits2d.Shape.Length == 3) logits2d.Reshape(input.Length, config.VocabSize);
            var loss = lossFn.Forward(logits2d, labels, ignoreIndex: -100);
            loss.Backward();
            cpuLoss = loss.Data.ToArray()[0];
        }

        silk.Forward(input);
        var (silkLoss, _) = silk.BackwardPromptMaskedCrossEntropy(labels, input.Length);
        silk.BackwardTransformerBlocks(input.Length);

        Assert.That(silkLoss, Is.EqualTo(cpuLoss).Within(2e-3f * (MathF.Abs(cpuLoss) + 1f)));
        AssertLoraGradsMatch(cpuModel, silk, config.NumHiddenLayers, relTol: 5e-2f);
    }
}
