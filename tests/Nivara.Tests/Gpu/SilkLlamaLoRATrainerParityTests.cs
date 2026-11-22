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
        Assert.That(maxAbs / scale, Is.LessThanOrEqualTo(2e-3f),
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
        Assert.That(silkLoss, Is.EqualTo(cpuLoss).Within(2e-3f * (MathF.Abs(cpuLoss) + 1f)),
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

    static void AssertLoraGradsMatch(LlamaForCausalLM<float, float> cpu, SilkLlamaLoRATrainer silk, int layers)
    {
        var cpuLayers = GetCpuLayers(cpu);
        int li = 0;
        foreach (var cpuLayer in cpuLayers)
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            CompareAdapterGrad(attn, "QAdapter", silk, li, SilkLlamaLoRATarget.Q);
            CompareAdapterGrad(attn, "VAdapter", silk, li, SilkLlamaLoRATarget.V);
            li++;
        }
        Assert.That(li, Is.EqualTo(layers));
    }

    static void CompareAdapterGrad(object attn, string prop, SilkLlamaLoRATrainer silk, int layer, SilkLlamaLoRATarget target)
    {
        var adapter = attn.GetType().GetProperty(prop)!.GetValue(attn);
        if (adapter is null) return;
        var (aGrad, bGrad) = GetAdapterGrads(adapter);
        var (silkGA, silkGB) = silk.DownloadLoraGrads(layer, target);

        float aScale = MaxAbs(aGrad) + 1e-6f;
        float bScale = MaxAbs(bGrad) + 1e-6f;
        float aRel = MaxAbsDiff(aGrad, silkGA) / aScale;
        float bRel = MaxAbsDiff(bGrad, silkGB) / bScale;
        Assert.That(aRel, Is.LessThanOrEqualTo(5e-2f), $"Layer {layer} {target} GradA rel diff {aRel} (maxAbs={MaxAbsDiff(aGrad, silkGA)})");
        Assert.That(bRel, Is.LessThanOrEqualTo(5e-2f), $"Layer {layer} {target} GradB rel diff {bRel} (maxAbs={MaxAbsDiff(bGrad, silkGB)})");
    }

    static void CopyCpuLoraToSilk(LlamaForCausalLM<float, float> cpu, SilkLlamaLoRATrainer silk, int layers)
    {
        var cpuLayers = GetCpuLayers(cpu);
        int li = 0;
        foreach (var cpuLayer in cpuLayers)
        {
            var attn = cpuLayer.GetType().GetProperty("Attention")!.GetValue(cpuLayer)!;
            CopyAdapter(attn, "QAdapter", silk, li, SilkLlamaLoRATarget.Q);
            CopyAdapter(attn, "VAdapter", silk, li, SilkLlamaLoRATarget.V);
            li++;
        }
        Assert.That(li, Is.EqualTo(layers));
    }

    static void CopyAdapter(object attn, string prop, SilkLlamaLoRATrainer silk, int layer, SilkLlamaLoRATarget target)
    {
        var adapter = attn.GetType().GetProperty(prop)!.GetValue(attn);
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
            ZeroAdapterGrad(attn, "VAdapter");
        }
    }

    static void ZeroAdapterGrad(object attn, string prop)
    {
        var adapter = attn.GetType().GetProperty(prop)!.GetValue(attn);
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
}
