using Nivara.Samples;
using NivaraInference;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Regression tests pinning the bias-gating behavior of
/// <see cref="Qwen.SynthesizeTensors"/> introduced in #426: q/k/v biases are emitted only
/// when <see cref="LlamaConfig.AttentionBias"/> is true (Qwen-style), and absent otherwise
/// (SmolLM/Llama-style). Both synthesized checkpoints must load via
/// <see cref="LlamaLoader"/> with the matching <c>qkvBias</c>.
/// </summary>
[TestFixture]
public class SynthesizeTensorsBiasGatingTests
{
    static LlamaConfig SmolLMConfig() => new()
    {
        HiddenSize = 4,
        NumHiddenLayers = 1,
        NumAttentionHeads = 2,
        NumKeyValueHeads = 1,
        IntermediateSize = 4,
        VocabSize = 8,
        AttentionBias = false,
    };

    static LlamaConfig QwenConfig() => new()
    {
        HiddenSize = 4,
        NumHiddenLayers = 1,
        NumAttentionHeads = 2,
        NumKeyValueHeads = 1,
        IntermediateSize = 4,
        VocabSize = 8,
        AttentionBias = true,
    };

    [Test]
    public void SynthesizeTensors_SmolLM_EmitsNoQkvBiasKeys()
    {
        var tensors = NivaraInference.Qwen.SynthesizeTensors(SmolLMConfig());

        Assert.That(tensors.Keys.Any(k => k.Contains("self_attn.q_proj.bias")), Is.False,
            "SmolLM synthesis must not emit q_proj.bias");
        Assert.That(tensors.Keys.Any(k => k.Contains("self_attn.k_proj.bias")), Is.False,
            "SmolLM synthesis must not emit k_proj.bias");
        Assert.That(tensors.Keys.Any(k => k.Contains("self_attn.v_proj.bias")), Is.False,
            "SmolLM synthesis must not emit v_proj.bias");
    }

    [Test]
    public void SynthesizeTensors_Qwen_EmitsQkvBiasKeys()
    {
        var tensors = NivaraInference.Qwen.SynthesizeTensors(QwenConfig());

        Assert.That(tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias"), Is.True,
            "Qwen synthesis must emit q_proj.bias");
        Assert.That(tensors.ContainsKey("model.layers.0.self_attn.k_proj.bias"), Is.True,
            "Qwen synthesis must emit k_proj.bias");
        Assert.That(tensors.ContainsKey("model.layers.0.self_attn.v_proj.bias"), Is.True,
            "Qwen synthesis must emit v_proj.bias");
    }

    [Test]
    public void SynthesizeTensors_LoadsCorrectQkvBias()
    {
        var smolTensors = NivaraInference.Qwen.SynthesizeTensors(SmolLMConfig());
        var smolModel = LlamaLoader.Load<float, float>(SmolLMConfig(), smolTensors);
        Assert.That(smolModel.layers[0].Attention.QProj.Bias, Is.Null,
            "SmolLM must load with qkvBias == false");

        var qwenTensors = NivaraInference.Qwen.SynthesizeTensors(QwenConfig());
        var qwenModel = LlamaLoader.Load<float, float>(QwenConfig(), qwenTensors);
        Assert.That(qwenModel.layers[0].Attention.QProj.Bias, Is.Not.Null,
            "Qwen must load with qkvBias == true");
    }
}
