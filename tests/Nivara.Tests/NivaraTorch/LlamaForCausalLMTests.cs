using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LlamaForCausalLMTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void LlamaForCausalLM_TiedHead_MatchesPyTorch()
    {
        const int vocab = 128;
        const int hidden = 32;
        const int layers = 2;
        const int heads = 4;
        const int kvHeads = 2;
        const int inter = 64;
        const int maxPos = 32;

        var inputIds = TestHelpers.LoadInt64Bin("llama_causal_lm_input_ids.bin");

        var embed = TestHelpers.LoadBin("llama_causal_lm_embed.bin");
        var finalNorm = TestHelpers.LoadBin("llama_causal_lm_final_norm.bin");
        var expectedLogits = TestHelpers.LoadBin("llama_causal_lm_logits.bin");
        var expectedEmbedGrad = TestHelpers.LoadBin("llama_causal_lm_embed_grad.bin");

        var layerInGamma = new float[layers][];
        var layerPostGamma = new float[layers][];
        var layerQw = new float[layers][];
        var layerKw = new float[layers][];
        var layerVw = new float[layers][];
        var layerOw = new float[layers][];
        var layerGate = new float[layers][];
        var layerUp = new float[layers][];
        var layerDown = new float[layers][];
        for (int i = 0; i < layers; i++)
        {
            layerInGamma[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_in_gamma.bin");
            layerPostGamma[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_post_gamma.bin");
            layerQw[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_qw.bin");
            layerKw[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_kw.bin");
            layerVw[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_vw.bin");
            layerOw[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_ow.bin");
            layerGate[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_gatew.bin");
            layerUp[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_upw.bin");
            layerDown[i] = TestHelpers.LoadBin($"llama_causal_lm_layer{i}_downw.bin");
        }

        using var model = new LlamaForCausalLM<float>(
            vocabSize: vocab, hiddenSize: hidden, numHiddenLayers: layers,
            numHeads: heads, numKeyValueHeads: kvHeads, intermediateSize: inter,
            rmsNormEps: 1e-5f, maxPositionEmbeddings: maxPos, ropeTheta: 10000f);

        var state = new Dictionary<string, ReverseGradTensor<float>>();

        // Embedding (Module_0) - requiresGrad true so the embedding gradient is tracked.
        state["Module_0.Weight"] = ReverseGradTensor<float>.FromMatrix(
            embed, vocab, hidden, requiresGrad: true);

        for (int i = 0; i < layers; i++)
        {
            string layerPrefix = $"Module_{i + 1}";
            // LlamaDecoderBlock registers: InputNorm(0), Attention(1), PostNorm(2), GateProj(3), UpProj(4), DownProj(5)
            state[$"{layerPrefix}.Module_0.Weight"] =
                ReverseGradTensor<float>.FromArray(layerInGamma[i], requiresGrad: false);
            // Attention (Module_1) registers: QProj(0), KProj(1), VProj(2), OProj(3), Rotary(4)
            int headDim = hidden / heads;
            int qWidth = heads * headDim;
            int kvWidth = kvHeads * headDim;
            state[$"{layerPrefix}.Module_1.Module_0.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerQw[i], qWidth, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_1.Module_1.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerKw[i], kvWidth, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_1.Module_2.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerVw[i], kvWidth, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_1.Module_3.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerOw[i], hidden, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_2.Weight"] =
                ReverseGradTensor<float>.FromArray(layerPostGamma[i], requiresGrad: false);
            state[$"{layerPrefix}.Module_3.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerGate[i], inter, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_4.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerUp[i], inter, hidden, requiresGrad: false);
            state[$"{layerPrefix}.Module_5.Weight"] =
                ReverseGradTensor<float>.FromMatrix(layerDown[i], hidden, inter, requiresGrad: false);
        }

        // Final norm (Module_{layers+1})
        state[$"Module_{layers + 1}.Weight"] =
            ReverseGradTensor<float>.FromArray(finalNorm, requiresGrad: false);

        model.LoadStateDict(state, strict: true);
        model.Eval();

        var logits = model.Forward(inputIds);

        Assert.That(logits.Shape, Is.EqualTo(new[] { inputIds.Length, vocab }));
        TestHelpers.AssertTensorEqual(expectedLogits, TestHelpers.ExtractOutput(logits),
            label: "LlamaForCausalLM_logits");

        // Embedding gradient parity (sum-backward).
        ReverseGradOperations.Sum(logits).Backward();
        var embedGrad = model.Embed.Weight!.Tensor.Grad
            ?? throw new InvalidOperationException("Embedding has no gradient");
        var embedGradArr = new float[embedGrad.Length];
        for (int i = 0; i < embedGrad.Length; i++) embedGradArr[i] = embedGrad[i];
        TestHelpers.AssertTensorEqual(expectedEmbedGrad, embedGradArr,
            absTol: 1e-4f, relTol: 1e-3f, label: "LlamaForCausalLM_embed_grad");
    }
}