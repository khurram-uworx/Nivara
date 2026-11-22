using Nivara.AutoDiff;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class QwenSyntheticSmokeTests
{
    [Test]
    public void ForwardPrefillAndCached_SyntheticWeights_AreDeterministicAndFinite()
    {
        var (tensors, config) = QwenTestWeights.Synthesize(seed: 1234);

        int[] prompt = [1, 2, 3, 4];
        var first = GreedyDecode(config, tensors, prompt, maxNewTokens: 4);
        var second = GreedyDecode(config, tensors, prompt, maxNewTokens: 4);

        Assert.That(second.GeneratedIds, Is.EqualTo(first.GeneratedIds));
        Assert.That(first.GeneratedIds, Has.Count.EqualTo(4));
        Assert.That(first.LastLogits.All(float.IsFinite), Is.True);
    }

    static (List<int> GeneratedIds, float[] LastLogits) GreedyDecode(
        LlamaConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int[] prompt,
        int maxNewTokens)
    {
        var model = LlamaLoader.Load<float, float>(config, tensors);
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        var prefill = model.ForwardPrefill(prompt, cache);
        var last = LastRow(prefill, config.VocabSize);
        int next = ArgMax(last);

        var generated = new List<int>(maxNewTokens);
        int position = prompt.Length;
        for (int t = 0; t < maxNewTokens; t++)
        {
            generated.Add(next);
            if (next == config.EosTokenId)
                return (generated, last);

            var step = model.ForwardCached(next, position++, cache);
            last = LastRow(step, config.VocabSize);
            next = ArgMax(last);
        }

        return (generated, last);
    }

    static float[] LastRow(ReverseGradTensor<float> logits, int vocabSize)
    {
        var row = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            row[i] = logits.Data[i];
        return row;
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
