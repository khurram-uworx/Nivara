using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaCausalKVCacheTests
{
    static LlamaForCausalLM<float> TinyModel()
        => new(vocabSize: 128, hiddenSize: 32, numHiddenLayers: 2, numHeads: 4, numKeyValueHeads: 2,
            intermediateSize: 64, rmsNormEps: 1e-5f, maxPositionEmbeddings: 32, ropeTheta: 10000f);

    static void AssertClose(float expected, float actual, float tol = 1e-5f)
        => Assert.That(actual, Is.EqualTo(expected).Within(tol), $"Expected {expected}, got {actual}.");

    void CompareCachedVsFull(LlamaForCausalLM<float> model, int[] tokens)
    {
        // Greedy reference: logits over the whole sequence at once.
        var full = model.Forward(tokens);

        // kvWidth = numKeyValueHeads(2) * headDim(hiddenSize/numHeads = 32/4 = 8) = 16.
        using var cache = new LlamaKVCache<float>(2, 16);

        for (int p = 0; p < tokens.Length; p++)
        {
            var step = model.ForwardCached(tokens[p], p, cache);

            // Full forward row p must equal the cached logits for token p.
            for (int v = 0; v < 128; v++)
            {
                float fullVal = full[p * 128 + v];
                float stepVal = step[0 * 128 + v];
                AssertClose(fullVal, stepVal);
            }
        }
    }

    [Test]
    public void ForwardCached_MatchesFullForward_WhenSeedingPromptTokenByToken()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];
        CompareCachedVsFull(model, tokens);
    }

    [Test]
    public void ForwardCached_GrowsCacheBeyondInitialCapacity()
    {
        using var model = TinyModel();
        // initialCapacity defaults to 16; force growth with a longer sequence.
        int[] tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        CompareCachedVsFull(model, tokens);
    }

    [Test]
    public void ForwardCached_ResetAllowsReuseForANewSequence()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45];
        int kvWidth = 2 * (32 / 4);

        using var cache = new LlamaKVCache<float>(2, kvWidth);
        CompareCachedVsFull(model, tokens);

        cache.Reset();

        CompareCachedVsFull(model, tokens);
    }
}
