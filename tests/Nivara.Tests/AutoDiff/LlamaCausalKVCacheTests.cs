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

[TestFixture]
public class SamplingTests
{
    static int Select(ReadOnlySpan<double> logits, float temperature, float topP, int vocab, Random rng)
    {
        if (temperature <= 0f)
        {
            int best = 0;
            for (int i = 1; i < vocab; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }

        double max = logits[0];
        for (int i = 1; i < vocab; i++)
            if (logits[i] > max) max = logits[i];

        var exp = new double[vocab];
        double sum = 0;
        for (int i = 0; i < vocab; i++)
        {
            exp[i] = Math.Exp((logits[i] - max) / temperature);
            sum += exp[i];
        }

        if (topP < 1f)
        {
            var order = new int[vocab];
            for (int i = 0; i < vocab; i++) order[i] = i;
            Array.Sort(order, (a, b) => exp[b].CompareTo(exp[a]));

            double cum = 0;
            int cut = vocab;
            for (int i = 0; i < vocab; i++)
            {
                cum += exp[order[i]];
                if (cum >= topP) { cut = i + 1; break; }
            }

            double keptSum = 0;
            for (int i = 0; i < cut; i++) keptSum += exp[order[i]];

            double r = rng.NextDouble() * keptSum;
            double acc = 0;
            for (int i = 0; i < cut; i++)
            {
                acc += exp[order[i]];
                if (r <= acc) return order[i];
            }
            return order[0];
        }

        double rFull = rng.NextDouble() * sum;
        double accFull = 0;
        for (int i = 0; i < vocab; i++)
        {
            accFull += exp[i];
            if (rFull <= accFull) return i;
        }
        return vocab - 1;
    }

    static double[] UniformLogits(int vocab)
    {
        var logits = new double[vocab];
        for (int i = 0; i < vocab; i++) logits[i] = i;
        return logits;
    }

    [Test]
    public void Sampling_Deterministic_GivenSameSeed()
    {
        var logits = UniformLogits(64);
        int tokA = Select(logits, temperature: 0.6f, topP: 0.9f, vocab: 64, new Random(42));
        int tokB = Select(logits, temperature: 0.6f, topP: 0.9f, vocab: 64, new Random(42));
        Assert.That(tokA, Is.EqualTo(tokB));
    }

    [Test]
    public void Sampling_DifferentSeeds_YieldDifferentTokens()
    {
        var logits = UniformLogits(100);
        const int vocab = 100;
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 20; seed++)
            seen.Add(Select(logits, temperature: 1.0f, topP: 1.0f, vocab, new Random(seed)));
        Assert.That(seen.Count, Is.GreaterThan(1), "Different seeds must produce at least 2 distinct tokens.");
    }

    [Test]
    public void Sampling_TemperatureOne_ProducesReasonableDistribution()
    {
        // With a strong signal (token 7 >> others), temperature=1 should sample 7 most often.
        var logits = new double[10];
        logits[7] = 5.0; // strong favorite
        int count = 0;
        for (int i = 0; i < 100; i++)
            count += Select(logits, temperature: 1.0f, topP: 1.0f, 10, new Random(i)) == 7 ? 1 : 0;
        Assert.That(count, Is.GreaterThan(70), $"Expected token 7 to be selected >70% of the time, got {count}/100.");
    }

    [Test]
    public void Sampling_TopP_LimitsChoices()
    {
        // tokens: 0=10.0, 1=9.0, 2=8.0, rest=0.0
        // topP=0.5 → only token 0 (probability ~10.0/(10+9+8) ≈ 37%, < 50%) should be in nucleus
        var logits = new double[10];
        logits[0] = 10.0;
        logits[1] = 9.0;
        logits[2] = 8.0;
        var selected = new HashSet<int>();
        for (int seed = 0; seed < 100; seed++)
            selected.Add(Select(logits, temperature: 1.0f, topP: 0.5f, 10, new Random(seed)));
        Assert.That(selected, Does.Contain(0), "Token 0 (highest logit) must be selectable with topP=0.5.");
        Assert.That(selected, Does.Not.Contain(2), "Token 2 (3rd) is outside the nucleus with topP=0.5.");
    }
}
