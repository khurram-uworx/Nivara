using NivaraChat.Shared;
using NUnit.Framework;
using System;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class TokenSamplingParityTests
{
    static int LocalSelect(ReadOnlySpan<double> logits, float temperature, float topP, int vocab, Random rng)
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

    [Test]
    public void Parity_UniformLogits()
    {
        var logits = new double[64];
        for (int i = 0; i < 64; i++) logits[i] = i;
        for (int seed = 0; seed < 50; seed++)
        {
            int a = LocalSelect(logits, 0.6f, 0.9f, 64, new Random(seed));
            int b = TokenSampling.Sample(logits, 0.6f, 0.9f, 64, new Random(seed));
            Assert.That(b, Is.EqualTo(a), $"mismatch seed {seed}");
        }
    }

    [Test]
    public void Parity_StrongSignal()
    {
        var logits = new double[10];
        logits[7] = 5.0;
        for (int seed = 0; seed < 50; seed++)
        {
            int a = LocalSelect(logits, 1.0f, 0.5f, 10, new Random(seed));
            int b = TokenSampling.Sample(logits, 1.0f, 0.5f, 10, new Random(seed));
            Assert.That(b, Is.EqualTo(a), $"mismatch seed {seed}");
        }
    }

    [Test]
    public void Parity_TopP1()
    {
        var logits = new double[100];
        for (int i = 0; i < 100; i++) logits[i] = i * 0.5;
        for (int seed = 0; seed < 50; seed++)
        {
            int a = LocalSelect(logits, 1.0f, 1.0f, 100, new Random(seed));
            int b = TokenSampling.Sample(logits, 1.0f, 1.0f, 100, new Random(seed));
            Assert.That(b, Is.EqualTo(a), $"mismatch seed {seed}");
        }
    }
}