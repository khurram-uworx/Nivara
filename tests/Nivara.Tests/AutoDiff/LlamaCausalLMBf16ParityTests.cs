using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using Nivara.Primitives;
using Nivara.Samples;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Model-level BFloat16-vs-float32 parity for the Qwen/Llama causal LM. BF16-on-disk weights
/// widened to F32 are bit-identical to BF16-native weights widened per lane inside the dot
/// kernel, so the only runtime differences are dot accumulation order and per-layer BF16
/// activation rounding. These tests pin greedy argmax equality and logits within a documented
/// relative-ish tolerance over prefill and KV-cached decode (plan: P1 — on-the-fly BF16).
/// </summary>
[TestFixture]
public class LlamaCausalLMBf16ParityTests
{
    const int Vocab = 128;
    const int DecodeSteps = 8;
    const int Layers = 2;
    const int KvWidth = 16; // kvHeads 2 * (hidden 32 / heads 4)

    static readonly int[] Prompt = [10, 23, 45, 12, 99, 3, 88, 55];

    static LlamaForCausalLM<T> Model<T>(int hidden = 32, int heads = 4, int kvHeads = 2, int maxPos = 64)
        where T : struct, IFloatingPointIeee754<T>
        => new(vocabSize: Vocab, hiddenSize: hidden, numHiddenLayers: Layers, numHeads: heads,
            numKeyValueHeads: kvHeads, intermediateSize: 2 * hidden, rmsNormEps: 1e-5f,
            maxPositionEmbeddings: maxPos, ropeTheta: 10000f);

    /// <summary>Seeds both models from a SINGLE deterministic random float pool (LCG, ±0.05) narrowed
    /// once to BF16 — mirrors real bf16 checkpoint storage so the two models share identical weight
    /// VALUES (bf16-rounded): the f32 model sees the pool widened back to float (widen-at-load pipe)
    /// and the bf16 model sees the raw bf16 values (on-the-fly widen pipe). The only remaining delta
    /// is per-layer BF16 activation rounding + dot accumulation order. The LCG keeps the pool
    /// identical across runs (module init uses <see cref="Random.Shared"/>, which would flake
    /// draw-tuned tolerances).</summary>
    static void SeedIdenticalBf16Pool(
        LlamaForCausalLM<float> f32,
        LlamaForCausalLM<BFloat16> bf16)
    {
        uint lcg = 0x9E3779B9u;
        var pool = new Dictionary<string, ReverseGradTensor<float>>();
        foreach (var (name, source) in f32.StateDict())
        {
            int n = source.Length;
            var values = new float[n];
            for (int i = 0; i < n; i++)
            {
                lcg = lcg * 1664525u + 1013904223u;
                values[i] = ((float)(lcg / (double)uint.MaxValue) - 0.5f) * 0.1f;
            }
            pool[name] = source.Shape.Length == 2
                ? ReverseGradTensor<float>.FromMatrix(values, (int)source.Shape[0], (int)source.Shape[1], requiresGrad: false)
                : new ReverseGradTensor<float>(NivaraColumn<float>.Create(values), requiresGrad: false);
        }
        f32.LoadStateDict(pool);
        bf16.LoadStateDict(pool.ToDictionary(
            kv => kv.Key,
            kv => TypeConverter.Convert<float, BFloat16>(kv.Value)));
    }

    static int ArgMaxLastRow<T>(ReverseGradTensor<T> logits)
        where T : struct, IFloatingPointIeee754<T>
    {
        logits.Data.TryGetSpan(out var span);
        int offset = span.Length - Vocab;
        int best = 0;
        var bestVal = span[offset];
        for (int i = 1; i < Vocab; i++)
        {
            if (span[offset + i] > bestVal)
            {
                bestVal = span[offset + i];
                best = i;
            }
        }
        return best;
    }

    /// <summary>BF16 last-row logits must track the f32 ones. Tolerance = 0.002 floor +
    /// <paramref name="relTol"/> of the largest |logit|. Observed max deltas on the deterministic
    /// pool: ~1e-4 through prefill/decode (fused GQA kernels keep attention weights in F32
    /// accumulation), ~1.2e-2 through the full-seq re-forward (BF16 attention-weight rounding
    /// compounds ~1% per layer) — so callers pass 0.002 for the production surface and 0.05 for
    /// the re-forward surface. Scale = max(1, max|logit|) so the bound has an absolute floor
    /// (diffs can concentrate on small-magnitude logits). Only delta is per-layer BF16 activation rounding; widen is
    /// lossless, so there is no systematic bias.</summary>
    static bool LogitsClose(ReverseGradTensor<float> f, ReverseGradTensor<BFloat16> b, float relTol)
    {
        f.Data.TryGetSpan(out var fs);
        b.Data.TryGetSpan(out var bs);
        int offset = fs.Length - Vocab;
        float maxAbsF = 0f;
        float maxDiff = 0f;
        for (int i = 0; i < Vocab; i++)
        {
            float fv = fs[offset + i];
            maxAbsF = Math.Max(maxAbsF, Math.Abs(fv));
            maxDiff = Math.Max(maxDiff, Math.Abs(fv - (float)bs[i]));
        }
        return maxDiff <= 0.002f + relTol * Math.Max(1f, maxAbsF);
    }

    [Test]
    public void Bf16_ForwardLastRow_ArgmaxAndLogits_MatchF32()
    {
        bool prior = NivaraPrimitives.UseWidenSimd;
        NivaraPrimitives.UseWidenSimd = true;
        try
        {
            using var f32 = Model<float>();
            using var bf16 = Model<BFloat16>();
            SeedIdenticalBf16Pool(f32, bf16);

            var f32Logits = f32.Forward(Prompt);
            var bf16Logits = bf16.Forward(Prompt);

            Assert.That(LogitsClose(f32Logits, bf16Logits, relTol: 0.05f), Is.True,
                "BF16 last-row logits diverge from F32 beyond the documented tolerance.");
            Assert.That(ArgMaxLastRow(bf16Logits), Is.EqualTo(ArgMaxLastRow(f32Logits)),
                "BF16 greedy argmax diverged from F32 on the prompt.");
        }
        finally
        {
            NivaraPrimitives.UseWidenSimd = prior;
        }
    }

    [Test]
    public void Bf16_PrefillThenDecode_ArgmaxSequenceAndLogits_MatchF32()
    {
        bool prior = NivaraPrimitives.UseWidenSimd;
        NivaraPrimitives.UseWidenSimd = true;
        try
        {
            using var f32 = Model<float>();
            using var bf16 = Model<BFloat16>();
            SeedIdenticalBf16Pool(f32, bf16);

            using var f32Cache = new LlamaKVCache<float>(Layers, KvWidth);
            using var bf16Cache = new LlamaKVCache<BFloat16>(Layers, KvWidth);

            var f32Logits = f32.ForwardPrefill(Prompt, f32Cache);
            var bf16Logits = bf16.ForwardPrefill(Prompt, bf16Cache);

            Assert.That(LogitsClose(f32Logits, bf16Logits, relTol: 0.002f), Is.True,
                "Prefill last-row logits diverge beyond the documented tolerance.");
            Assert.That(ArgMaxLastRow(bf16Logits), Is.EqualTo(ArgMaxLastRow(f32Logits)),
                "Prefill argmax diverged.");

            int position = Prompt.Length;
            var f32Seq = new List<int>();
            var bf16Seq = new List<int>();
            for (int t = 0; t < DecodeSteps; t++)
            {
                int next32 = ArgMaxLastRow(f32Logits);
                int nextBf = ArgMaxLastRow(bf16Logits);
                Assert.That(nextBf, Is.EqualTo(next32), $"Decode step {t}: argmax diverged.");
                f32Seq.Add(next32);
                bf16Seq.Add(nextBf);

                f32Logits = f32.ForwardCached(next32, position++, f32Cache);
                bf16Logits = bf16.ForwardCached(nextBf, position++, bf16Cache);

                Assert.That(LogitsClose(f32Logits, bf16Logits, relTol: 0.002f), Is.True,
                    $"Decode step {t}: logits diverge beyond the documented tolerance.");
            }

            Assert.That(f32Seq, Is.EqualTo(bf16Seq), "Greedy decode sequences diverged.");
        }
        finally
        {
            NivaraPrimitives.UseWidenSimd = prior;
        }
    }
}