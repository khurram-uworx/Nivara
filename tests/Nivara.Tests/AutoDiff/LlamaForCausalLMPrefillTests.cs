using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaForCausalLMPrefillTests
{
    static int KvWidth(int heads, int kvHeads, int hidden) => kvHeads * (hidden / heads);

    static LlamaForCausalLM<float> TinyModel(int hidden = 32, int heads = 4, int kvHeads = 2, int maxPos = 64)
        => new(vocabSize: 128, hiddenSize: hidden, numHiddenLayers: 2, numHeads: heads, numKeyValueHeads: kvHeads,
            intermediateSize: 2 * hidden, rmsNormEps: 1e-5f, maxPositionEmbeddings: maxPos, ropeTheta: 10000f);

    static void AssertClose(float expected, float actual, float tol = 1e-5f)
        => Assert.That(actual, Is.EqualTo(expected).Within(tol), $"Expected {expected}, got {actual}.");

    static void AssertLastRowClose(ReverseGradTensor<float> full, ReverseGradTensor<float> lastRow, int vocab)
    {
        full.Data.TryGetSpan(out var fullSpan);
        lastRow.Data.TryGetSpan(out var lastSpan);
        int fullOffset = fullSpan.Length - vocab;
        Assert.That(lastSpan.Length, Is.EqualTo(vocab), "Expected [1, vocab] last-row logits.");
        for (int v = 0; v < vocab; v++)
            AssertClose(fullSpan[fullOffset + v], lastSpan[v]);
    }

    static bool CacheBitEqual(float[] a, float[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (Unsafe.As<float, int>(ref a[i]) != Unsafe.As<float, int>(ref b[i]))
                return false;
        return true;
    }

    static bool CacheClose(float[] a, float[] b, float tol)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (Math.Abs(a[i] - b[i]) > tol)
                return false;
        return true;
    }

    static bool RangeBitEqual(float[] a, float[] b, int start, int end)
    {
        for (int i = start; i < end; i++)
            if (Unsafe.As<float, int>(ref a[i]) != Unsafe.As<float, int>(ref b[i]))
                return false;
        return true;
    }

    static bool RangeAllZero(float[] a, int start, int end)
    {
        for (int i = start; i < end; i++)
            if (a[i] != 0f) return false;
        return true;
    }

    static bool HasAnyNonZero(float[] a)
    {
        foreach (var x in a)
            if (x != 0f) return true;
        return false;
    }

    [Test]
    public void ForwardPrefill_SeedMatchesTokenByTokenSeed_KvCacheAndLogits()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];
        int kvWidth = KvWidth(4, 2, 32);

        using var batchedCache = new LlamaKVCache<float>(2, kvWidth);
        var logits = model.ForwardPrefill(tokens, batchedCache);

        using var walkCache = new LlamaKVCache<float>(2, kvWidth);
        for (int p = 0; p < tokens.Length; p++)
            _ = model.ForwardCached(tokens[p], p, walkCache);

        // Layer 0 K/V must be bit-exact: the batched row p is the same per-row projection + RoPE
        // as the single-token walk (P0-2 locked single-row == row-of-batch; RoPE per-row math).
        // Deeper layers agree within the existing cache-vs-full 1e-5 convention: the batched
        // prefill attention runs MultiHeadAttention while the token-by-token walk runs the fused
        // DecodeAttention kernel, whose outputs match within 1e-5 and feed the next layer's input.
        Assert.That(CacheBitEqual(batchedCache.keys[0], walkCache.keys[0]), Is.True,
            "Layer 0 K cache rows are not bit-equal between batched prefill and the per-token walk.");
        Assert.That(CacheBitEqual(batchedCache.values[0], walkCache.values[0]), Is.True,
            "Layer 0 V cache rows are not bit-equal between batched prefill and the per-token walk.");
        for (int l = 1; l < 2; l++)
        {
            Assert.That(CacheClose(batchedCache.keys[l], walkCache.keys[l], 1e-5f), Is.True,
                $"Layer {l} K cache rows exceed the 1e-5 cache-vs-full tolerance.");
            Assert.That(CacheClose(batchedCache.values[l], walkCache.values[l], 1e-5f), Is.True,
                $"Layer {l} V cache rows exceed the 1e-5 cache-vs-full tolerance.");
        }

        // Last-token logits vs the token-by-token walk's final logits (cache-vs-full convention).
        AssertLastRowClose(model.Forward(tokens), logits, 128);
    }

    [Test]
    public void ForwardPrefill_LastRowLogits_MatchFullForwardLastRow()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];
        using var cache = new LlamaKVCache<float>(2, KvWidth(4, 2, 32));

        var logits = model.ForwardPrefill(tokens, cache);

        // [1, vocab] must equal row L-1 of the full [L, vocab] head (single-row LM-head parity).
        AssertLastRowClose(model.Forward(tokens), logits, 128);
    }

    [Test]
    public void ForwardPrefill_SeedThenCachedDecode_MatchesFullForward()
    {
        using var model = TinyModel();
        int[] prompt = [3, 12, 44, 78, 99];
        int[] gen = [7, 21, 55];
        using var cache = new LlamaKVCache<float>(2, KvWidth(4, 2, 32));

        var logits = model.ForwardPrefill(prompt, cache);
        AssertLastRowClose(model.Forward(prompt), logits, 128);

        var prefix = new List<int>(prompt);
        foreach (var g in gen)
        {
            prefix.Add(g);
            logits = model.ForwardCached(g, prefix.Count - 1, cache);
            AssertLastRowClose(model.Forward(prefix.ToArray()), logits, 128);
        }
    }

    [Test]
    public void ForwardPrefill_PlainPromptLength_SeedThenCachedDecode_MatchesFullForward()
    {
        // The Qwen plain (no-tools) prompt (system + user + generation prompt) renders to 36
        // tokens (measured from the real checkpoint via qwen benchmark --plain); pin
        // seed-then-decode == full Forward at that length so the #408 plain surface and the
        // #403 fused prefill attention share one always-run 1e-5 numeric seat (no checkpoint
        // required).
        using var model = TinyModel();
        int[] prompt = Enumerable.Range(1, 36).ToArray();
        int[] gen = [7, 21, 55];
        using var cache = new LlamaKVCache<float>(2, KvWidth(4, 2, 32));

        var logits = model.ForwardPrefill(prompt, cache);
        AssertLastRowClose(model.Forward(prompt), logits, 128);

        var prefix = new List<int>(prompt);
        foreach (var g in gen)
        {
            prefix.Add(g);
            logits = model.ForwardCached(g, prefix.Count - 1, cache);
            AssertLastRowClose(model.Forward(prefix.ToArray()), logits, 128);
        }
    }

    [Test]
    public void ForwardPrefill_CapturesAllKvRows_AtCorrectOffsets()
    {
        // Model-level (offset 0): rows [0, L) captured, rows [L, capacity) untouched fresh zeros.
        using var model = TinyModel();
        int[] tokens = [2, 9, 17, 88];
        int kvWidth = KvWidth(4, 2, 32);
        using var cache = new LlamaKVCache<float>(2, kvWidth, initialCapacity: 8);

        _ = model.ForwardPrefill(tokens, cache);

        for (int l = 0; l < 2; l++)
        {
            Assert.That(HasAnyNonZero(
                    cache.keys[l].AsSpan(0, tokens.Length * kvWidth).ToArray()), Is.True,
                $"Layer {l} K rows [0, L) must be filled by prefill.");
            Assert.That(RangeAllZero(cache.keys[l], tokens.Length * kvWidth, cache.keys[l].Length), Is.True,
                $"Layer {l} K rows beyond the prompt must stay untouched.");
            Assert.That(HasAnyNonZero(
                    cache.values[l].AsSpan(0, tokens.Length * kvWidth).ToArray()), Is.True,
                $"Layer {l} V rows [0, L) must be filled by prefill.");
            Assert.That(RangeAllZero(cache.values[l], tokens.Length * kvWidth, cache.values[l].Length), Is.True,
                $"Layer {l} V rows beyond the prompt must stay untouched.");
        }

        // Decoder-block level (offset 2): rows [2, 5) must match the per-token walk writing the
        // same absolute positions, and rows [0, 2) / [5, capacity) must stay untouched.
        var block = new LlamaDecoderBlock<float>(hiddenSize: 32, numHeads: 4, numKeyValueHeads: 2,
            intermediateSize: 64, rmsNormEps: 1e-5f, maxPositionEmbeddings: 64);
        int capacity = 12;
        var prefillK = new float[capacity * kvWidth];
        var prefillV = new float[capacity * kvWidth];
        var data = new float[3 * 32];
        for (int i = 0; i < data.Length; i++) data[i] = 0.01f * (i + 1);
        var input = ReverseGradTensor<float>.FromMatrix(data, 3, 32, requiresGrad: false);

        _ = block.ForwardPrefill(input, positionOffset: 2, kCache: prefillK, vCache: prefillV);

        var walkK = new float[capacity * kvWidth];
        var walkV = new float[capacity * kvWidth];
        for (int p = 0; p < 3; p++)
        {
            var rowData = new float[32];
            Array.Copy(data, p * 32, rowData, 0, 32);
            var row = ReverseGradTensor<float>.FromMatrix(rowData, 1, 32, requiresGrad: false);
            _ = block.ForwardCached(row, positionOffset: 2 + p, kCache: walkK, vCache: walkV, cacheLen: 2 + p);
        }

        Assert.That(RangeBitEqual(prefillK, walkK, 2 * kvWidth, 5 * kvWidth), Is.True,
            "K rows at a non-zero offset must equal the per-token walk at the same absolute positions.");
        Assert.That(RangeBitEqual(prefillV, walkV, 2 * kvWidth, 5 * kvWidth), Is.True,
            "V rows at a non-zero offset must equal the per-token walk at the same absolute positions.");
        Assert.That(RangeAllZero(prefillK, 0, 2 * kvWidth), Is.True, "K rows before the offset must stay untouched.");
        Assert.That(RangeAllZero(prefillK, 5 * kvWidth, capacity * kvWidth), Is.True, "K rows after the prompt must stay untouched.");
        Assert.That(RangeAllZero(prefillV, 0, 2 * kvWidth), Is.True, "V rows before the offset must stay untouched.");
        Assert.That(RangeAllZero(prefillV, 5 * kvWidth, capacity * kvWidth), Is.True, "V rows after the prompt must stay untouched.");
    }

    [Test]
    public void ForwardPrefill_OutsideGrad_BuildsNoGraphNode()
    {
        using var model = TinyModel();
        int[] tokens = [1, 5, 9];
        using var cache = new LlamaKVCache<float>(2, KvWidth(4, 2, 32));

        var logits = model.ForwardPrefill(tokens, cache);

        Assert.That(logits.IsLeaf, Is.True, "Outside Grad() prefill must build no graph node.");
        Assert.That(HasAnyNonZero(cache.keys[0]), Is.True, "Cache capture must write real K rows even outside Grad().");
    }

    [Test]
    public void ForwardPrefill_Parity_AcrossGqaRatios()
    {
        // Every GQA ratio exercises the fused BatchedAttention prefill path (outside Grad,
        // cache present) against the full Forward reference within 1e-5 — the acceptance gate
        // for #403 (GQA parity across 4/2, 8/2, 14/2).
        foreach (var (hidden, heads, kvHeads) in new[] { (32, 4, 2), (64, 8, 2), (112, 14, 2) })
        {
            using var model = TinyModel(hidden, heads, kvHeads);
            int[] tokens = [1, 12, 45, 78, 99];
            using var cache = new LlamaKVCache<float>(2, KvWidth(heads, kvHeads, hidden));

            var logits = model.ForwardPrefill(tokens, cache);

            AssertLastRowClose(model.Forward(tokens), logits, 128);
        }
    }

    [Test]
    public void ForwardPrefill_InsideGrad_MatchesFullForwardAndBuildsGraph()
    {
        // Verify the fused BatchedAttention branch is never taken inside Grad(): the Grad-enabled
        // path must use MHA (GqaRepeatKV + CreateCausalMask + MultiHeadAttention) and match full
        // Forward (also Grad-enabled) within 1e-5. Graph nodes must be built (leaf == false) —
        // the fused branch only ever returns leaf tensors, so IsLeaf going false is the proof
        // that the inference-only branch was skipped.
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];
        using var gradScope = GradientUtils.Grad();

        using var cache = new LlamaKVCache<float>(2, KvWidth(4, 2, 32));
        var logitsPrefill = model.ForwardPrefill(tokens, cache);
        var logitsFull = model.Forward(tokens);

        Assert.That(logitsPrefill.IsLeaf, Is.False,
            "Grad-enabled ForwardPrefill must take the MHA path (fused branch is inference-only).");
        Assert.That(logitsFull.IsLeaf, Is.False,
            "Grad-enabled Forward must build graph nodes.");

        // Numerically the same MHA path on the same data.
        AssertLastRowClose(logitsFull, logitsPrefill, 128);
    }

    [Test]
    public void BatchedAttention_Kernel_ParityAcrossGqaRatios()
    {
        // Isolated kernel-level test: fused BatchedAttention produces identical output (1e-5)
        // to the MHA reference (GqaRepeatKV + CreateCausalMask + MultiHeadAttention) over the
        // same random Q/K/V data, for every (qLen, GQA-ratio) pair. Exercises the virtual head
        // mapping, reduced-range causal softmax, and weighted-sum accumulation in isolation.
        foreach (var (heads, kvHeads, qLen) in new[]
        {
            (4, 2, 1), (4, 2, 5), (8, 2, 3), (8, 2, 8), (14, 2, 6)
        })
        {
            int headDim = 8;
            int kvWidth = kvHeads * headDim;
            int D = heads * headDim;
            var rng = new Random(42);

            var qArr = new float[qLen * D];
            var kArr = new float[qLen * kvWidth];
            var vArr = new float[qLen * kvWidth];
            for (int i = 0; i < qArr.Length; i++) qArr[i] = rng.NextSingle() - 0.5f;
            for (int i = 0; i < kArr.Length; i++) kArr[i] = rng.NextSingle() - 0.5f;
            for (int i = 0; i < vArr.Length; i++) vArr[i] = rng.NextSingle() - 0.5f;

            float scale = 1.0f / MathF.Sqrt(headDim);

            // Fused path.
            var fusedOut = new float[qLen * D];
            AttentionKernels<float>.BatchedAttention(qArr, kArr, vArr, fusedOut, qLen, heads, kvHeads, headDim, scale);

            // Reference path: GqaRepeatKV K/V, create causal mask, call MultiHeadAttention.
            var qTensor = ReverseGradTensor<float>.FromMatrix(qArr, qLen, D, requiresGrad: false);
            var kTensor = ReverseGradTensor<float>.FromMatrix(kArr, qLen, kvWidth, requiresGrad: false);
            var vTensor = ReverseGradTensor<float>.FromMatrix(vArr, qLen, kvWidth, requiresGrad: false);
            var kRep = ReverseGradOperations.GqaRepeatKV(kTensor, heads, kvHeads);
            var vRep = ReverseGradOperations.GqaRepeatKV(vTensor, heads, kvHeads);
            var mask = ModuleHelpers<float>.CreateCausalMask(qLen, qLen);
            var refOut = ReverseGradOperations.MultiHeadAttention(qTensor, kRep, vRep, heads, scale, mask);
            refOut.Data.TryGetSpan(out var refSpan);

            for (int i = 0; i < fusedOut.Length; i++)
                AssertClose(refSpan[i], fusedOut[i]);
        }
    }

    [Test]
    public void ForwardCached_FusedMatchesPerOp_Within1e5()
    {
        // A/B parity for the model-level fused decode: with LlamaFusedKernels.DecoderBlockFused
        // off, ForwardCached runs the per-op block chain; with it on, the fused single-token
        // kernel (both gate on !GradientUtils.IsGradEnabled). The logits agree within 1e-5 and
        // the layer-0 K/V rows are bit-equal (both paths run the same projection/rope/attention
        // kernels outside Grad).
        using var model = TinyModel();
        int kvWidth = KvWidth(4, 2, 32);
        int[] tokens = [1, 12, 45];

        using var perOpCache = new LlamaKVCache<float>(2, kvWidth);
        ReverseGradTensor<float>? perOp = null;
        try
        {
            LlamaFusedKernels.DecoderBlockFused = false;
            for (int p = 0; p < tokens.Length; p++)
                perOp = model.ForwardCached(tokens[p], p, perOpCache);
        }
        finally
        {
            LlamaFusedKernels.DecoderBlockFused = true;
        }

        using var fusedCache = new LlamaKVCache<float>(2, kvWidth);
        ReverseGradTensor<float>? fused = null;
        for (int p = 0; p < tokens.Length; p++)
            fused = model.ForwardCached(tokens[p], p, fusedCache);

        perOp!.Data.TryGetSpan(out var perSpan);
        fused!.Data.TryGetSpan(out var fusedSpan);
        Assert.That(fusedSpan.Length, Is.EqualTo(perSpan.Length));
        for (int i = 0; i < perSpan.Length; i++)
            AssertClose(perSpan[i], fusedSpan[i]);

        Assert.That(CacheBitEqual(fusedCache.keys[0], perOpCache.keys[0]), Is.True,
            "Fused decode layer-0 K rows must match the per-op walk bit-for-bit.");
        Assert.That(CacheBitEqual(fusedCache.values[0], perOpCache.values[0]), Is.True,
            "Fused decode layer-0 V rows must match the per-op walk bit-for-bit.");
        Assert.That(CacheClose(fusedCache.keys[1], perOpCache.keys[1], 1e-5f), Is.True,
            "A/B decode layer-1 K rows must agree within 1e-5.");
        Assert.That(CacheClose(fusedCache.values[1], perOpCache.values[1], 1e-5f), Is.True,
            "A/B decode layer-1 V rows must agree within 1e-5.");
    }

    [Test]
    public void FinalStage_NormAndHead_FusedMatchesPerOpLogits_BitExact()
    {
        // Pins the issue #413 acceptance: the fused final norm + tied-head GEMV must be
        // bit-identical to the per-op RMSNorm.Forward + MatMulTransposedB path (same kernels,
        // same reduction order).
        const int hidden = 32, vocab = 128;
        var rng = new Random(42);
        var row = new float[hidden];
        for (int i = 0; i < hidden; i++) row[i] = (float)(rng.NextDouble() * 2 - 1);
        var weights = new float[vocab * hidden];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

        using var norm = new RMSNorm<float>(hidden, 1e-5f);
        using var head = ReverseGradTensor<float>.FromMatrix(weights, vocab, hidden, requiresGrad: false);
        var perOpH = norm.Forward(ReverseGradTensor<float>.FromMatrix(row, 1, hidden, requiresGrad: false));
        var perOpLogits = ReverseGradOperations.MatMulTransposedB(perOpH, head);
        perOpLogits.Reshape(1, vocab);
        perOpLogits.Data.TryGetSpan(out var perOpSpan);
        var perOpArr = perOpSpan.ToArray();

        var fusedRow = (float[])row.Clone();
        norm.Weight!.Tensor.Data.TryGetSpan(out var gamma);
        LlamaFusedKernels.RMSNormForwardInPlace(fusedRow, gamma, 1, hidden, double.CreateChecked(norm.Eps));
        var fusedLogits = new float[vocab];
        head.Data.TryGetSpan(out var weightSpan);
        LlamaFusedKernels.MatMulTransposedB(fusedRow, weightSpan, fusedLogits, 1, hidden, vocab);

        Assert.That(RangeBitEqual(fusedLogits, perOpArr, 0, vocab), Is.True,
            "Fused final norm + head logits differ bit-for-bit from the per-op norm + head path.");
    }

    [Test]
    public void FinalStage_MultiRow_FusedMatchesPerOpLastRowLogits_BitExact()
    {
        // Pins the issue #413 prefill final stage: a fused in-place norm over the whole
        // [L, hidden] chunk plus last-row GEMV must equal the per-op chunk norm + last-row head.
        const int rows = 4, hidden = 32, vocab = 128;
        var rng = new Random(7);
        var chunk = new float[rows * hidden];
        for (int i = 0; i < chunk.Length; i++) chunk[i] = (float)(rng.NextDouble() * 2 - 1);
        var weights = new float[vocab * hidden];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

        using var norm = new RMSNorm<float>(hidden, 1e-5f);
        using var head = ReverseGradTensor<float>.FromMatrix(weights, vocab, hidden, requiresGrad: false);
        var perOpH = norm.Forward(ReverseGradTensor<float>.FromMatrix(chunk, rows, hidden, requiresGrad: false));
        perOpH.Data.TryGetSpan(out var perOpHSpan);
        var lastRow = new float[hidden];
        perOpHSpan.Slice((rows - 1) * hidden, hidden).CopyTo(lastRow);
        var perOpLogits = ReverseGradOperations.MatMulTransposedB(
            ReverseGradTensor<float>.FromMatrix(lastRow, 1, hidden, requiresGrad: false), head);
        perOpLogits.Reshape(1, vocab);
        perOpLogits.Data.TryGetSpan(out var perOpSpan);
        var perOpArr = perOpSpan.ToArray();

        var fusedChunk = (float[])chunk.Clone();
        norm.Weight!.Tensor.Data.TryGetSpan(out var gamma);
        LlamaFusedKernels.RMSNormForwardInPlace(fusedChunk, gamma, rows, hidden, double.CreateChecked(norm.Eps));
        var fusedLogits = new float[vocab];
        head.Data.TryGetSpan(out var weightSpan);
        LlamaFusedKernels.MatMulTransposedB(fusedChunk.AsSpan((rows - 1) * hidden, hidden), weightSpan, fusedLogits, 1, hidden, vocab);

        Assert.That(RangeBitEqual(fusedLogits, perOpArr, 0, vocab), Is.True,
            "Fused multi-row norm + last-row head logits differ bit-for-bit from the per-op path.");
    }

    [Test]
    public void ForwardPrefill_FusedMatchesPerOp_Within1e5()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];
        int kvWidth = KvWidth(4, 2, 32);

        using var perOpCache = new LlamaKVCache<float>(2, kvWidth);
        ReverseGradTensor<float>? perOp = null;
        try
        {
            LlamaFusedKernels.DecoderBlockFused = false;
            perOp = model.ForwardPrefill(tokens, perOpCache);
        }
        finally
        {
            LlamaFusedKernels.DecoderBlockFused = true;
        }

        using var fusedCache = new LlamaKVCache<float>(2, kvWidth);
        var fused = model.ForwardPrefill(tokens, fusedCache);

        perOp!.Data.TryGetSpan(out var perSpan);
        fused.Data.TryGetSpan(out var fusedSpan);
        Assert.That(fusedSpan.Length, Is.EqualTo(perSpan.Length));
        for (int i = 0; i < perSpan.Length; i++)
            AssertClose(perSpan[i], fusedSpan[i]);

        Assert.That(CacheBitEqual(fusedCache.keys[0], perOpCache.keys[0]), Is.True,
            "Fused prefill layer-0 K rows must match the per-op prefill bit-for-bit.");
        Assert.That(CacheBitEqual(fusedCache.values[0], perOpCache.values[0]), Is.True,
            "Fused prefill layer-0 V rows must match the per-op prefill bit-for-bit.");
        Assert.That(CacheClose(fusedCache.keys[1], perOpCache.keys[1], 1e-5f), Is.True,
            "A/B prefill layer-1 K rows must agree within 1e-5.");
        Assert.That(CacheClose(fusedCache.values[1], perOpCache.values[1], 1e-5f), Is.True,
            "A/B prefill layer-1 V rows must agree within 1e-5.");
    }
}
