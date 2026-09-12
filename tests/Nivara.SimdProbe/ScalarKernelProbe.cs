using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Nivara.SimdProbe;

/// <summary>
/// AVX-512 scalar hot-path probe for Qwen2.5-0.5B F32 decode. The per-token path
/// is already TensorPrimitives-backed — and TensorPrimitives is vectorized
/// (AVX-512-capable) inside the .NET runtime, so replacing those calls would be
/// reinventing the wheel. The only kernels still running <b>scalar</b> are RoPE
/// forward and the decode-attention V-weighted accumulation. This probe measures
/// whether hand-rolled <see cref="Vector512{T}"/> branches beat the current scalar
/// loops at the exact Qwen shapes, and re-confirms the TensorPrimitives GEMV
/// baseline for the "already AVX-512 optimized" claim.
/// Decision rule: promote into <c>src/Nivara</c> only if a kernel can move
/// ≥ ~1% of per-token time AND AVX-512 gives ≥ 2× on it.
/// </summary>
internal static class ScalarKernelProbe
{
    static readonly bool Avx512Available = Vector512.IsHardwareAccelerated && Avx512F.IsSupported;

    // Qwen2.5-0.5B config.json: hidden 896, intermediate 4864, vocab 151936,
    // 24 layers, 14 query heads, 2 KV heads, headDim 64.
    const int Layers = 24;
    const int Heads = 14;
    const int KvHeads = 2;
    const int HeadDim = 64;
    const int PromptLen = 216;
    const int MaxKvLen = 376; // 216-token prompt + 160 generated

    static readonly (string Label, int Out, int In)[] GemvShapes =
    [
        ("lm_head [151936 x  896]", 151936, 896),
        ("gate/up [ 4864 x  896]",   4864, 896),
        ("qkv/o   [  896 x  896]",    896, 896),
        ("down    [  896 x 4864]",    896, 4864),
    ];

    public static int Run()
    {
        int failures = 0;
        Console.WriteLine("=== AVX-512 scalar hot-path probe (Qwen2.5-0.5B F32 decode) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Vector512.IsHardwareAccelerated: {Vector512.IsHardwareAccelerated}");
        Console.WriteLine($"Vector512<float>.Count: {Vector512<float>.Count}");
        Console.WriteLine($"Avx512F.IsSupported: {Avx512F.IsSupported}   Avx512F.VL.IsSupported: {Avx512F.VL.IsSupported}");
        Console.WriteLine($"Vector512 kernel is {(Avx512Available ? "ACTIVE" : "UNAVAILABLE — baseline-only run")}.");
        Console.WriteLine();
        Console.WriteLine("Audit: QKV/O, gate/up, down, LM head, SiLU, RMSNorm, softmax and attention");
        Console.WriteLine("scores all route through TensorPrimitives (already AVX-512 via the runtime).");
        Console.WriteLine("Only RoPE forward and the decode-attention V-weighted accumulation are scalar.");
        Console.WriteLine();
        Console.WriteLine($"Style note: per-token compute is ~1.05 GFLOP; RoPE is ~0.005% of it,");
        Console.WriteLine($"the V-accumulation ~1.5% at kvLen={MaxKvLen}. Context for the numbers below.");
        Console.WriteLine();

        failures += RunCorrectness();
        failures += RunGemvBaseline();
        failures += RunRopeBench();
        failures += RunAttnVBench();
        Console.WriteLine(failures == 0 ? "ScalarKernelProbe PASSED." : $"{failures} check(s) FAILED.");
        return failures;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Correctness
    // ═══════════════════════════════════════════════════════════════

    static int RunCorrectness()
    {
        int failures = 0;
        Console.WriteLine("=== Correctness (AVX-512 vs scalar reference) ===");

        foreach (int p in new[] { 8, 32, 37 })
        {
            var x = Pattern(2 * p, seed: 1);
            var cos = Pattern(p, seed: 2);
            var sin = Pattern(p, seed: 3);
            var scalar = new float[2 * p];
            var avx = new float[2 * p];
            RotScalar(x, cos, sin, scalar);
            if (Avx512Available) Rot512(x, cos, sin, avx);
            else scalar.CopyTo(avx);
            bool ok = NearlyMatch(scalar, avx);
            Console.WriteLine($"  {(ok ? "PASS" : "FAIL")} RoPE p={p,3} (headDim {2 * p,3})  |ref-avx512|={MaxDiff(scalar, avx):E2}");
            if (!ok) failures++;
        }

        foreach (int kvLen in new[] { 216, 376 })
        {
            foreach (int headDim in new[] { 40, 64 })
            {
                int kvWidth = KvHeads * headDim;
                var scores = Pattern(kvLen, seed: 4);
                var vCache = Pattern(kvLen * kvWidth, seed: 5);
                var scalar = new float[Heads * headDim];
                var avx = new float[Heads * headDim];
                AttnVScalar(scores, vCache, scalar, kvLen, Heads, KvHeads, headDim);
                if (Avx512Available) AttnV512(scores, vCache, avx, kvLen, Heads, KvHeads, headDim);
                else scalar.CopyTo(avx);
                bool ok = NearlyMatch(scalar, avx);
                Console.WriteLine($"  {(ok ? "PASS" : "FAIL")} AttnV kvLen={kvLen,3} headDim={headDim}  |ref-avx512|={MaxDiff(scalar, avx):E2}");
                if (!ok) failures++;
            }
        }

        // Integrity spot check: the TensorPrimitives GEMV path vs a scalar double dot.
        {
            const int bCols = 896, aCols = 896;
            var act = Pattern(aCols, seed: 6);
            var w = Pattern(bCols * aCols, seed: 7);
            var tp = new float[bCols];
            var scl = new float[bCols];
            RefGemv(act, w, tp, bCols, aCols);
            DotDoubleRef(act, w, scl, bCols, aCols);
            bool ok = NearlyMatch(tp, scl);
            Console.WriteLine($"  {(ok ? "PASS" : "FAIL")} TP GEMV vs scalar-double (qkv/o shape)  |tp-scalar|={MaxDiff(tp, scl):E2}");
            if (!ok) failures++;
        }

        Console.WriteLine();
        return failures;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Kernels under test
    // ═══════════════════════════════════════════════════════════════

    /// <summary>Verbatim shape of <c>GradKernels.RotaryForward</c> (scalar pair rotation).</summary>
    static void RotScalar(ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> output)
    {
        int p = cos.Length;
        for (int i = 0; i < p; i++)
        {
            int i0 = i;
            int i1 = i + p;
            float c = cos[i];
            float s = sin[i];
            float x0 = x[i0];
            float x1 = x[i1];
            output[i0] = x0 * c - x1 * s;
            output[i1] = x0 * s + x1 * c;
        }
    }

    /// <summary>AVX-512 RoPE: two FMA chains over the halves (out-lo from
    /// <c>x0*c − x1*s</c>, out-hi from <c>x0*s + x1*c</c>).</summary>
    static void Rot512(ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> output)
    {
        int p = cos.Length;
        int width = Vector512<float>.Count;
        ref float xRef = ref MemoryMarshal.GetReference(x);
        ref float cRef = ref MemoryMarshal.GetReference(cos);
        ref float sRef = ref MemoryMarshal.GetReference(sin);
        ref float oRef = ref MemoryMarshal.GetReference(output);
        int i = 0;
        for (; i + width <= p; i += width)
        {
            var xLo = Vector512.LoadUnsafe(ref xRef, (nuint)i);
            var xHi = Vector512.LoadUnsafe(ref xRef, (nuint)(i + p));
            var c = Vector512.LoadUnsafe(ref cRef, (nuint)i);
            var s = Vector512.LoadUnsafe(ref sRef, (nuint)i);
            var negS = Vector512.Negate(s);
            Vector512.StoreUnsafe(Vector512.FusedMultiplyAdd(xLo, c, Vector512.Multiply(xHi, negS)), ref oRef, (nuint)i);
            Vector512.StoreUnsafe(Vector512.FusedMultiplyAdd(xLo, s, Vector512.Multiply(xHi, c)), ref oRef, (nuint)(i + p));
        }
        for (; i < p; i++)
        {
            int i0 = i;
            int i1 = i + p;
            float c = cos[i];
            float s = sin[i];
            float x0 = x[i0];
            float x1 = x[i1];
            output[i0] = x0 * c - x1 * s;
            output[i1] = x0 * s + x1 * c;
        }
    }

    /// <summary>Verbatim shape of the decode-attention V-phase:
    /// <c>output[d] += score[j] * vRow[j, d]</c> per query head (GQA mapping).</summary>
    static void AttnVScalar(ReadOnlySpan<float> scores, ReadOnlySpan<float> vCache, Span<float> output, int kvLen, int numHeads, int numKvHeads, int headDim)
    {
        int repeat = numHeads / numKvHeads;
        int kvWidth = numKvHeads * headDim;
        for (int qh = 0; qh < numHeads; qh++)
        {
            int kvHead = qh / repeat;
            var outSpan = output.Slice(qh * headDim, headDim);
            for (int j = 0; j < kvLen; j++)
            {
                float w = scores[j];
                var vRow = vCache.Slice(j * kvWidth + kvHead * headDim, headDim);
                for (int d = 0; d < headDim; d++)
                    outSpan[d] += w * vRow[d];
            }
        }
    }

    /// <summary>AVX-512 decode-attention V-phase: d-blocked broadcast-FMA. Each
    /// <c>headDim</c> block of 16 lanes accumulates across all <c>kvLen</c> rows in
    /// one zmm register (weights stream once per head); tail handled scalar.</summary>
    static void AttnV512(ReadOnlySpan<float> scores, ReadOnlySpan<float> vCache, Span<float> output, int kvLen, int numHeads, int numKvHeads, int headDim)
    {
        int repeat = numHeads / numKvHeads;
        int kvWidth = numKvHeads * headDim;
        int width = Vector512<float>.Count;
        ref float vRef = ref MemoryMarshal.GetReference(vCache);
        ref float oRef = ref MemoryMarshal.GetReference(output);
        for (int qh = 0; qh < numHeads; qh++)
        {
            int kvHead = qh / repeat;
            int oOff = qh * headDim;
            int d = 0;
            for (; d + width <= headDim; d += width)
            {
                var acc = Vector512<float>.Zero;
                for (int j = 0; j < kvLen; j++)
                {
                    var wVec = Vector512.Create(scores[j]);
                    var vRow = Vector512.LoadUnsafe(ref vRef, (nuint)(j * kvWidth + kvHead * headDim + d));
                    acc = Vector512.FusedMultiplyAdd(wVec, vRow, acc);
                }
                Vector512.StoreUnsafe(acc, ref oRef, (nuint)(oOff + d));
            }
            for (; d < headDim; d++)
            {
                float s = 0;
                for (int j = 0; j < kvLen; j++)
                    s += scores[j] * vCache[j * kvWidth + kvHead * headDim + d];
                output[oOff + d] = s;
            }
        }
    }

    /// <summary>Current tensors.MultiplyCore GEMV shape: one
    /// <see cref="TensorPrimitives.Dot"/> per output column.</summary>
    static void RefGemv(ReadOnlySpan<float> act, ReadOnlySpan<float> w, Span<float> res, int bCols, int aCols)
    {
        var aRow = act.Slice(0, aCols);
        for (int j = 0; j < bCols; j++)
            res[j] = TensorPrimitives.Dot(aRow, w.Slice(j * aCols, aCols));
    }

    static void DotDoubleRef(ReadOnlySpan<float> act, ReadOnlySpan<float> w, Span<float> res, int bCols, int aCols)
    {
        for (int j = 0; j < bCols; j++)
        {
            double s = 0;
            int off = j * aCols;
            for (int i = 0; i < aCols; i++)
                s += (double)act[i] * w[off + i];
            res[j] = (float)s;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Benchmarks
    // ═══════════════════════════════════════════════════════════════

    /// <summary>Evidence for "already AVX-512 optimized": the SAME TensorPrimitives
    /// GEMV path the model runs today, measured at the Qwen shapes.</summary>
    static int RunGemvBaseline()
    {
        Console.WriteLine("=== Baseline: current TensorPrimitives GEMV path (already AVX-512 via runtime) ===");
        foreach (var (label, outRows, inCols) in GemvShapes)
        {
            var act = Pattern(inCols, seed: 11);
            var w = Pattern(outRows * inCols, seed: 22);
            var res = new float[outRows];

            long bytes = (long)outRows * inCols * sizeof(float);
            long macs = (long)outRows * inCols;
            int reps = outRows switch { >= 65536 => 3, >= 4096 => 20, _ => 200 };
            ulong ns = TimeIt(() => RefGemv(act, w, res, outRows, inCols), reps);
            Console.WriteLine($"  {label}  {Ms(ns),7:F2} ms  {(double)bytes / (ns / 1e9) / 1e9,7:F1} GB/s  {2.0 * macs / (ns / 1e9) / 1e9,7:F1} GFLOPS");
        }
        Console.WriteLine();
        return 0;
    }

    static int RunRopeBench()
    {
        Console.WriteLine("=== RoPE forward: scalar (today) vs AVX-512 ===");
        if (!Avx512Available)
        {
            Console.WriteLine("  AVX-512 unavailable — skipping comparison.");
            Console.WriteLine();
            return 0;
        }

        int p = HeadDim / 2;
        var x = Pattern(HeadDim, seed: 33);
        var cos = Pattern(p, seed: 34);
        var sin = Pattern(p, seed: 35);
        var buf = new float[HeadDim];

        ulong scalarNs = TimeIt(() => RotScalar(x, cos, sin, buf), reps: 2000);
        ulong avxNs = TimeIt(() => Rot512(x, cos, sin, buf), reps: 2000);

        long perTokenRots = Layers * (Heads + KvHeads);      // 384
        long prefillRots = perTokenRots * PromptLen;          // 82944
        double tokenScalarMs = (double)scalarNs * perTokenRots / 1e6;
        double tokenAvxMs = (double)avxNs * perTokenRots / 1e6;
        double prefillScalarMs = (double)scalarNs * prefillRots / 1e6;
        double prefillAvxMs = (double)avxNs * prefillRots / 1e6;

        Console.WriteLine($"  one rotation (p={p}):  scalar={scalarNs,6} ns  avx512={avxNs,7} ns  ({Speedup(scalarNs, avxNs)})");
        Console.WriteLine($"  per token (24 layers): scalar={tokenScalarMs,6:F3} ms  avx512={tokenAvxMs,6:F3} ms  ({Speedup(tokenScalarMs, tokenAvxMs)})  [{100.0 * tokenScalarMs / 441:F3}% of 441 ms/token]");
        Console.WriteLine($"  prefill (216 tok):     scalar={prefillScalarMs,6:F1} ms  avx512={prefillAvxMs,6:F1} ms");
        Console.WriteLine();
        return 0;
    }

    static int RunAttnVBench()
    {
        Console.WriteLine("=== Decode-attention V-phase: scalar (today) vs AVX-512 ===");
        if (!Avx512Available)
        {
            Console.WriteLine("  AVX-512 unavailable — skipping comparison.");
            Console.WriteLine();
            return 0;
        }

        foreach (int kvLen in new[] { PromptLen, MaxKvLen })
        {
            int kvWidth = KvHeads * HeadDim;
            var scores = Pattern(kvLen, seed: 44);
            var vCache = Pattern(kvLen * kvWidth, seed: 45);
            var scalar = new float[Heads * HeadDim];
            var avx = new float[Heads * HeadDim];

            ulong scalarNs = TimeIt(() => AttnVScalar(scores, vCache, scalar, kvLen, Heads, KvHeads, HeadDim), reps: 100);
            ulong avxNs = TimeIt(() => AttnV512(scores, vCache, avx, kvLen, Heads, KvHeads, HeadDim), reps: 100);

            double perTokenScalarMs = scalarNs * Layers / 1e6;
            double perTokenAvxMs = avxNs * Layers / 1e6;
            Console.WriteLine($"  kvLen={kvLen,3}: scalar={scalarNs,8:N0} ns  avx512={avxNs,8:N0} ns  ({Speedup(scalarNs, avxNs)})");
            Console.WriteLine($"      per token (x24 layers): scalar={perTokenScalarMs,6:F3} ms  avx512={perTokenAvxMs,6:F3} ms  [{100.0 * perTokenScalarMs / 441:F3}% of 441 ms/token]");
        }
        Console.WriteLine();
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Harness helpers (median-of-trials, deterministic data)
    // ═══════════════════════════════════════════════════════════════

    static string Speedup(double scalar, double simd)
        => scalar >= simd && simd > 0 ? $"{scalar / simd:F2}x" : "slower";

    static string Ms(ulong ns) => (ns / 1e6).ToString("F2");

    static ulong TimeIt(Action action, int reps)
    {
        int trials = 7;
        var samples = new ulong[trials];
        // JIT warmup.
        for (int i = 0; i < Math.Min(3, Math.Max(1, reps / 10)); i++) action();
        for (int t = 0; t < trials; t++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < reps; i++) action();
            sw.Stop();
            samples[t] = (ulong)(sw.Elapsed.TotalNanoseconds / reps);
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    static float[] Pattern(int n, int seed)
    {
        var a = new float[n];
        unchecked
        {
            for (int i = 0; i < n; i++)
            {
                uint h = (uint)i * 2654435761u ^ (uint)seed;
                a[i] = ((h & 0xFFFF) / 32768f) - 1f;
            }
        }
        return a;
    }

    static bool NearlyMatch(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float scale = 1f + MaxAbs(a);
        return MaxDiff(a, b) <= 1e-3f * scale;
    }

    static float MaxAbs(ReadOnlySpan<float> a)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float v = Math.Abs(a[i]);
            if (v > m) m = v;
        }
        return m;
    }

    static float MaxDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = Math.Abs(a[i] - b[i]);
            if (d > m) m = d;
        }
        return m;
    }
}