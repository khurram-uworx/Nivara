using ILGPU;
using ILGPU.Runtime;
using Nivara.Samples.Gpu;

namespace Nivara.PerformanceTests;

/// <summary>
/// The <c>--silk-kernels</c> gate: runs the OpenCL C BERT-family kernels
/// (samples/Nivara.Samples/Gpu/SilkKernels.cs) that the <c>--silk</c> sample path uses against the
/// incumbent ILGPU kernels of the same shape, on the same inputs, at each model's real dimensions
/// (MiniLM and DistilBERT) — and answers the one question that decides whether the Silk path is
/// worth building out: do the two backends compute the same numbers?
///
/// <para>
/// The pure-multiply cells (<c>gemm_row4_bias</c>, <c>gemm_row4_qkv</c>, <c>gemm_row4_relu</c>,
/// <c>gather_rows</c>) gate on
/// <b>byte-identity</b>: they add the same f32 values to the same accumulators in the same
/// ascending-K order as the ILGPU Row4 family, so a bit difference means the accumulation order
/// or the contraction changed — a design bug, not rounding (docs/ACCELERATION.md lesson 19,
/// GUIDELINES.md "Assert Exactness When Exactness Is Available"). That is not a theoretical
/// hazard: the first version of these kernels was fused (the OpenCL driver contracts
/// <c>a*b+c</c> by default; ILGPU does not) and diverged from ILGPU on ~81% of outputs at
/// rounding scale — invisible to every tolerance, fatal to a bit-for-bit claim.
/// </para>
///
/// <para>
/// The remaining cells gate on the model's numeric bound, <c>|silk - ilgpu| &lt;= 1e-3 *
/// (1 + |ilgpu|)</c>, and report their byte status on the same line. That is not leniency for its
/// own sake: <c>gemm_row4_gelu</c> and <c>batched_attention</c> call <c>exp</c>, and ILGPU's
/// <c>XMath.Exp</c> and OpenCL's <c>exp</c> are independent implementations of the same function
/// that may differ by an ulp — a difference no exactness check could tell apart from a real bug.
/// Every cell currently comes out bit-identical, GELU and softmax included; the bound is what
/// stops a legitimate future ulp from being reported as a design defect.
/// </para>
///
/// <para>
/// Each GEMM shape also gets a host replay of its dot product — <c>mul+add</c>, <c>fma</c> and
/// <c>f64</c> in the same ascending-K order — with the byte-match counts against both backends
/// printed above the table. It is the difference between "the byte gate disagrees" and "the byte
/// gate disagrees <i>because</i> the driver fused": it is how the contraction above was pinned
/// down, and it is the standing evidence for the accumulation claim in <see cref="SilkKernels"/>.
/// </para>
///
/// <para>
/// Per-kernel timing is printed for the informal A/B the plan asks for and is deliberately
/// <em>not</em> a verdict: this gate's exit code counts correctness failures only, and a ratio
/// outside the ±10–20% band is a red flag to report to the human, not a failure to encode. The
/// figures are not committed to any doc.
/// </para>
/// </summary>
internal static class SilkKernelsGate
{
    const int ExitUnbuilt = 10;
    const double Bound = 1e-3;
    const int Warmups = 3;
    const int Rounds = 10;
    const double BandLow = 0.8;
    const double BandHigh = 1.2;

    // MiniLM (nreimers/MiniLM-L6-H384-uncased, samples/data/minilm/config.json):
    // hidden 384, intermediate 1536, num_attention_heads 12 -> headDim 32, layer_norm_eps 1e-12.
    // DistilBERT (distilbert-base-uncased, samples/data/distilbert/config.json):
    // hidden 768, intermediate 3072, num_attention_heads 12 -> headDim 64, layer_norm_eps 1e-12.
    // DistilBERT is the stressing case: headDim 64 == SilkKernels.MaxHeadDim, so it drives the
    // attention kernel's per-thread local-memory tile to its compiled bound.
    sealed record ModelShape(
        string Name,
        int SeqLen,
        int Hidden,
        int NumHeads,
        int HeadDim,
        int Vocab,
        float Eps,
        (string Name, int ARows, int ACols, int BCols)[] Gemms);

    static readonly ModelShape[] Shapes =
    [
        new ModelShape("minilm", 128, 384, 12, 32, 30522, 1e-12f,
        [
            ("qkv", 128, 384, 1152),
            ("o/fc2", 128, 384, 384),
            ("fc1", 128, 384, 1536),
        ]),
        new ModelShape("distilbert", 128, 768, 12, 64, 30522, 1e-12f,
        [
            ("qkv", 128, 768, 2304),
            ("o/fc2", 128, 768, 768),
            ("fc1", 128, 768, 3072),
        ]),
    ];

    /// <summary>One measured kernel cell: its reference output (ILGPU), the output under test
    /// (Silk), and the best-of-N timings of the two launches.</summary>
    sealed class Leg
    {
        public required string Kernel { get; init; }
        public required string Shape { get; init; }
        public required bool RequireBytes { get; init; }
        public required float[] Reference { get; init; }
        public required float[] Actual { get; init; }
        public double IlgpuUs { get; init; }
        public double SilkUs { get; init; }
    }

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("Silk.NET OpenCL BERT-family kernel gate: Silk kernels vs the ILGPU originals");
        Console.WriteLine($"  Numeric bound : |silk - ilgpu| <= {Bound:E0} * (1 + |ilgpu|)   (exp ulp diverger)");
        Console.WriteLine("  Byte-identity : REQUIRED for gemm_row4_bias, gemm_row4_qkv, gemm_row4_relu, gather_rows");
        Console.WriteLine($"  Timing        : best-of-{Rounds} after {Warmups} warmups; NOT a verdict (x{BandLow:F1}..x{BandHigh:F1} is a red flag only)");
        Console.WriteLine();

        if (!GemmBenchmark.TryCreateRuntime(out IlgpuRuntime ilgpu, out string ilgpuReason))
        {
            Console.WriteLine("UNBUILT: --silk-kernels needs the ILGPU leg too; no OpenCL GPU device was found.");
            Console.WriteLine($"  {ilgpuReason}");
            return ExitUnbuilt;
        }

        SilkRuntime silk;
        try
        {
            silk = new SilkRuntime();
        }
        catch (Exception ex)
        {
            ilgpu.Dispose();
            Console.WriteLine("UNBUILT: --silk-kernels needs the Silk leg; the OpenCL program/device was not available.");
            Console.WriteLine($"  {ex.GetType().Name}: {ex.Message}");
            return ExitUnbuilt;
        }

        using (ilgpu)
        using (silk)
        {
            Console.WriteLine($"  ILGPU : {ilgpu.DeviceName}");
            Console.WriteLine($"  Silk  : {silk.Describe()}");
            Console.WriteLine();

            var buffers = new List<nint>();
            try
            {
                var legs = new List<Leg>();
                foreach (var shape in Shapes)
                {
                    legs.AddRange(MeasureGemms(ilgpu, silk, shape, buffers));
                    legs.Add(MeasureAttention(ilgpu, silk, shape, buffers));
                    legs.Add(MeasureLayerNormResidual(ilgpu, silk, shape, buffers));
                    legs.Add(MeasureEmbedSum(ilgpu, silk, shape, buffers));
                    legs.Add(MeasureGather(ilgpu, silk, shape, buffers));
                }
                return Report(legs);
            }
            finally
            {
                foreach (nint buffer in buffers)
                    silk.Release(buffer);
            }
        }
    }

    // ── GEMM: byte-identity on the fused epilogues ────────────────

    static List<Leg> MeasureGemms(IlgpuRuntime ilgpu, SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        var legs = new List<Leg>();
        uint seed = 0x5EED1234u;
        var acc = ilgpu.Accelerator;
        var stream = ilgpu.Stream;

        var biasKernel = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Bias);
        var reluKernel = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Relu);
        var geluKernel = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Gelu);
        var qkvKernel = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Qkv);

        foreach (var (name, aRows, aCols, bCols) in shape.Gemms)
        {
            string cell = $"{shape.Name} {name}";
            var a = new float[aRows * aCols];
            var w = new float[bCols * aCols];
            var bias = new float[bCols];
            for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < w.Length; i++) w[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < bias.Length; i++) bias[i] = GemmBenchmark.NextUniformF(ref seed);

            // Same [in, out] transpose the runners upload, so both backends read the same Bt.
            var bt = new float[aCols * bCols];
            for (int j = 0; j < bCols; j++)
                for (int k = 0; k < aCols; k++)
                    bt[k * bCols + j] = w[j * aCols + k];

            int elements = aRows * bCols;
            var cfg = GpuBuffers.GemmCfg(aRows, bCols);

            using var aBuf = ilgpu.Allocate1D(a.Length);
            using var bBuf = ilgpu.Allocate1D(bt.Length);
            using var cBuf = ilgpu.Allocate1D(elements);
            using var biasBuf = ilgpu.Allocate1D(bias.Length);
            aBuf.CopyFromCPU(a);
            bBuf.CopyFromCPU(bt);
            biasBuf.CopyFromCPU(bias);

            var aView = aBuf.View;
            var bView = bBuf.View;
            var cView = cBuf.View;
            var biasView = biasBuf.View;

            nint sa = silk.Allocate(a.Length); buffers.Add(sa);
            nint sb = silk.Allocate(bt.Length); buffers.Add(sb);
            nint sc = silk.Allocate(elements); buffers.Add(sc);
            nint sbias = silk.Allocate(bias.Length); buffers.Add(sbias);
            silk.Upload(sa, a);
            silk.Upload(sb, bt);
            silk.Upload(sbias, bias);

            Leg biasLeg = Measure(
                SilkKernels.GemmBias, cell, requireBytes: true,
                () => { biasKernel(stream, cfg, aView, bView, cView, biasView, aRows, aCols, bCols); ilgpu.Synchronize(); },
                () => cBuf.AsContiguous().GetAsArray(),
                () => { SetGemmArgs(silk, SilkKernels.GemmBias, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false); RunGemm(silk, SilkKernels.GemmBias, aRows, bCols); },
                () => silk.Download(sc, elements));
            legs.Add(biasLeg);
            PrintAccumulationModel(cell, a, bt, bias, aRows, aCols, bCols, biasLeg.Reference, biasLeg.Actual);

            // pre_classifier-shaped ReLU epilogue: pure add + clamp, so byte-identity is required.
            legs.Add(Measure(
                SilkKernels.GemmRelu, cell, requireBytes: true,
                () => { reluKernel(stream, cfg, aView, bView, cView, biasView, aRows, aCols, bCols); ilgpu.Synchronize(); },
                () => cBuf.AsContiguous().GetAsArray(),
                () => { SetGemmArgs(silk, SilkKernels.GemmRelu, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false); RunGemm(silk, SilkKernels.GemmRelu, aRows, bCols); },
                () => silk.Download(sc, elements)));

            legs.Add(Measure(
                SilkKernels.GemmGelu, cell, requireBytes: false,
                () => { geluKernel(stream, cfg, aView, bView, cView, biasView, aRows, aCols, bCols); ilgpu.Synchronize(); },
                () => cBuf.AsContiguous().GetAsArray(),
                () => { SetGemmArgs(silk, SilkKernels.GemmGelu, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false); RunGemm(silk, SilkKernels.GemmGelu, aRows, bCols); },
                () => silk.Download(sc, elements)));

            legs.Add(Measure(
                SilkKernels.GemmQkv, cell, requireBytes: true,
                () => { qkvKernel(stream, cfg, aView, bView, cView, biasView, aRows, aCols, bCols, bCols / 3); ilgpu.Synchronize(); },
                () => cBuf.AsContiguous().GetAsArray(),
                () => { SetGemmArgs(silk, SilkKernels.GemmQkv, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: true); RunGemm(silk, SilkKernels.GemmQkv, aRows, bCols); },
                () => silk.Download(sc, elements)));
        }

        return legs;
    }

    /// <summary>Classifies each backend's GEMM accumulation by replaying the same dot product on
    /// the host three ways — f32 multiply-then-add, f32 fused-multiply-add, and f64 accumulate —
    /// in exactly the ascending-K order the kernels use, and counting byte-for-byte matches. This
    /// is what turns "the byte gate disagrees" into "the byte gate disagrees because <i>this</i>",
    /// and it is the evidence behind (or against) the bit-identity claim in
    /// <see cref="SilkKernels"/>. Prints one line per backend.</summary>
    static void PrintAccumulationModel(
        string shape, float[] a, float[] bt, float[] bias, int aRows, int aCols, int bCols, float[] ilgpu, float[] silk)
    {
        int total = aRows * bCols;
        Span<int> hits = stackalloc int[6];
        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                float mulAdd = 0f, fma = 0f;
                double dbl = 0.0;
                for (int k = 0; k < aCols; k++)
                {
                    float av = a[i * aCols + k];
                    float bv = bt[k * bCols + j];
                    mulAdd += av * bv;
                    fma = MathF.FusedMultiplyAdd(av, bv, fma);
                    dbl += (double)av * bv;
                }
                int mulAddBits = BitConverter.SingleToInt32Bits(mulAdd + bias[j]);
                int fmaBits = BitConverter.SingleToInt32Bits(fma + bias[j]);
                int dblBits = BitConverter.SingleToInt32Bits((float)dbl + bias[j]);
                int refBits = BitConverter.SingleToInt32Bits(ilgpu[i * bCols + j]);
                int actBits = BitConverter.SingleToInt32Bits(silk[i * bCols + j]);
                if (refBits == mulAddBits) hits[0]++;
                if (refBits == fmaBits) hits[1]++;
                if (refBits == dblBits) hits[2]++;
                if (actBits == mulAddBits) hits[3]++;
                if (actBits == fmaBits) hits[4]++;
                if (actBits == dblBits) hits[5]++;
            }
        }

        Console.WriteLine(
            $"  accumulation {shape,-8} ilgpu: mul+add {hits[0],7}/{total}  fma {hits[1],7}/{total}  f64 {hits[2],7}/{total}");
        Console.WriteLine(
            $"  accumulation {shape,-8} silk : mul+add {hits[3],7}/{total}  fma {hits[4],7}/{total}  f64 {hits[5],7}/{total}");
    }

    static void SetGemmArgs(SilkRuntime silk, string kernel, nint a, nint b, nint c, nint bias, int aRows, int aCols, int bCols, bool qkv)
    {
        silk.SetBuffer(kernel, 0, a);
        silk.SetBuffer(kernel, 1, b);
        silk.SetBuffer(kernel, 2, c);
        silk.SetBuffer(kernel, 3, bias);
        silk.SetUInt32(kernel, 4, (uint)aRows);
        silk.SetUInt32(kernel, 5, (uint)aCols);
        silk.SetUInt32(kernel, 6, (uint)bCols);
        if (qkv) silk.SetUInt32(kernel, 7, (uint)(bCols / 3));
    }

    static void RunGemm(SilkRuntime silk, string kernel, int aRows, int bCols)
        => silk.Run(
            kernel,
            [(nuint)SilkKernels.GemmGlobalRows(aRows), (nuint)SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);

    // ── attention ─────────────────────────────────────────────────

    static Leg MeasureAttention(IlgpuRuntime ilgpu, SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        const int batch = 2;
        int seqLen = shape.SeqLen;
        int numHeads = shape.NumHeads;
        int headDim = shape.HeadDim;
        int d = numHeads * headDim;
        int rows = batch * seqLen;
        int total = batch * numHeads * seqLen;
        float scale = (float)(1.0 / Math.Sqrt(headDim));

        uint seed = 0xA77E1711u;
        var q = new float[rows * d];
        var k = new float[rows * d];
        var v = new float[rows * d];
        for (int i = 0; i < q.Length; i++) q[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < k.Length; i++) k[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < v.Length; i++) v[i] = GemmBenchmark.NextUniformF(ref seed);

        // Batch 0 scrambles a few keys so the mask actually suppresses something without ever
        // emptying a row; batch 1 suppresses every key, which drives a row max of -inf and must
        // produce zeros (the safe-softmax clamp both kernels carry).
        var mask = new float[rows];
        Array.Fill(mask, 1f);
        for (int i = 0; i < seqLen; i++)
            if (i % 7 == 3) mask[i] = 0f;
        for (int i = seqLen; i < rows; i++) mask[i] = 0f;

        silk.EnsureAttentionLocalMemory(headDim);

        var cfg = GpuBuffers.Cfg1D(total, GpuBuffers.AttentionGroupSize);
        var stream = ilgpu.Stream;
        var attention = ilgpu.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float>(
            AttentionKernels.BatchedAttention);

        using var qBuf = ilgpu.Allocate1D(q.Length);
        using var kBuf = ilgpu.Allocate1D(k.Length);
        using var vBuf = ilgpu.Allocate1D(v.Length);
        using var maskBuf = ilgpu.Allocate1D(mask.Length);
        using var outBuf = ilgpu.Allocate1D(q.Length);
        qBuf.CopyFromCPU(q);
        kBuf.CopyFromCPU(k);
        vBuf.CopyFromCPU(v);
        maskBuf.CopyFromCPU(mask);

        var qView = qBuf.View;
        var kView = kBuf.View;
        var vView = vBuf.View;
        var maskView = maskBuf.View;
        var outView = outBuf.View;

        nint sq = silk.Allocate(q.Length); buffers.Add(sq);
        nint sk = silk.Allocate(k.Length); buffers.Add(sk);
        nint sv = silk.Allocate(v.Length); buffers.Add(sv);
        nint smask = silk.Allocate(mask.Length); buffers.Add(smask);
        nint sout = silk.Allocate(q.Length); buffers.Add(sout);
        silk.Upload(sq, q);
        silk.Upload(sk, k);
        silk.Upload(sv, v);
        silk.Upload(smask, mask);

        return Measure(
            SilkKernels.Attention, $"{shape.Name} b{batch} s{seqLen} h{numHeads} d{headDim}", requireBytes: false,
            () =>
            {
                attention(stream, cfg, qView, kView, vView, maskView, outView,
                    batch, seqLen, numHeads, headDim, GpuBuffers.GlobalAttentionBand, scale);
                ilgpu.Synchronize();
            },
            () => outBuf.AsContiguous().GetAsArray(),
            () =>
            {
                silk.SetBuffer(SilkKernels.Attention, 0, sq);
                silk.SetBuffer(SilkKernels.Attention, 1, sk);
                silk.SetBuffer(SilkKernels.Attention, 2, sv);
                silk.SetBuffer(SilkKernels.Attention, 3, smask);
                silk.SetBuffer(SilkKernels.Attention, 4, sout);
                silk.SetUInt32(SilkKernels.Attention, 5, batch);
                silk.SetUInt32(SilkKernels.Attention, 6, (uint)seqLen);
                silk.SetUInt32(SilkKernels.Attention, 7, (uint)numHeads);
                silk.SetUInt32(SilkKernels.Attention, 8, (uint)headDim);
                silk.SetInt32(SilkKernels.Attention, 9, GpuBuffers.GlobalAttentionBand);
                silk.SetFloat32(SilkKernels.Attention, 10, scale);
                silk.Run(
                    SilkKernels.Attention,
                    [(nuint)SilkKernels.AttentionGlobal(total)],
                    [(nuint)SilkKernels.AttentionGroupSize]);
            },
            () => silk.Download(sout, q.Length));
    }

    // ── layer norm + residual ─────────────────────────────────────

    static Leg MeasureLayerNormResidual(IlgpuRuntime ilgpu, SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        int rows = shape.SeqLen, cols = shape.Hidden, elements = rows * cols;
        uint seed = 0x1A71CE12u;
        var a = new float[elements];
        var b = new float[elements];
        var gamma = new float[cols];
        var beta = new float[cols];
        for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < b.Length; i++) b[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < gamma.Length; i++) gamma[i] = 1f + 0.5f * GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < beta.Length; i++) beta[i] = GemmBenchmark.NextUniformF(ref seed);

        var cfg = GpuBuffers.Cfg1D(rows);
        var stream = ilgpu.Stream;
        var layerNormResidual = ilgpu.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            ElementwiseKernels.LayerNormResidual1D);

        using var aBuf = ilgpu.Allocate1D(a.Length);
        using var bBuf = ilgpu.Allocate1D(b.Length);
        using var gammaBuf = ilgpu.Allocate1D(gamma.Length);
        using var betaBuf = ilgpu.Allocate1D(beta.Length);
        using var yBuf = ilgpu.Allocate1D(elements);
        aBuf.CopyFromCPU(a);
        bBuf.CopyFromCPU(b);
        gammaBuf.CopyFromCPU(gamma);
        betaBuf.CopyFromCPU(beta);

        var aView = aBuf.View;
        var bView = bBuf.View;
        var gammaView = gammaBuf.View;
        var betaView = betaBuf.View;
        var yView = yBuf.View;

        nint sa = silk.Allocate(elements); buffers.Add(sa);
        nint sb = silk.Allocate(elements); buffers.Add(sb);
        nint sgamma = silk.Allocate(cols); buffers.Add(sgamma);
        nint sbeta = silk.Allocate(cols); buffers.Add(sbeta);
        nint sy = silk.Allocate(elements); buffers.Add(sy);
        silk.Upload(sa, a);
        silk.Upload(sb, b);
        silk.Upload(sgamma, gamma);
        silk.Upload(sbeta, beta);

        return Measure(
            SilkKernels.LayerNormResidual, $"{shape.Name} r{rows} c{cols}", requireBytes: false,
            () =>
            {
                layerNormResidual(stream, cfg, aView, bView, gammaView, betaView, yView, rows, cols, shape.Eps);
                ilgpu.Synchronize();
            },
            () => yBuf.AsContiguous().GetAsArray(),
            () =>
            {
                silk.SetBuffer(SilkKernels.LayerNormResidual, 0, sa);
                silk.SetBuffer(SilkKernels.LayerNormResidual, 1, sb);
                silk.SetBuffer(SilkKernels.LayerNormResidual, 2, sgamma);
                silk.SetBuffer(SilkKernels.LayerNormResidual, 3, sbeta);
                silk.SetBuffer(SilkKernels.LayerNormResidual, 4, sy);
                silk.SetUInt32(SilkKernels.LayerNormResidual, 5, (uint)rows);
                silk.SetUInt32(SilkKernels.LayerNormResidual, 6, (uint)cols);
                silk.SetFloat32(SilkKernels.LayerNormResidual, 7, shape.Eps);
                silk.Run(
                    SilkKernels.LayerNormResidual,
                    [(nuint)SilkKernels.LinearGlobal(rows)],
                    [(nuint)SilkKernels.LinearGroupSize]);
            },
            () => silk.Download(sy, elements));
    }

    // ── embedding gather/sum ──────────────────────────────────────

    static Leg MeasureEmbedSum(IlgpuRuntime ilgpu, SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        int rows = shape.SeqLen, hidden = shape.Hidden, elements = rows * hidden;
        uint seed = 0x3EEB1234u;
        var wordEmb = new float[shape.Vocab * hidden];
        var posEmb = new float[rows * hidden];
        var tokenType = new float[hidden];
        var ids = new int[rows];
        var posIds = new int[rows];
        for (int i = 0; i < wordEmb.Length; i++) wordEmb[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < posEmb.Length; i++) posEmb[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < tokenType.Length; i++) tokenType[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < rows; i++)
        {
            ids[i] = (int)((GemmBenchmark.NextUniformF(ref seed) + 1f) * 0.5f * shape.Vocab) % shape.Vocab;
            posIds[i] = i;
        }

        var cfg = GpuBuffers.Cfg1D(elements);
        var stream = ilgpu.Stream;
        var embeddingSum = ilgpu.Accelerator.LoadKernel<
            ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            ElementwiseKernels.EmbeddingSum);

        using var idsBuf = GpuBuffers.AllocInt(ilgpu.Accelerator, rows);
        using var posIdsBuf = GpuBuffers.AllocInt(ilgpu.Accelerator, rows);
        using var wordBuf = ilgpu.Allocate1D(wordEmb.Length);
        using var posBuf = ilgpu.Allocate1D(posEmb.Length);
        using var ttBuf = ilgpu.Allocate1D(tokenType.Length);
        using var yBuf = ilgpu.Allocate1D(elements);
        idsBuf.CopyFromCPU(ids);
        posIdsBuf.CopyFromCPU(posIds);
        wordBuf.CopyFromCPU(wordEmb);
        posBuf.CopyFromCPU(posEmb);
        ttBuf.CopyFromCPU(tokenType);

        var idsView = idsBuf.View;
        var posIdsView = posIdsBuf.View;
        var wordView = wordBuf.View;
        var posView = posBuf.View;
        var ttView = ttBuf.View;
        var yView = yBuf.View;

        nint sids = silk.AllocateBytes((ulong)rows * sizeof(int)); buffers.Add(sids);
        nint sposIds = silk.AllocateBytes((ulong)rows * sizeof(int)); buffers.Add(sposIds);
        nint sword = silk.Allocate(wordEmb.Length); buffers.Add(sword);
        nint spos = silk.Allocate(posEmb.Length); buffers.Add(spos);
        nint stt = silk.Allocate(tokenType.Length); buffers.Add(stt);
        nint sy = silk.Allocate(elements); buffers.Add(sy);
        silk.Upload(sids, ids);
        silk.Upload(sposIds, posIds);
        silk.Upload(sword, wordEmb);
        silk.Upload(spos, posEmb);
        silk.Upload(stt, tokenType);

        return Measure(
            SilkKernels.EmbedSum, $"{shape.Name} r{rows} h{hidden}", requireBytes: false,
            () =>
            {
                embeddingSum(stream, cfg, idsView, posIdsView, wordView, posView, ttView, yView, hidden, 0);
                ilgpu.Synchronize();
            },
            () => yBuf.AsContiguous().GetAsArray(),
            () =>
            {
                silk.SetBuffer(SilkKernels.EmbedSum, 0, sids);
                silk.SetBuffer(SilkKernels.EmbedSum, 1, sposIds);
                silk.SetBuffer(SilkKernels.EmbedSum, 2, sword);
                silk.SetBuffer(SilkKernels.EmbedSum, 3, spos);
                silk.SetBuffer(SilkKernels.EmbedSum, 4, stt);
                silk.SetBuffer(SilkKernels.EmbedSum, 5, sy);
                silk.SetUInt32(SilkKernels.EmbedSum, 6, (uint)elements);
                silk.SetUInt32(SilkKernels.EmbedSum, 7, (uint)hidden);
                silk.SetUInt32(SilkKernels.EmbedSum, 8, 0u);
                silk.Run(
                    SilkKernels.EmbedSum,
                    [(nuint)SilkKernels.LinearGlobal(elements)],
                    [(nuint)SilkKernels.LinearGroupSize]);
            },
            () => silk.Download(sy, elements));
    }

    // ── classifier-head row gather ────────────────────────────────

    static Leg MeasureGather(IlgpuRuntime ilgpu, SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        const int batch = 8;
        const int tableRows = 1024;
        int hidden = shape.Hidden;
        int rows = batch;
        int elements = rows * hidden;
        uint seed = 0x6A7B1234u;
        var table = new float[tableRows * hidden];
        var ids = new int[rows];
        for (int i = 0; i < table.Length; i++) table[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < rows; i++)
            ids[i] = (int)((GemmBenchmark.NextUniformF(ref seed) + 1f) * 0.5f * tableRows) % tableRows;

        var cfg = GpuBuffers.Cfg1D(elements);
        var stream = ilgpu.Stream;
        var gather = ilgpu.Accelerator.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(
            ElementwiseKernels.Gather);

        using var tableBuf = ilgpu.Allocate1D(table.Length);
        using var idsBuf = GpuBuffers.AllocInt(ilgpu.Accelerator, rows);
        using var outBuf = ilgpu.Allocate1D(elements);
        tableBuf.CopyFromCPU(table);
        idsBuf.CopyFromCPU(ids);

        var tableView = tableBuf.View;
        var idsView = idsBuf.View;
        var outView = outBuf.View;

        nint stable = silk.Allocate(table.Length); buffers.Add(stable);
        nint sids = silk.AllocateBytes((ulong)rows * sizeof(int)); buffers.Add(sids);
        nint sout = silk.Allocate(elements); buffers.Add(sout);
        silk.Upload(stable, table);
        silk.Upload(sids, ids);

        return Measure(
            SilkKernels.GatherRows, $"{shape.Name} r{rows} h{hidden}", requireBytes: true,
            () =>
            {
                gather(stream, cfg, tableView, idsView, outView, hidden);
                ilgpu.Synchronize();
            },
            () => outBuf.AsContiguous().GetAsArray(),
            () =>
            {
                silk.SetBuffer(SilkKernels.GatherRows, 0, stable);
                silk.SetBuffer(SilkKernels.GatherRows, 1, sids);
                silk.SetBuffer(SilkKernels.GatherRows, 2, sout);
                silk.SetUInt32(SilkKernels.GatherRows, 3, (uint)elements);
                silk.SetUInt32(SilkKernels.GatherRows, 4, (uint)hidden);
                silk.Run(
                    SilkKernels.GatherRows,
                    [(nuint)SilkKernels.LinearGlobal(elements)],
                    [(nuint)SilkKernels.LinearGroupSize]);
            },
            () => silk.Download(sout, elements));
    }

    // ── measurement + verdict ─────────────────────────────────────

    static Leg Measure(
        string kernel,
        string shape,
        bool requireBytes,
        Action runIlgpu,
        Func<float[]> readIlgpu,
        Action runSilk,
        Func<float[]> readSilk)
    {
        runIlgpu();
        runSilk();

        double ilgpuUs = GemmBenchmark.TimeBest(runIlgpu, Warmups, Rounds);
        double silkUs = GemmBenchmark.TimeBest(runSilk, Warmups, Rounds);

        return new Leg
        {
            Kernel = kernel,
            Shape = shape,
            RequireBytes = requireBytes,
            Reference = readIlgpu(),
            Actual = readSilk(),
            IlgpuUs = ilgpuUs,
            SilkUs = silkUs,
        };
    }

    static int Report(List<Leg> legs)
    {
        Console.WriteLine();
        Console.WriteLine($"{"kernel",-20} {"shape",-26} {"bytes",-12} {"%bound",-9} {"ilgpu us",-9} {"silk us",-9} {"x",-6}");
        Console.WriteLine(new string('-', 102));

        int failures = 0;
        int outOfBand = 0;
        int byteIdentical = 0;
        foreach (var leg in legs)
        {
            int byteDiffs = ByteDiffs(leg.Actual, leg.Reference);
            double worstRatio = WorstRatio(leg.Actual, leg.Reference);
            bool byteOk = byteDiffs == 0;
            bool numericOk = worstRatio <= 1.0;
            bool failed = leg.RequireBytes ? !byteOk : !numericOk;
            failures += failed ? 1 : 0;
            if (byteOk) byteIdentical++;

            double ratio = leg.IlgpuUs > 0 ? leg.SilkUs / leg.IlgpuUs : 0;
            bool band = ratio < BandLow || ratio > BandHigh;
            outOfBand += band ? 1 : 0;

            string bytes = byteOk
                ? (leg.RequireBytes ? "IDENTICAL" : "identical")
                : $"{byteDiffs} diff";
            Console.WriteLine(
                $"{leg.Kernel,-20} {leg.Shape,-26} {bytes,-12} {worstRatio * 100.0,-9:F3} " +
                $"{leg.IlgpuUs,-9:F1} {leg.SilkUs,-9:F1} {ratio,-6:F2}{(failed ? "  FAIL" : "")}{(band ? "  OUT-OF-BAND" : "")}");
        }

        Console.WriteLine(new string('-', 102));
        Console.WriteLine($"  {byteIdentical} of {legs.Count} kernel outputs are bit-identical to ILGPU; {outOfBand} of {legs.Count} outside the x{BandLow:F1}..x{BandHigh:F1} timing band.");
        Console.WriteLine(outOfBand > 0
            ? "  A timing ratio outside the band is a red flag to report, not a failure — this gate's exit code is correctness only."
            : "  Every kernel is inside the informal timing band.");

        if (failures == 0)
        {
            Console.WriteLine($"Gate PASS — all {legs.Count} cells met their criterion (byte-identity where required, otherwise {Bound:E0} * (1 + |ilgpu|)).");
            return 0;
        }

        int requiredByte = legs.Count(l => l.RequireBytes && ByteDiffs(l.Actual, l.Reference) != 0);
        int numeric = legs.Count(l => !l.RequireBytes && WorstRatio(l.Actual, l.Reference) > 1.0);
        if (requiredByte > 0)
            Console.WriteLine($"Byte-identity FAIL — {requiredByte} of the {legs.Count(l => l.RequireBytes)} required GEMM cells differ from ILGPU; the accumulation order or the contraction changed (a design bug, not rounding).");
        if (numeric > 0)
            Console.WriteLine($"Numeric FAIL — {numeric} cells exceed {Bound:E0} * (1 + |ilgpu|).");
        return failures;
    }

    static int ByteDiffs(float[] actual, float[] reference)
    {
        int diffs = 0;
        for (int i = 0; i < reference.Length; i++)
            if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(reference[i]))
                diffs++;
        return diffs;
    }

    /// <summary>Worst element as a fraction of the model bound <c>1e-3 * (1 + |reference|)</c>.
    /// &lt;= 1.0 means every element is inside the bound.</summary>
    static double WorstRatio(float[] actual, float[] reference)
    {
        double worst = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            double bound = Bound * (1.0 + Math.Abs(reference[i]));
            double ratio = Math.Abs(actual[i] - reference[i]) / bound;
            if (ratio > worst) worst = ratio;
        }
        return worst;
    }
}
