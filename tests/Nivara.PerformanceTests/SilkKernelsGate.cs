using System;

namespace Nivara.PerformanceTests;

/// <summary>
/// The <c>--silk-kernels</c> gate: runs the OpenCL C BERT-family kernels
/// (src/Nivara.Extensions/Silk/SilkKernels.cs) that the <c>--silk</c> sample path uses against a
/// CPU reference of the same shape, on the same inputs, at each model's real dimensions
/// (MiniLM and DistilBERT) — and answers the one question that decides whether the Silk path is
/// worth building out: do the two backends compute the same numbers?
///
/// <para>
/// The pure-multiply cells (<c>gemm_row4</c>, <c>gemm_row4_bias</c>, <c>gemm_row4_qkv</c>,
/// <c>gemm_row4_relu</c>, <c>gather_rows</c>, <c>add</c>, <c>add_bias</c>, <c>split_columns</c>,
/// <c>rotary</c>) gate on
/// <b>byte-identity</b>: they add the same f32 values to the same accumulators in the same
/// ascending-K order, so a bit difference means the accumulation order or the contraction
/// changed — a design bug, not rounding (docs/ILGPU.md §10.2,
/// GUIDELINES.md "Assert Exactness When Exactness Is Available").
/// </para>
///
/// <para>
/// The remaining cells gate on the model's numeric bound, <c>|silk - cpu| &lt;= 1e-3 *
/// (1 + |cpu|)</c>, and report their byte status on the same line.
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

    sealed class Leg
    {
        public required string Kernel { get; init; }
        public required string Shape { get; init; }
        public required bool RequireBytes { get; init; }
        public required float[] Reference { get; init; }
        public required float[] Actual { get; init; }
        public double ReferenceUs { get; init; }
        public double SilkUs { get; init; }
    }

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("Silk.NET OpenCL BERT-family kernel gate: Silk kernels vs CPU reference");
        Console.WriteLine($"  Numeric bound : |silk - cpu| <= {Bound:E0} * (1 + |cpu|)   (exp ulp diverger)");
        Console.WriteLine("  Byte-identity : REQUIRED for gemm_row4, gemm_row4_bias, gemm_row4_qkv, gemm_row4_relu,");
        Console.WriteLine("                  gather_rows, add, add_bias, split_columns, rotary");
        Console.WriteLine($"  Timing        : best-of-{Rounds} after {Warmups} warmups; NOT a verdict (x{BandLow:F1}..x{BandHigh:F1} is a red flag only)");
        Console.WriteLine();

        Nivara.Silk.SilkRuntime silk;
        try
        {
            silk = new Nivara.Silk.SilkRuntime();
        }
        catch (Exception ex)
        {
            Console.WriteLine("UNBUILT: --silk-kernels needs the Silk leg; the OpenCL program/device was not available.");
            Console.WriteLine($"  {ex.GetType().Name}: {ex.Message}");
            return ExitUnbuilt;
        }

        using (silk)
        {
            Console.WriteLine($"  Silk  : {silk.Describe()}");
            Console.WriteLine();

            var buffers = new List<nint>();
            try
            {
                var legs = new List<Leg>();
                foreach (var shape in Shapes)
                {
                    legs.AddRange(MeasureGemms(silk, shape, buffers));
                    legs.Add(MeasureAttention(silk, shape, buffers));
                    legs.Add(MeasureLayerNormResidual(silk, shape, buffers));
                    legs.Add(MeasureEmbedSum(silk, shape, buffers));
                    legs.Add(MeasureGather(silk, shape, buffers));
                }

                legs.AddRange(MeasureModernBertPorts(silk, buffers));
                legs.AddRange(MeasureSmolLMKernels(silk, buffers));
                MeasureDecodeShapeAb(silk, buffers);
                return Report(legs);
            }
            finally
            {
                foreach (nint buffer in buffers)
                    silk.Release(buffer);
            }
        }
    }

    // ── GEMM ──────────────────────────────────────────────────────

    static List<Leg> MeasureGemms(Nivara.Silk.SilkRuntime silk, ModelShape shape, List<nint> buffers)
    {
        var legs = new List<Leg>();
        uint seed = 0x5EED1234u;

        foreach (var (name, aRows, aCols, bCols) in shape.Gemms)
        {
            string cell = $"{shape.Name} {name}";
            var a = new float[aRows * aCols];
            var w = new float[bCols * aCols];
            var bias = new float[bCols];
            for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < w.Length; i++) w[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < bias.Length; i++) bias[i] = GemmBenchmark.NextUniformF(ref seed);

            var bt = new float[aCols * bCols];
            for (int j = 0; j < bCols; j++)
                for (int k = 0; k < aCols; k++)
                    bt[k * bCols + j] = w[j * aCols + k];

            int elements = aRows * bCols;

            nint sa = silk.Allocate(a.Length); buffers.Add(sa);
            nint sb = silk.Allocate(bt.Length); buffers.Add(sb);
            nint sc = silk.Allocate(elements); buffers.Add(sc);
            nint sbias = silk.Allocate(bias.Length); buffers.Add(sbias);
            silk.Upload(sa, a);
            silk.Upload(sb, bt);
            silk.Upload(sbias, bias);

            // bias
            {
                var reference = new float[elements];
                Action runCpu = () =>
                {
                    for (int i = 0; i < aRows; i++)
                        for (int j = 0; j < bCols; j++)
                        {
                            float acc = 0f;
                            for (int k = 0; k < aCols; k++)
                                acc += a[i * aCols + k] * bt[k * bCols + j];
                            reference[i * bCols + j] = acc + bias[j];
                        }
                };
                Func<float[]> readCpu = () => reference;
                Action runSilk = () =>
                {
                    SetGemmArgs(silk, Nivara.Silk.SilkKernels.GemmBias, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false);
                    RunGemm(silk, Nivara.Silk.SilkKernels.GemmBias, aRows, bCols);
                };
                Func<float[]> readSilk = () => silk.Download(sc, elements);
                legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.GemmBias, cell, true, runCpu, readCpu, runSilk, readSilk));
            }

            // relu
            {
                var reference = new float[elements];
                Action runCpu = () =>
                {
                    for (int i = 0; i < aRows; i++)
                        for (int j = 0; j < bCols; j++)
                        {
                            float acc = 0f;
                            for (int k = 0; k < aCols; k++)
                                acc += a[i * aCols + k] * bt[k * bCols + j];
                            float v = acc + bias[j];
                            reference[i * bCols + j] = v > 0f ? v : 0f;
                        }
                };
                Func<float[]> readCpu = () => reference;
                Action runSilk = () =>
                {
                    SetGemmArgs(silk, Nivara.Silk.SilkKernels.GemmRelu, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false);
                    RunGemm(silk, Nivara.Silk.SilkKernels.GemmRelu, aRows, bCols);
                };
                Func<float[]> readSilk = () => silk.Download(sc, elements);
                legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.GemmRelu, cell, true, runCpu, readCpu, runSilk, readSilk));
            }

            // gelu
            {
                var reference = new float[elements];
                Action runCpu = () =>
                {
                    for (int i = 0; i < aRows; i++)
                        for (int j = 0; j < bCols; j++)
                        {
                            float acc = 0f;
                            for (int k = 0; k < aCols; k++)
                                acc += a[i * aCols + k] * bt[k * bCols + j];
                            reference[i * bCols + j] = GeluExact(acc + bias[j]);
                        }
                };
                Func<float[]> readCpu = () => reference;
                Action runSilk = () =>
                {
                    SetGemmArgs(silk, Nivara.Silk.SilkKernels.GemmGelu, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: false);
                    RunGemm(silk, Nivara.Silk.SilkKernels.GemmGelu, aRows, bCols);
                };
                Func<float[]> readSilk = () => silk.Download(sc, elements);
                legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.GemmGelu, cell, false, runCpu, readCpu, runSilk, readSilk));
            }

            // qkv
            {
                var reference = new float[elements];
                int blockWidth = bCols / 3;
                Action runCpu = () =>
                {
                    for (int i = 0; i < aRows; i++)
                        for (int j = 0; j < bCols; j++)
                        {
                            float acc = 0f;
                            for (int k = 0; k < aCols; k++)
                                acc += a[i * aCols + k] * bt[k * bCols + j];
                            int dest = QkvDest(j, i, aRows, blockWidth);
                            reference[dest] = acc + bias[j];
                        }
                };
                Func<float[]> readCpu = () => reference;
                Action runSilk = () =>
                {
                    SetGemmArgs(silk, Nivara.Silk.SilkKernels.GemmQkv, sa, sb, sc, sbias, aRows, aCols, bCols, qkv: true);
                    RunGemm(silk, Nivara.Silk.SilkKernels.GemmQkv, aRows, bCols);
                };
                Func<float[]> readSilk = () => silk.Download(sc, elements);
                legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.GemmQkv, cell, true, runCpu, readCpu, runSilk, readSilk));
            }
        }

        return legs;
    }

    static void SetGemmArgs(Nivara.Silk.SilkRuntime silk, string kernel, nint a, nint b, nint c, nint bias, int aRows, int aCols, int bCols, bool qkv)
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

    static void RunGemm(Nivara.Silk.SilkRuntime silk, string kernel, int aRows, int bCols)
    {
        silk.Run(
            kernel,
            [(nuint)Nivara.Silk.SilkKernels.GemmGlobalRows(aRows), (nuint)Nivara.Silk.SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)Nivara.Silk.SilkKernels.GemmTileSize, (nuint)Nivara.Silk.SilkKernels.GemmTileSize]);
        silk.Finish();
    }

    // ── attention ─────────────────────────────────────────────────

    static Leg MeasureAttention(Nivara.Silk.SilkRuntime silk, ModelShape shape, List<nint> buffers, int band = -1)
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

        var mask = new float[rows];
        Array.Fill(mask, 1f);
        for (int i = 0; i < seqLen; i++)
            if (i % 7 == 3) mask[i] = 0f;
        for (int i = seqLen; i < rows; i++) mask[i] = 0f;

        silk.EnsureAttentionLocalMemory(headDim);

        nint sq = silk.Allocate(q.Length); buffers.Add(sq);
        nint sk = silk.Allocate(k.Length); buffers.Add(sk);
        nint sv = silk.Allocate(v.Length); buffers.Add(sv);
        nint smask = silk.Allocate(mask.Length); buffers.Add(smask);
        nint sout = silk.Allocate(q.Length); buffers.Add(sout);
        silk.Upload(sq, q);
        silk.Upload(sk, k);
        silk.Upload(sv, v);
        silk.Upload(smask, mask);

        int b = band < 0 ? -1 : band;
        string label = band < 0
            ? $"{shape.Name} b{batch} s{seqLen} h{numHeads} d{headDim}"
            : $"{shape.Name} band{band} s{seqLen}";

        var cpuOut = new float[q.Length];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < total; idx++)
            {
                int bb = idx / (numHeads * seqLen);
                int rem = idx - bb * (numHeads * seqLen);
                int h = rem / seqLen;
                int qPos = rem - h * seqLen;
                int D = numHeads * headDim;
                int dOffset = h * headDim;
                int qRow = bb * seqLen + qPos;
                int maskBase = bb * seqLen;
                int outBase = qRow * D;

                int jFirst = b < 0 ? 0 : Math.Max(0, qPos - b);
                int jLast = b < 0 ? seqLen - 1 : Math.Min(seqLen - 1, qPos + b);

                float max = float.NegativeInfinity;
                for (int j = jFirst; j <= jLast; j++)
                {
                    float s = CpuRowScore(q, k, qRow, bb * seqLen + j, dOffset, D, headDim, scale);
                    if (!CpuKeep(mask, maskBase, qPos, j, b)) s = float.NegativeInfinity;
                    if (s > max) max = s;
                }

                if (max == float.NegativeInfinity)
                {
                    for (int dd = 0; dd < headDim; dd++)
                        cpuOut[outBase + dOffset + dd] = 0f;
                    continue;
                }

                float sum = 0f;
                for (int j = jFirst; j <= jLast; j++)
                {
                    float s = CpuRowScore(q, k, qRow, bb * seqLen + j, dOffset, D, headDim, scale);
                    if (!CpuKeep(mask, maskBase, qPos, j, b)) s = float.NegativeInfinity;
                    sum += MathF.Exp(s - max);
                }

                var acc = new float[headDim];
                for (int j = jFirst; j <= jLast; j++)
                {
                    float s = CpuRowScore(q, k, qRow, bb * seqLen + j, dOffset, D, headDim, scale);
                    if (!CpuKeep(mask, maskBase, qPos, j, b)) s = float.NegativeInfinity;
                    float p = MathF.Exp(s - max) / sum;
                    int vBase = (bb * seqLen + j) * D + dOffset;
                    for (int dd = 0; dd < headDim; dd++)
                        acc[dd] += p * v[vBase + dd];
                }

                for (int dd = 0; dd < headDim; dd++)
                    cpuOut[outBase + dOffset + dd] = acc[dd];
            }
        };

        Func<float[]> readCpu = () => cpuOut;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.Attention, 0, sq);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Attention, 1, sk);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Attention, 2, sv);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Attention, 3, smask);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Attention, 4, sout);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Attention, 5, batch);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Attention, 6, (uint)seqLen);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Attention, 7, (uint)numHeads);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Attention, 8, (uint)headDim);
            silk.SetInt32(Nivara.Silk.SilkKernels.Attention, 9, b);
            silk.SetFloat32(Nivara.Silk.SilkKernels.Attention, 10, scale);
            silk.Run(
                Nivara.Silk.SilkKernels.Attention,
                [(nuint)Nivara.Silk.SilkKernels.AttentionGlobal(total)],
                [(nuint)Nivara.Silk.SilkKernels.AttentionGroupSize]);
            silk.Finish();
        };
        Func<float[]> readSilk = () => silk.Download(sout, q.Length);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.Attention, label, false, runCpu, readCpu, runSilk, readSilk);
    }

    static int QkvDest(int j, int outRow, int aRows, int blockWidth)
    {
        int block = j / blockWidth;
        int colInBlock = j - block * blockWidth;
        return block * aRows * blockWidth + outRow * blockWidth + colInBlock;
    }

    static float CpuRowScore(float[] q, float[] k, int qRow, int jRow, int dOffset, int D, int headDim, float scale)
    {
        int qBase = qRow * D + dOffset;
        int kBase = jRow * D + dOffset;
        float acc = 0f;
        for (int dd = 0; dd < headDim; dd++)
            acc += q[qBase + dd] * k[kBase + dd];
        return acc * scale;
    }

    static bool CpuKeep(float[] mask, int maskBase, int qPos, int j, int band)
        => mask[maskBase + j] >= 0.5f && (band < 0 || Math.Abs(qPos - j) <= band);

    // ── layer norm + residual ─────────────────────────────────────

    static Leg MeasureLayerNormResidual(Nivara.Silk.SilkRuntime silk, ModelShape shape, List<nint> buffers)
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

        nint sa = silk.Allocate(elements); buffers.Add(sa);
        nint sb = silk.Allocate(elements); buffers.Add(sb);
        nint sgamma = silk.Allocate(cols); buffers.Add(sgamma);
        nint sbeta = silk.Allocate(cols); buffers.Add(sbeta);
        nint sy = silk.Allocate(elements); buffers.Add(sy);
        silk.Upload(sa, a);
        silk.Upload(sb, b);
        silk.Upload(sgamma, gamma);
        silk.Upload(sbeta, beta);

        var cpuOut = new float[elements];
        Action runCpu = () =>
        {
            for (int r = 0; r < rows; r++)
            {
                int off = r * cols;
                float sum = 0f;
                for (int c = 0; c < cols; c++) sum += a[off + c] + b[off + c];
                float mean = sum / cols;
                float sumSq = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float dd = a[off + c] + b[off + c] - mean;
                    sumSq += dd * dd;
                }
                float invStd = 1f / MathF.Sqrt(sumSq / cols + shape.Eps);
                for (int c = 0; c < cols; c++)
                    cpuOut[off + c] = (a[off + c] + b[off + c] - mean) * invStd * gamma[c] + beta[c];
            }
        };
        Func<float[]> readCpu = () => cpuOut;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNormResidual, 0, sa);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNormResidual, 1, sb);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNormResidual, 2, sgamma);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNormResidual, 3, sbeta);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNormResidual, 4, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.LayerNormResidual, 5, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.LayerNormResidual, 6, (uint)cols);
            silk.SetFloat32(Nivara.Silk.SilkKernels.LayerNormResidual, 7, shape.Eps);
            silk.Run(
                Nivara.Silk.SilkKernels.LayerNormResidual,
                [(nuint)Nivara.Silk.SilkKernels.LinearGlobal(rows)],
                [(nuint)Nivara.Silk.SilkKernels.LinearGroupSize]);
            silk.Finish();
        };
        Func<float[]> readSilk = () => silk.Download(sy, elements);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.LayerNormResidual, $"{shape.Name} r{rows} c{cols}", false, runCpu, readCpu, runSilk, readSilk);
    }

    // ── embedding gather/sum ──────────────────────────────────────

    static Leg MeasureEmbedSum(Nivara.Silk.SilkRuntime silk, ModelShape shape, List<nint> buffers)
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

        var cpuOut = new float[elements];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < elements; idx++)
            {
                int row = idx / hidden;
                int col = idx - row * hidden;
                float v = wordEmb[ids[row] * hidden + col] + posEmb[posIds[row] * hidden + col];
                cpuOut[idx] = v;
            }
        };
        Func<float[]> readCpu = () => cpuOut;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 0, sids);
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 1, sposIds);
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 2, sword);
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 3, spos);
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 4, stt);
            silk.SetBuffer(Nivara.Silk.SilkKernels.EmbedSum, 5, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.EmbedSum, 6, (uint)elements);
            silk.SetUInt32(Nivara.Silk.SilkKernels.EmbedSum, 7, (uint)hidden);
            silk.SetUInt32(Nivara.Silk.SilkKernels.EmbedSum, 8, 0u);
            silk.Run(
                Nivara.Silk.SilkKernels.EmbedSum,
                [(nuint)Nivara.Silk.SilkKernels.LinearGlobal(elements)],
                [(nuint)Nivara.Silk.SilkKernels.LinearGroupSize]);
            silk.Finish();
        };
        Func<float[]> readSilk = () => silk.Download(sy, elements);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.EmbedSum, $"{shape.Name} r{rows} h{hidden}", false, runCpu, readCpu, runSilk, readSilk);
    }

    // ── classifier-head row gather ────────────────────────────────

    static Leg MeasureGather(Nivara.Silk.SilkRuntime silk, ModelShape shape, List<nint> buffers)
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

        nint stable = silk.Allocate(table.Length); buffers.Add(stable);
        nint sids = silk.AllocateBytes((ulong)rows * sizeof(int)); buffers.Add(sids);
        nint sout = silk.Allocate(elements); buffers.Add(sout);
        silk.Upload(stable, table);
        silk.Upload(sids, ids);

        var cpuOut = new float[elements];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < elements; idx++)
            {
                int row = idx / hidden;
                int col = idx - row * hidden;
                cpuOut[idx] = table[ids[row] * hidden + col];
            }
        };
        Func<float[]> readCpu = () => cpuOut;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.GatherRows, 0, stable);
            silk.SetBuffer(Nivara.Silk.SilkKernels.GatherRows, 1, sids);
            silk.SetBuffer(Nivara.Silk.SilkKernels.GatherRows, 2, sout);
            silk.SetUInt32(Nivara.Silk.SilkKernels.GatherRows, 3, (uint)elements);
            silk.SetUInt32(Nivara.Silk.SilkKernels.GatherRows, 4, (uint)hidden);
            silk.Run(
                Nivara.Silk.SilkKernels.GatherRows,
                [(nuint)Nivara.Silk.SilkKernels.LinearGlobal(elements)],
                [(nuint)Nivara.Silk.SilkKernels.LinearGroupSize]);
            silk.Finish();
        };
        Func<float[]> readSilk = () => silk.Download(sout, elements);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.GatherRows, $"{shape.Name} r{rows} h{hidden}", true, runCpu, readCpu, runSilk, readSilk);
    }

    // ── ModernBERT / Laya ports ───────────────────────────────────

    static List<Leg> MeasureModernBertPorts(Nivara.Silk.SilkRuntime silk, List<nint> buffers)
    {
        const int seq = 128;
        const int hidden = 1024;
        const int heads = 16;
        const int headDim = 64;
        const int intermediate = 2624;
        const float eps = 1e-5f;

        var legs = new List<Leg>();
        var shape = new ModelShape("modernbert", seq, hidden, heads, headDim, 50368, eps, []);
        legs.Add(MeasureAttention(silk, shape, buffers, band: 64));
        legs.AddRange(MeasurePlainGemm(silk, buffers, seq, hidden, intermediate));
        legs.Add(MeasureAdd(silk, buffers, seq, hidden));
        legs.Add(MeasureAddBias(silk, buffers, seq, hidden));
        legs.Add(MeasureSplit(silk, buffers, seq, hidden));
        legs.Add(MeasureLayerNorm(silk, buffers, seq, hidden, eps));
        legs.Add(MeasureRotary(silk, buffers, seq, heads, headDim));
        legs.Add(MeasureGeGlu(silk, buffers, seq, intermediate));
        return legs;
    }

    static List<Leg> MeasurePlainGemm(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int seq, int hidden, int intermediate)
    {
        var legs = new List<Leg>();
        uint seed = 0x6E4D0001u;

        (string Name, int ARows, int ACols, int BCols)[] gemms =
        [
            ("qkv", seq, hidden, 3 * hidden),
            ("o", seq, hidden, hidden),
            ("wi", seq, hidden, 2 * intermediate),
            ("wo", seq, intermediate, hidden),
        ];

        foreach (var (name, aRows, aCols, bCols) in gemms)
        {
            var a = new float[aRows * aCols];
            var bt = new float[aCols * bCols];
            for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < bt.Length; i++) bt[i] = GemmBenchmark.NextUniformF(ref seed);

            int elements = aRows * bCols;
            nint sa = silk.Allocate(a.Length); buffers.Add(sa);
            nint sb = silk.Allocate(bt.Length); buffers.Add(sb);
            nint sc = silk.Allocate(elements); buffers.Add(sc);
            silk.Upload(sa, a);
            silk.Upload(sb, bt);

            var reference = new float[elements];
            Action runCpu = () =>
            {
                for (int i = 0; i < aRows; i++)
                    for (int j = 0; j < bCols; j++)
                    {
                        float acc = 0f;
                        for (int k = 0; k < aCols; k++)
                            acc += a[i * aCols + k] * bt[k * bCols + j];
                        reference[i * bCols + j] = acc;
                    }
            };
            Func<float[]> readCpu = () => reference;

            Action runSilk = () =>
            {
                silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 0, sa);
                silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 1, sb);
                silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 2, sc);
                silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 3, (uint)aRows);
                silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 4, (uint)aCols);
                silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 5, (uint)bCols);
                RunGemmPlain(silk, aRows, bCols);
            };
            Func<float[]> readSilk = () => silk.Download(sc, elements);

            legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.Gemm, $"modernbert {name}", true, runCpu, readCpu, runSilk, readSilk));
        }

        return legs;
    }

    static void RunGemmPlain(Nivara.Silk.SilkRuntime silk, int aRows, int bCols)
    {
        silk.Run(
            Nivara.Silk.SilkKernels.Gemm,
            [(nuint)Nivara.Silk.SilkKernels.GemmGlobalRows(aRows), (nuint)Nivara.Silk.SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)Nivara.Silk.SilkKernels.GemmTileSize, (nuint)Nivara.Silk.SilkKernels.GemmTileSize]);
        silk.Finish();
    }

    static Leg MeasureAdd(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int cols)
    {
        int n = rows * cols;
        uint seed = 0xADD00001u;
        var a = new float[n];
        var b = new float[n];
        for (int i = 0; i < n; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < n; i++) b[i] = GemmBenchmark.NextUniformF(ref seed);

        nint sa = silk.Allocate(n); buffers.Add(sa);
        nint sb = silk.Allocate(n); buffers.Add(sb);
        nint sy = silk.Allocate(n); buffers.Add(sy);
        silk.Upload(sa, a);
        silk.Upload(sb, b);

        var reference = new float[n];
        Action runCpu = () =>
        {
            for (int i = 0; i < n; i++) reference[i] = a[i] + b[i];
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.Add, 0, sa);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Add, 1, sb);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Add, 2, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Add, 3, (uint)n);
            RunLinear(silk, Nivara.Silk.SilkKernels.Add, n);
        };
        Func<float[]> readSilk = () => silk.Download(sy, n);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.Add, $"modernbert n{n}", true, runCpu, readCpu, runSilk, readSilk);
    }

    static Leg MeasureAddBias(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int cols)
    {
        int n = rows * cols;
        uint seed = 0xB1A50001u;
        var x = new float[n];
        var bias = new float[cols];
        for (int i = 0; i < n; i++) x[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < cols; i++) bias[i] = GemmBenchmark.NextUniformF(ref seed);

        nint sx = silk.Allocate(n); buffers.Add(sx);
        nint sb = silk.Allocate(cols); buffers.Add(sb);
        nint sy = silk.Allocate(n); buffers.Add(sy);
        silk.Upload(sx, x);
        silk.Upload(sb, bias);

        var reference = new float[n];
        Action runCpu = () =>
        {
            for (int i = 0; i < n; i++) reference[i] = x[i] + bias[i % cols];
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.AddBias, 0, sx);
            silk.SetBuffer(Nivara.Silk.SilkKernels.AddBias, 1, sb);
            silk.SetBuffer(Nivara.Silk.SilkKernels.AddBias, 2, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.AddBias, 3, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.AddBias, 4, (uint)cols);
            RunLinear(silk, Nivara.Silk.SilkKernels.AddBias, n);
        };
        Func<float[]> readSilk = () => silk.Download(sy, n);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.AddBias, $"modernbert r{rows} c{cols}", true, runCpu, readCpu, runSilk, readSilk);
    }

    static Leg MeasureSplit(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int blockCols)
    {
        const int parts = 3;
        const int part = 1;
        int srcN = rows * blockCols * parts;
        int dstN = rows * blockCols;
        uint seed = 0x59170001u;
        var src = new float[srcN];
        for (int i = 0; i < srcN; i++) src[i] = GemmBenchmark.NextUniformF(ref seed);

        nint ssrc = silk.Allocate(srcN); buffers.Add(ssrc);
        nint sdst = silk.Allocate(dstN); buffers.Add(sdst);
        silk.Upload(ssrc, src);

        var reference = new float[dstN];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < dstN; idx++)
            {
                int row = idx / blockCols;
                int col = idx - row * blockCols;
                reference[idx] = src[row * blockCols * parts + part * blockCols + col];
            }
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.SplitColumns, 0, ssrc);
            silk.SetBuffer(Nivara.Silk.SilkKernels.SplitColumns, 1, sdst);
            silk.SetUInt32(Nivara.Silk.SilkKernels.SplitColumns, 2, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.SplitColumns, 3, (uint)blockCols);
            silk.SetUInt32(Nivara.Silk.SilkKernels.SplitColumns, 4, parts);
            silk.SetUInt32(Nivara.Silk.SilkKernels.SplitColumns, 5, part);
            RunLinear(silk, Nivara.Silk.SilkKernels.SplitColumns, dstN);
        };
        Func<float[]> readSilk = () => silk.Download(sdst, dstN);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.SplitColumns, $"modernbert part{part}", true, runCpu, readCpu, runSilk, readSilk);
    }

    static Leg MeasureLayerNorm(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int cols, float eps)
    {
        int n = rows * cols;
        uint seed = 0x1A7E0001u;
        var x = new float[n];
        var gamma = new float[cols];
        var beta = new float[cols];
        for (int i = 0; i < n; i++) x[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < cols; i++) gamma[i] = 1f + 0.5f * GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < cols; i++) beta[i] = GemmBenchmark.NextUniformF(ref seed);

        nint sx = silk.Allocate(n); buffers.Add(sx);
        nint sg = silk.Allocate(cols); buffers.Add(sg);
        nint sb = silk.Allocate(cols); buffers.Add(sb);
        nint sy = silk.Allocate(n); buffers.Add(sy);
        silk.Upload(sx, x);
        silk.Upload(sg, gamma);
        silk.Upload(sb, beta);

        var reference = new float[n];
        Action runCpu = () =>
        {
            for (int r = 0; r < rows; r++)
            {
                int off = r * cols;
                float sum = 0f;
                for (int c = 0; c < cols; c++) sum += x[off + c];
                float mean = sum / cols;
                float sumSq = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float dd = x[off + c] - mean;
                    sumSq += dd * dd;
                }
                float invStd = 1f / MathF.Sqrt(sumSq / cols + eps);
                for (int c = 0; c < cols; c++)
                    reference[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
            }
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNorm, 0, sx);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNorm, 1, sg);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNorm, 2, sb);
            silk.SetBuffer(Nivara.Silk.SilkKernels.LayerNorm, 3, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.LayerNorm, 4, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.LayerNorm, 5, (uint)cols);
            silk.SetFloat32(Nivara.Silk.SilkKernels.LayerNorm, 6, eps);
            RunLinear(silk, Nivara.Silk.SilkKernels.LayerNorm, rows);
        };
        Func<float[]> readSilk = () => silk.Download(sy, n);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.LayerNorm, $"modernbert r{rows} c{cols}", false, runCpu, readCpu, runSilk, readSilk);
    }

    static Leg MeasureRotary(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int numHeads, int headDim)
    {
        int half = headDim / 2;
        int n = rows * numHeads * headDim;
        int table = rows * half;
        uint seed = 0x20FE0001u;
        var x = new float[n];
        var cos = new float[table];
        var sin = new float[table];
        var pos = new int[rows];
        for (int i = 0; i < n; i++) x[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < table; i++) cos[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < table; i++) sin[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < rows; i++) pos[i] = i;

        nint sx = silk.Allocate(n); buffers.Add(sx);
        nint spos = silk.AllocateBytes((ulong)rows * sizeof(int)); buffers.Add(spos);
        nint scos = silk.Allocate(table); buffers.Add(scos);
        nint ssin = silk.Allocate(table); buffers.Add(ssin);
        nint sy = silk.Allocate(n); buffers.Add(sy);
        silk.Upload(sx, x);
        silk.Upload(spos, pos);
        silk.Upload(scos, cos);
        silk.Upload(ssin, sin);

        var reference = new float[n];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < rows * numHeads * half; idx++)
            {
                int i = idx % half;
                int rem = idx / half;
                int h = rem % numHeads;
                int row = rem / numHeads;
                int t = pos[row] * half + i;
                float c = cos[t];
                float s = sin[t];
                int b = row * (numHeads * headDim) + h * headDim;
                float x0 = x[b + i];
                float x1 = x[b + i + half];
                reference[b + i] = x0 * c - x1 * s;
                reference[b + i + half] = x0 * s + x1 * c;
            }
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.Rotary, 0, sx);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Rotary, 1, spos);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Rotary, 2, scos);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Rotary, 3, ssin);
            silk.SetBuffer(Nivara.Silk.SilkKernels.Rotary, 4, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Rotary, 5, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Rotary, 6, (uint)numHeads);
            silk.SetUInt32(Nivara.Silk.SilkKernels.Rotary, 7, (uint)headDim);
            RunLinear(silk, Nivara.Silk.SilkKernels.Rotary, rows * numHeads * half);
        };
        Func<float[]> readSilk = () => silk.Download(sy, n);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.Rotary, $"modernbert h{numHeads} d{headDim}", true, runCpu, readCpu, runSilk, readSilk);
    }

    static Leg MeasureGeGlu(Nivara.Silk.SilkRuntime silk, List<nint> buffers, int rows, int intermediate)
    {
        int srcN = rows * 2 * intermediate;
        int dstN = rows * intermediate;
        uint seed = 0x6E610001u;
        var x = new float[srcN];
        for (int i = 0; i < srcN; i++) x[i] = GemmBenchmark.NextUniformF(ref seed);

        nint sx = silk.Allocate(srcN); buffers.Add(sx);
        nint sy = silk.Allocate(dstN); buffers.Add(sy);
        silk.Upload(sx, x);

        var reference = new float[dstN];
        Action runCpu = () =>
        {
            for (int idx = 0; idx < dstN; idx++)
            {
                int col = idx % intermediate;
                int row = idx / intermediate;
                int xBase = row * 2 * intermediate;
                reference[idx] = GeluExact(x[xBase + col]) * x[xBase + intermediate + col];
            }
        };
        Func<float[]> readCpu = () => reference;

        Action runSilk = () =>
        {
            silk.SetBuffer(Nivara.Silk.SilkKernels.GeGlu, 0, sx);
            silk.SetBuffer(Nivara.Silk.SilkKernels.GeGlu, 1, sy);
            silk.SetUInt32(Nivara.Silk.SilkKernels.GeGlu, 2, (uint)rows);
            silk.SetUInt32(Nivara.Silk.SilkKernels.GeGlu, 3, (uint)intermediate);
            RunLinear(silk, Nivara.Silk.SilkKernels.GeGlu, dstN);
        };
        Func<float[]> readSilk = () => silk.Download(sy, dstN);

        return MeasureCpuReference(Nivara.Silk.SilkKernels.GeGlu, $"modernbert i{intermediate}", false, runCpu, readCpu, runSilk, readSilk);
    }

    // ── SmolLM (already CPU reference) ────────────────────────────

    static List<Leg> MeasureSmolLMKernels(Nivara.Silk.SilkRuntime silk, List<nint> buffers)
    {
        // unchanged from original (CPU reference) — keep all existing code
        var legs = new List<Leg>();

        const int hidden = 576;
        const int intermediate = 1536;
        const int numHeads = 9;
        const int numKvHeads = 3;
        const int headDim = 64;
        const float eps = 1e-5f;
        const int seq = 32;
        const int rows = 128;

        uint seed;

        // rms_norm
        {
            int n = rows * hidden;
            var src = new float[n];
            seed = 0x50700001u;
            for (int i = 0; i < n; i++) src[i] = GemmBenchmark.NextUniformF(ref seed);
            var gamma = new float[hidden];
            Array.Fill(gamma, 1f);
            var cpuOut = new float[n];
            Action runCpu = () => Nivara.AutoDiff.Nn.RMSNormKernel<float>.PerRowRMSNormForwardKernel(src, cpuOut, rows, hidden, eps);
            Func<float[]> readCpu = () => cpuOut;

            nint sx = silk.Allocate(n); buffers.Add(sx);
            nint sg = silk.Allocate(hidden); buffers.Add(sg);
            nint sy = silk.Allocate(n); buffers.Add(sy);
            silk.Upload(sx, src);
            silk.Upload(sg, gamma);

            Action runSilk = () =>
            {
                silk.SetBuffer(Nivara.Silk.SilkKernels.RmsNorm, 0, sx);
                silk.SetBuffer(Nivara.Silk.SilkKernels.RmsNorm, 1, sg);
                silk.SetBuffer(Nivara.Silk.SilkKernels.RmsNorm, 2, sy);
                silk.SetUInt32(Nivara.Silk.SilkKernels.RmsNorm, 3, (uint)rows);
                silk.SetUInt32(Nivara.Silk.SilkKernels.RmsNorm, 4, (uint)hidden);
                silk.SetFloat32(Nivara.Silk.SilkKernels.RmsNorm, 5, eps);
                silk.Run(Nivara.Silk.SilkKernels.RmsNorm,
                    [(nuint)Nivara.Silk.SilkKernels.LinearGlobal(rows)],
                    [(nuint)Nivara.Silk.SilkKernels.LinearGroupSize]);
                silk.Finish();
            };
            Func<float[]> readSilk = () => silk.Download(sy, n);

            legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.RmsNorm, $"smollm r{rows} c{hidden}", runCpu, readCpu, runSilk, readSilk));
        }

        // silu
        {
            int n = rows * intermediate;
            var src = new float[n];
            seed = 0x51100001u;
            for (int i = 0; i < n; i++) src[i] = GemmBenchmark.NextUniformF(ref seed);
            var cpuOut = new float[n];
            Action runCpu = () => Nivara.AutoDiff.Operations.GradKernels.Silu<float>(src, cpuOut);
            Func<float[]> readCpu = () => cpuOut;

            nint sx = silk.Allocate(n); buffers.Add(sx);
            nint sy = silk.Allocate(n); buffers.Add(sy);
            silk.Upload(sx, src);

            Action runSilk = () =>
            {
                Nivara.Silk.SilkLauncher.Silu(silk, sx, sy, n);
                silk.Finish();
            };
            Func<float[]> readSilk = () => silk.Download(sy, n);

            legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.Silu, $"smollm n{n}", runCpu, readCpu, runSilk, readSilk));
        }

        // swiglu
        {
            int n = rows * intermediate;
            var gate = new float[n];
            var up = new float[n];
            seed = 0x57100001u;
            for (int i = 0; i < n; i++) gate[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < n; i++) up[i] = GemmBenchmark.NextUniformF(ref seed);
            var temp = new float[n];
            var cpuOut = new float[n];
            Action runCpu = () =>
            {
                Nivara.AutoDiff.Operations.GradKernels.Silu<float>(gate, temp);
                System.Numerics.Tensors.TensorPrimitives.Multiply(temp.AsSpan(), up.AsSpan(), cpuOut.AsSpan());
            };
            Func<float[]> readCpu = () => cpuOut;

            nint sg = silk.Allocate(n); buffers.Add(sg);
            nint su = silk.Allocate(n); buffers.Add(su);
            nint sy = silk.Allocate(n); buffers.Add(sy);
            silk.Upload(sg, gate);
            silk.Upload(su, up);

            Action runSilk = () => { Nivara.Silk.SilkLauncher.SwiGlu(silk, sg, su, sy, rows, intermediate); silk.Finish(); };
            Func<float[]> readSilk = () => silk.Download(sy, n);

            legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.SwiGlu, $"smollm r{rows} i{intermediate}", runCpu, readCpu, runSilk, readSilk));
        }

        // causal_gqa_attention
        {
            int qLen = seq;
            int d = numHeads * headDim;
            int kvWidth = numKvHeads * headDim;
            int qn = qLen * d;
            int kn = qLen * kvWidth;
            var q = new float[qn];
            var k = new float[kn];
            var v = new float[kn];
            seed = 0x5CA90001u;
            for (int i = 0; i < q.Length; i++) q[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < k.Length; i++) k[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < v.Length; i++) v[i] = GemmBenchmark.NextUniformF(ref seed);
            float scale = 1f / MathF.Sqrt(headDim);
            var cpuOut = new float[qn];
            Action runCpu = () => Nivara.AutoDiff.Operations.AttentionKernels<float>.BatchedAttention(q, k, v, cpuOut, qLen, numHeads, numKvHeads, headDim, scale);
            Func<float[]> readCpu = () => cpuOut;

            nint sq = silk.Allocate(qn); buffers.Add(sq);
            nint sk = silk.Allocate(kn); buffers.Add(sk);
            nint sv = silk.Allocate(kn); buffers.Add(sv);
            nint so = silk.Allocate(qn); buffers.Add(so);
            silk.Upload(sq, q);
            silk.Upload(sk, k);
            silk.Upload(sv, v);

            silk.EnsureAttentionLocalMemory(headDim);
            Action runSilk = () => { Nivara.Silk.SilkLauncher.CausalGqaAttention(silk, sq, sk, sv, so, qLen, numHeads, numKvHeads, headDim, scale); silk.Finish(); };
            Func<float[]> readSilk = () => silk.Download(so, qn);

            legs.Add(MeasureCpuReference(Nivara.Silk.SilkKernels.CausalGqaAttention, $"smollm s{qLen} h{numHeads}/{numKvHeads} d{headDim}", runCpu, readCpu, runSilk, readSilk));
        }

        // rms_norm_gemv biased 3-slot
        {
            int kvWidth = hidden / 3;
            var x = Randoms(hidden, 0x67000001u);
            var gamma = Randoms(hidden, 0x67000002u);
            var w0 = Randoms(hidden * hidden, 0x67000003u);
            var w1 = Randoms(hidden * kvWidth, 0x67000004u);
            var w2 = Randoms(hidden * kvWidth, 0x67000005u);
            var b0 = Randoms(hidden, 0x67000006u);
            var b1 = Randoms(kvWidth, 0x67000007u);
            var b2 = Randoms(kvWidth, 0x67000008u);
            var c0 = new float[hidden];
            var c1 = new float[kvWidth];
            var c2 = new float[kvWidth];

            Action runCpu = () =>
            {
                RmsNormGemvCpu(x, gamma, w0, b0, c0, hidden, eps);
                RmsNormGemvCpu(x, gamma, w1, b1, c1, kvWidth, eps);
                RmsNormGemvCpu(x, gamma, w2, b2, c2, kvWidth, eps);
            };
            Func<float[]> readCpu = () => [.. c0, .. c1, .. c2];

            nint sx = silk.Allocate(hidden); buffers.Add(sx);
            nint sg = silk.Allocate(hidden); buffers.Add(sg);
            nint sw0 = silk.Allocate(w0.Length); buffers.Add(sw0);
            nint sw1 = silk.Allocate(w1.Length); buffers.Add(sw1);
            nint sw2 = silk.Allocate(w2.Length); buffers.Add(sw2);
            nint sb0 = silk.Allocate(b0.Length); buffers.Add(sb0);
            nint sb1 = silk.Allocate(b1.Length); buffers.Add(sb1);
            nint sb2 = silk.Allocate(b2.Length); buffers.Add(sb2);
            nint sc0 = silk.Allocate(c0.Length); buffers.Add(sc0);
            nint sc1 = silk.Allocate(c1.Length); buffers.Add(sc1);
            nint sc2 = silk.Allocate(c2.Length); buffers.Add(sc2);
            silk.Upload(sx, x);
            silk.Upload(sg, gamma);
            silk.Upload(sw0, w0);
            silk.Upload(sw1, w1);
            silk.Upload(sw2, w2);
            silk.Upload(sb0, b0);
            silk.Upload(sb1, b1);
            silk.Upload(sb2, b2);

            Action runSilk = () =>
            {
                Nivara.Silk.SilkLauncher.RmsNormGemv(silk, sx, sg,
                    [new(sw0, sb0, sc0, hidden), new(sw1, sb1, sc1, kvWidth), new(sw2, sb2, sc2, kvWidth)],
                    hasBias: true, aRows: 1, aCols: hidden, eps: eps);
                silk.Finish();
            };
            Func<float[]> readSilk = () =>
                [.. silk.Download(sc0, hidden), .. silk.Download(sc1, kvWidth), .. silk.Download(sc2, kvWidth)];

            legs.Add(MeasureCpuReference(
                Nivara.Silk.SilkKernels.RmsNormGemv,
                $"qkv h{hidden} q{hidden} kv{kvWidth} biased",
                runCpu, readCpu, runSilk, readSilk));
        }

        // rms_norm_gemv bias-free 2-slot
        {
            var x = Randoms(hidden, 0x67100001u);
            var gamma = Randoms(hidden, 0x67000002u);
            var w0 = Randoms(hidden * intermediate, 0x67000003u);
            var w1 = Randoms(hidden * intermediate, 0x67000004u);
            var c0 = new float[intermediate];
            var c1 = new float[intermediate];

            Action runCpu = () =>
            {
                RmsNormGemvCpu(x, gamma, w0, null, c0, intermediate, eps);
                RmsNormGemvCpu(x, gamma, w1, null, c1, intermediate, eps);
            };
            Func<float[]> readCpu = () => [.. c0, .. c1];

            nint sx = silk.Allocate(hidden); buffers.Add(sx);
            nint sg = silk.Allocate(hidden); buffers.Add(sg);
            nint sw0 = silk.Allocate(w0.Length); buffers.Add(sw0);
            nint sw1 = silk.Allocate(w1.Length); buffers.Add(sw1);
            nint sc0 = silk.Allocate(c0.Length); buffers.Add(sc0);
            nint sc1 = silk.Allocate(c1.Length); buffers.Add(sc1);
            silk.Upload(sx, x);
            silk.Upload(sg, gamma);
            silk.Upload(sw0, w0);
            silk.Upload(sw1, w1);

            Action runSilk = () =>
            {
                Nivara.Silk.SilkLauncher.RmsNormGemv(silk, sx, sg,
                    [new(sw0, 0, sc0, intermediate), new(sw1, 0, sc1, intermediate)],
                    hasBias: false, aRows: 1, aCols: hidden, eps: eps);
                silk.Finish();
            };
            Func<float[]> readSilk = () => [.. silk.Download(sc0, intermediate), .. silk.Download(sc1, intermediate)];

            legs.Add(MeasureCpuReference(
                Nivara.Silk.SilkKernels.RmsNormGemv, $"gate-up h{hidden} i{intermediate}", runCpu, readCpu, runSilk, readSilk));
        }

        // gemm_gemv_add
        {
            int aCols = hidden;
            int bCols = hidden;
            var a = Randoms(aCols, 0x68100001u);
            var b = Randoms(aCols * bCols, 0x68100002u);
            var residual = Randoms(bCols, 0x68100003u);
            var cpuOut = new float[bCols];

            Action runCpu = () =>
            {
                for (int j = 0; j < bCols; j++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < aCols; k++)
                        acc += a[k] * b[k * bCols + j];
                    cpuOut[j] = acc + residual[j];
                }
            };
            Func<float[]> readCpu = () => cpuOut;

            nint sa = silk.Allocate(a.Length); buffers.Add(sa);
            nint sb = silk.Allocate(b.Length); buffers.Add(sb);
            nint sres = silk.Allocate(residual.Length); buffers.Add(sres);
            nint sc = silk.Allocate(bCols); buffers.Add(sc);
            silk.Upload(sa, a);
            silk.Upload(sb, b);
            silk.Upload(sres, residual);

            Action runSilk = () =>
            {
                Nivara.Silk.SilkLauncher.GemmGemvAdd(silk, sa, sb, sc, sres, aCols, bCols);
                silk.Finish();
            };
            Func<float[]> readSilk = () => silk.Download(sc, bCols);

            legs.Add(MeasureCpuReference(
                Nivara.Silk.SilkKernels.GemmGemvAdd, $"o-proj a{aCols} b{bCols}", runCpu, readCpu, runSilk, readSilk));
        }

        return legs;
    }

    static float[] Randoms(int count, uint seed)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = GemmBenchmark.NextUniformF(ref seed);
        return values;
    }

    static void RmsNormGemvCpu(
        float[] x, float[] gamma, float[] weight, float[]? bias, float[] output, int cols, float eps)
    {
        int aCols = x.Length;
        float ssq = 0.0f;
        for (int k = 0; k < aCols; k++)
        {
            float v = x[k];
            ssq += v * v;
        }
        float scale = 1.0f / MathF.Sqrt(ssq / aCols + eps);
        for (int j = 0; j < cols; j++)
        {
            float acc = 0.0f;
            for (int k = 0; k < aCols; k++)
            {
                float xv = x[k] * scale * gamma[k];
                acc += xv * weight[k * cols + j];
            }
            output[j] = bias is null ? acc : acc + bias[j];
        }
    }

    // ── measurement helpers ───────────────────────────────────────

    static Leg MeasureCpuReference(
        string kernel,
        string shape,
        Action runCpu,
        Func<float[]> readCpu,
        Action runSilk,
        Func<float[]> readSilk)
        => MeasureCpuReference(kernel, shape, false, runCpu, readCpu, runSilk, readSilk);

    static Leg MeasureCpuReference(
        string kernel,
        string shape,
        bool requireBytes,
        Action runCpu,
        Func<float[]> readCpu,
        Action runSilk,
        Func<float[]> readSilk)
    {
        runCpu();
        runSilk();
        double cpuUs = GemmBenchmark.TimeBest(runCpu, Warmups, Rounds);
        double silkUs = GemmBenchmark.TimeBest(runSilk, Warmups, Rounds);
        return new Leg
        {
            Kernel = kernel,
            Shape = shape,
            RequireBytes = requireBytes,
            Reference = readCpu(),
            Actual = readSilk(),
            ReferenceUs = cpuUs,
            SilkUs = silkUs,
        };
    }

    static void RunLinear(Nivara.Silk.SilkRuntime silk, string kernel, int total)
    {
        silk.Run(
            kernel,
            [(nuint)Nivara.Silk.SilkKernels.LinearGlobal(total)],
            [(nuint)Nivara.Silk.SilkKernels.LinearGroupSize]);
        silk.Finish();
    }

    static void MeasureDecodeShapeAb(Nivara.Silk.SilkRuntime silk, List<nint> buffers)
    {
        Console.WriteLine();
        Console.WriteLine("=== decode-shape kernel geometry A/B: rms_norm_lm_head (tile, aRows=1) vs rms_norm_gemv (flat, one slot) (#682) ===");
        Console.WriteLine();

        const float eps = 1e-5f;
        const int warmups = 5;
        const int rounds = 10;

        (int hidden, int width)[] shapes =
        [
            (896, 151936),
            (896, 96),
            (896, 24),
        ];

        foreach (var (hidden, width) in shapes)
        {
            var x = new float[hidden];
            var gamma = new float[hidden];
            var w = new float[hidden * width];
            uint seed = 0x68200001u;
            for (int i = 0; i < x.Length; i++) x[i] = GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < gamma.Length; i++) gamma[i] = 1f + 0.5f * GemmBenchmark.NextUniformF(ref seed);
            for (int i = 0; i < w.Length; i++) w[i] = GemmBenchmark.NextUniformF(ref seed);

            nint sx = silk.Allocate(hidden); buffers.Add(sx);
            nint sg = silk.Allocate(hidden); buffers.Add(sg);
            nint sw = silk.Allocate(w.Length); buffers.Add(sw);
            nint soutTile = silk.Allocate(width); buffers.Add(soutTile);
            nint soutFlat = silk.Allocate(width); buffers.Add(soutFlat);
            silk.Upload(sx, x);
            silk.Upload(sg, gamma);
            silk.Upload(sw, w);

            Action runTile = () =>
            {
                Nivara.Silk.SilkLauncher.RmsNormLmHead(silk, sx, sg, sw, soutTile, 1, hidden, width, eps);
                silk.Finish();
            };
            Action runFlat = () =>
            {
                Nivara.Silk.SilkLauncher.RmsNormGemv(
                    silk, sx, sg,
                    [new Nivara.Silk.SilkLauncher.GemvSlot(sw, 0, soutFlat, width)],
                    hasBias: false, aRows: 1, aCols: hidden, eps: eps);
                silk.Finish();
            };

            for (int wu = 0; wu < warmups; wu++)
            {
                runTile();
                runFlat();
            }

            var tileTicks = new double[rounds];
            var flatTicks = new double[rounds];
            for (int r = 0; r < rounds; r++)
            {
                if (r % 2 == 0)
                {
                    tileTicks[r] = Time(() => runTile());
                    flatTicks[r] = Time(() => runFlat());
                }
                else
                {
                    flatTicks[r] = Time(() => runFlat());
                    tileTicks[r] = Time(() => runTile());
                }
            }

            var ratios = new List<double>(rounds);
            for (int r = 0; r < rounds; r++)
            {
                if (flatTicks[r] == 0.0) continue;
                ratios.Add(tileTicks[r] / flatTicks[r]);
            }
            double medianRatio = ratios.Count > 0 ? Median(ratios.ToArray()) : double.NaN;
            double medianTileMs = Median(tileTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            double medianFlatMs = Median(flatTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

            Console.WriteLine($"  hidden={hidden} width={width}: median tile/flat ratio = {medianRatio:F3}  " +
                              $"(tile {medianTileMs:F3} ms, flat {medianFlatMs:F3} ms)");
        }

        Console.WriteLine();
    }

    static long Time(Action action)
    {
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        action();
        return System.Diagnostics.Stopwatch.GetTimestamp() - start;
    }

    static double Median(double[] values)
    {
        double[] sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    static int Report(List<Leg> legs)
    {
        Console.WriteLine();
        Console.WriteLine($"{"kernel",-20} {"shape",-26} {"bytes",-12} {"%bound",-9} {"cpu us",-9} {"silk us",-9} {"x",-6}");
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

            double ratio = leg.ReferenceUs > 0 ? leg.SilkUs / leg.ReferenceUs : 0;
            bool band = ratio < BandLow || ratio > BandHigh;
            outOfBand += band ? 1 : 0;

            string bytes = byteOk
                ? (leg.RequireBytes ? "IDENTICAL" : "identical")
                : $"{byteDiffs} diff";
            Console.WriteLine(
                $"{leg.Kernel,-20} {leg.Shape,-26} {bytes,-12} {worstRatio * 100.0,-9:F3} " +
                $"{leg.ReferenceUs,-9:F1} {leg.SilkUs,-9:F1} {ratio,-6:F2}{(failed ? "  FAIL" : "")}{(band ? "  OUT-OF-BAND" : "")}");
        }

        Console.WriteLine(new string('-', 102));
        Console.WriteLine($"  {byteIdentical} of {legs.Count} kernel outputs are bit-identical to CPU; {outOfBand} of {legs.Count} outside the x{BandLow:F1}..x{BandHigh:F1} timing band.");
        Console.WriteLine(outOfBand > 0
            ? "  A timing ratio outside the band is a red flag to report, not a failure — this gate's exit code is correctness only."
            : "  Every kernel is inside the informal timing band.");

        if (failures == 0)
        {
            Console.WriteLine($"Gate PASS — all {legs.Count} cells met their criterion (byte-identity where required, otherwise {Bound:E0} * (1 + |cpu|)).");
            return 0;
        }

        int requiredByte = legs.Count(l => l.RequireBytes && ByteDiffs(l.Actual, l.Reference) != 0);
        int numeric = legs.Count(l => !l.RequireBytes && WorstRatio(l.Actual, l.Reference) > 1.0);
        if (requiredByte > 0)
            Console.WriteLine($"Byte-identity FAIL — {requiredByte} of the {legs.Count(l => l.RequireBytes)} required cells differ from CPU; the accumulation order or the contraction changed (a design bug, not rounding).");
        if (numeric > 0)
            Console.WriteLine($"Numeric FAIL — {numeric} cells exceed {Bound:E0} * (1 + |cpu|).");
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

    static float GeluExact(float v)
    {
        float z = v * 0.7071067811865475f;
        float az = MathF.Abs(z);
        float t = 1f / (1f + 0.3275911f * az);
        float p = 1.061405429f * t - 1.453152027f;
        p = p * t + 1.421413741f;
        p = p * t - 0.284496736f;
        p = p * t + 0.254829592f;
        float erf = 1f - p * t * MathF.Exp(-az * az);
        if (z < 0f) erf = -erf;
        return 0.5f * v * (1f + erf);
    }
}
