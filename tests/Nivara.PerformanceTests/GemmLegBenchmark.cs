using ILGPU;
using ILGPU.Runtime;
using Nivara.Samples.Gpu;

namespace Nivara.PerformanceTests;

/// <summary>
/// Phase 1 of #440: measures how much of a ModernBERT/Laya GPU forward is GEMM and how much
/// is everything else, by timing each non-GEMM kernel in isolation at the exact launch
/// configuration <c>ModernBertGpuRunner.ForwardOnDevice</c> uses and multiplying by the
/// per-forward launch counts read off that method (15 launches per encoder layer: 4 GEMM,
/// 2 LayerNorm, 3 SplitColumns, 2 Rotary, 1 attention, 1 GeGlu, 2 Add, plus 3 once-per-forward).
///
/// <para>
/// Of those 3 once-per-forward launches only <c>gather</c> is timed. The other two are
/// <c>LayerNorm1D</c> (the embedding projection and the final norm), so they are folded into
/// the per-layer LayerNorm count rather than listed separately. That is deliberate and not
/// worth a separate leg: the LayerNorm leg totals ~28 ms of a ~2221 ms Laya forward, so the
/// omission moves every published share by less than 0.05%.
/// </para>
///
/// <para>
/// Why this exists: the #440 issue comment argues that GEMM is ~99.9% of Laya's *arithmetic*,
/// so kernel throughput is essentially the whole cost. That is a share of arithmetic, not of
/// time. This probe turns the arithmetic share into a measured time share, which is what
/// actually decides whether the tile-32/2x2 GEMM work is the leading lever.
/// </para>
///
/// <para>
/// This is a measurement, not a gate — nothing here asserts a bound and a healthy run returns
/// 0. Both sides of the split (GEMM and non-GEMM) are timed in the same session on the same
/// device, so the ratio carries no cross-run load contamination.
/// </para>
///
/// <para>
/// The totals are comparable to <c>modernbert --gpu benchmark</c> but **not** to
/// <c>laya --gpu benchmark</c>: this probe times the <em>encoder</em> only, while the laya
/// benchmark figure covers the whole model including the decision head. For laya the residual
/// against that benchmark is dispatch, readback <em>and</em> the head, so do not read it as
/// unattributed kernel time. <c>modernbert --gpu benchmark</c> is encoder-only and does line up.
/// </para>
///
/// <para>
/// AC power only, for the same reason as <c>--gemm</c>: the iGPU throttles flat on battery.
/// <see cref="GemmBenchmark.PrintPowerState"/> warns when it detects that.
/// </para>
/// </summary>
internal static class GemmLegBenchmark
{
    const int Warmups = 1;
    const int TimedRounds = 25;
    const int ExitUnbuilt = 10;
    const float Eps = 1e-5f;

    /// <summary>
    /// One encoder's shape and depth. <c>Rows</c> is the padded prompt length; <c>Ffn</c> is
    /// the feed-forward width (ModernBERT's <c>Wi</c> produces <c>2 * Ffn</c> for the fused
    /// input|gate GeGLU pair).
    /// </summary>
    internal sealed record ModelShape(
        string Name, int Rows, int Hidden, int Heads, int HeadDim, int Ffn, int Layers);

    // Laya is ModernBERT-large: 28L d=1024, Wi=5248 (fused), ffn=2624, 16 heads.
    // ModernBERT base is the control at d=768, so the two differ in width as well as depth.
    internal static readonly ModelShape[] s_models =
    [
        new("modernbert base", 512, 768, 12, 64, 3072, 22),
        new("laya large", 512, 1024, 16, 64, 2624, 28),
    ];

    /// <summary>One measured kernel: its time per launch and how many times a forward issues it.</summary>
    readonly record struct Leg(string Name, string Shape, double Us, int LaunchesPerForward)
    {
        public double PerForwardMs => Us * LaunchesPerForward / 1000.0;
    }

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("Kernel leg split (#440 Phase 1): per-forward time by kernel, at ModernBertGpuRunner launch configs");

        if (!GemmBenchmark.TryCreateRuntime(out var runtime, out string unbuiltReason))
        {
            Console.WriteLine();
            Console.WriteLine("UNBUILT: --gemm-legs requires an OpenCL GPU device (Intel Arc iGPU); none was found.");
            Console.WriteLine($"  {unbuiltReason}");
            return ExitUnbuilt;
        }

        using (runtime)
            return RunSplit(runtime);
    }

    static int RunSplit(IlgpuRuntime runtime)
    {
        var acc = runtime.Accelerator;
        Console.WriteLine($"  Device: {runtime.DeviceName}");
        Console.WriteLine($"  Limits: {acc.MaxNumThreadsPerGroup} threads/group, {acc.MaxSharedMemoryPerGroup} B shared/group");
        Console.WriteLine($"  Timing: {Warmups} warmup + best-of-{TimedRounds} synchronized launches per leg");
        Console.WriteLine();

        foreach (var model in s_models)
            RunModel(runtime, model);

        return 0;
    }

    static void RunModel(IlgpuRuntime runtime, ModelShape m)
    {
        var acc = runtime.Accelerator;
        int rows = m.Rows, hidden = m.Hidden, heads = m.Heads, headDim = m.HeadDim, ffn = m.Ffn, layers = m.Layers;
        int halfDim = headDim / 2;
        float scale = 1f / MathF.Sqrt(headDim);

        var gather = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(ElementwiseKernels.Gather);
        var layerNorm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(ElementwiseKernels.LayerNorm1D);
        var splitColumns = acc.LoadKernel<ArrayView<float>, ArrayView<float>, int, int, int, int>(ElementwiseKernels.SplitColumns);
        var rotary = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(ElementwiseKernels.Rotary);
        var attention = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float>(AttentionKernels.BatchedAttention);
        var geGlu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, int, int>(ElementwiseKernels.GeGlu);
        var add = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.Add);
        var gemm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GemmKernels.TiledGemmKernelRow4);

        // Scratch laid out exactly as ModernBertGpuRunner.EnsureWorkspace does.
        using var xBuf = runtime.Allocate1D(rows * hidden);
        using var normedBuf = runtime.Allocate1D(rows * hidden);
        using var projectedBuf = runtime.Allocate1D(rows * hidden);
        using var contextBuf = runtime.Allocate1D(rows * hidden);
        using var mlpOutBuf = runtime.Allocate1D(rows * hidden);
        using var qkvBuf = runtime.Allocate1D(rows * 3 * hidden);
        using var qkvSplitBuf = runtime.Allocate1D(rows * 3 * hidden);
        using var mlpInBuf = runtime.Allocate1D(rows * 2 * ffn);
        using var gatedBuf = runtime.Allocate1D(rows * ffn);
        using var maskBuf = runtime.Allocate1D(rows * rows);
        using var gammaBuf = runtime.Allocate1D(hidden);
        using var betaBuf = runtime.Allocate1D(hidden);
        using var cosBuf = runtime.Allocate1D(rows * halfDim);
        using var sinBuf = runtime.Allocate1D(rows * halfDim);
        using var posBuf = GpuBuffers.AllocInt(acc, rows);

        // Zeroed, not random: these legs are timed, not validated, and zeros keep the
        // arithmetic free of denormals and NaNs. The mask is the exception — BatchedAttention
        // early-returns on an all-suppressed row, so an all-zero mask would measure the
        // cheap path and understate attention badly.
        Zero(xBuf); Zero(normedBuf); Zero(projectedBuf); Zero(contextBuf); Zero(mlpOutBuf);
        Zero(qkvBuf); Zero(qkvSplitBuf); Zero(mlpInBuf); Zero(gatedBuf);
        Zero(gammaBuf); Zero(betaBuf); Zero(cosBuf); Zero(sinBuf);
        Fill(maskBuf, 1f);
        posBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, new int[rows]);

        var x = xBuf.View;
        var normed = normedBuf.View;
        var projected = projectedBuf.View;
        var context = contextBuf.View;
        var qkv = qkvBuf.View;
        var qkvSplit = qkvSplitBuf.View;
        var mlpIn = mlpInBuf.View;
        var gated = gatedBuf.View;
        var mask = maskBuf.View;
        var gamma = gammaBuf.View;
        var beta = betaBuf.View;
        var cos = cosBuf.View;
        var sin = sinBuf.View;
        var pos = posBuf.View;

        // Global attention (band < 0) is the more expensive of ModernBERT's 3:1 global:sliding
        // layer mix, so the attention leg here is the pessimistic one.
        int band = GpuBuffers.GlobalAttentionBand;
        var q = qkvSplit.SubView(0, rows * hidden);
        var k = qkvSplit.SubView(rows * hidden, rows * hidden);
        var v = qkvSplit.SubView(2 * rows * hidden, rows * hidden);

        var stream = runtime.Stream;
        double Time(Action launch) => GemmBenchmark.TimeBest(launch, Warmups, TimedRounds);

        var legs = new List<Leg>
        {
            new("GEMM qkv", $"[{rows}x{hidden}x{3 * hidden}]",
                TimeGemm(runtime, gemm, rows, hidden, 3 * hidden), layers),

            new("GEMM attn out", $"[{rows}x{hidden}x{hidden}]",
                TimeGemm(runtime, gemm, rows, hidden, hidden), layers),

            new("GEMM Wi", $"[{rows}x{hidden}x{2 * ffn}]",
                TimeGemm(runtime, gemm, rows, hidden, 2 * ffn), layers),

            new("GEMM WoMlp", $"[{rows}x{ffn}x{hidden}]",
                TimeGemm(runtime, gemm, rows, ffn, hidden), layers),

            new("LayerNorm1D", $"[{rows}x{hidden}]",
                Time(() => { layerNorm(stream, GpuBuffers.Cfg1D(rows), x, gamma, beta, normed, rows, hidden, Eps); runtime.Synchronize(); }),
                2 * layers),

            new("SplitColumns", $"[{rows}x{hidden}]",
                Time(() =>
                {
                    for (int part = 0; part < 3; part++)
                        splitColumns(stream, GpuBuffers.Cfg1D(rows * hidden), qkv,
                            qkvSplit.SubView(part * rows * hidden, rows * hidden), rows, hidden, 3, part);
                    runtime.Synchronize();
                }),
                3 * layers),

            new("Rotary", $"[{rows}x{heads}x{headDim}]",
                Time(() => { rotary(stream, GpuBuffers.Cfg1D(rows * heads * halfDim), q, pos, cos, sin, q, rows, heads, headDim); runtime.Synchronize(); }),
                2 * layers),

            new("BatchedAttention", $"[{heads}x{rows}x{rows}x{headDim}]",
                Time(() =>
                {
                    attention(stream, GpuBuffers.Cfg1D(rows * heads, GpuBuffers.AttentionGroupSize),
                        q, k, v, mask, context, 1, rows, heads, headDim, band, scale);
                    runtime.Synchronize();
                }),
                layers),

            new("GeGlu", $"[{rows}x{ffn}]",
                Time(() => { geGlu(stream, GpuBuffers.Cfg1D(rows * ffn), mlpIn, gated, rows, ffn); runtime.Synchronize(); }),
                layers),

            new("Add", $"[{rows}x{hidden}]",
                Time(() => { add(stream, GpuBuffers.Cfg1D(rows * hidden), x, projected, x); runtime.Synchronize(); }),
                2 * layers),

            new("gather (once)", $"[{rows}x{hidden}]",
                Time(() => { gather(stream, GpuBuffers.Cfg1D(rows * hidden), normed, pos, x, hidden); runtime.Synchronize(); }),
                1),
        };

        double total = legs.Sum(l => l.PerForwardMs);
        double gemmMs = legs.Where(l => l.Name.StartsWith("GEMM", StringComparison.Ordinal)).Sum(l => l.PerForwardMs);

        Console.WriteLine($"── {m.Name}: rows={rows} hidden={hidden} heads={heads} headDim={headDim} ffn={ffn} layers={layers}");
        Console.WriteLine($"  {"leg",-16} {"shape",-26} {"us/launch",-11} {"n/fwd",-7} {"ms/fwd",-9} {"share",-7}");
        foreach (var leg in legs)
            Console.WriteLine($"  {leg.Name,-16} {leg.Shape,-26} {leg.Us,-11:F1} {leg.LaunchesPerForward,-7} {leg.PerForwardMs,-9:F2} {100 * leg.PerForwardMs / total,-6:F1}%");
        Console.WriteLine($"  {100 * gemmMs / total:F1}% GEMM, {100 - 100 * gemmMs / total:F1}% non-GEMM, {total:F2} ms/fwd total");
        Console.WriteLine();
    }

    /// <summary>
    /// Times <see cref="GemmKernels.TiledGemmKernelRow4"/> at the shape and launch config the
    /// runners use, so the GEMM side of the split comes from the same session and device as
    /// the non-GEMM legs. Fresh buffers per call: the weight matrix is [aCols x bCols] and
    /// differs per shape, so there is nothing to share.
    /// </summary>
    static double TimeGemm(
        IlgpuRuntime runtime,
        Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> kernel,
        int aRows, int aCols, int bCols)
    {
        var a = new float[aRows * aCols];
        var b = new float[aCols * bCols];
        uint seed = 0x85EBCA6Bu;
        for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
        for (int i = 0; i < b.Length; i++) b[i] = GemmBenchmark.NextUniformF(ref seed);

        using var aBuf = runtime.Allocate1D(a.Length);
        using var bBuf = runtime.Allocate1D(b.Length);
        using var cBuf = runtime.Allocate1D(aRows * bCols);
        aBuf.CopyFromCPU(a);
        bBuf.CopyFromCPU(b);

        var stream = runtime.Stream;
        var cfg = GpuBuffers.GemmCfg(aRows, bCols);
        return GemmBenchmark.TimeBest(
            () => { kernel(stream, cfg, aBuf.View, bBuf.View, cBuf.View, aRows, aCols, bCols); runtime.Synchronize(); },
            Warmups, TimedRounds);
    }

    static void Zero(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        => buffer.CopyFromCPU(new float[buffer.Length]);

    static void Fill(MemoryBuffer1D<float, Stride1D.Dense> buffer, float value)
    {
        var host = new float[buffer.Length];
        Array.Fill(host, value);
        buffer.CopyFromCPU(host);
    }
}
