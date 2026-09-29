using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Nivara.Samples.Gpu;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand tiled-GEMM regression gate (#435): the lasting promotion of the deleted
/// %TEMP%\opencode\gemm-measure harness. Runs all ten ILGPU GEMM kernels
/// (samples/Nivara.Samples/Gpu/GemmKernels.cs — OneToOne, Row4, the M2 fused siblings
/// Row4Bias/Row4Gelu/Row4Relu/Row4Qkv, and the four measured-and-rejected #440 tile
/// geometries) over the model + padded-edge + Laya shapes, gating each
/// (kernel, shape) cell against a host double-precision truth (maxAbs ≤ 1e-3) and reporting
/// best-of-N synced-launch GMAC/s. Run only on explicit request (<c>--gemm</c>) — it is
/// GPU-dependent, so it is never part of the default scenario suite or the
/// <c>--json</c>/<c>--compare</c> gate.
/// </summary>
internal static class GemmBenchmark
{
    const int Warmups = 1;
    const int TimedRounds = 25;
    const double GateMaxAbs = 1e-3;
    const int ExitUnbuilt = 10;

    enum GemmVariant { OneToOne, Row4, Row4Bias, Row4Gelu, Row4Relu, Row4Qkv, Reg2x2K16, Reg2x2K32, Reg4x2K32, Reg1x8K16 }

    /// <summary>
    /// The #440 register-blocked geometries under test, in attribution order. The 2x2 factor
    /// appears at both K-tile widths so the blocking effect and the K-tile effect can be told
    /// apart rather than bundled into "the new kernel".
    /// </summary>
    static readonly GemmVariant[] s_regVariants =
    [
        GemmVariant.Reg2x2K16, GemmVariant.Reg2x2K32, GemmVariant.Reg4x2K32, GemmVariant.Reg1x8K16,
    ];

    /// <summary>The tile geometry a #440 variant implements (GpuBuffers is the single authority).</summary>
    static GpuBuffers.GemmGeometry GeometryFor(GemmVariant variant) => variant switch
    {
        GemmVariant.OneToOne => GpuBuffers.OneToOneGeometry,
        GemmVariant.Reg2x2K16 => GpuBuffers.GemmGeometries[0],
        GemmVariant.Reg2x2K32 => GpuBuffers.GemmGeometries[1],
        GemmVariant.Reg4x2K32 => GpuBuffers.GemmGeometries[2],
        GemmVariant.Reg1x8K16 => GpuBuffers.GemmGeometries[3],
        _ => GpuBuffers.Row4Geometry,
    };

    internal readonly record struct GemmShape(string Name, int Arows, int Acols, int Bcols)
    {
        public long Macs => (long)Arows * Acols * Bcols;
    }

    internal static readonly GemmShape[] s_shapes =
    [
        new("distilbert qkv/o", 128, 768, 768),
        new("distilbert fc1", 128, 768, 3072),
        new("distilbert fc2", 128, 3072, 768),
        new("distilbert head", 128, 768, 2),
        new("minilm qkv/o", 128, 384, 384),
        new("minilm fc1", 128, 384, 1536),
        new("minilm fc2", 128, 1536, 384),
        new("minilm head", 128, 384, 2),
        new("edge padded rows", 100, 770, 70),
        new("edge padded K", 64, 1032, 130),

        // #440 edge shapes for the wider tiles. A 32- or 64-row tile with a non-multiple row
        // count, and a 128-column tile against a narrow N, exercise the halo guards the
        // existing "edge padded rows" row (N=70) only half-reaches: 70 crosses a 64-column
        // tile but not a 128-column one, and neither existing row has a K below a K-tile step.
        new("edge 1x8 N=100", 100, 512, 100),      // 1x8 tile: N<128, A rows not a multiple of 16
        new("edge 4x2 M=20 N=40", 20, 96, 40),     // 4x2 tile: M<64 and K<KT32, both tiles partial
        new("edge 2x2 K=17", 64, 17, 64),          // K shorter than one K-tile step

        // Laya / ModernBERT-large: 28L d=1024, Wi=5248 (fused input|gate), ffn=2624.
        // The DistilBERT/MiniLM rows above are S=128 at d=768/384; Laya is d=1024 at
        // S=512, so its GEMMs are 4-30x larger and dominate the model. Row count is
        // the padded prompt length; K/N come from the checkpoint's tensor shapes.
        new("laya qkv", 512, 1024, 3072),        // Wqkv — fused 3x d, 3072/3 = 1024 exact
        new("laya attn out", 512, 1024, 1024),   // Wo
        new("laya fc1 (Wi)", 512, 1024, 5248),   // fused input|gate -> GeGLU
        new("laya fc2 (Wo)", 512, 2624, 1024),   // back down from ffn 2624
        new("laya head ff1", 512, 1024, 4096),   // decision head linear1
        new("laya head ff2", 512, 4096, 1024),   // decision head linear2 (ReLU head)
        new("laya act 1", 512, 1028, 256),       // d + 4 act features
        new("laya scorer 1", 8, 1024, 1024),     // k option markers -> d
        new("laya scorer 2", 8, 1024, 1),        // d -> 1 logit: GEMV-shaped, bandwidth-bound
        new("laya qkv@128", 128, 1024, 3072),    // S=128 control, comparable to the BERT rows
    ];

    [StructLayout(LayoutKind.Sequential)]
    struct SystemPowerStatus
    {
        public byte AclLineStatus;
        public byte BatteryFlag;
        public byte BatteryLifePercent;
        public byte SystemStatusFlag;
        public uint BatteryLifeTime;
        public uint BatteryFullLifeTime;
    }

    [DllImport("kernel32.dll")]
    static extern bool GetSystemPowerStatus(out SystemPowerStatus status);

    public static int Run(string[] args, bool writeFingerprintBaseline = false)
    {
        PrintPowerState();
        Console.WriteLine("Tiled GEMM regression gate (#435): ten ILGPU kernels vs double-precision truth");

        if (!TryCreateRuntime(out var runtime, out string unbuiltReason))
        {
            Console.WriteLine();
            Console.WriteLine("UNBUILT: --gemm requires an OpenCL GPU device (Intel Arc iGPU); none was found.");
            Console.WriteLine($"  {unbuiltReason}");
            return ExitUnbuilt;
        }

        using (runtime)
            return RunGate(runtime, writeFingerprintBaseline);
    }

    static int RunGate(IlgpuRuntime runtime, bool writeFingerprintBaseline)
    {
        CLDevice device = runtime.Device;
        Console.WriteLine($"  Device: {runtime.DeviceName}");
        Console.WriteLine($"  OpenCL: {ToolchainKey(device)}");
        Console.WriteLine($"  Gate  : maxAbs(gpu - dp-truth) <= {GateMaxAbs} (f32-vs-DP floor is 4.4e-5..1.6e-4 at K=768..3072; real kernel bugs land ~40-77)");
        Console.WriteLine($"  Timing: 1 JIT + {Warmups} warmup + best-of-{TimedRounds} synchronized launches");
        Console.WriteLine();

        var plainKernel = runtime.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernel);
        var row4Kernel = runtime.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4);
        var fusedKernels = new Dictionary<GemmVariant,
            Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>>
        {
            [GemmVariant.Row4Bias] = runtime.Accelerator.LoadKernel<
                ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                GemmKernels.TiledGemmKernelRow4Bias),
            [GemmVariant.Row4Gelu] = runtime.Accelerator.LoadKernel<
                ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                GemmKernels.TiledGemmKernelRow4Gelu),
            [GemmVariant.Row4Relu] = runtime.Accelerator.LoadKernel<
                ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                GemmKernels.TiledGemmKernelRow4Relu),
        };
        var qkvKernel = runtime.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Qkv);

        // #440 register-blocked geometries. Plain kernels only at this stage: the epilogue is
        // one extra register add over the same accumulators in the same ascending-K order, so
        // it is orthogonal to the blocking factor and the winner's siblings get ported after
        // the geometry is chosen.
        //
        // Each is loaded independently so one geometry the device cannot host (shared tile
        // too large, work-group limit) is reported and skipped rather than taking the whole
        // gate down with it — the other three still give the attribution the matrix exists for.
        var regKernels = new Dictionary<GemmVariant, Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>>();
        var unavailable = new HashSet<GemmVariant>();
        var loadFailures = new List<string>();
        foreach (var variant in s_regVariants)
        {
            var geometry = GeometryFor(variant);

            // A geometry the device genuinely cannot host (shared footprint over the reported
            // per-group limit) is reported and skipped. A kernel that *fails to compile* is a
            // different thing entirely and must fail the gate: #468 landed exactly there, as a
            // bare CLException from LoadKernel, and routing that down the skip path would have
            // reported a green gate over three of four missing kernels.
            try
            {
                GpuBuffers.ValidateGemmSharedMemory(runtime.Accelerator, geometry);
            }
            catch (Exception ex)
            {
                unavailable.Add(variant);
                Console.WriteLine($"SKIP  {geometry.Name} ({geometry.SharedBytes} B shared/group) — device cannot host it: {ex.Message}");
                continue;
            }

            try
            {
                regKernels[variant] = variant switch
                {
                    GemmVariant.Reg2x2K16 => runtime.Accelerator.LoadKernel<
                        ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                        GemmKernels.TiledGemmKernelReg2x2K16),
                    GemmVariant.Reg2x2K32 => runtime.Accelerator.LoadKernel<
                        ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                        GemmKernels.TiledGemmKernelReg2x2K32),
                    GemmVariant.Reg4x2K32 => runtime.Accelerator.LoadKernel<
                        ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                        GemmKernels.TiledGemmKernelReg4x2K32),
                    _ => runtime.Accelerator.LoadKernel<
                        ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                        GemmKernels.TiledGemmKernelReg1x8K16),
                };
            }
            catch (Exception ex)
            {
                loadFailures.Add($"  {geometry.Name}: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine($"LOAD-FAIL  {geometry.Name} — kernel did not compile: {ex.GetType().Name}: {ex.Message}");
            }
        }
        if (unavailable.Count > 0)
            Console.WriteLine($"      ({s_regVariants.Length - unavailable.Count} of {s_regVariants.Length} #440 geometries available on this device.)");

        Console.WriteLine($"{"kernel",-9} {"shape",-20} {"maxAbs",-11} {"gate(1e-3)",-12} {"best us",-9} {"GMAC/s",-9}");
        int failures = 0;
        int identicalFailures = 0;
        // Every cell's bit-exact result, compared against the committed baseline after the loop.
        // A SortedDictionary so the written file is ordered by key and diffs cleanly.
        var fingerprints = new SortedDictionary<string, GemmFingerprint.Cell>(StringComparer.Ordinal);
        // Row4's output per shape, kept so each #440 geometry can be compared bit-for-bit.
        // VariantsFor always emits Row4 before the #440 variants, so the reference is populated.
        var row4Reference = new Dictionary<string, float[]>();
        foreach (var shape in s_shapes)
        {
            uint seed = 0x9E3779B9u;
            var a = new float[shape.Arows * shape.Acols];
            var w = new float[shape.Bcols * shape.Acols];
            for (int i = 0; i < a.Length; i++) a[i] = NextUniformF(ref seed);
            for (int i = 0; i < w.Length; i++) w[i] = NextUniformF(ref seed);
            var bias = new float[shape.Bcols];
            for (int i = 0; i < bias.Length; i++) bias[i] = NextUniformF(ref seed);

            var bt = new float[shape.Acols * shape.Bcols];
            for (int j = 0; j < shape.Bcols; j++)
                for (int k = 0; k < shape.Acols; k++)
                    bt[k * shape.Bcols + j] = w[j * shape.Acols + k];

            double[] core = DpCore(shape, a, bt);

            foreach (var variant in WithRegVariants(VariantsFor(shape), regKernels.Keys))
            {
                double[] truth = ApplyEpilogue(shape, variant, core, bias);

                using var aBuffer = runtime.Allocate1D(shape.Arows * shape.Acols);
                using var bBuffer = runtime.Allocate1D(shape.Acols * shape.Bcols);
                using var cBuffer = runtime.Allocate1D(shape.Arows * shape.Bcols);
                using var biasBuffer = runtime.Allocate1D(shape.Bcols);
                aBuffer.CopyFromCPU(a);
                bBuffer.CopyFromCPU(bt);
                biasBuffer.CopyFromCPU(bias);

                // Every geometry's launch config comes from GpuBuffers, so a cell cannot
                // silently measure a kernel with the wrong grid. OneToOne is a 1x1 block over
                // the same 16x16 group, so it is a named geometry there like the rest rather
                // than an inline construction in the hot loop.
                var cfg = GpuBuffers.GemmCfg(shape.Arows, shape.Bcols, GeometryFor(variant));
                var stream = runtime.Stream;
                var aView = aBuffer.View;
                var bView = bBuffer.View;
                var cView = cBuffer.View;
                var biasView = biasBuffer.View;

                Action launch = variant switch
                {
                    GemmVariant.OneToOne => () => { plainKernel(stream, cfg, aView, bView, cView, shape.Arows, shape.Acols, shape.Bcols); runtime.Synchronize(); }
                    ,
                    GemmVariant.Row4 => () => { row4Kernel(stream, cfg, aView, bView, cView, shape.Arows, shape.Acols, shape.Bcols); runtime.Synchronize(); }
                    ,
                    GemmVariant.Row4Qkv => () => { qkvKernel(stream, cfg, aView, bView, cView, biasView, shape.Arows, shape.Acols, shape.Bcols, shape.Bcols / 3); runtime.Synchronize(); }
                    ,
                    _ when regKernels.ContainsKey(variant) => () => { regKernels[variant](stream, cfg, aView, bView, cView, shape.Arows, shape.Acols, shape.Bcols); runtime.Synchronize(); }
                    ,
                    _ => () => { fusedKernels[variant](stream, cfg, aView, bView, cView, biasView, shape.Arows, shape.Acols, shape.Bcols); runtime.Synchronize(); }
                    ,
                };

                launch();
                double bestUs = TimeBest(launch, Warmups, TimedRounds);

                float[] gpu = cBuffer.AsContiguous().GetAsArray();
                double maxAbs = 0.0;
                for (int i = 0; i < truth.Length; i++)
                    maxAbs = Math.Max(maxAbs, Math.Abs(gpu[i] - truth[i]));

                bool failed = maxAbs > GateMaxAbs;
                failures += failed ? 1 : 0;

                // Captured for every cell regardless of its tolerance verdict: the fingerprint
                // answers a different question ("are the bits the same as the ones we committed,
                // when nothing was supposed to change them?") and a cell can be inside tolerance
                // and still not be the result this kernel is contracted to produce.
                fingerprints[GemmFingerprint.Key(variant.ToString(), shape.Name)] = GemmFingerprint.Of(gpu);

                // #440 byte-identity: the plain geometries accumulate strictly ascending k over
                // the same f32 values as Row4, so every output element must be bit-identical.
                // A violation means the accumulation order changed - a design bug, not rounding -
                // so it is counted separately from the tolerance check above. Folding it into
                // `failures` made the FAIL line claim cells exceeded the tolerance when none had.
                bool identical = true;
                if (variant == GemmVariant.Row4)
                    row4Reference[shape.Name] = gpu.ToArray();
                else if (s_regVariants.Contains(variant))
                {
                    identical = row4Reference.TryGetValue(shape.Name, out var reference)
                        && ByteIdentical(gpu, reference);
                    if (!identical) identicalFailures++;
                }

                double gmacPerSec = shape.Macs / (bestUs * 1000.0);
                Console.WriteLine($"{variant,-9} {shape.Name,-20} {maxAbs,-11:E2} {(failed ? "FAIL" : "PASS"),-12} {bestUs,-9:F1} {gmacPerSec,-9:F0}{(identical ? "" : "  BYTE-DIVERGED")}");
            }
        }

        int cells = CountCells(regKernels.Count);
        Console.WriteLine();

        // #440 coverage is stated unconditionally. Omitting it when zero geometries load would
        // make "all 75 cells passed" read exactly like the pre-#440 gate - the same output for a
        // very different amount of testing. Verified: with the variant list emptied every
        // remaining cell still passes, so this line and the exit code are the only things
        // distinguishing the two, which is why the message cannot itself read as a pass.
        Console.WriteLine(regKernels.Count == s_regVariants.Length
            ? $"#440 coverage: all {s_regVariants.Length} tile geometries loaded."
            : $"#440 coverage: {regKernels.Count} of {s_regVariants.Length} tile geometries loaded"
              + (unavailable.Count > 0 ? $", {unavailable.Count} skipped as unhostable on this device" : "")
              + (loadFailures.Count > 0 ? $", {loadFailures.Count} FAILED TO COMPILE" : "")
              + (regKernels.Count == 0 ? " - NO #440 COVERAGE AT ALL, which fails the gate rather than passing it." : "."));

        if (loadFailures.Count > 0)
        {
            Console.WriteLine($"Gate FAIL — {loadFailures.Count} #440 kernel(s) failed to compile, which is a defect rather than a skip:");
            foreach (var failure in loadFailures)
                Console.WriteLine(failure);
        }

        if (failures == 0 && identicalFailures == 0)
            Console.WriteLine($"Gate PASS — all {cells} (kernel, shape) cells within {GateMaxAbs} of double-precision truth.");
        else
        {
            if (failures > 0)
                Console.WriteLine($"Tolerance FAIL — {failures} of {cells} cells exceed {GateMaxAbs} vs double-precision truth.");
            if (identicalFailures > 0)
                Console.WriteLine($"Byte-identity FAIL — {identicalFailures} of {CountCells(regKernels.Count)} #440 cells are not bit-identical to Row4; the accumulation order changed, which is a design bug rather than rounding.");
        }

        if (regKernels.Count > 0 && identicalFailures == 0)
            Console.WriteLine("Byte-identity PASS — all #440 cells bit-identical to Row4 (ascending-K accumulation preserved).");

        int fingerprintFailures = ReportFingerprints(device, fingerprints, writeFingerprintBaseline);

        // Every distinct way this gate can fail contributes to the exit code: a cell outside
        // tolerance, a cell that is not bit-identical, a kernel that would not compile, #440
        // coverage that quietly dropped to zero, and a cell whose f32 bits are not the ones the
        // committed baseline recorded.
        return failures + identicalFailures + loadFailures.Count + (regKernels.Count == 0 ? 1 : 0) + fingerprintFailures;
    }

    /// <summary>
    /// What the f32 results of this run were produced by, in the one string the baseline is keyed
    /// on. The driver version is the part that matters and the part ILGPU does not expose, so it
    /// is read from the ICD; when that fails the key says so rather than quietly degrading to the
    /// OpenCL version, which a driver bump does not change.
    /// </summary>
    static string ToolchainKey(CLDevice device)
    {
        string? driver = ClDriverVersion.TryGet(device);
        return driver is null
            ? $"OpenCL {device.DeviceVersion}, driver version UNREADABLE — this key cannot see a driver bump"
            : $"OpenCL {device.DeviceVersion}, driver {driver}";
    }

    /// <summary>
    /// Records or checks every cell's bit-exact result against the committed baseline, and
    /// returns how many distinct failure reasons that produced.
    /// </summary>
    /// <remarks>
    /// A normal run only ever compares. Regenerating is behind an explicit flag, because a gate
    /// that can rewrite its own reference is a gate whose green means nothing — the exact defect
    /// the exactness assertion exists to prevent. An absent baseline is a failure rather than a
    /// pass for the same reason: with nothing to be exact against, the run verified nothing.
    /// </remarks>
    static int ReportFingerprints(
        CLDevice device,
        SortedDictionary<string, GemmFingerprint.Cell> measured,
        bool writeBaseline)
    {
        Console.WriteLine();

        string toolchain = ToolchainKey(device);

        if (writeBaseline)
        {
            string path = GemmFingerprint.Locate() ?? GemmFingerprint.SearchedPaths[0];
            GemmFingerprint.Write(path, device.Name ?? "?", toolchain, measured);
            Console.WriteLine($"Fingerprint baseline WRITTEN — {measured.Count} cells -> {path}");
            Console.WriteLine("  This is not a verdict. The cells above are the reference from here on; a later");
            Console.WriteLine("  run that disagrees with this file fails.");
            return 0;
        }

        string? located = GemmFingerprint.Locate();
        GemmFingerprint.Baseline? baseline = null;
        if (located is not null)
        {
            try
            {
                baseline = GemmFingerprint.Read(located);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fingerprint FAIL — cannot read baseline {located}: {ex.GetType().Name}: {ex.Message}");
                return 1;
            }
        }

        var verdict = GemmFingerprint.Compare(baseline, measured, device.Name ?? "?", toolchain);

        if (verdict.BaselineAbsent)
        {
            Console.WriteLine($"Fingerprint FAIL — no committed baseline ({GemmFingerprint.FileName}) in any of:");
            foreach (string path in GemmFingerprint.SearchedPaths)
                Console.WriteLine($"  {path}");
            Console.WriteLine($"  Nothing to be exact against, so the {measured.Count} cells this run measured are");
            Console.WriteLine($"  unverified. Record the reference deliberately with --gemm {GemmFingerprint.WriteFlag}.");
            return verdict.Failures;
        }

        // Coverage is stated as "matched of baseline rows", never as a bare count of what this
        // run happened to measure: a geometry the device cannot host shrinks the denominator, and
        // a shrunken run that reads like a full one is the same defect as a silent skip.
        int baselineRows = baseline!.Cells.Count;
        string notExercised = verdict.NotExercised.Count == 0
            ? ""
            : $", {verdict.NotExercised.Count} baseline rows not exercised this run ({string.Join(", ", verdict.NotExercised)})";

        if (verdict.ToolchainDifference is not null)
        {
            Console.WriteLine($"Fingerprint NOT VERIFIED — {verdict.ToolchainDifference}.");
            Console.WriteLine("  f32 results can legitimately differ across drivers (FMA contraction in the generated");
            Console.WriteLine("  OpenCL C changes the rounding with no change to the kernel), so this run's bits are");
            Console.WriteLine("  compared against a reference from a different toolchain. Per-cell differences below are");
            Console.WriteLine("  reported, not judged. Re-record deliberately with --gemm " + GemmFingerprint.WriteFlag + ".");
        }

        if (verdict.Mismatched.Count > 0)
        {
            Console.WriteLine($"Fingerprint FAIL — {verdict.Mismatched.Count} of {baselineRows} baseline cells are not bit-identical:");
            foreach (string row in verdict.Mismatched)
                Console.WriteLine($"  {row}");
        }

        if (verdict.Missing.Count > 0)
        {
            Console.WriteLine($"Fingerprint FAIL — {verdict.Missing.Count} measured cells have no baseline row:");
            foreach (string key in verdict.Missing)
                Console.WriteLine($"  {key}");
            Console.WriteLine($"  A cell with no reference cannot be exact about anything. Re-record deliberately with --gemm {GemmFingerprint.WriteFlag}.");
        }

        if (verdict.Mismatched.Count == 0 && verdict.Missing.Count == 0 && verdict.ToolchainDifference is null)
            Console.WriteLine($"Fingerprint PASS — {verdict.Matched} of {baselineRows} baseline cells bit-identical{notExercised}.");

        return verdict.Failures;
    }

    /// <summary>Bit-for-bit comparison of a #440 variant's output against Row4's.</summary>
    static bool ByteIdentical(float[] a, float[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                return false;
        return true;
    }

    internal static bool TryCreateRuntime(out IlgpuRuntime runtime, out string reason)
    {
        try
        {
            runtime = new IlgpuRuntime();
            reason = "";
            return true;
        }
        catch (Exception ex)
        {
            runtime = null!;
            reason = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>Total (kernel, shape) cells the gate measures, for its PASS/FAIL count.</summary>
    static int CountCells(int regVariantsTested)
    {
        int count = 0;
        foreach (var shape in s_shapes)
            count += VariantsFor(shape).Count + regVariantsTested;
        return count;
    }

    internal static void PrintPowerState()
    {
        if (!GetSystemPowerStatus(out var status))
        {
            Console.WriteLine("Power  : GetSystemPowerStatus unavailable — AC state unknown");
            return;
        }

        if (status.AclLineStatus == 0)
        {
            Console.WriteLine("WARNING: on battery — the iGPU throttles flat, GMAC/s numbers are INVALID (docs/BERT-GPU.md battery lesson).");
            Console.WriteLine("         The correctness gate is battery-safe and still runs.");
        }
        else if (status.AclLineStatus == 1)
        {
            Console.WriteLine("Power  : AC line — GMAC/s valid");
        }
        else
        {
            Console.WriteLine("Power  : unknown AC status (255)");
        }
    }

    internal static float NextUniformF(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state / 4294967295.0f) * 2.0f - 1.0f;
    }

    /// <summary>Double-precision C = A·Bt in row-major [aRows × bCols] (variant-agnostic core).</summary>
    static double[] DpCore(GemmShape shape, float[] a, float[] bt)
    {
        int aRows = shape.Arows, k = shape.Acols, bCols = shape.Bcols;
        double[] ad = Array.ConvertAll(a, x => (double)x);
        double[] btd = Array.ConvertAll(bt, x => (double)x);
        var core = new double[aRows * bCols];

        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++)
                    sum += ad[i * k + kk] * btd[kk * bCols + j];
                core[i * bCols + j] = sum;
            }
        }

        return core;
    }

    /// <summary>Applies the fused epilogue (or none) to the double-precision core. The plain
    /// kernels take no bias; Qkv scatters into the same block-separated layout the kernel
    /// writes (GemmKernels.QkvDest mapping; private there, so mirrored here).</summary>
    static double[] ApplyEpilogue(GemmShape shape, GemmVariant variant, double[] core, float[] bias)
    {
        int aRows = shape.Arows, bCols = shape.Bcols;
        int blockWidth = variant == GemmVariant.Row4Qkv ? bCols / 3 : 0;
        var truth = new double[aRows * bCols];

        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                double raw = core[i * bCols + j] + bias[j];
                double v = variant switch
                {
                    GemmVariant.OneToOne or GemmVariant.Row4
                        or GemmVariant.Reg2x2K16 or GemmVariant.Reg2x2K32
                        or GemmVariant.Reg4x2K32 or GemmVariant.Reg1x8K16 => core[i * bCols + j],
                    GemmVariant.Row4Gelu => GeluD(raw),
                    GemmVariant.Row4Relu => Math.Max(0.0, raw),
                    _ => raw,
                };
                int idx = variant == GemmVariant.Row4Qkv
                    ? QkvIndex(j, i, aRows, blockWidth)
                    : i * bCols + j;
                truth[idx] = v;
            }
        }

        return truth;
    }

    static int QkvIndex(int j, int outRow, int aRows, int blockWidth)
    {
        int block = j / blockWidth;
        int colInBlock = j - block * blockWidth;
        return block * aRows * blockWidth + outRow * blockWidth + colInBlock;
    }

    static double GeluD(double v)
    {
        double z = v * 0.7071067811865475;
        double az = Math.Abs(z);
        double t = 1.0 / (1.0 + 0.3275911 * az);
        double p = 1.061405429 * t - 1.453152027;
        p = p * t + 1.421413741;
        p = p * t - 0.284496736;
        p = p * t + 0.254829592;
        double erf = 1.0 - p * t * Math.Exp(-az * az);
        if (z < 0.0) erf = -erf;
        return 0.5 * v * (1.0 + erf);
    }

    static IReadOnlyList<GemmVariant> VariantsFor(GemmShape shape) => shape.Name switch
    {
        "distilbert qkv/o" or "minilm qkv/o" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Qkv],
        "distilbert fc1" or "minilm fc1" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Gelu],
        "distilbert head" or "minilm head" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Relu],
        // Laya picks the fused epilogue its real activation uses, so the rows measure the
        // kernels this model would actually run rather than a generic bias add.
        "laya qkv" or "laya qkv@128" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Qkv],
        "laya fc1 (Wi)" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Gelu],
        "laya head ff2" => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias, GemmVariant.Row4Relu],
        _ => [GemmVariant.OneToOne, GemmVariant.Row4, GemmVariant.Row4Bias],
    };

    /// <summary>
    /// The four #440 geometries, appended to every shape's variant list after Row4 (which the
    /// byte-identity check needs as its reference). Appending to the existing list rather than
    /// replacing it keeps each shape's real epilogue rows intact, so the new kernels are
    /// measured under exactly the shapes and neighbours the incumbent is.
    /// </summary>
    static IReadOnlyList<GemmVariant> WithRegVariants(IReadOnlyList<GemmVariant> baseVariants, ICollection<GemmVariant> available)
        => [.. baseVariants, .. s_regVariants.Where(available.Contains)];

    /// <summary>Best-of-<paramref name="rounds"/> wall time of a synchronized launch, in µs.
    /// Shared with <see cref="GemmLegBenchmark"/> so both probes time legs identically.</summary>
    internal static double TimeBest(Action launch, int warmups, int rounds)
    {
        for (int w = 0; w < warmups; w++)
            launch();

        double bestUs = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            var sw = Stopwatch.StartNew();
            launch();
            sw.Stop();
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMilliseconds * 1000.0);
        }

        return bestUs;
    }
}
