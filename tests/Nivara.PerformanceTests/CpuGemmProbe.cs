using Nivara.AutoDiff.Operations;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand CPU GEMM ceiling probe for the Laya backend decision (docs/TODO.md, leg 3).
///
/// This is the decisive measurement. The question is not "is the CPU slow" — it is
/// "how fast could the CPU be, given a properly blocked GEMM", because the GPU is
/// measured as-is while the CPU side has no in-tree implementation of the fix that
/// grounding identified. Deciding CPU on an unfixed-GPU-vs-unfixed-CPU comparison would
/// be circular.
///
/// Grounding (docs/TODO.md § Grounding corrections) established three facts that shape
/// this probe:
///
/// 1. There is no BCL matrix multiply to compare against. <c>TensorPrimitives.Dot</c> is
///    the *vector* dot product, and <c>Tensor.MatrixMultiply</c> has not shipped
///    (dotnet/runtime#95863, noted at TensorsHelper.cs:30-36). So the CPU ceiling has to
///    be written here.
/// 2. The in-tree GEMM is already row-parallel. <c>GradKernels.MatMulTransposedB</c>
///    delegates to <c>TensorsHelper.MultiplyCore</c>, which reaches
///    <c>Parallel.For</c> (TensorsHelper.cs:191) behind <c>ShouldParallelize</c>. Issue
///    #456 reported a missing <c>Parallel.For</c>; it is not missing, and #456 is closed.
/// 3. The remaining suspect is the kernel's *shape*: <c>MultiplyRowFloat</c> computes
///    every output element as its own <c>TensorPrimitives.Dot</c> over K
///    (TensorsHelper.cs:293), so a 512x1024 @ 1024x3072 GEMM runs 1.57M independent
///    horizontal reductions of length 1024 instead of holding an output tile in
///    registers across the K loop.
///
/// Leg 3 (<see cref="Blocked"/>) is the shape that fact 3 implies: <c>Parallel.For</c>
/// over disjoint output rows, JB output columns accumulated in register-resident
/// <see cref="Vector{T}"/>s across the whole K loop, storing only at the end. It also
/// drops the two per-call preps the in-tree path pays — a full <c>RentCopy</c> of A
/// (TensorsHelper.cs:341) and a pooled staging copy of B — neither of which is
/// necessary, since output rows are disjoint and A/B are read-only.
///
/// Every leg is gated against the same host double-precision truth the GPU gate uses
/// (maxAbs ≤ 1e-3), so a fast-but-wrong reference cannot win the comparison.
///
/// Run only on explicit request (<c>--cpu-gemm</c>). Not GPU-dependent, but still
/// excluded from the default scenario suite and the <c>--json</c>/<c>--compare</c> gate
/// because it is a diagnostic, not a regression threshold.
/// </summary>
internal static class CpuGemmProbe
{
    const int Warmups = 1;
    const int TimedRounds = 5;
    const double GateMaxAbs = 1e-3;

    /// <summary>Output columns accumulated per register block (leg 3).</summary>
    const int BlockCols = 4;

    enum Leg
    {
        /// <summary>GradKernels.MatMulTransposedB — what the AutoDiff path calls today.</summary>
        AutoDiff,

        /// <summary>GradKernels.MatMul — same kernel, but transposes B per call.</summary>
        BTranspose,

        /// <summary>Probe-local register-blocked Parallel.For reference (the ceiling).</summary>
        Blocked,
    }

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("CPU GEMM ceiling probe: in-tree path vs a register-blocked reference");
        Console.WriteLine();
        Console.WriteLine($"  Legs     : {Leg.AutoDiff} (GradKernels.MatMulTransposedB), {Leg.BTranspose} (GradKernels.MatMul), {Leg.Blocked} (Parallel.For + register accumulators)");
        Console.WriteLine($"  Gate     : maxAbs(leg - dp-truth) <= {GateMaxAbs}");
        Console.WriteLine($"  Timing   : {Warmups} warmup + best-of-{TimedRounds}");
        Console.WriteLine($"  Layout   : the two in-tree legs differ only in B's layout — MatMul takes [K x N]");
        Console.WriteLine($"             and transposes per call, MatMulTransposedB takes [N x K] ready-made.");
        Console.WriteLine($"             Both compute A[M x K] · B[K x N]. Blocked reads the same [N x K].");
        Console.WriteLine();

        Console.WriteLine($"{"shape",-16} {"M",-5} {"K",-5} {"N",-6} {"leg",-12} {"maxAbs",-10} {"gate",-6} {"best ms",-9} {"GMAC/s",-8} {"x vs MatMul",-12}");
        int failures = 0;
        var bestMs = new Dictionary<(string, Leg), double>();

        foreach (var shape in GemmBenchmark.s_shapes)
        {
            int m = shape.Arows, k = shape.Acols, n = shape.Bcols;
            var a = new float[m * k];
            var b = new float[k * n];
            uint seed = 0x9E3779B9u;
            for (int i = 0; i < a.Length; i++) a[i] = Uniform(ref seed);
            for (int i = 0; i < b.Length; i++) b[i] = Uniform(ref seed);

            // B^T, [N x K] row-major, so every leg reads each output column contiguously —
            // the layout the in-tree path builds internally via Transpose/CopyTo.
            var bt = new float[n * k];
            for (int j = 0; j < n; j++)
                for (int kk = 0; kk < k; kk++)
                    bt[j * k + kk] = b[kk * n + j];

            double[] truth = DpTruth(a, bt, m, k, n);

            // Measure every leg before printing any: the speedup column is relative to
            // MatMul, so the baseline has to exist before the first row is formatted.
            var order = new[] { Leg.AutoDiff, Leg.BTranspose, Leg.Blocked };
            var legMs = new double[order.Length];
            var legAbs = new double[order.Length];

            for (int li = 0; li < order.Length; li++)
            {
                var leg = order[li];
                var result = new float[m * n];
                // MatMulTransposedB means "B is already transposed": it copies b into its
                // staging buffer verbatim and then slices bT[j*K .. +K] as output column j,
                // so b must be [N x K]. MatMul takes [K x N] and transposes per call. Both
                // compute A[M x K] · B[K x N]; Blocked reads the same [N x K] as the former.
                Action run = leg switch
                {
                    Leg.AutoDiff => () => GradKernels.MatMulTransposedB(a, bt, result, m, k, n),
                    Leg.BTranspose => () => GradKernels.MatMul(a, b, result, m, k, n),
                    _ => () => BlockedGemm(a, bt, result, m, k, n),
                };

                run();
                legMs[li] = TimeBest(run, Warmups, TimedRounds);

                double maxAbs = 0.0;
                for (int i = 0; i < truth.Length; i++)
                    maxAbs = Math.Max(maxAbs, Math.Abs(result[i] - truth[i]));
                legAbs[li] = maxAbs;

                if (maxAbs > GateMaxAbs)
                    failures++;

                bestMs[(shape.Name, leg)] = legMs[li];
            }

            double baselineGmacs = shape.Macs / (legMs[Array.IndexOf(order, Leg.BTranspose)] * 1e6);
            for (int li = 0; li < order.Length; li++)
            {
                double gmacs = shape.Macs / (legMs[li] * 1e6);
                string speedup = order[li] == Leg.BTranspose ? "-" : $"{gmacs / baselineGmacs:F2}x";
                bool failed = legAbs[li] > GateMaxAbs;
                Console.WriteLine($"{shape.Name,-16} {m,-5} {k,-5} {n,-6} {order[li],-12} {legAbs[li],-10:E2} {(failed ? "FAIL" : "PASS"),-6} {legMs[li],-9:F3} {gmacs,-8:F1} {speedup,-12}");
            }
        }

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? $"Gate PASS — all legs within {GateMaxAbs} of double-precision truth."
            : $"Gate FAIL — {failures} leg(s) exceed {GateMaxAbs}.");

        ProjectLayaForward(bestMs);
        return failures;
    }

    /// <summary>
    /// Rolls the per-shape numbers up into a projected Laya forward pass, GEMM only.
    /// This is the figure the backend decision actually turns on, because it is the only
    /// one that accounts for the model's shape: 28 encoder layers of four GEMMs each,
    /// plus the 2-layer decision head. Attention, LayerNorm, GeGLU and dispatch are
    /// excluded, so this is a lower bound on wall-clock for every leg equally.
    /// </summary>
    static void ProjectLayaForward(Dictionary<(string, Leg), double> bestMs)
    {
        string[] encoderLayer = ["laya qkv", "laya attn out", "laya fc1 (Wi)", "laya fc2 (Wo)"];
        string[] headLayer = ["laya head ff1", "laya head ff2"];
        string[] headTail = ["laya act 1", "laya scorer 1", "laya scorer 2"];

        Console.WriteLine();
        Console.WriteLine("Projected Laya forward, GEMM only (attention / norms / dispatch excluded):");
        Console.WriteLine($"{"leg",-12} {"enc layer",-12} {"x28 layers",-13} {"head",-10} {"total",-10} {"GMAC/s",-9} {"vs MatMul",-12}");

        var legs = new[] { Leg.AutoDiff, Leg.BTranspose, Leg.Blocked };
        var totals = new double[legs.Length];
        var layerMs = new double[legs.Length];
        var headMs = new double[legs.Length];
        var encoderMs = new double[legs.Length];

        for (int i = 0; i < legs.Length; i++)
        {
            layerMs[i] = encoderLayer.Sum(s => bestMs[(s, legs[i])]);
            encoderMs[i] = layerMs[i] * 28;
            headMs[i] = headLayer.Sum(s => bestMs[(s, legs[i])]) * 2
                      + headTail.Sum(s => bestMs[(s, legs[i])]);
            totals[i] = encoderMs[i] + headMs[i];
        }

        double baseline = totals[Array.IndexOf(legs, Leg.BTranspose)];
        const long totalMacs = 175_700_000_000L + 8_600_000_000L;

        for (int i = 0; i < legs.Length; i++)
        {
            double gmacs = totalMacs / (totals[i] * 1e6);
            string speedup = legs[i] == Leg.BTranspose ? "-" : $"{baseline / totals[i]:F2}x";
            Console.WriteLine($"{legs[i],-12} {layerMs[i],-12:F1} {encoderMs[i],-13:F0} {headMs[i],-10:F1} {totals[i],-10:F0} {gmacs,-9:F0} {speedup,-12}");
        }

        Console.WriteLine();
        Console.WriteLine("  enc layer = qkv + attn out + Wi + Wo at S=512; 28 layers.");
        Console.WriteLine("  head      = (ff1 + ff2) x 2 layers + act + scorer.");
    }

    /// <summary>
    /// C = A · B^T in double precision, where <paramref name="bt"/> is B^T as [N x K]
    /// row-major. Parallel over output rows, which are disjoint, so the result is
    /// bit-identical to the serial form: each (i, j) accumulates k in ascending order
    /// on a single thread.
    /// </summary>
    static double[] DpTruth(float[] a, float[] bt, int m, int k, int n)
    {
        var ad = new double[a.Length];
        var btd = new double[bt.Length];
        for (int i = 0; i < ad.Length; i++) ad[i] = a[i];
        for (int i = 0; i < btd.Length; i++) btd[i] = bt[i];

        var truth = new double[m * n];
        Parallel.For(0, m, i =>
        {
            int rowBase = i * k;
            int outBase = i * n;
            for (int j = 0; j < n; j++)
            {
                int bBase = j * k;
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++)
                    sum += ad[rowBase + kk] * btd[bBase + kk];
                truth[outBase + j] = sum;
            }
        });
        return truth;
    }

    /// <summary>
    /// Register-blocked reference GEMM: C[i, j] = dot(A[i, :], Bt[j, :]) with
    /// <see cref="BlockCols"/> output columns accumulated in <see cref="Vector{T}"/>
    /// registers across the entire K loop, so the horizontal reduction happens once per
    /// output element at the end rather than once per (element, K) pair. A and Bt are
    /// read in place — no staging copy, no RentCopy — and <c>Parallel.For</c> partitions
    /// output rows, which are disjoint.
    /// </summary>
    static void BlockedGemm(float[] a, float[] bt, float[] result, int m, int k, int n)
    {
        int width = Vector<float>.Count;
        int kVector = k - (k % width);

        Parallel.For(0, m, i =>
        {
            int aBase = i * k;
            int outBase = i * n;

            for (int j = 0; j < n; j += BlockCols)
            {
                int cols = Math.Min(BlockCols, n - j);

                if (cols == BlockCols)
                {
                    var b0 = j * k;
                    var b1 = b0 + k;
                    var b2 = b1 + k;
                    var b3 = b2 + k;
                    var acc0 = Vector<float>.Zero;
                    var acc1 = Vector<float>.Zero;
                    var acc2 = Vector<float>.Zero;
                    var acc3 = Vector<float>.Zero;

                    for (int kk = 0; kk < kVector; kk += width)
                    {
                        var av = new Vector<float>(a, aBase + kk);
                        acc0 += av * new Vector<float>(bt, b0 + kk);
                        acc1 += av * new Vector<float>(bt, b1 + kk);
                        acc2 += av * new Vector<float>(bt, b2 + kk);
                        acc3 += av * new Vector<float>(bt, b3 + kk);
                    }

                    result[outBase + j] = Vector.Sum(acc0);
                    result[outBase + j + 1] = Vector.Sum(acc1);
                    result[outBase + j + 2] = Vector.Sum(acc2);
                    result[outBase + j + 3] = Vector.Sum(acc3);

                    for (int kk = kVector; kk < k; kk++)
                    {
                        float av = a[aBase + kk];
                        result[outBase + j] += av * bt[b0 + kk];
                        result[outBase + j + 1] += av * bt[b1 + kk];
                        result[outBase + j + 2] += av * bt[b2 + kk];
                        result[outBase + j + 3] += av * bt[b3 + kk];
                    }
                }
                else
                {
                    for (int c = 0; c < cols; c++)
                    {
                        int bBase = (j + c) * k;
                        var acc = Vector<float>.Zero;
                        for (int kk = 0; kk < kVector; kk += width)
                            acc += new Vector<float>(a, aBase + kk) * new Vector<float>(bt, bBase + kk);

                        float sum = Vector.Sum(acc);
                        for (int kk = kVector; kk < k; kk++)
                            sum += a[aBase + kk] * bt[bBase + kk];
                        result[outBase + j + c] = sum;
                    }
                }
            }
        });
    }

    static float Uniform(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state / 4294967295.0f) * 2.0f - 1.0f;
    }

    static double TimeBest(Action run, int warmups, int rounds)
    {
        for (int w = 0; w < warmups; w++)
            run();

        double best = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            var watch = Stopwatch.StartNew();
            run();
            watch.Stop();
            best = Math.Min(best, watch.Elapsed.TotalMilliseconds);
        }
        return best;
    }
}
