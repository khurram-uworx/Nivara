using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.SimdProbe;

/// <summary>
/// A/B probe for the #136 transpose kernel swap (<c>TensorsHelper.Transpose</c>).
///
/// <para><b>Question.</b> <c>Tensor.Transpose&lt;T&gt;</c> shipped in .NET 11 but
/// returns a zero-copy strided <i>view</i>; Nivara consumers need contiguous
/// row-major output, so the swap route must pay Tensor construction plus
/// <c>FlattenTo</c> materialization per call. Does that still cost more than the
/// handwritten cache-tiled kernel?</para>
///
/// <para><b>Why this is a probe and not a unit-test assertion (#482).</b>
/// <c>Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization</c>
/// asserted the ordering from a single best-of-5 comparison of two sequentially
/// measured routes. It failed 3 of 5 runs on a clean tree, once on a 0.2% margin —
/// so it could not tell a regression from machine noise and provided no coverage.
/// The defects were structural, not statistical:</para>
/// <list type="number">
///   <item>best-of-5 taken <i>inside</i> each route, then the two summaries compared
///   once — two noisy estimates meet and one number decides;</item>
///   <item>routes measured sequentially with <c>tiled</c> first, so any frequency or
///   load drift across the window always landed on <c>tiled</c> (systematic bias,
///   not variance);</item>
///   <item>no warmup, so tiered JIT and first-touch page faults landed inside
///   someone's measurement.</item>
/// </list>
///
/// <para>This probe fixes the design rather than the threshold: it interleaves the
/// routes and alternates which goes first so first-measured drift cancels, warms
/// both routes up front, and reports a distribution (win rate + median ratio) over
/// several shapes instead of a single point estimate. It answers "do the routes
/// differ, and by how much" — a question a distribution can answer and a single
/// timing comparison cannot.</para>
///
/// <para><b>Reading the verdict.</b> <c>median ratio &gt; 1</c> means tiled is
/// SLOWER. A ratio inside roughly ±3% is machine noise, not a result: the honest
/// conclusion there is that the two routes are comparable and the swap should stay
/// parked.</para>
/// </summary>
internal static class TransposeKernelProbe
{
    // The failing gate's shape, plus neighbours that stress different regimes:
    // square, tall, wide, and a tile-multiple vs non-multiple pair.
    static readonly (int Rows, int Cols, string Label)[] Shapes =
    [
        (1024, 1024, "square, the #482 gate shape"),
        (512, 512, "square, smaller"),
        (2048, 1024, "tall"),
        (1024, 2048, "wide"),
        (129, 257, "non-tile-multiple (tail handling)"),
    ];

    const int Rounds = 30;      // interleaved A/B pairs per shape
    const int Warmups = 5;      // untimed passes per route, before any measurement

    public static int Run()
    {
        Console.WriteLine("=== #136 transpose route A/B: tiled kernel vs BCL view+flatten ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Interleaved design: {Rounds} A/B rounds per shape, first route alternates every round,");
        Console.WriteLine($"{Warmups} untimed warmups per route. A ratio > 1 means tiled is SLOWER.");
        Console.WriteLine();

        int failures = 0;
        var coldStart = new List<string>();
        var allRatios = new List<double>();
        var allWins = 0;
        var allRounds = 0;

        foreach (var (rows, cols, label) in Shapes)
        {
            var (wins, bclWins, ties, ratios, tiledMs, bclMs, seqTiled, seqBcl) = MeasureShape(rows, cols, coldStart);
            allRatios.AddRange(ratios);
            allWins += wins;
            allRounds += wins + bclWins + ties;

            var verdict = Classify(ratios);
            double ticksToMs = 1000.0 / Stopwatch.Frequency;

            Console.WriteLine($"  {rows,5}x{cols,-5} {label}  " +
                              $"({rows * cols / 1e6:F2} Melem, {rows * cols * 8 / 1e6:F1} MB moved)");
            Console.WriteLine($"          interleaved: tiled wins {wins,2} | bcl wins {bclWins,2} | ties {ties,2}  " +
                              $"| median ratio {Median(ratios):F3} (range {ratios.Min():F3}..{ratios.Max():F3})");
            Console.WriteLine($"          interleaved: median tiled {tiledMs,6:F2} ms | median bcl {bclMs,6:F2} ms");
            Console.WriteLine($"          [#482 methodology, reproduced] best-of-5, sequential, tiled first: " +
                              $"tiled {seqTiled * ticksToMs,6:F2} ms | bcl {seqBcl * ticksToMs,6:F2} ms | " +
                              $"ratio {(double)seqTiled / seqBcl:F3} -> {(seqTiled < seqBcl ? "PASS" : "FAIL")}");
            Console.WriteLine($"          VERDICT: {verdict}");
            Console.WriteLine();

            if (verdict == "TILE BOUNDARY BROKEN — tiled loses consistently")
                failures++;
        }

        double medianAll = Median(allRatios.ToArray());
        double winRate = allRounds > 0 ? allWins / (double)allRounds : 0;
        Console.WriteLine($"  ALL SHAPES: tiled wins {allWins}/{allRounds} rounds ({winRate * 100:F1}%), " +
                          $"median ratio {medianAll:F3}");
        Console.WriteLine();
        Console.WriteLine("  Cold-start (first call, before warmup) — the JIT asymmetry:");
        foreach (var line in coldStart)
            Console.WriteLine($"          {line}");
        Console.WriteLine("          Tiled is JIT'd here and starts at tier 0; Tensor.Transpose/FlattenTo");
        Console.WriteLine("          ship ReadyToRun and are already optimized. A no-warmup gate measures");
        Console.WriteLine("          an unoptimized hand kernel against optimized framework code.");
        Console.WriteLine();
        Console.WriteLine(Describe(medianAll, winRate));
        return failures;
    }

    /// <summary>Interleaved A/B for one shape.</summary>
    static (int TiledWins, int BclWins, int Ties, double[] Ratios, double TiledMs, double BclMs, long SeqTiled, long SeqBcl)
        MeasureShape(int rows, int cols, List<string> coldStart)
    {
        var src = FillRowMajor(rows, cols, seed: 42);
        var tiledBuf = new float[rows * cols];

        // Correctness first: a timing comparison between routes that disagree is meaningless.
        Tiled(src, tiledBuf, rows, cols);
        if (!ViaBcl(src, rows, cols).AsSpan().SequenceEqual(tiledBuf))
            throw new InvalidOperationException($"routes disagree at {rows}x{cols} — refusing to time them");

        // ── Cold-start diagnostics ────────────────────────────────────────────────
        // The hand-written tiled kernel is JIT'd on first call and starts at tier 0
        // (quick JIT, no optimizations) until the call-count threshold promotes it.
        // Tensor.Transpose/FlattenTo live in System.Numerics.Tensors, which ships
        // ReadyToRun — already optimized, never tier-0. So a best-of-5-with-no-warmup
        // gate compares an *unoptimized* hand kernel against *optimized* framework
        // code. These first-call timings make that asymmetry visible.
        var coldTiled = Time(() => Tiled(src, tiledBuf, rows, cols));
        var coldBcl = Time(() => _ = ViaBcl(src, rows, cols));
        double toMs = 1000.0 / Stopwatch.Frequency;

        // Warm up BOTH routes before any measurement so tiered JIT and first-touch
        // page faults do not land in one route's sample.
        for (int w = 0; w < Warmups; w++)
        {
            Tiled(src, tiledBuf, rows, cols);
            _ = ViaBcl(src, rows, cols);
        }

        coldStart.Add($"{rows}x{cols}: cold(1st call) tiled={coldTiled * toMs:F2} ms vs bcl={coldBcl * toMs:F2} ms");

        var tiledTicks = new double[Rounds];
        var bclTicks = new double[Rounds];

        for (int r = 0; r < Rounds; r++)
        {
            // Alternate which route is measured first. This is what cancels the
            // first-measured systematic bias that made the unit-test gate flaky.
            if (r % 2 == 0)
            {
                tiledTicks[r] = Time(() => Tiled(src, tiledBuf, rows, cols));
                bclTicks[r] = Time(() => _ = ViaBcl(src, rows, cols));
            }
            else
            {
                bclTicks[r] = Time(() => _ = ViaBcl(src, rows, cols));
                tiledTicks[r] = Time(() => Tiled(src, tiledBuf, rows, cols));
            }
        }

        int tiledWins = 0, bclWins = 0, ties = 0;
        var ratios = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            ratios[r] = tiledTicks[r] / bclTicks[r];
            if (tiledTicks[r] < bclTicks[r]) tiledWins++;
            else if (bclTicks[r] < tiledTicks[r]) bclWins++;
            else ties++;
        }

        // Reproduce the superseded #482 unit-test methodology for contrast: sequential
        // best-of-5, tiled measured first, compared exactly once.
        var seqTiled = BestOfFive(() => Tiled(src, tiledBuf, rows, cols));
        var seqBcl = BestOfFive(() => _ = ViaBcl(src, rows, cols));

        double ticksToMs = 1000.0 / Stopwatch.Frequency;
        return (tiledWins, bclWins, ties, ratios, Median(tiledTicks) * ticksToMs, Median(bclTicks) * ticksToMs,
            seqTiled, seqBcl);
    }

    /// <summary>The old, defective cross-route estimator: minimum of N, taken per route and compared once.</summary>
    static long BestOfFive(Action action)
    {
        long best = long.MaxValue;
        for (int i = 0; i < 5; i++)
            best = Math.Min(best, Time(action));
        return best;
    }

    /// <summary>Classifies a per-shape ratio distribution against the observed noise floor.</summary>
    static string Classify(double[] ratios)
    {
        double median = Median(ratios);
        if (median > 1.10)
            return "TILE BOUNDARY BROKEN — tiled loses consistently";
        if (median < 0.90)
            return "tiled consistently faster — the tiled kernel is earning its keep";
        if (median >= 0.97)
            return "comparable, slight edge to tiled (inside noise)";
        if (median <= 1.03)
            return "comparable, slight edge to BCL (inside noise)";
        return "comparable — within noise, no swap signal";
    }

    static string Describe(double medianRatio, double winRate)
    {
        if (medianRatio > 1.10)
            return "Conclusion: the BCL view+flatten route is materially faster. Re-evaluate the #136 swap — "
                 + "do NOT restate this as a flaky test, the ordering claim is genuinely violated.";
        if (medianRatio < 0.90)
            return "Conclusion: the tiled kernel is materially faster. Keep it; the #136 swap stays parked "
                 + "until Tensor.Transpose materializes contiguous output.";
        return $"Conclusion: the two routes are comparable (median ratio {medianRatio:F3}, tiled win rate "
             + $"{winRate * 100:F0}%). This is what the #482 unit-test gate was really measuring — noise. "
             + "The honest record is that neither route wins; the tiled kernel stays because it is the only "
             + "span-based materializer, not because it is faster.";
    }

    /// <summary>Verbatim copy of <c>TensorsHelper.Transpose</c> (src/Nivara/Tensors/TensorsHelper.cs).</summary>
    static void Tiled(float[] src, float[] dst, int rows, int cols)
    {
        const int tile = 32;
        for (int i0 = 0; i0 < rows; i0 += tile)
        {
            int iMax = Math.Min(i0 + tile, rows);
            for (int j0 = 0; j0 < cols; j0 += tile)
            {
                int jMax = Math.Min(j0 + tile, cols);
                for (int i = i0; i < iMax; i++)
                {
                    int srcRow = i * cols;
                    for (int j = j0; j < jMax; j++)
                        dst[j * rows + i] = src[srcRow + j];
                }
            }
        }
    }

    /// <summary>The #136 swap route: strided view plus <c>FlattenTo</c> materialization.</summary>
    static float[] ViaBcl(float[] src, int rows, int cols)
    {
        var tensor = Tensor.Create(src, new ReadOnlySpan<nint>([rows, cols]));
        var view = Tensor.Transpose(tensor);
        var dst = new float[rows * cols];
        view.FlattenTo(dst.AsSpan());
        return dst;
    }

    static long Time(Action action)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        return sw.ElapsedTicks;
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    static float[] FillRowMajor(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1.0);
        return data;
    }
}