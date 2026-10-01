using Nivara.AutoDiff.Operations;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Nivara.PerformanceTests;

/// <summary>
/// <c>--mask</c>: measures <c>AttentionKernels&lt;T&gt;.ApplyMask</c> against the
/// <c>TensorPrimitives.Add</c> it replaced (#480).
///
/// <para><b>Question.</b> #448 changed the additive mask from a vectorized
/// <c>TensorPrimitives.Add</c> over the <c>[qLen, kvLen]</c> score buffer to a compare-and-select
/// loop, because BCL has no select/blend primitive. The justification on record is an inference
/// from a MAC count, never a stopwatch reading, so this probe supplies the reading.</para>
///
/// <para><b>Why a probe and not a scenario row.</b> Two reasons, both about what the harness can
/// answer. The scenario table measures each row independently and sequentially, so it cannot
/// produce a ratio within one run - a ratio there means comparing against a recorded baseline
/// JSON, which is cross-day and cross-machine drift. And the superseded transpose gate (#482)
/// failed 3 of 5 runs on a clean tree on a 0.2% margin because its <em>design</em> was wrong: it
/// took best-of-5 inside each route and compared the two summaries once, measured the routes
/// sequentially with one always first, and warmed neither up. This probe keeps
/// <c>TransposeKernelProbe</c>'s design - interleaved rounds, alternating which route is measured
/// first, both routes warmed before any sample, verdict on a median ratio against a stated band.</para>
///
/// <para><b>Why absolute throughput is reported next to the ratio.</b> Both routes read the same
/// two buffers and write the same one, so they move identical bytes. A bare ratio cannot
/// distinguish "both are bandwidth-bound and equal" from "both are compute-bound and equal", and
/// neither can it show that <c>scores[i] = MaskedCell(...)</c> is a select the JIT already
/// auto-vectorizes - which would make "scalar" a misnomer and part of the issue's premise wrong.
/// ns/op and GB/s make that falsifiable rather than assumed.</para>
///
/// <para><b>The two routes legitimately disagree, by design.</b> They cannot be required to agree:
/// assigning <c>-inf</c> rather than summing into it <em>is</em> the #448 fix, because
/// <c>NaN + (-inf) = NaN</c> lets a diverged score escape suppression. So parity is asserted only
/// where it is exact - all-finite scores with a mask in <c>{0, -inf}</c> - and the intentional
/// divergences are documented here rather than papered over. Outside that domain the routes differ
/// on exactly two cases, both of which #448 exists to fix: a <c>NaN</c> score and a <c>+inf</c>
/// score under a <c>-inf</c> mask cell. Open #489 disputes <c>MaskedCell</c>'s <c>dead</c> rule;
/// it does not touch this parity contract.</para>
///
/// <para><b>Steady-state repetition is representative.</b> The only data-dependent branch is
/// <c>mask == NegativeInfinity</c>, which reads the mask, not the score. Over a mask in
/// <c>{0, -inf}</c> the map is idempotent - <c>x + 0 == x</c> and <c>-inf + -inf == -inf</c> - so
/// repeated timed iterations re-apply the identical branch pattern with no drift and the buffers
/// need no re-seeding between rounds.</para>
///
/// <para><b>Reading a verdict.</b> A median ratio above 1 means <c>ApplyMask</c> is SLOWER than the
/// add it replaced. Inside roughly +/-3% is machine noise, not a result. Cells whose faster route
/// spans fewer than <see cref="MinReliableTicks"/> ticks print <c>n/a</c> and are excluded from the
/// aggregate, because at that scale the ratio is clock quantization rather than cost.</para>
///
/// <para><b>No percentage of end-to-end forward is printed.</b> The leg profile reports only what
/// this run measures - the mask's share of attention, and the absolute ms per forward a revert
/// would return. The two recorded forward-time figures do not reconcile with the leg profile
/// (attention alone exceeds the recorded whole forward), so dividing by either would manufacture
/// precision that the evidence does not carry. The reconciliation is printed so the conflict is
/// visible instead of hidden.</para>
/// </summary>
static class ApplyMaskProbe
{
    /// <summary>Noise band, reused from <c>TransposeKernelProbe.Classify</c> so two probes do not invent two thresholds.</summary>
    const double NoiseBand = 0.03;

    const int Rounds = 30;
    const int Warmups = 5;

    /// <summary>
    /// A cell whose faster route still resolves to fewer ticks than this is below the clock's
    /// resolution, and its ratio is quantization rather than cost. <c>[1, 512]</c> is the case
    /// that matters: the whole call is ~10 Stopwatch ticks, which is why it reports ratios of
    /// 5-10x. Those cells are labelled and excluded from the aggregate rather than published,
    /// because "ApplyMask is 6x slower at decode" read off a rounded tick count is exactly the
    /// false claim this probe exists to avoid.
    /// </summary>
    const int MinReliableTicks = 1000;

    /// <summary>The issue's three shapes. See the decode caveat printed at the end of the run.</summary>
    static readonly (int Rows, int Cols, string Label)[] Shapes =
    [
        (512, 512, "Laya prefill S=512"),
        (2048, 2048, "wide prefill"),
        (1, 512, "decode-shaped (API only)"),
    ];

    // Laya-shaped forward leg profile. ModernBERT-large: 28 layers, d=1024, nhead=16, headDim=64,
    // S=512 (docs/LAYA.md). Laya's CPU attention is the per-sequence path,
    // ModernBertModel.cs -> ReverseGradOperations.MultiHeadAttention, and at B=1 the per-sequence
    // and batched loops are the same work.
    const int LayaQLen = 512, LayaD = 1024, LayaHeads = 16, LayaHeadDim = 64, LayaLayers = 28;

    // Recorded cross-run figures, NOT measured here and NOT used as denominators. They are printed
    // only so a reader can see that they do not reconcile with the leg profile measured in this
    // run; see the reconciliation block at the end of RunLegProfile.
    //
    //   RecordedLayaForwardMs      - docs/LAYA.md's ~4.1 s measured Laya CPU forward.
    //   RecordedLayaGemmProjectedMs - tests/Nivara.PerformanceTests README's 2414 ms CPU projection,
    //                                 which explicitly EXCLUDES attention, norms and dispatch, so it
    //                                 was never a forward total in the first place.
    //
    // The README also warns the harness is load-sensitive, which is a second reason not to divide
    // a measured delta by either of these.
    const double RecordedLayaForwardMs = 4100.0;
    const double RecordedLayaGemmProjectedMs = 2414.0;

    public static int Run()
    {
        Console.WriteLine("=== #480 ApplyMask cost vs the TensorPrimitives.Add it replaced ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Interleaved design: {Rounds} A/B rounds per cell, first route alternates every round, " +
                          $"{Warmups} untimed warmups per route per type.");
        Console.WriteLine($"Noise band: +/-{NoiseBand:P0}. A ratio > 1 means ApplyMask is SLOWER.");
        Console.WriteLine($"Vector128/256/512 accelerated: {Vector128.IsHardwareAccelerated}/{Vector256.IsHardwareAccelerated}/{Vector512.IsHardwareAccelerated}");
        if (IsDebugBuild())
        {
            Console.WriteLine();
            Console.WriteLine("  *** DEBUG BUILD - the numbers below are NOT evidence. See the build-configuration check at the end.");
        }
        Console.WriteLine();

        // One instantiation per element type: ApplyMask<T> is generic over
        // IFloatingPointIeee754<T>, so it is JIT-specialized per T and each type needs its own
        // warmup. float is the shipping path; Half and BFloat16 are in ADR-001's domain too, and
        // they are the types a ConditionalSelect fast path could not cover.
        var cells = new List<CellResult>();
        cells.AddRange(MeasureForType<float>("float"));
        cells.AddRange(MeasureForType<double>("double"));
        cells.AddRange(MeasureForType<Half>("Half"));
        cells.AddRange(MeasureForType<BFloat16>("BFloat16"));

        // Coverage is stated as N of M, and a shortfall is reported as a failure rather than a
        // neutral event: a run that tested three quarters of what it claims must not print the
        // same summary as one that tested all of it.
        int parityExact = cells.Count(c => c.ParityExact);
        Console.WriteLine("Parity - the routes must agree bit-for-bit where the contract says they must");
        Console.WriteLine("  (all-finite scores, mask in {0,-inf}; outside that domain they differ by design, #448)");
        Console.WriteLine($"  {parityExact} of {cells.Count} cells bit-identical");
        if (parityExact != cells.Count)
            Console.WriteLine("  *** COVERAGE SHORTFALL - cells below disagree where the contract says they must not.");
        Console.WriteLine();

        foreach (string type in new[] { "float", "double", "Half", "BFloat16" })
        {
            var forType = cells.Where(c => c.TypeName == type).ToList();
            Console.WriteLine($"--- {type} ---");
            Console.WriteLine($"  {"shape",-25} {"regime",-7} {"flags",-8} {"ratio",7} {"verdict",-14} {"wins",7} " +
                              $"{"ApplyMask",11} {"add",11} {"A/B GB/s",15}");
            foreach (CellResult c in forType)
            {
                string ratioText = c.Measurable ? $"{c.MedianRatio,7:F3}" : "   n/a ";
                Console.WriteLine($"  {c.ShapeLabel,-25} {c.Regime,-7} {c.Flags,-8} {ratioText} " +
                                  $"{Classify(c),-14} {c.WinText,-7} {c.MaskMs,9:F3}ms {c.AddMs,9:F3}ms " +
                                  $"{c.MaskGBs,6:F2}/{c.AddGBs,6:F2}"
                                  + (c.Measurable ? "" : "   (below timer resolution)"));
            }
            Console.WriteLine();
        }

        // The aggregate covers measurable cells only, and says so with its own denominator. A
        // summary that mixed unmeasurable cells in would report a ratio drawn partly from clock
        // quantization without revealing it.
        var measurable = cells.Where(c => c.Measurable).ToList();
        var unmeasurable = cells.Where(c => !c.Measurable).ToList();
        double overall = Median(measurable.Select(c => c.MedianRatio).ToArray());
        int slower = measurable.Count(c => Classify(c) == MaskSlower);
        int faster = measurable.Count(c => Classify(c) == AddFaster);
        int noise = measurable.Count(c => Classify(c) == WithinNoise);
        Console.WriteLine($"Aggregate over measurable cells: median ratio {overall:F3}.");
        Console.WriteLine($"  ApplyMask SLOWER in {slower}, FASTER in {faster}, within noise in {noise} " +
                          $"(of {measurable.Count}).");
        Console.WriteLine("  Read this only with the per-type tables. The aggregate mixes types that");
        Console.WriteLine("  disagree in DIRECTION - on Half the BCL add is the slower route - so a");
        Console.WriteLine("  median near 1.0 here is a coincidence of averaging, not a null result.");
        Console.WriteLine($"Coverage: {measurable.Count} of {cells.Count} cells measurable. " +
                          $"{unmeasurable.Count} excluded as below the clock's resolution (~{MinReliableTicks} ticks).");
        if (unmeasurable.Count > 0)
        {
            Console.WriteLine("  Excluded shapes: " +
                              string.Join(", ", unmeasurable.Select(c => $"{c.TypeName}/{c.ShapeLabel}/{c.Regime}").Distinct()));
            Console.WriteLine("  These are reported as n/a, not as ratios. Publishing \"ApplyMask is 6x");
            Console.WriteLine("  slower at decode\" off a rounded tick count would be a false claim.");
        }
        Console.WriteLine();

        RunLegProfile();

        Console.WriteLine(BuildConfigurationWarning());
        Console.WriteLine();
        Console.WriteLine("Decode caveat: the [1,512] row measures the mask API at decode shape. The decode");
        Console.WriteLine("paths (AttentionKernels.DecodeAttention and BatchedAttention) use a Keep predicate");
        Console.WriteLine("and never call ApplyMask, so that row is not the decode path's cost.");
        Console.WriteLine();

        // Separate exit paths: a structural failure (routes disagreeing where the contract says
        // they must) is never reported as a numeric verdict, and vice versa.
        return parityExact == cells.Count ? 0 : 1;
    }

    // ───────────────────────────── kernel A/B ─────────────────────────────

    static IEnumerable<CellResult> MeasureForType<T>(string typeName) where T : struct, IFloatingPointIeee754<T>
    {
        var results = new List<CellResult>();
        foreach (var (rows, cols, label) in Shapes)
        {
            foreach (MaskRegime regime in Enum.GetValues<MaskRegime>())
            {
                foreach (bool track in new[] { false, true })
                {
                    results.Add(MeasureCell<T>(typeName, rows, cols, label, regime, track));
                }
            }
        }
        return results;
    }

    enum MaskRegime
    {
        /// <summary>No suppression. The perfectly-predicted branch case.</summary>
        None,
        /// <summary>Suppress the strict upper triangle, so ~50% of cells take each branch - the
        /// worst case for the predictor, and the regime a causal/padding mask actually produces.</summary>
        Causal,
        /// <summary>Every cell suppressed. A perfectly-predicted branch, but the saturation case.</summary>
        All,
    }

    sealed record CellResult(
        string TypeName, string ShapeLabel, string Regime, string Flags,
        double MedianRatio, double MaskMs, double AddMs, double MaskGBs, double AddGBs,
        int MaskWins, int AddWins, int Ties, bool ParityExact, bool Measurable)
    {
        public string WinText => $"{MaskWins}/{MaskWins + AddWins + Ties}";
    }

    static CellResult MeasureCell<T>(string typeName, int rows, int cols, string shapeLabel,
                                     MaskRegime regime, bool track)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = rows * cols;
        T[] scores = BuildScores<T>(n);
        T[] mask = BuildMask<T>(rows, cols, regime);

        // Correctness first. A timing comparison between routes that disagree where the contract
        // says they must is a comparison of nothing, so parity is settled before any clock runs.
        bool parityExact = ParityHolds<T>(scores, mask, rows, cols, track);

        // Preallocated outside the timing loop: the production path passes a span into a rented
        // array, so a per-call stackalloc would measure stack zeroing rather than the kernel.
        var rowFlags = new bool[rows];

        // Warm BOTH routes before any sample, so tiered JIT and first-touch page faults do not
        // land in one route's sample.
        for (int w = 0; w < Warmups; w++)
        {
            RunAdd(scores, mask);
            RunMask<T>(scores, mask, rowFlags, rows, cols, track);
        }

        var maskTicks = new double[Rounds];
        var addTicks = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            // Alternate which route goes first. This is what cancels the first-measured
            // systematic bias that made the superseded #482 gate coin-flip on a 0.2% margin.
            if (r % 2 == 0)
            {
                maskTicks[r] = Time(() => RunMask<T>(scores, mask, rowFlags, rows, cols, track));
                addTicks[r] = Time(() => RunAdd(scores, mask));
            }
            else
            {
                addTicks[r] = Time(() => RunAdd(scores, mask));
                maskTicks[r] = Time(() => RunMask<T>(scores, mask, rowFlags, rows, cols, track));
            }
        }

        // Only valid rounds contribute to the ratio, because Array.Sort's NaN ordering is not a
        // contract to rely on and a poisoned median would read as a result.
        var ratios = new List<double>(Rounds);
        int maskWins = 0, addWins = 0, ties = 0;
        for (int r = 0; r < Rounds; r++)
        {
            // A zero-tick sample means the call fell between clock reads. Dividing by it would
            // yield Infinity, so the round is recorded as a tie and the cell is demoted by the
            // Measurable check below rather than misreported.
            if (addTicks[r] == 0.0) { ties++; continue; }

            ratios.Add(maskTicks[r] / addTicks[r]);
            if (maskTicks[r] < addTicks[r]) maskWins++;
            else if (addTicks[r] < maskTicks[r]) addWins++;
            else ties++;
        }

        double toMs = 1000.0 / Stopwatch.Frequency;
        double medianMaskTicks = Median(maskTicks);
        double medianAddTicks = Median(addTicks);
        double maskMs = medianMaskTicks * toMs;
        double addMs = medianAddTicks * toMs;

        // A cell is only measurable when its FASTER route still spans enough ticks for the ratio
        // to mean something. [1,512] is the case that matters: the entire call is ~10 ticks, so
        // its ratio is quantization. Flagged, printed, and excluded from the aggregate - never
        // published as a "N times slower at decode" figure, which is a false claim.
        bool measurable = Math.Max(medianMaskTicks, medianAddTicks) >= MinReliableTicks
                          && medianAddTicks > 0.0;

        // Both routes read two buffers and write one, so both move 3 * sizeof(T) * n bytes per
        // call. Reporting GB/s is what shows whether either route is bandwidth-saturated - which
        // is the difference between "equal because both hit the memory ceiling" and "equal
        // because both are compute-bound", and a ratio alone cannot tell those apart.
        double bytes = 3.0 * Unsafe.SizeOf<T>() * n;

        return new CellResult(
            typeName, shapeLabel, regime.ToString(), track ? "tracked" : "empty",
            measurable ? Median(ratios.ToArray()) : double.NaN, maskMs, addMs,
            bytes / (maskMs / 1000.0) / 1e9,
            bytes / (addMs / 1000.0) / 1e9,
            maskWins, addWins, ties, parityExact, measurable);
    }

    // ───────────────────────────── routes ─────────────────────────────

    static void RunAdd<T>(Span<T> scores, ReadOnlySpan<T> mask) where T : struct, IFloatingPointIeee754<T>
        // Exact aliasing is explicitly permitted: the TensorPrimitives.Add contract throws only
        // when the inputs "reference overlapping memory locations and do not begin at the same
        // location". This in-place call is what #448 replaced.
        => TensorPrimitives.Add(scores, mask, scores);

    static void RunMask<T>(Span<T> scores, ReadOnlySpan<T> mask, Span<bool> rowFlags, int rows, int cols, bool track)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (track) AttentionKernels<T>.ApplyMask(scores, mask, rowFlags, rows, cols);
        else AttentionKernels<T>.ApplyMask(scores, mask);
    }

    /// <summary>
    /// Bit-for-bit comparison of the two routes over the exact domain where they are required to
    /// agree: all-finite scores and a mask in <c>{0, -inf}</c>.
    /// </summary>
    static bool ParityHolds<T>(T[] scores, T[] mask, int rows, int cols, bool track)
        where T : struct, IFloatingPointIeee754<T>
    {
        T[] viaMask = (T[])scores.Clone();
        T[] viaAdd = (T[])scores.Clone();

        RunMask<T>(viaMask, mask, new bool[rows], rows, cols, track);
        TensorPrimitives.Add(viaAdd, mask, viaAdd);

        return BitIdentical(viaMask, viaAdd);
    }

    /// <summary>
    /// Exactness by raw bytes rather than per-type equality, so no element type gets a weaker
    /// comparison than another and no tolerance can creep in. Every type here is an unmanaged
    /// struct, which is what makes the cast legal.
    /// </summary>
    static bool BitIdentical<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : struct
        => MemoryMarshal.Cast<T, byte>(a).SequenceEqual(MemoryMarshal.Cast<T, byte>(b));

    // ───────────────────────────── data ─────────────────────────────

    /// <summary>
    /// Finite scores only: the parity contract is stated over finite scores, and a NaN or +inf
    /// here would move the cell out of the domain the two routes must agree in. Values sit near
    /// unity so Half and BFloat16 stay well inside range and clear of subnormals - otherwise the
    /// narrow-type rows would be measuring denormal handling rather than the mask.
    /// </summary>
    static T[] BuildScores<T>(int n) where T : struct, IFloatingPointIeee754<T>
    {
        var data = new T[n];
        for (int i = 0; i < n; i++)
        {
            // Deterministic, no RNG state: every run must measure the same branch pattern.
            uint h = unchecked((uint)i) * 2654435761u;
            double v = (h % 2003u) / 2003.0 * 2.0 - 1.0;
            data[i] = T.CreateChecked(v);
        }
        return data;
    }

    static T[] BuildMask<T>(int rows, int cols, MaskRegime regime) where T : struct, IFloatingPointIeee754<T>
    {
        var mask = new T[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                mask[r * cols + c] = regime switch
                {
                    MaskRegime.None => T.Zero,
                    MaskRegime.Causal => c > r ? T.NegativeInfinity : T.Zero,
                    _ => T.NegativeInfinity,
                };
            }
        }
        return mask;
    }

    // ───────────────────────────── end-to-end leg profile ─────────────────────────────

    /// <summary>
    /// Attributes one Laya-shaped attention forward to its individual legs, so the kernel ratio
    /// converts into a materiality verdict. A ratio alone cannot answer "is this within noise end
    /// to end" - only a share of a real call can.
    ///
    /// <para>Legs mirror <c>ReverseGradOperations.MultiHeadAttention</c> exactly: PackHeads x3 once
    /// per layer, then per head MatMulTransposedB (QK^T), the scale multiply, ApplyMask,
    /// SoftmaxRows, MatMul (PV) and ScatterHead.</para>
    /// </summary>
    static void RunLegProfile()
    {
        Console.WriteLine($"--- Laya-shaped attention leg profile (B=1, S={LayaQLen}, d={LayaD}, " +
                          $"H={LayaHeads}, headDim={LayaHeadDim}, {LayaLayers} layers) ---");

        const int qLen = LayaQLen, kvLen = LayaQLen, D = LayaD, heads = LayaHeads, headDim = LayaHeadDim;
        int scoreLen = qLen * kvLen;
        float scale = 1f / MathF.Sqrt(headDim);

        float[] q = BuildScores<float>(qLen * D);
        float[] k = BuildScores<float>(kvLen * D);
        float[] v = BuildScores<float>(kvLen * D);
        float[] mask = BuildMask<float>(qLen, kvLen, MaskRegime.Causal);

        int qHeadsLen = heads * qLen * headDim;
        int kvHeadsLen = heads * kvLen * headDim;
        var qHeads = new float[qHeadsLen];
        var kHeads = new float[kvHeadsLen];
        var vHeads = new float[kvHeadsLen];
        var scores = new float[scoreLen];
        var outHead = new float[qLen * headDim];
        var output = new float[qLen * D];

        // One full forward warms every leg and leaves the buffers in their steady state.
        RunOneHead(q, k, v, qHeads, kHeads, vHeads, qHeadsLen, kvHeadsLen, qLen, kvLen, D, heads, headDim,
                   scale, scores, mask, outHead, output, out bool finiteAfterWarmup);
        if (!finiteAfterWarmup)
        {
            // A NaN here would put every leg on the slow NaN path and make the timings below a
            // measurement of NaN handling rather than of the mask. That is a structural void, so
            // it is reported as one rather than folded into a numeric verdict.
            Console.WriteLine("  *** VOID: the warmup forward produced a non-finite score. Every leg below");
            Console.WriteLine("      would be timing NaN handling. The leg numbers are not printed.");
            Console.WriteLine();
            return;
        }

        var packMs = TimeLeg(() =>
        {
            AttentionKernels<float>.PackHeads(q, qHeads.AsSpan(0, qHeadsLen), qLen, heads, headDim);
            AttentionKernels<float>.PackHeads(k, kHeads.AsSpan(0, kvHeadsLen), kvLen, heads, headDim);
            AttentionKernels<float>.PackHeads(v, vHeads.AsSpan(0, kvHeadsLen), kvLen, heads, headDim);
        });
        var qkMs = TimeLeg(() => GradKernels.MatMulTransposedB(qHeads, kHeads, scores, qLen, headDim, kvLen));
        var scaleMs = TimeLeg(() => TensorPrimitives.Multiply(scores, scale, scores));
        var maskMs = TimeLeg(() => AttentionKernels<float>.ApplyMask(scores, mask));
        var softmaxMs = TimeLeg(() => AttentionKernels<float>.SoftmaxRows(scores, qLen, kvLen));
        var pvMs = TimeLeg(() => GradKernels.MatMul(scores, vHeads, outHead, qLen, kvLen, headDim));
        var scatterMs = TimeLeg(() =>
            AttentionKernels<float>.ScatterHead(outHead.AsSpan(0, qLen * headDim), output.AsSpan(), qLen, D, 0, headDim));

        double perHeadTotal = qkMs + scaleMs + maskMs + softmaxMs + pvMs + scatterMs;
        double layerTotal = packMs + heads * perHeadTotal;

        Console.WriteLine($"  {"leg",-22} {"per call":>11} {"per layer":>12} {"share":>8}");
        // The multiplier is explicit rather than inferred from the label: PackHeads runs once per
        // layer, every other leg once per head. Deriving it from a string comparison would break
        // silently the first time a label was reworded.
        PrintLeg("PackHeads (x3)", packMs, 1, layerTotal);
        PrintLeg("QK^T matmul", qkMs, heads, layerTotal);
        PrintLeg("scale multiply", scaleMs, heads, layerTotal);
        PrintLeg("ApplyMask", maskMs, heads, layerTotal);
        PrintLeg("softmax rows", softmaxMs, heads, layerTotal);
        PrintLeg("PV matmul", pvMs, heads, layerTotal);
        PrintLeg("ScatterHead", scatterMs, heads, layerTotal);
        Console.WriteLine($"  {"attention total":-22} {"":11} {layerTotal,10:F3}ms {1.0,8:P1}");

        // The counterfactual the issue actually asks for: if the mask went back to the vectorized
        // add, how much of the forward would come back? Measured, not inferred from a MAC count.
        double maskBytes = 3.0 * sizeof(float) * scoreLen;
        double maskGBs = maskBytes / (maskMs / 1000.0) / 1e9;
        double addMs = TimeLeg(() => TensorPrimitives.Add(scores, mask, scores));
        double addGBs = maskBytes / (addMs / 1000.0) / 1e9;
        double deltaPerLayer = heads * (maskMs - addMs);
        double deltaPerForward = deltaPerLayer * LayaLayers;

        Console.WriteLine();
        Console.WriteLine($"  ApplyMask   {maskMs:F4} ms/head  {maskGBs:F2} GB/s");
        Console.WriteLine($"  add instead {addMs:F4} ms/head  {addGBs:F2} GB/s  (ratio {maskMs / addMs:F3})");
        Console.WriteLine($"  difference  {deltaPerLayer:F4} ms/layer  ->  {deltaPerForward:F2} ms per {LayaLayers}-layer forward");
        Console.WriteLine();
        Console.WriteLine($"  Materiality, expressed only in quantities measured IN THIS RUN:");
        Console.WriteLine($"    ApplyMask is {heads * maskMs / layerTotal:P2} of attention; the vectorized add would be {heads * addMs / layerTotal:P2}.");
        Console.WriteLine($"    Reverting would return {deltaPerLayer:F3} ms/layer, {deltaPerForward:F2} ms per {LayaLayers}-layer forward.");
        Console.WriteLine();

        // A percentage of end-to-end time is deliberately NOT printed here. No forward total was
        // measured in this run, and the two recorded candidates do not reconcile with this run:
        // the leg profile above already puts attention alone at layerTotal ms/layer, which over
        // LayaLayers layers exceeds docs/LAYA.md's recorded ~4.1 s forward on its own. Dividing
        // a measured delta by a denominator this run contradicts would produce a precise-looking
        // number with no defensible meaning, which is worse than reporting none.
        Console.WriteLine("  *** The forward-time reconciliation does not close, and this probe will not paper over it:");
        Console.WriteLine($"      measured attention alone    : {layerTotal * LayaLayers,10:F0} ms per forward (from the legs above)");
        Console.WriteLine($"      recorded measured forward   : {RecordedLayaForwardMs,10:F0} ms (docs/LAYA.md)");
        Console.WriteLine($"      recorded GEMM-only projection: {RecordedLayaGemmProjectedMs,9:F0} ms (tests README - EXCLUDES attention)");
        Console.WriteLine("      Attention alone is larger than the recorded whole forward, so the recorded figures");
        Console.WriteLine("      cannot be the denominator for this delta. Possible causes: the recorded forward");
        Console.WriteLine("      was measured on a different code state or a faster machine; the leg profile is");
        Console.WriteLine("      inflated by single-threaded skinny matmuls (QK^T at K=64 runs at ~1 GMAC/s,");
        Console.WriteLine("      versus ~20 GMAC/s for the recorded [512x1024x3072] projection); or the");
        Console.WriteLine("      recorded forward ran a smaller effective S. NOT RESOLVED HERE - it is not this");
        Console.WriteLine("      issue's question. Tracked separately so the number is not lost.");
        Console.WriteLine("      Until it is resolved, treat the ABSOLUTE ms above as the finding and any");
        Console.WriteLine("      percentage-of-forward as unmeasured.");
        Console.WriteLine();

        static void PrintLeg(string name, double perCallMs, int multiplier, double layerTotalMs)
            => Console.WriteLine($"  {name,-22} {perCallMs,9:F4}ms {perCallMs * multiplier,10:F3}ms " +
                                 $"{perCallMs * multiplier / layerTotalMs,8:P1}");
    }

    /// <summary>
    /// One head's leg sequence, verbatim in structure from
    /// <c>ReverseGradOperations.MultiHeadAttention</c>. Forward only: empty row flags and no
    /// saved-weights copy, which is what an inference pass does.
    /// </summary>
    static void RunOneHead(float[] q, float[] k, float[] v, float[] qHeads, float[] kHeads, float[] vHeads,
                           int qHeadsLen, int kvHeadsLen,
                           int qLen, int kvLen, int D, int heads, int headDim, float scale,
                           float[] scores, float[] mask, float[] outHead, float[] output, out bool finite)
    {
        // The SAME q/k/v the timed legs use. An earlier revision built fresh inputs here, which
        // meant the warmup and the measured legs ran on different data - structurally identical
        // and equally deterministic, so the finite check still held, but it need not have. The
        // finite check exists to catch NaN reaching a leg, so it should inspect the exact buffers
        // that get timed.
        AttentionKernels<float>.PackHeads(q, qHeads.AsSpan(0, qHeadsLen), qLen, heads, headDim);
        AttentionKernels<float>.PackHeads(k, kHeads.AsSpan(0, kvHeadsLen), kvLen, heads, headDim);
        AttentionKernels<float>.PackHeads(v, vHeads.AsSpan(0, kvHeadsLen), kvLen, heads, headDim);

        for (int h = 0; h < heads; h++)
        {
            var qh = qHeads.AsSpan(h * qLen * headDim, qLen * headDim);
            var kh = kHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);
            var vh = vHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);

            GradKernels.MatMulTransposedB(qh, kh, scores, qLen, headDim, kvLen);
            TensorPrimitives.Multiply(scores, scale, scores);
            AttentionKernels<float>.ApplyMask(scores, mask);
            AttentionKernels<float>.SoftmaxRows(scores, qLen, kvLen);
            GradKernels.MatMul(scores, vh, outHead, qLen, kvLen, headDim);
            AttentionKernels<float>.ScatterHead(
                outHead.AsSpan(0, qLen * headDim), output.AsSpan(), qLen, D, h, headDim);
        }

        finite = true;
        foreach (float s in scores)
        {
            if (!float.IsFinite(s)) { finite = false; break; }
        }
    }

    /// <summary>Median of <see cref="Rounds"/> timed passes, after the same warmup both routes get.</summary>
    static double TimeLeg(Action action)
    {
        for (int w = 0; w < Warmups; w++) action();
        var samples = new double[Rounds];
        for (int r = 0; r < Rounds; r++) samples[r] = Time(action);
        return Median(samples) * (1000.0 / Stopwatch.Frequency);
    }

    // ───────────────────────────── helpers ─────────────────────────────

    static long Time(Action action)
    {
        long start = Stopwatch.GetTimestamp();
        action();
        return Stopwatch.GetTimestamp() - start;
    }

    static double Median(double[] values)
    {
        double[] sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    const string MaskSlower = "MASK SLOWER";
    const string AddFaster = "add faster";
    const string WithinNoise = "within noise";

    /// <summary>
    /// The verdict, and the reason the noise band exists at all: a raw ratio invites reading 1.03
    /// as a 3% regression when it is indistinguishable from drift. Classification is deliberately
    /// symmetric so a win is not reported as a neutral event.
    /// </summary>
    static string Classify(CellResult c)
    {
        if (!c.Measurable) return "not measured";
        if (c.MedianRatio > 1.0 + NoiseBand) return MaskSlower;
        if (c.MedianRatio < 1.0 - NoiseBand) return AddFaster;
        return WithinNoise;
    }

    static bool IsDebugBuild()
    {
#if DEBUG
        return true;
#else
        return false;
#endif
    }

    static string BuildConfigurationWarning()
    {
        bool isDebug = IsDebugBuild();
        return "  Build configuration check:\n"
             + $"          Optimized: {!isDebug}"
             + (isDebug
                 ? "   (DEBUG BUILD - these numbers are NOT evidence. The handwritten loop runs\n"
                   + "            unoptimized while System.Numerics.Tensors ships ReadyToRun and stays\n"
                   + "            optimized, which collapses both routes toward parity. Re-run with:\n"
                   + "            dotnet run -c Release --project tests/Nivara.PerformanceTests -- --mask)"
                 : "   (Release - numbers are meaningful.)");
    }
}