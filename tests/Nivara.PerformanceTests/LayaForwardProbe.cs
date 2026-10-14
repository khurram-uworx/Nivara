using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.Diagnostics;
using Nivara.Tensors;
using System.Diagnostics;
using System.Numerics.Tensors;

namespace Nivara.PerformanceTests;

/// <summary>
/// <c>--laya-forward</c> (#492): a Laya CPU forward and an attention leg profile in one process,
/// plus the <c>MatMulTransposedB</c> / <c>MatMul</c> comparison at the attention shapes.
///
/// <para>A probe, not a gate. It exits 0 when every planned cell ran, 1 when a structural guard
/// failed (a matmul span that was not one head, a non-finite hidden state or score), and 2 on a
/// Debug build, whose numbers are not evidence. It does not divide by the recorded 4.1 s or the
/// recorded 2414 ms GEMM projection. Those were not measured in this process.</para>
/// </summary>
internal static class LayaForwardProbe
{
    const int Seq = 512;
    const int Heads = 16;
    const int HeadDim = 64;
    const int Hidden = 1024;
    const int LegRounds = 30;
    const int LegWarmups = 5;
    const int ForwardWarmup = 1;
    const int ForwardIters = 3;
    const int LayoutRounds = 30;
    const int LayoutWarmups = 5;
    const double NoiseBand = 0.03;
    const int PairValid = 40;

    static readonly int[] ValidLengths = [40, 119, 48, 39, 512];
    static readonly int[] PairMarkers = [4, 12];

    public static int Run()
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("=== #492 Laya CPU forward, same-run attention legs, attention-shape layout ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Processors: {Environment.ProcessorCount}");
        Console.WriteLine("This probe does not divide by the recorded 4.1 s or the recorded 2414 ms GEMM projection.");
        Console.WriteLine("A battery figure is not comparable to that AC forward. The residual below is this process only.");
        Console.WriteLine();

#if DEBUG
        Console.WriteLine("DEBUG BUILD - timings would not be evidence. Not measured.");
        return 2;
#else
        var (encoder, head, config, weightSource) = Load();
        Console.WriteLine($"Weights: {weightSource}");
        Console.WriteLine();

        int voids = 0;
        var forwards = TimeForwards(encoder, head, config, weightSource, ref voids);
        var legs = TimeLegs(config, ref voids);
        PrintRollup(forwards, legs, config, weightSource);
        int layoutVoids = TimeLayout();
        voids += layoutVoids;
        if (encoder is not null)
            TimeDiagnostics(encoder, config);

        int plannedForwards = encoder is null ? 0 : ValidLengths.Length + 2;
        int measuredForwards = forwards.Count(f => f.Measured);
        int plannedLegs = 6;
        int measuredLegs = legs.Count(l => l.Measured);
        int measuredLayout = layoutVoids == 0 ? 4 : 0;
        Console.WriteLine($"Coverage: forwards {measuredForwards} of {plannedForwards}, " +
                          $"leg cells {measuredLegs} of {plannedLegs}, layout cells {measuredLayout} of 4.");
        Console.WriteLine("A shortfall is a failed run, not a smaller experiment.");
        if (voids > 0 || measuredForwards != plannedForwards || measuredLegs != plannedLegs)
        {
            Console.WriteLine($"*** {voids} structural void(s). The numeric lines above that say VOID are not readings.");
            return 1;
        }
        return 0;
#endif
    }

    sealed record ForwardCell(string Name, bool Measured, double MedianMs, double MinMs, double MaxMs);

    sealed record LegCell(string Name, bool Measured, double LayerMs, int FullyMaskedRows);

    static (ModernBertEncoder<float>? Encoder, LayaDecisionHead<float>? Head, ModernBertConfig Config, string Source) Load()
    {
        if (!TryResolveCheckpoint(out string modelPath, out string configPath))
        {
            Console.WriteLine("Checkpoint absent. Kernel legs and the layout matrix still run.");
            Console.WriteLine("The forward is SYNTHETIC (Kaiming init) and is not a denominator against the recorded 4.1 s.");
            var synthetic = new ModernBertConfig();
            var encoder = new ModernBertEncoder<float>(synthetic);
            var head = new LayaDecisionHead<float>(synthetic.HiddenSize, headLayerCount: 2);
            encoder.Eval();
            head.Eval();
            return (encoder, head, synthetic, "SYNTHETIC");
        }

        Console.WriteLine($"Checkpoint: {modelPath}");
        var sw = Stopwatch.StartNew();
        var tensors = SafeTensorsLoader.Read(modelPath);
        var config = ModernBertConfig.FromJson(File.ReadAllText(configPath));
        var loaded = ModernBertEncoder<float>.LoadWeights(tensors, config, prefix: "encoder");
        var loadedHead = LayaDecisionHead<float>.LoadWeights<float, float>(tensors, headLayerCount: 2);
        sw.Stop();
        Console.WriteLine($"Load: {sw.ElapsedMilliseconds} ms (not part of any forward below)");
        return (loaded, loadedHead, config, "REAL");
    }

    static bool TryResolveCheckpoint(out string modelPath, out string configPath)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
        {
            string model = Path.Combine(dir.FullName, "samples", "data", "laya", "model.safetensors");
            string config = Path.Combine(dir.FullName, "samples", "data", "laya", "encoder", "config.json");
            if (File.Exists(model) && File.Exists(config))
            {
                modelPath = model;
                configPath = config;
                return true;
            }
        }
        modelPath = "";
        configPath = "";
        return false;
    }

    static List<ForwardCell> TimeForwards(
        ModernBertEncoder<float>? encoder, LayaDecisionHead<float>? head, ModernBertConfig config,
        string weightSource, ref int voids)
    {
        var cells = new List<ForwardCell>();
        Console.WriteLine($"--- forward ({weightSource}; 1 warmup + {ForwardIters} timed, median) ---");
        Console.WriteLine("Ids are random in range, pad past validLength. Same shape as the benchmark, not its prompt.");
        if (encoder is null || head is null)
        {
            Console.WriteLine("  forward UNMEASURED");
            Console.WriteLine();
            return cells;
        }

        foreach (int valid in ValidLengths)
        {
            var ids = BuildIds(valid, config.VocabSize, config.PadTokenId);
            var (median, min, max, finite) = TimeEncoder(encoder, ids, valid);
            if (!finite)
            {
                voids++;
                Console.WriteLine($"  encoder valid={valid,-3}  VOID (non-finite hidden)");
                cells.Add(new ForwardCell($"encoder/{valid}", false, 0, 0, 0));
                continue;
            }
            cells.Add(new ForwardCell($"encoder/{valid}", true, median, min, max));
            Console.WriteLine($"  encoder valid={valid,-3}  median {median,8:F0} ms   min {min,8:F0}   max {max,8:F0}");
        }

        var pairIds = BuildIds(PairValid, config.VocabSize, config.PadTokenId);
        var hidden = encoder.Forward(pairIds, PairValid);
        var (headMedian, headMin, headMax, headFinite) = TimeHead(head, hidden, PairValid);
        if (!headFinite)
        {
            voids++;
            Console.WriteLine($"  head    valid={PairValid,-3}  VOID (non-finite)");
            cells.Add(new ForwardCell("head", false, 0, 0, 0));
        }
        else
        {
            cells.Add(new ForwardCell("head", true, headMedian, headMin, headMax));
            Console.WriteLine($"  head    valid={PairValid,-3}  median {headMedian,8:F0} ms   min {headMin,8:F0}   max {headMax,8:F0}  " +
                              $"(2 markers; not inside the encoder number)");
        }

        var (pairMedian, pairMin, pairMax, pairFinite) = TimePair(encoder, head, pairIds, PairValid);
        if (!pairFinite)
        {
            voids++;
            Console.WriteLine($"  pair    valid={PairValid,-3}  VOID (non-finite)");
            cells.Add(new ForwardCell("pair", false, 0, 0, 0));
        }
        else
        {
            cells.Add(new ForwardCell("pair", true, pairMedian, pairMin, pairMax));
            Console.WriteLine($"  pair    valid={PairValid,-3}  median {pairMedian,8:F0} ms   min {pairMin,8:F0}   max {pairMax,8:F0}  " +
                              "(encoder+head, one stopwatch; the benchmark's unit of work, one question)");
        }
        Console.WriteLine();
        return cells;
    }

    static (double Median, double Min, double Max, bool Finite) TimeEncoder(
        ModernBertEncoder<float> encoder, int[] ids, int valid)
    {
        for (int i = 0; i < ForwardWarmup; i++)
            encoder.Forward(ids, valid);
        var samples = new double[ForwardIters];
        bool finite = true;
        for (int i = 0; i < ForwardIters; i++)
        {
            long start = Stopwatch.GetTimestamp();
            var hidden = encoder.Forward(ids, valid);
            samples[i] = Stopwatch.GetTimestamp() - start;
            finite &= AllFinite(hidden.AsSpan());
        }
        return Summarize(samples, finite);
    }

    static (double Median, double Min, double Max, bool Finite) TimeHead(
        LayaDecisionHead<float> head, Nivara.AutoDiff.ReverseGradTensor<float> hidden, int valid)
    {
        for (int i = 0; i < ForwardWarmup; i++)
            head.Forward(hidden, LayaQuestionType.Choice, PairMarkers, valid);
        var samples = new double[ForwardIters];
        bool finite = true;
        for (int i = 0; i < ForwardIters; i++)
        {
            long start = Stopwatch.GetTimestamp();
            var output = head.Forward(hidden, LayaQuestionType.Choice, PairMarkers, valid);
            samples[i] = Stopwatch.GetTimestamp() - start;
            finite &= double.IsFinite(output.ActionProbability);
        }
        return Summarize(samples, finite);
    }

    static (double Median, double Min, double Max, bool Finite) TimePair(
        ModernBertEncoder<float> encoder, LayaDecisionHead<float> head, int[] ids, int valid)
    {
        for (int i = 0; i < ForwardWarmup; i++)
        {
            var warm = encoder.Forward(ids, valid);
            head.Forward(warm, LayaQuestionType.Choice, PairMarkers, valid);
        }
        var samples = new double[ForwardIters];
        bool finite = true;
        for (int i = 0; i < ForwardIters; i++)
        {
            long start = Stopwatch.GetTimestamp();
            var hidden = encoder.Forward(ids, valid);
            var output = head.Forward(hidden, LayaQuestionType.Choice, PairMarkers, valid);
            samples[i] = Stopwatch.GetTimestamp() - start;
            finite &= AllFinite(hidden.AsSpan()) && double.IsFinite(output.ActionProbability);
        }
        return Summarize(samples, finite);
    }

    static List<LegCell> TimeLegs(ModernBertConfig config, ref int voids)
    {
        int band = config.SlidingWindow;
        Console.WriteLine($"--- attention legs (per-head spans; {LegWarmups} warmup + median of {LegRounds}) ---");
        Console.WriteLine("SoftmaxSingle skips Exp only when the row max is -inf, i.e. every key is suppressed.");
        Console.WriteLine("A mostly-masked row still pays Exp. The fully-masked count is that test, not a story.");
        Console.WriteLine();

        var cells = new List<LegCell>
        {
            TimeLeg("causal/512", BuildCausal(Seq), ref voids),
            TimeLeg("full/512", MaskData(Seq, -1, Seq), ref voids),
            TimeLeg("full/40", MaskData(Seq, -1, 40), ref voids),
            TimeLeg("full/119", MaskData(Seq, -1, 119), ref voids),
            TimeLeg($"sliding/40 band={band}", MaskData(Seq, band, 40), ref voids),
            TimeLeg($"sliding/512 band={band}", MaskData(Seq, band, Seq), ref voids),
        };
        Console.WriteLine();
        return cells;
    }

    static LegCell TimeLeg(string name, float[] mask, ref int voids)
    {
        const int qLen = Seq, kvLen = Seq, headDim = HeadDim, heads = Heads;
        int headQ = qLen * headDim;
        int headKV = kvLen * headDim;
        int scoreLen = qLen * kvLen;
        if (mask.Length != scoreLen)
        {
            voids++;
            Console.WriteLine($"  {name,-22}  VOID (mask length {mask.Length} != {scoreLen})");
            return new LegCell(name, false, 0, 0);
        }

        float[] q = Fill(qLen * Hidden);
        float[] k = Fill(kvLen * Hidden);
        float[] v = Fill(kvLen * Hidden);
        var qHeads = new float[heads * headQ];
        var kHeads = new float[heads * headKV];
        var vHeads = new float[heads * headKV];
        var scores = new float[scoreLen];
        var outHead = new float[headQ];
        var output = new float[qLen * Hidden];
        float scale = 1f / MathF.Sqrt(headDim);
        int fullyMasked = CountFullyMasked(mask, qLen, kvLen);

        bool once = RunChain(q, k, v, qHeads, kHeads, vHeads, scores, mask, outHead, output,
            qLen, kvLen, heads, headDim, scale, sample: null);
        if (!once)
        {
            voids++;
            Console.WriteLine($"  {name,-22}  VOID (matmul span was not one head)");
            return new LegCell(name, false, 0, fullyMasked);
        }

        for (int w = 0; w < LegWarmups; w++)
            RunChain(q, k, v, qHeads, kHeads, vHeads, scores, mask, outHead, output,
                qLen, kvLen, heads, headDim, scale, sample: null);

        var samples = new double[LegRounds];
        for (int r = 0; r < LegRounds; r++)
        {
            if (!RunChain(q, k, v, qHeads, kHeads, vHeads, scores, mask, outHead, output,
                    qLen, kvLen, heads, headDim, scale, samples.AsSpan(r, 1)))
            {
                voids++;
                Console.WriteLine($"  {name,-22}  VOID (matmul span was not one head)");
                return new LegCell(name, false, 0, fullyMasked);
            }
        }

        if (!AllFinite(scores))
        {
            voids++;
            Console.WriteLine($"  {name,-22}  VOID (non-finite score)");
            return new LegCell(name, false, 0, fullyMasked);
        }

        double layerMs = Median(samples) * 1000.0 / Stopwatch.Frequency;
        Console.WriteLine($"  {name,-22}  {layerMs,8:F2} ms/layer   fully-masked rows {fullyMasked} of {qLen}");
        return new LegCell(name, true, layerMs, fullyMasked);
    }

    /// <summary>
    /// One layer, heads summed. Returns false when a span passed to the matmul is not one head:
    /// <c>MultiplyCore</c> rents and clears by <c>span.Length</c>, so a packed multi-head buffer
    /// is not a head's cost. Writes the layer's ticks into <paramref name="sample"/> when non-empty.
    /// </summary>
    static bool RunChain(
        float[] q, float[] k, float[] v, float[] qHeads, float[] kHeads, float[] vHeads,
        float[] scores, float[] mask, float[] outHead, float[] output,
        int qLen, int kvLen, int heads, int headDim, float scale, Span<double> sample)
    {
        int headQ = qLen * headDim;
        int headKV = kvLen * headDim;
        if (qHeads.Length != heads * headQ || kHeads.Length != heads * headKV || vHeads.Length != heads * headKV)
            return false;

        long start = Stopwatch.GetTimestamp();
        AttentionKernels<float>.PackHeads(q, qHeads, qLen, heads, headDim);
        AttentionKernels<float>.PackHeads(k, kHeads, kvLen, heads, headDim);
        AttentionKernels<float>.PackHeads(v, vHeads, kvLen, heads, headDim);

        for (int h = 0; h < heads; h++)
        {
            var qh = qHeads.AsSpan(h * headQ, headQ);
            var kh = kHeads.AsSpan(h * headKV, headKV);
            var vh = vHeads.AsSpan(h * headKV, headKV);
            if (qh.Length != headQ || kh.Length != headKV || vh.Length != headKV)
                return false;

            GradKernels.MatMulTransposedB(qh, kh, scores, qLen, headDim, kvLen);
            TensorPrimitives.Multiply(scores, scale, scores);
            AttentionKernels<float>.ApplyMask(scores, mask);
            AttentionKernels<float>.SoftmaxRows(scores, qLen, kvLen);
            GradKernels.MatMul(scores, vh, outHead, qLen, kvLen, headDim);
            AttentionKernels<float>.ScatterHead(outHead, output, qLen, Hidden, h, headDim);
        }

        if (!sample.IsEmpty)
            sample[0] = Stopwatch.GetTimestamp() - start;
        return true;
    }

    static void PrintRollup(List<ForwardCell> forwards, List<LegCell> legs, ModernBertConfig config, string weightSource)
    {
        int full = 0, sliding = 0;
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            if (config.IsFullAttention(i)) full++;
            else sliding++;
        }
        Console.WriteLine($"--- roll-up ({full} full + {sliding} sliding, from the config, not assumed) ---");
        Console.WriteLine("The roll-up is attention kernels only. The residual is not called GEMM: this run did not time GEMM.");

        PrintOneRollup("valid=40", forwards, legs, "encoder/40", "full/40", SlidingName(legs, "sliding/40"), full, sliding, weightSource);
        PrintOneRollup("valid=512", forwards, legs, "encoder/512", "full/512", SlidingName(legs, "sliding/512"), full, sliding, weightSource);
        Console.WriteLine();
    }

    static string SlidingName(List<LegCell> legs, string prefix)
        => legs.FirstOrDefault(l => l.Name.StartsWith(prefix, StringComparison.Ordinal))?.Name ?? prefix;

    static void PrintOneRollup(
        string label, List<ForwardCell> forwards, List<LegCell> legs,
        string forwardName, string fullName, string slidingName, int fullLayers, int slidingLayers, string weightSource)
    {
        var fullCell = legs.FirstOrDefault(l => l.Name == fullName);
        var slidingCell = legs.FirstOrDefault(l => l.Name == slidingName);
        var forward = forwards.FirstOrDefault(f => f.Name == forwardName);
        if (fullCell is not { Measured: true } || slidingCell is not { Measured: true })
        {
            Console.WriteLine($"  {label,-12}  attention roll-up VOID (a leg cell did not measure)");
            return;
        }

        double attention = fullLayers * fullCell.LayerMs + slidingLayers * slidingCell.LayerMs;
        Console.WriteLine($"  {label,-12}  attention roll-up {attention,8:F0} ms  " +
                          $"({fullLayers}x {fullCell.LayerMs:F2} + {slidingLayers}x {slidingCell.LayerMs:F2})");
        if (forward is not { Measured: true })
        {
            Console.WriteLine($"  {label,-12}  forward UNMEASURED ({weightSource})");
            return;
        }
        double residual = forward.MedianMs - attention;
        Console.WriteLine($"  {label,-12}  forward {forward.MedianMs,8:F0} ms ({weightSource})   residual {residual,8:F0} ms  " +
                          "(forward minus attention; not GEMM, not a percentage of a recorded figure)");
    }

    static int TimeLayout()
    {
        Console.WriteLine($"--- layout matrix (interleaved, {LayoutWarmups} warmup + median of {LayoutRounds}; spans are exactly the shape) ---");
        Console.WriteLine("Same logical B. transposed=true receives it already [N x K]; transposed=false pays the transpose per call.");
        Console.WriteLine();

        const int qM = 512, qK = 64, qN = 512;
        const int pM = 512, pK = 512, pN = 64;
        float[] aQ = Fill(qM * qK), bQ = Fill(qK * qN), btQ = new float[qN * qK], rQ = new float[qM * qN];
        float[] aP = Fill(pM * pK), bP = Fill(pK * pN), btP = new float[pN * pK], rP = new float[pM * pN];
        TensorsHelper.Transpose(bQ, btQ, qK, qN);
        TensorsHelper.Transpose(bP, btP, pK, pN);

        var cells = new (string Name, int M, int K, int N, bool Transposed, float[] A, float[] B, float[] Result)[]
        {
            ("QK^T transposed-B", qM, qK, qN, true, aQ, btQ, rQ),
            ("QK^T per-call transpose", qM, qK, qN, false, aQ, bQ, rQ),
            ("PV transposed-B", pM, pK, pN, true, aP, btP, rP),
            ("PV per-call transpose", pM, pK, pN, false, aP, bP, rP),
        };

        foreach (var cell in cells)
        {
            if (cell.A.Length != cell.M * cell.K || cell.B.Length != cell.K * cell.N || cell.Result.Length != cell.M * cell.N)
            {
                Console.WriteLine("  *** VOID: a layout span is not the declared shape. No ratio is printed.");
                return 1;
            }
        }

        for (int w = 0; w < LayoutWarmups; w++)
            foreach (var cell in cells)
                Launch(cell.A, cell.B, cell.Result, cell.M, cell.K, cell.N, cell.Transposed);

        var samples = new double[cells.Length][];
        for (int i = 0; i < cells.Length; i++) samples[i] = new double[LayoutRounds];
        for (int r = 0; r < LayoutRounds; r++)
        {
            for (int step = 0; step < cells.Length; step++)
            {
                int i = (r + step) % cells.Length;
                var cell = cells[i];
                long start = Stopwatch.GetTimestamp();
                Launch(cell.A, cell.B, cell.Result, cell.M, cell.K, cell.N, cell.Transposed);
                samples[i][r] = Stopwatch.GetTimestamp() - start;
            }
        }

        var ms = new double[cells.Length];
        double ticksToMs = 1000.0 / Stopwatch.Frequency;
        for (int i = 0; i < cells.Length; i++)
        {
            ms[i] = Median(samples[i]) * ticksToMs;
            double macs = (double)cells[i].M * cells[i].K * cells[i].N;
            double gmacs = macs / (ms[i] / 1000.0) / 1e9;
            Console.WriteLine($"  {cells[i].Name,-26}  {ms[i],8:F3} ms   {gmacs,6:F1} GMAC/s");
        }

        double shape = ms[0] / ms[2];
        double qkTranspose = ms[1] / ms[0];
        double pvTranspose = ms[3] / ms[2];
        Console.WriteLine();
        Console.WriteLine($"  shape, transposed-B held constant : QK^T/PV = {shape:F3}  ({Moved(shape)})");
        Console.WriteLine($"  per-call transpose at QK^T        : plain/transposed = {qkTranspose:F3}  ({Moved(qkTranspose)})");
        Console.WriteLine($"  per-call transpose at PV          : plain/transposed = {pvTranspose:F3}  ({Moved(pvTranspose)})");
        Console.WriteLine($"  Noise band +/-{NoiseBand:P0}, the same band --mask uses. A ratio inside it did not move this run.");
        Console.WriteLine("  One run. Do not cite a ratio from here as the settlement until three runs agree.");
        Console.WriteLine();
        return 0;
    }

    static void Launch(float[] a, float[] b, float[] result, int m, int k, int n, bool transposed)
    {
        if (transposed)
            GradKernels.MatMulTransposedB(a, b, result, m, k, n);
        else
            GradKernels.MatMul(a, b, result, m, k, n);
    }

    static string Moved(double ratio) => Math.Abs(ratio - 1.0) <= NoiseBand ? "within noise" : "moved";

    static void TimeDiagnostics(ModernBertEncoder<float> encoder, ModernBertConfig config)
    {
        Console.WriteLine("--- diagnostics pass (shares of the instrumented total only) ---");
        Console.WriteLine("The tracker takes a lock per op. These durations are not the forward above and are not a denominator.");
        Console.WriteLine("Ops that do not call AutoDiffDiagnostics.Measure (norms, activations) are absent. Absence is not a zero.");
        var ids = BuildIds(PairValid, config.VocabSize, config.PadTokenId);
        DiagnosticsTracker.ClearRecordedOperations();
        DiagnosticsTracker.IsEnabled = true;
        AutoDiffDiagnostics.Enabled = true;
        try
        {
            encoder.Forward(ids, PairValid);
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            AutoDiffDiagnostics.Enabled = false;
        }

        var ops = DiagnosticsTracker.GetRecordedOperations();
        DiagnosticsTracker.ClearRecordedOperations();
        if (ops.Length == 0)
        {
            Console.WriteLine("  no operations recorded");
            Console.WriteLine();
            return;
        }

        double totalMs = ops.Sum(op => op.Elapsed.TotalMilliseconds);
        foreach (var group in ops.GroupBy(op => op.OperationType).OrderByDescending(g => g.Sum(op => op.Elapsed.TotalMilliseconds)))
        {
            double ms = group.Sum(op => op.Elapsed.TotalMilliseconds);
            Console.WriteLine($"  {group.Key,-32}  {group.Count(),5} calls   {ms,10:F1} ms   {ms / totalMs,7:P1} of instrumented");
        }
        Console.WriteLine($"  {"instrumented total",-32}  {ops.Length,5} calls   {totalMs,10:F1} ms");
        Console.WriteLine();
    }

    static int[] BuildIds(int valid, int vocab, int pad)
    {
        var ids = new int[Seq];
        var rng = new Random(492 + valid);
        for (int i = 0; i < valid; i++)
            ids[i] = rng.Next(vocab);
        for (int i = valid; i < Seq; i++)
            ids[i] = pad;
        return ids;
    }

    static float[] MaskData(int seq, int band, int valid)
    {
        var tensor = ModernBertMasks.Build<float>(seq, band, valid);
        var data = new float[tensor.AsSpan().Length];
        tensor.AsSpan().CopyTo(data);
        return data;
    }

    static float[] BuildCausal(int n)
    {
        var mask = new float[n * n];
        for (int r = 0; r < n; r++)
            for (int c = r + 1; c < n; c++)
                mask[r * n + c] = float.NegativeInfinity;
        return mask;
    }

    static int CountFullyMasked(float[] mask, int rows, int cols)
    {
        int count = 0;
        for (int r = 0; r < rows; r++)
        {
            int row = r * cols;
            bool full = true;
            for (int c = 0; c < cols; c++)
            {
                if (mask[row + c] != float.NegativeInfinity)
                {
                    full = false;
                    break;
                }
            }
            if (full) count++;
        }
        return count;
    }

    static float[] Fill(int n)
    {
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            uint h = unchecked((uint)i) * 2654435761u;
            data[i] = (h % 2003u) / 2003f * 2f - 1f;
        }
        return data;
    }

    static (double Median, double Min, double Max, bool Finite) Summarize(double[] ticks, bool finite)
    {
        double toMs = 1000.0 / Stopwatch.Frequency;
        double min = ticks.Min() * toMs;
        double max = ticks.Max() * toMs;
        return (Median(ticks) * toMs, min, max, finite);
    }

    static double Median(double[] values)
    {
        double[] sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    static bool AllFinite(ReadOnlySpan<float> data)
    {
        foreach (float x in data)
            if (!float.IsFinite(x)) return false;
        return true;
    }
}
