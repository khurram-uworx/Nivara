using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Diagnostics;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Storage;
using Nivara.Tensors;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Nivara.PerformanceTests;

static class Program
{
    static readonly List<ScenarioDefinition> s_scenarios = [];

    const double DefaultMinOpsFraction = 0.90;
    const double MaxAllocationFraction = 1.01;
    const double Gen0Tolerance = 0.05;

    static readonly JsonSerializerOptions s_jsonOptions = new(JsonSerializerDefaults.Web)
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = true,
    };

    static int Main(string[] args)
    {
        var (jsonPath, comparePath, runs, minOpsFraction) = ParseArgs(args);

        if (runs > 1)
        {
            var results = MeasureAcrossProcesses(runs);
            if (results is null)
                return 2;

            PrintTable(results);

            if (jsonPath is not null)
                WriteJson(jsonPath, results, runs);

            if (comparePath is not null)
                return Compare(comparePath, results, minOpsFraction);

            return 0;
        }

        PrintHeader();
        RegisterScenarios();

        var singleResults = new List<ScenarioResult>();
        foreach (var scenario in s_scenarios)
        {
            var result = MeasureScenario(scenario, 1);
            singleResults.Add(result);
            PrintRow(result);
        }

        if (jsonPath is not null)
            WriteJson(jsonPath, singleResults, 1);

        if (comparePath is not null)
            return Compare(comparePath, singleResults, minOpsFraction);

        return 0;
    }

    static void PrintHeader()
    {
        Console.WriteLine("Nivara storage plan benchmark");
        Console.WriteLine($"  Runtime : {Environment.Version}");
        Console.WriteLine($"  Machine : {Environment.ProcessorCount} logical processors, {(Environment.Is64BitProcess ? "x64" : "x86")}");
        Console.WriteLine();
        Console.WriteLine($"{"Scenario",-46} {"ops/s",12} {"ns/op",8} {"B/op",12} {"gen0/op",7}");
        Console.WriteLine(new string('-', 92));
    }

    static void PrintTable(List<ScenarioResult> results)
    {
        PrintHeader();
        foreach (var r in results)
            PrintRow(r);
    }

    static List<ScenarioResult>? MeasureAcrossProcesses(int runs)
    {
        var exe = Environment.ProcessPath;
        if (exe is null)
        {
            Console.Error.WriteLine("Cannot resolve harness executable path.");
            return null;
        }

        var tmpFiles = new string[runs];
        try
        {
            for (int i = 0; i < runs; i++)
            {
                tmpFiles[i] = Path.Combine(Path.GetTempPath(), $"nivara-perf-{Guid.NewGuid():N}.json");
                var psi = new ProcessStartInfo
                {
                    FileName = exe,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                psi.ArgumentList.Add("--runs");
                psi.ArgumentList.Add("1");
                psi.ArgumentList.Add("--json");
                psi.ArgumentList.Add(tmpFiles[i]);
                using var child = Process.Start(psi);
                if (child is null)
                {
                    Console.Error.WriteLine($"Failed to start measurement process {i + 1}.");
                    return null;
                }
                child.WaitForExit();
                if (child.ExitCode != 0)
                {
                    Console.Error.WriteLine($"Measurement process {i + 1} exited with {child.ExitCode}.");
                    return null;
                }
            }

            var runsByName = new Dictionary<string, List<ScenarioResult>>();
            foreach (var file in tmpFiles)
            {
                var report = JsonSerializer.Deserialize<HarnessReport>(File.ReadAllText(file), s_jsonOptions);
                if (report is null)
                    return null;
                foreach (var r in report.Results)
                {
                    if (!runsByName.TryGetValue(r.Name, out var list))
                        runsByName[r.Name] = list = new List<ScenarioResult>();
                    list.Add(r);
                }
            }

            var medians = new List<ScenarioResult>();
            foreach (var (name, list) in runsByName)
            {
                var ops = list.Select(r => r.OpsPerSec).OrderBy(v => v).ToArray();
                var ns = list.Select(r => r.NsPerOp).OrderBy(v => v).ToArray();
                var bytes = list.Select(r => r.BytesPerOp).OrderBy(v => v).ToArray();
                var gen0 = list.Select(r => r.Gen0PerOp).OrderBy(v => v).ToArray();
                int mid = runs / 2;
                medians.Add(new ScenarioResult(name, ops[mid], ns[mid], bytes[mid], gen0[mid]));
            }
            return medians;
        }
        finally
        {
            foreach (var file in tmpFiles)
            {
                if (file is not null && File.Exists(file))
                    File.Delete(file);
            }
        }
    }

    static void RegisterScenarios()
    {
        Run("ColumnAdd 1M x float", 5, 200,
            () =>
            {
                var a = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var b = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                return () => a.Add(b);
            });

        Run("ColumnSigmoid 1M x float", 5, 200,
            () =>
            {
                var a = Fill(new float[1_000_000]);
                var dest = new float[1_000_000];
                return () => TensorPrimitives.Sigmoid(a, dest);
            });

        Run("Span chain 1M x 3 ops (raw)", 5, 100,
            () =>
            {
                var a = Fill(new float[1_000_000]);
                var b = Fill(new float[1_000_000]);
                var c = Fill(new float[1_000_000]);
                var d = Fill(new float[1_000_000]);
                var t1 = new float[1_000_000];
                var t2 = new float[1_000_000];
                var result = new float[1_000_000];
                return () =>
                {
                    TensorPrimitives.Add(a, b, t1);
                    TensorPrimitives.Multiply(t1, c, t2);
                    TensorPrimitives.Subtract(t2, d, result);
                };
            });

        Run("Column chain 1M x 3 ops (wrapper)", 5, 100,
            () =>
            {
                var a = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var b = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var c = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var d = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                return () =>
                {
                    var t1 = a.Add(b);
                    var t2 = t1.Multiply(c);
                    _ = t2.Subtract(d);
                };
            });

        Run("Fused chain 1M x (Salary*1.1)+1000-Tax", 5, 50,
            () => CreateFusedChainScenario(1_000_000));

        Run("Linear forward [32x256] -> [32x256]", 5, 100,
            () =>
            {
                var linear = new Linear<float>(256, 256);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 256]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(32, 256);
                    linear.Forward(input);
                };
            });

        Run("Linear forward+backward [32x256]", 5, 20,
            () =>
            {
                var linear = new Linear<float>(256, 256);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 256]));
                var ones = Fill(new float[32 * 256]);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: true);
                        input.Reshape(32, 256);
                        var output = linear.Forward(input);
                        var gradient = new ReverseGradTensor<float>(NivaraColumn<float>.Create(ones), requiresGrad: false);
                        gradient.Reshape(32, 256);
                        output.Backward(gradient);
                    }
                };
            });

        Run("TransformerBlock forward [32x64, 4 heads]", 5, 30,
            () =>
            {
                var block = new TransformerBlock<float>(64, 4, dropout: 0.0, maxSeqLen: 32, normType: NormType.RMSNorm);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 64]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(32, 64);
                    block.Forward(input);
                };
            });

        RunBatchedAttentionScenarios();
        RunRowScoringScenarios();
    }

    static void RunRowScoringScenarios()
    {
        const int rows = 10_000, cols = 128;

        Run("RowScore per-row copy+dot [10k x 128]", 5, 20,
            () =>
            {
                var frame = BuildScoreFrame(rows, cols);
                var columns = frame.ColumnNames.Select(frame.GetColumn<float>).ToArray();
                var query = Fill(new float[cols]);
                var scratch = new float[cols];
                return () =>
                {
                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < cols; c++)
                            scratch[c] = columns[c][r];
                        _ = TensorPrimitives.Dot(scratch, query);
                    }
                };
            });

        Run("Frame RowDot [10k x 128]", 5, 20,
            () =>
            {
                var frame = BuildScoreFrame(rows, cols);
                var query = NivaraSeries<float>.Create(Fill(new float[cols]));
                return () => frame.RowDot(query);
            });

        Run("Frame Slice [10k x 128]", 5, 100,
            () =>
            {
                var frame = BuildScoreFrame(rows, cols);
                return () => frame.Slice(0, 5_000);
            });

        Run("RowDot kernel raw [10k x 128]", 5, 50,
            () =>
            {
                var buffer = Fill(new float[rows * cols]);
                var query = Fill(new float[cols]);
                var output = new float[rows];
                var outputMask = new bool[rows];
                var fullMask = new bool[rows * cols];
                return () => TensorsHelper.RowDot(
                    buffer, fullMask,
                    query, ReadOnlySpan<bool>.Empty,
                    output, outputMask, rows, cols);
            });

        Run("RowCosineSimilarity kernel raw [10k x 128]", 5, 50,
            () =>
            {
                var buffer = Fill(new float[rows * cols]);
                var query = Fill(new float[cols]);
                var output = new float[rows];
                var outputMask = new bool[rows];
                return () => TensorsHelper.RowCosineSimilarity(
                    buffer, ReadOnlySpan<bool>.Empty,
                    query, ReadOnlySpan<bool>.Empty,
                    output, outputMask, rows, cols);
            });
    }

    static NivaraFrame BuildScoreFrame(int rows, int cols)
    {
        var columns = new (string Name, IColumn Column)[cols];
        for (int c = 0; c < cols; c++)
        {
            var data = new float[rows];
            for (int r = 0; r < rows; r++)
                data[r] = (r * cols + c) * 0.001f;
            columns[c] = ($"C{c}", NivaraColumn<float>.Create(data));
        }
        return new NivaraFrame(columns);
    }

    static void RunBatchedAttentionScenarios()
    {
        const int B = 16, L = 128, D = 64, H = 4;
        float scale = 1f / MathF.Sqrt(D / H);

        var qData = Fill(new float[B * L * D]);
        var kData = Fill(new float[B * L * D]);
        var vData = Fill(new float[B * L * D]);
        var dOut = Fill(new float[B * L * D]);
        var causalPerSeq = BuildCausalMask(L);
        var causalBatched = BuildCausalMask(B, L);

        Run($"Attn per-seq forward [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var mask = ReverseGradTensor<float>.FromMatrix(causalPerSeq, L, L, requiresGrad: false);
                return () =>
                {
                    for (int b = 0; b < B; b++)
                    {
                        var q = Mat2D(Slice(qData, b, L, D), L, D, false);
                        var k = Mat2D(Slice(kData, b, L, D), L, D, false);
                        var v = Mat2D(Slice(vData, b, L, D), L, D, false);
                        ReverseGradOperations.MultiHeadAttention(q, k, v, H, scale, mask);
                    }
                };
            });

        Run($"Attn batched forward [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var q = Mat3D(qData, B, L, D, false);
                var k = Mat3D(kData, B, L, D, false);
                var v = Mat3D(vData, B, L, D, false);
                var mask = Mat3D(causalBatched, B, L, L, false);
                return () => { ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask); };
            });

        Run($"Attn per-seq fwd+bwd [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var mask = ReverseGradTensor<float>.FromMatrix(causalPerSeq, L, L, requiresGrad: false);
                var ones = Fill(new float[L * D]);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        for (int b = 0; b < B; b++)
                        {
                            var q = Mat2D(Slice(qData, b, L, D), L, D, true);
                            var k = Mat2D(Slice(kData, b, L, D), L, D, true);
                            var v = Mat2D(Slice(vData, b, L, D), L, D, true);
                            var output = ReverseGradOperations.MultiHeadAttention(q, k, v, H, scale, mask);
                            output.Backward(Mat2D(ones, L, D, false));
                        }
                    }
                };
            });

        Run($"Attn batched fwd+bwd [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var q = Mat3D(qData, B, L, D, true);
                var k = Mat3D(kData, B, L, D, true);
                var v = Mat3D(vData, B, L, D, true);
                var mask = Mat3D(causalBatched, B, L, L, false);
                var dout = Mat3D(dOut, B, L, D, false);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var output = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);
                        output.Backward(dout);
                    }
                };
            });
    }

    static float[] BuildCausalMask(int l)
    {
        var mask = new float[l * l];
        for (int i = 0; i < l; i++)
            for (int j = i + 1; j < l; j++)
                mask[i * l + j] = float.NegativeInfinity;
        return mask;
    }

    static float[] BuildCausalMask(int b, int l)
    {
        var mask = new float[b * l * l];
        var perSeq = BuildCausalMask(l);
        for (int i = 0; i < b * l * l; i++)
            mask[i] = perSeq[i % (l * l)];
        return mask;
    }

    static float[] Slice(float[] data, int b, int rows, int cols)
    {
        var slice = new float[rows * cols];
        Array.Copy(data, b * rows * cols, slice, 0, rows * cols);
        return slice;
    }

    static ReverseGradTensor<float> Mat2D(float[] data, int rows, int cols, bool requiresGrad)
        => ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);

    static ReverseGradTensor<float> Mat3D(float[] data, int b, int l, int d, bool requiresGrad)
    {
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(b, l, d);
        return tensor;
    }

    static void Run(string name, int warmup, int iterations, Func<Action> createOp)
        => s_scenarios.Add(new ScenarioDefinition(name, warmup, iterations, createOp));

    static ScenarioResult MeasureScenario(ScenarioDefinition scenario, int runs)
    {
        if (runs <= 1)
            return MeasureOnce(scenario);

        var ops = new double[runs];
        var ns = new double[runs];
        var bytes = new double[runs];
        var gen0 = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            var r = MeasureOnce(scenario);
            ops[i] = r.OpsPerSec;
            ns[i] = r.NsPerOp;
            bytes[i] = r.BytesPerOp;
            gen0[i] = r.Gen0PerOp;
        }

        Array.Sort(ops);
        Array.Sort(ns);
        Array.Sort(bytes);
        Array.Sort(gen0);
        return new ScenarioResult(scenario.Name, ops[runs / 2], ns[runs / 2], bytes[runs / 2], gen0[runs / 2]);
    }

    static ScenarioResult MeasureOnce(ScenarioDefinition scenario)
    {
        var op = scenario.Create();

        for (int i = 0; i < scenario.Warmup; i++)
            op();

        long bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        int gen0Before = GC.CollectionCount(0);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < scenario.Iterations; i++)
            op();
        sw.Stop();
        long bytesAfter = GC.GetAllocatedBytesForCurrentThread();
        int gen0After = GC.CollectionCount(0);

        double nsPerOp = sw.Elapsed.TotalNanoseconds / scenario.Iterations;
        double opsPerSec = 1e9 / nsPerOp;
        double bytesPerOp = (double)(bytesAfter - bytesBefore) / scenario.Iterations;
        double gen0PerOp = (double)(gen0After - gen0Before) / scenario.Iterations;

        return new ScenarioResult(scenario.Name, opsPerSec, nsPerOp, bytesPerOp, gen0PerOp);
    }

    static void PrintRow(ScenarioResult r)
        => Console.WriteLine($"{r.Name,-46} {r.OpsPerSec,12:N0} {r.NsPerOp,8:N0} {r.BytesPerOp,12:N0} {r.Gen0PerOp,7:N2}");

    static (string? JsonPath, string? ComparePath, int Runs, double MinOpsFraction) ParseArgs(string[] args)
    {
        string? jsonPath = null, comparePath = null;
        int runs = 1;
        double minOpsFraction = DefaultMinOpsFraction;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--json" when i + 1 < args.Length:
                    jsonPath = args[++i];
                    break;
                case "--compare" when i + 1 < args.Length:
                    comparePath = args[++i];
                    break;
                case "--runs" when i + 1 < args.Length:
                    runs = int.Parse(args[++i]);
                    break;
                case "--tolerance" when i + 1 < args.Length:
                    minOpsFraction = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture) / 100.0;
                    break;
                default:
                    Console.Error.WriteLine($"Unknown argument: {args[i]}");
                    Console.Error.WriteLine("Usage: Nivara.PerformanceTests [--json <path>] [--compare <baseline.json>] [--runs <n>] [--tolerance <pct>]");
                    Environment.Exit(2);
                    break;
            }
        }

        return (jsonPath, comparePath, runs, minOpsFraction);
    }

    static void WriteJson(string path, List<ScenarioResult> results, int runs)
    {
        var report = new HarnessReport
        {
            Runtime = Environment.Version.ToString(),
            Machine = $"{Environment.ProcessorCount} logical processors, {(Environment.Is64BitProcess ? "x64" : "x86")}",
            Timestamp = DateTimeOffset.UtcNow,
            Runs = runs,
            Results = results,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(report, s_jsonOptions));
        Console.WriteLine($"Wrote {path}");
    }

    static int Compare(string baselinePath, List<ScenarioResult> results, double minOpsFraction)
    {
        HarnessReport baseline;
        try
        {
            baseline = JsonSerializer.Deserialize<HarnessReport>(File.ReadAllText(baselinePath), s_jsonOptions)
                ?? throw new InvalidDataException("empty baseline");
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"Cannot read baseline {baselinePath}: {e.Message}");
            return 2;
        }

        var baselineByName = baseline.Results.ToDictionary(r => r.Name);
        Console.WriteLine();
        Console.WriteLine($"No-regression gate vs {Path.GetFileName(baselinePath)} (minOps {minOpsFraction:P0}, maxAlloc {(1 - MaxAllocationFraction):P0} slack, gen0 +{Gen0Tolerance:N2}):");

        int failures = 0;
        foreach (var r in results)
        {
            if (!baselineByName.TryGetValue(r.Name, out var b))
            {
                Console.WriteLine($"  {r.Name,-46}  NEW   (no baseline row; not gated)");
                continue;
            }

            bool opsOk = r.OpsPerSec >= b.OpsPerSec * minOpsFraction;
            bool bytesOk = r.BytesPerOp <= b.BytesPerOp * MaxAllocationFraction;
            bool gen0Ok = r.Gen0PerOp <= b.Gen0PerOp + Gen0Tolerance;
            bool ok = opsOk && bytesOk && gen0Ok;
            if (!ok)
                failures++;

            Console.WriteLine(
                $"  {(ok ? "PASS" : "FAIL")}  {r.Name,-46}  ops/s {r.OpsPerSec,9:N0} vs {b.OpsPerSec,9:N0}  B/op {r.BytesPerOp,12:N0} vs {b.BytesPerOp,12:N0}  gen0 {r.Gen0PerOp,5:N2} vs {b.Gen0PerOp,5:N2}");
        }

        Console.WriteLine();
        if (failures == 0)
        {
            Console.WriteLine("Gate PASS — no regressions.");
            return 0;
        }

        Console.WriteLine($"Gate FAIL — {failures} scenario(s) outside tolerance.");
        return 1;
    }

    internal sealed record ScenarioDefinition(string Name, int Warmup, int Iterations, Func<Action> Create);

    internal sealed record ScenarioResult(string Name, double OpsPerSec, double NsPerOp, double BytesPerOp, double Gen0PerOp);

    internal sealed class HarnessReport
    {
        public string Runtime { get; set; } = "";
        public string Machine { get; set; } = "";
        public DateTimeOffset Timestamp { get; set; }
        public int Runs { get; set; }
        public List<ScenarioResult> Results { get; set; } = [];
    }

    /// <summary>
    /// Builds the fused-evaluator chain scenario for (Salary * 1.1) + 1000 - Tax. The scenario
    /// gates on the vectorized kernel heuristic (KernelSelector length >= vectorSize * 4) so the
    /// fused compiled target is exercised at a vectorized length.
    /// </summary>
    static Action CreateFusedChainScenario(int length)
    {
        var salary = NivaraColumn<double>.Create(Fill(new double[length]));
        var tax = NivaraColumn<double>.Create(Fill(new double[length]));
        var input = new Dictionary<string, IColumn> { ["Salary"] = salary, ["Tax"] = tax };
        var expression = ColumnExpressions.Col("Salary") * 1.1 + 1000 - ColumnExpressions.Col("Tax");

        if (KernelSelector.DetermineKernelType(length, ColumnStorageFactory.IsVectorizable<double>()) != KernelType.Vectorized)
        {
            throw new InvalidOperationException(
                $"Fused-chain gate requires the vectorized kernel heuristic at length {length} (length >= vectorSize * 4)");
        }

        var fused = new FusedExpressionEvaluator();

        for (int i = 0; i < 3; i++)
        {
            fused.Evaluate(expression, input);
        }

        return () => fused.Evaluate(expression, input);
    }

    /// <summary>
    /// Measures the mean execution time of an operation after a warmup phase.
    /// </summary>
    static double MeasureNs(Action op, int warmup, int iterations)
    {
        for (int i = 0; i < warmup; i++)
            op();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            op();
        sw.Stop();

        return sw.Elapsed.TotalNanoseconds / iterations;
    }

    static float[] Fill(float[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = i * 0.001f;
        return values;
    }

    static double[] Fill(double[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = i * 0.001;
        return values;
    }
}

