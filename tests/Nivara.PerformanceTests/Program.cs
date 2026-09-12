using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Diagnostics;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;
using Nivara.Samples;
using Nivara.Storage;
using Nivara.Tensors;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Nivara.PerformanceTests;

static class Program
{
    static readonly List<ScenarioDefinition> s_scenarios = [];

    static readonly JsonSerializerOptions s_jsonOptions = new(JsonSerializerDefaults.Web)
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = true,
    };

    static int Main(string[] args)
    {
        var (jsonPath, comparePath, runs, minOpsFraction, only, datasetTest, safetensorsMmap) = ParseArgs(args);

        if (datasetTest)
        {
            IncidentLabBenchmark.RunDatasetGeneratorTests(args);
            return 0;
        }

        if (safetensorsMmap)
        {
            SafeTensorsLoadBenchmark.Run(args);
            return 0;
        }

        if (runs > 1)
        {
            var results = MeasureAcrossProcesses(runs, only);
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

        if (only is not null)
            s_scenarios.RemoveAll(s => !s.Name.Contains(only, StringComparison.OrdinalIgnoreCase));

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
        Console.WriteLine($"  Machine : {MachineIdentityDescription()}");
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

    static List<ScenarioResult>? MeasureAcrossProcesses(int runs, string? only)
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
                if (only is not null)
                {
                    psi.ArgumentList.Add("--only");
                    psi.ArgumentList.Add(only);
                }
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

        Run("Fused chain chunked 1M x 64k rows", 5, 50,
            () => CreateFusedChunkedChainScenario(1_000_000, 65_536));

        Run("Fused single-op TP 1M x (Salary*1.1)", 5, 50,
            () => CreateFusedSingleOpScenario(1_000_000));

        Run("Column mul-scalar 1M (wrapper)", 5, 100,
            () => CreateColumnMulScalarScenario(1_000_000));

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
        RunWindowAllocationScenarios();
        RunRowWhereScenarios();
        RunStreamingCancellationScenarios();
        RunAutoDiffSimdScenarios();
        RunQwenDecodeMatMulScenarios();
        RunQwenDecodeAttentionScenarios();
        RunQwenDecodeBlockScenarios();
        RunQwenDecodeForwardScenarios();
        RunQwenPrefillScenarios();
    }

    static void RunRowWhereScenarios()
    {
        // Issue #347 gate: row GetValue<int> over a nullable-element column (NivaraColumn<int?>)
        // must not allocate per read. The cached delegate path makes the read cost ~0 B/row; the
        // residual B/op is the Where result-frame construction (FilterByMask), which is fixed
        // separately in issue #349. Registered as a NEW baseline row so --compare gates it once a
        // baseline is recorded for this harness revision.
        Run("Row.Where nullable-element GetValue 100k", 5, 20,
            () =>
            {
                var values = new int?[100_000];
                for (int i = 0; i < values.Length; i++)
                    values[i] = i % 100 == 0 ? null : i;
                var frame = NivaraFrame.Create(
                    ("Name", NivaraColumn<string>.CreateForReferenceType(Enumerable.Repeat("x", values.Length).ToArray())),
                    ("Age", NivaraColumn<int?>.Create(values)));
                return () => frame.Where(row => row.GetValue<int>("Age") > 15_000);
            });
    }

    static void RunWindowAllocationScenarios()
    {
        Run("RollingSum null-free 1M x int (w10)", 5, 50,
            () =>
            {
                var data = NivaraColumn<int>.Create(FillInt(new int[1_000_000]));
                return () => data.RollingSum(10);
            });

        Run("RollingSum nulls 1M x int (w10)", 5, 50,
            () =>
            {
                var data = FillInt(new int[1_000_000]);
                var mask = new bool[1_000_000];
                for (int i = 0; i < mask.Length; i++)
                    mask[i] = i % 7 == 0;
                var col = NivaraColumn<int>.CreateFromSpans(data, mask);
                return () => col.RollingSum(10, nullHandler: () => 0);
            });

        Run("RankKernel RowNumber 100k x int", 5, 50,
            () =>
            {
                var columns = new Dictionary<string, IColumn> { ["v"] = NivaraColumn<int>.Create(FillInt(new int[100_000])) };
                var orderBy = new[] { new SortKey("v", SortDirection.Ascending) };
                return () => RankKernel.Compute(columns, [], orderBy, RankKind.RowNumber);
            });

        Run("GroupBy 1M rows x 1000 keys (typed)", 5, 20,
            () =>
            {
                var keys = new int[1_000_000];
                for (int i = 0; i < keys.Length; i++)
                    keys[i] = i % 1000;
                var columns = new Dictionary<string, IColumn> { ["k"] = NivaraColumn<int>.Create(keys) };
                return () => GroupByOperation.CreateGroupsInternal(columns, new[] { "k" });
            });

        Run("GroupBy 1M rows x 100 string keys (typed)", 5, 20,
            () =>
            {
                var groups = new string[1_000_000];
                for (int i = 0; i < groups.Length; i++)
                    groups[i] = (i % 100).ToString();
                var columns = new Dictionary<string, IColumn> { ["g"] = NivaraColumn<string>.CreateForReferenceType(groups) };
                return () => GroupByOperation.CreateGroupsInternal(columns, new[] { "g" });
            });

        Run("PartitionedWindow RollingSum 1M x 100 parts", 5, 20,
            () =>
            {
                var data = new int[1_000_000];
                var groups = new string[1_000_000];
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = i;
                    groups[i] = (i % 100).ToString();
                }

                var columns = new Dictionary<string, IColumn>
                {
                    ["g"] = NivaraColumn<string>.CreateForReferenceType(groups),
                    ["v"] = NivaraColumn<int>.Create(data),
                };
                var spec = new WindowSpec().PartitionBy("g");
                return () => PartitionedWindowEngine.Compute(
                    columns, columns["v"], spec,
                    col => ((NivaraColumn<int>)col).RollingSum(10, 1));
            });
    }

    static void RunStreamingCancellationScenarios()
    {
        Run("Streaming cancel mid-stream 200k rows x 10k chunk", 3, 15,
            () => CreateStreamingCancellationScenario(totalRows: 200_000, chunkSize: 10_000, cancelAfterChunks: 3));
    }

    static void RunAutoDiffSimdScenarios()
    {
        const int n = 1_000_000;

        Run("AutoDiff Pow(2.5) fwd+bwd 1M x float", 5, 20,
            () =>
            {
                var data = NivaraColumn<float>.Create(Fill(new float[n]));
                var gradOnesCol = NivaraColumn<float>.Create(Fill(new float[n]));
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var input = new ReverseGradTensor<float>(data, requiresGrad: true);
                        var output = ReverseGradOperations.Pow(input, 2.5);
                        var gradOnes = new ReverseGradTensor<float>(gradOnesCol);
                        output.Backward(gradOnes);
                    }
                };
            });

        Run("AutoDiff Pow(2.5) scalar baseline 1M x float", 5, 20,
            () =>
            {
                var data = Fill(new float[n]);
                var grad = new float[n];
                return () =>
                {
                    for (int i = 0; i < n; i++)
                    {
                        var val = data[i];
                        var powVal = (float)Math.Pow(val, 2.5);
                        grad[i] = powVal * 2.5f / (val + 1e-7f);
                    }
                };
            });

        Run("AutoDiff RMSNorm fwd+bwd 1M x float", 5, 20,
            () =>
            {
                var data = NivaraColumn<float>.Create(Fill(new float[n]));
                var gradOnesCol = NivaraColumn<float>.Create(Fill(new float[n]));
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var input = new ReverseGradTensor<float>(data, requiresGrad: true);
                        var output = ReverseGradOperations.RMSNorm(input);
                        var gradOnes = new ReverseGradTensor<float>(gradOnesCol);
                        output.Backward(gradOnes);
                    }
                };
            });

        Run("AutoDiff RMSNorm scalar baseline 1M x float", 5, 20,
            () =>
            {
                var data = Fill(new float[n]);
                var grad = new float[n];
                return () =>
                {
                    float sumSq = 0;
                    for (int i = 0; i < n; i++)
                        sumSq += data[i] * data[i];
                    float rms = MathF.Sqrt(sumSq / n + 1e-5f);
                    for (int i = 0; i < n; i++)
                    {
                        var normed = data[i] / rms;
                        grad[i] = (1.0f / rms) * (1.0f - normed * normed / n);
                    }
                };
            });

        // #418 gate: the RMSNorm<T> module grad closure (gamma multiply + gamma-grad accumulation),
        // exercised end-to-end. The "AutoDiff RMSNorm fwd+bwd" row above covers the functional op
        // (ReverseGradOperations.RMSNorm) only, not the affine module.
        Run("AutoDiff RMSNormModule fwd+bwd 1M x float", 5, 20,
            () =>
            {
                const int rows = 256, cols = 4096;
                var rms = new RMSNorm<float>(cols, eps: 1e-5f);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[rows * cols]));
                var ones = Fill(new float[rows * cols]);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: true);
                        input.Reshape(rows, cols);
                        var output = rms.Forward(input);
                        var gradient = new ReverseGradTensor<float>(NivaraColumn<float>.Create(ones), requiresGrad: false);
                        gradient.Reshape(rows, cols);
                        output.Backward(gradient);
                    }
                };
            });
    }

    static void RunQwenDecodeMatMulScenarios()
    {
        // Qwen2.5-0.5B decode hot shapes (docs/QWEN.md → "Making Qwen fast"). Every decode matmul is a
        // single-row multiply against row-major [out, in] weights (bTransposed: true), so
        // these rows gate P0-2: the kernel must read each weight once — no rent + identity
        // copy — and allocate ~0 B/op with a preallocated result. Benchmarked at the real
        // caller sizes per the .NET SIMD guidance ("benchmark the input sizes your callers
        // actually use", learn.microsoft.com/dotnet/standard/simd).

        Run("Qwen LM head matmul [1x896 @ 151936x896]", 2, 8,
            () =>
            {
                const int aCols = 896, bCols = 151_936;
                var a = Fill(new float[aCols]);
                var b = Fill(new float[aCols * bCols]);
                var result = new float[bCols];
                return () => GradKernels.MatMulTransposedB(a, b, result, 1, aCols, bCols);
            });

        Run("Qwen FFN gate/up/down x3", 5, 30,
            () =>
            {
                const int hidden = 896, ff = 4864;
                var h = Fill(new float[hidden]);
                var ffHidden = Fill(new float[ff]);
                var wGate = Fill(new float[hidden * ff]);
                var wUp = Fill(new float[hidden * ff]);
                var wDown = Fill(new float[ff * hidden]);
                var ffOut = new float[ff];
                var hiddenOut = new float[hidden];
                return () =>
                {
                    GradKernels.MatMulTransposedB(h, wGate, ffOut, 1, hidden, ff);
                    GradKernels.MatMulTransposedB(h, wUp, ffOut, 1, hidden, ff);
                    GradKernels.MatMulTransposedB(ffHidden, wDown, hiddenOut, 1, ff, hidden);
                };
            });

        Run("Qwen attn Q/K/V/O proj", 5, 30,
            () =>
            {
                const int hidden = 896, oCols = 896, kvCols = 128;
                var h = Fill(new float[hidden]);
                var wQ = Fill(new float[hidden * oCols]);
                var wK = Fill(new float[hidden * kvCols]);
                var wV = Fill(new float[hidden * kvCols]);
                var wO = Fill(new float[hidden * oCols]);
                var outO = new float[oCols];
                var outKV = new float[kvCols];
                return () =>
                {
                    GradKernels.MatMulTransposedB(h, wQ, outO, 1, hidden, oCols);
                    GradKernels.MatMulTransposedB(h, wK, outKV, 1, hidden, kvCols);
                    GradKernels.MatMulTransposedB(h, wV, outKV, 1, hidden, kvCols);
                    GradKernels.MatMulTransposedB(h, wO, outO, 1, hidden, oCols);
                };
            });

        Run("Qwen Linear fwd [1x896 -> 2688]", 5, 30,
            () =>
            {
                var linear = new Linear<float>(896, 2688);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[896]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(1, 896);
                    linear.Forward(input);
                };
            });

        Run("Qwen LM head fwd [1x896 -> 151936]", 2, 8,
            () =>
            {
                var head = new Linear<float>(896, 151_936, bias: false);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[896]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(1, 896);
                    head.Forward(input);
                };
            });
    }

    static void RunQwenDecodeAttentionScenarios()
    {
        // Qwen2.5-0.5B decode attention (docs/QWEN.md → "Making Qwen fast"). Each per-token
        // decode step in LlamaCausalAttention.ForwardCached currently BlockCopies the whole
        // cached KV prefix, runs GqaRepeatKV x2 (14->2 heads), and re-packs K/V head-major via
        // MultiHeadAttention.PackHeads — O(newLen * numHeads * headDim) copies per layer per
        // token (~1 MB/op at kvLen=128; ~25 MB/token across 24 layers). The fused GQA
        // single-query kernel replaces that with zero-copy cache reads. These rows gate that
        // P0: B/op should drop from ~1 MB/op (kvLen=128) toward the residual op-boxing allocs
        // (~26 KB, tracked separately by the P1 fused-decoder-block item).

        Run("Qwen decode-attn fwd [1x896 @ kvLen=64]", 3, 30,
            () => CreateQwenDecodeAttentionScenario(64));
        Run("Qwen decode-attn fwd [1x896 @ kvLen=128]", 3, 30,
            () => CreateQwenDecodeAttentionScenario(128));
        Run("Qwen decode-attn fwd [1x896 @ kvLen=256]", 3, 20,
            () => CreateQwenDecodeAttentionScenario(256));
    }

    // Qwen2.5-0.5B shapes: hidden 896, heads 14, kvHeads 2, headDim 64. Pre-populates a cache
    // with kvLen positions (seeded, deterministic) and runs one single-token decode step. Each
    // run reuses the same pre-seeded cache so the measured op is exactly the decode attention.
    static Action CreateQwenDecodeAttentionScenario(int kvLen)
    {
        const int hidden = 896, numHeads = 14, numKvHeads = 2;
        var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 2048);
        int kvWidth = numKvHeads * (hidden / numHeads);
        var seed = new Random(42 + kvLen);
        // Capacity for kvLen cached positions plus the new decode row (required by ForwardCached).
        var kCache = new float[(kvLen + 1) * kvWidth];
        var vCache = new float[(kvLen + 1) * kvWidth];
        for (int i = 0; i < kvLen * kvWidth; i++)
        {
            kCache[i] = (float)(seed.NextDouble() * 2 - 1);
            vCache[i] = (float)(seed.NextDouble() * 2 - 1);
        }

        // A single throwaway decode warms the RoPE tables (cached lazily) so the timed op is
        // purely the decode attention, not first-use table construction.
        var warmToken = new ReverseGradTensor<float>(NivaraColumn<float>.Create(Fill(new float[hidden])), requiresGrad: false);
        warmToken.Reshape(1, hidden);
        attn.ForwardCached(warmToken, kvLen, kCache, vCache, kvLen);

        return () =>
        {
            var input = new ReverseGradTensor<float>(NivaraColumn<float>.Create(Fill(new float[hidden])), requiresGrad: false);
            input.Reshape(1, hidden);
            attn.ForwardCached(input, kvLen, kCache, vCache, kvLen);
        };
    }

    static void RunQwenDecodeBlockScenarios()
    {
        // Qwen2.5-0.5B single-token decoder-block decode (issue #404, docs/QWEN.md → "Making Qwen fast"). Row
        // name is identical on both branches — the baseline body measured the per-op block chain
        // (block.ForwardCached); this branch measures the fused span kernel
        // (block.ForwardCachedFused) — so --compare gates the true before/after. Target: B/op
        // drops from 131,985 (per-op chain) toward ~0 steady-state.
        Run("Qwen decode block [1x896 @ kvLen=64]", 3, 30,
            () => CreateQwenDecodeBlockScenario(64));
    }

    // Qwen2.5-0.5B shapes: hidden 896, heads 14, kvHeads 2, headDim 64 (kvWidth 128), FFN 4864.
    // Pre-populates a kvLen-row seeded cache and runs one single-token decode step through one
    // full decoder block. Steady-state body: the fused kernel reuses the per-block scratch + the
    // same input/output buffers and pre-seeded cache every call (zero heap allocations).
    static Action CreateQwenDecodeBlockScenario(int kvLen)
    {
        const int hidden = 896, numHeads = 14, numKvHeads = 2, intermediate = 4864;
        var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate);
        int kvWidth = numKvHeads * (hidden / numHeads);
        var seed = new Random(42 + kvLen);
        // Capacity for kvLen cached positions plus the new decode row (required by ForwardCachedFused).
        var kCache = new float[(kvLen + 1) * kvWidth];
        var vCache = new float[(kvLen + 1) * kvWidth];
        for (int i = 0; i < kvLen * kvWidth; i++)
        {
            kCache[i] = (float)(seed.NextDouble() * 2 - 1);
            vCache[i] = (float)(seed.NextDouble() * 2 - 1);
        }

        // A single throwaway decode warms the lazy RoPE tables + the per-block fused scratch so
        // the timed op is purely the fused block step, not first-use table/scratch allocation.
        var inBuf = Fill(new float[hidden]);
        var outBuf = new float[hidden];
        block.ForwardCachedFused(inBuf, outBuf, kvLen, kCache, vCache, kvLen);

        return () => block.ForwardCachedFused(inBuf, outBuf, kvLen, kCache, vCache, kvLen);
    }

    static void RunQwenDecodeForwardScenarios()
    {
        // Model-level single-token decode after a 64-token prefill (issue #404's per-token cost,
        // docs/QWEN.md → "Making Qwen fast"). Same row name on both branches — the model body routes to the
        // fused block path outside Grad once wired, so --compare gates the before/after. The
        // ~2 GB model is built once per row outside timing, like CreateQwenPrefillScenario.
        Run("Qwen decode fwd [1 step]", 1, 6, () => CreateQwenDecodeForwardScenario());
    }

    static Action CreateQwenDecodeForwardScenario()
    {
        const int hidden = 896, kvLen = 64;
        var model = new LlamaForCausalLM<float>(151_936, 896, 24, 14, 2, 4864);
        var cache = new LlamaKVCache<float>(24, 2 * (hidden / 14));
        var rng = new Random(42 + kvLen);
        var prompt = new int[kvLen];
        for (int i = 0; i < kvLen; i++)
            prompt[i] = rng.Next(151_936);
        // Seed outside timing so the measured op is exactly one decode step (KV append + attend).
        model.ForwardPrefill(prompt, cache);
        int nextTok = rng.Next(151_936);
        return () => model.ForwardCached(nextTok, kvLen, cache);
    }

    static void RunQwenPrefillScenarios()
    {
        // Qwen2.5-0.5B prompt prefill (docs/QWEN.md → "Making Qwen fast"). "Qwen prefill
        // seed [L]" measures seeding the KV cache for an L-token prompt: on main the row runs the
        // current token-by-token SeedCache loop (L full-model walks — each re-reads every weight,
        // ~2 GB F32); on this branch the same row runs one batched model.ForwardPrefill (single
        // weight pass + K/V capture at post-RoPE, pre-GQA-repeat). Row names are identical on both
        // branches — only PrefillInto's body swaps — so --compare gates the true before/after.
        // The L = 64/256 seed rows are batched-only NEW rows added by the implementation commit
        // (the loop cost ~30-100 s/op there; the L = 64 before/after is carried by the E2E split
        // prefill/decode timing, docs/TODO.md §C). "Qwen full fwd" rows are unchanged
        // model.Forward(ids) on both branches — no-regression siblings.
        Run("Qwen prefill seed [8 tok]", 1, 6, () => CreateQwenPrefillScenario(8));
        Run("Qwen prefill seed [16 tok]", 1, 6, () => CreateQwenPrefillScenario(16));
        Run("Qwen prefill seed [64 tok]", 1, 6, () => CreateQwenPrefillScenario(64));
        Run("Qwen prefill seed [256 tok]", 1, 6, () => CreateQwenPrefillScenario(256));
        Run("Qwen full fwd [64 tok]", 1, 6, () => CreateQwenFullForwardScenario(64));
        Run("Qwen full fwd [256 tok]", 1, 6, () => CreateQwenFullForwardScenario(256));
    }

    // Qwen2.5-0.5B shapes: vocab 151,936, hidden 896, 24 layers, 14 heads, 2 KV heads (headDim
    // 64, kvWidth 128), intermediate 4864 — ~2 GB resident F32, built once per row outside
    // timing. Deterministic token ids via a seeded RNG (per-row seed) keep runs reproducible.
    static Action CreateQwenPrefillScenario(int L)
    {
        var model = new LlamaForCausalLM<float>(151_936, 896, 24, 14, 2, 4864);
        var cache = new LlamaKVCache<float>(24, 2 * (896 / 14));
        var ids = new int[L];
        var rng = new Random(42 + L);
        for (int i = 0; i < L; i++)
            ids[i] = rng.Next(151_936);
        return () => PrefillInto(model, ids, cache);
    }

    static Action CreateQwenFullForwardScenario(int L)
    {
        var model = new LlamaForCausalLM<float>(151_936, 896, 24, 14, 2, 4864);
        var ids = new int[L];
        var rng = new Random(42 + L);
        for (int i = 0; i < L; i++)
            ids[i] = rng.Next(151_936);
        return () => model.Forward(ids);
    }

    // Baseline (on main): the token-by-token SeedCache loop this P0 replaces. On this branch the
    // body is the batched model.ForwardPrefill — same row names, both JSONs — so --compare gates
    // the true before/after.
    static void PrefillInto(LlamaForCausalLM<float> model, int[] ids, LlamaKVCache<float> cache)
        => model.ForwardPrefill(ids, cache);

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

    static (string? JsonPath, string? ComparePath, int Runs, double MinOpsFraction, string? Only, bool DatasetTest, bool SafetensorsMmap) ParseArgs(string[] args)
    {
        string? jsonPath = null, comparePath = null, only = null;
        int runs = 1;
        double minOpsFraction = GateEvaluator.DefaultMinOpsFraction;
        bool datasetTest = false;
        bool safetensorsMmap = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dataset-test":
                    datasetTest = true;
                    break;
                case "--safetensors-mmap":
                    safetensorsMmap = true;
                    break;
                case "--only" when i + 1 < args.Length:
                    only = args[++i];
                    break;
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
                    Console.Error.WriteLine("Usage: Nivara.PerformanceTests [--dataset-test] [--safetensors-mmap [<path>]] [--only <substring>] [--json <path>] [--compare <baseline.json>] [--runs <n>] [--tolerance <pct>]");
                    Environment.Exit(2);
                    break;
            }
        }

        return (jsonPath, comparePath, runs, minOpsFraction, only, datasetTest, safetensorsMmap);
    }

    static void WriteJson(string path, List<ScenarioResult> results, int runs)
    {
        var report = new HarnessReport
        {
            Runtime = Environment.Version.ToString(),
            Machine = MachineIdentityDescription(),
            Timestamp = DateTimeOffset.UtcNow,
            Runs = runs,
            Results = results,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(report, s_jsonOptions));
        Console.WriteLine($"Wrote {path}");
    }

    /// <summary>Self-attesting machine identity for apples-to-apples A/B claims: CPU
    /// identifier, logical processor count, process architecture, and OS. Windows exposes
    /// the CPU brand via the PROCESSOR_IDENTIFIER environment variable; other platforms fall
    /// back to /proc/cpuinfo (Linux) or "unknown".</summary>
    static string MachineIdentityDescription()
    {
        string cpu = CpuIdentifier();
        string arch = RuntimeInformation.ProcessArchitecture.ToString();
        string os = RuntimeInformation.OSDescription;
        return $"{cpu} · {Environment.ProcessorCount} logical processors · {arch} · {os}";
    }

    static string CpuIdentifier()
    {
        var env = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER");
        if (!string.IsNullOrWhiteSpace(env)) return env.Trim();
        try
        {
            if (File.Exists("/proc/cpuinfo"))
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                {
                    if (line.StartsWith("model name", StringComparison.OrdinalIgnoreCase))
                        return line[(line.IndexOf(':') + 1)..].Trim();
                }
            }
        }
        catch
        {
            // Fall through to "unknown" — identity capture must never fail the harness.
        }
        return "unknown CPU";
    }

    /// <summary>
    /// True for bandwidth-bound scenarios whose ops/s swings ~2.5-3x with machine state
    /// (single-row GEMV / memory-streaming kernels — the Qwen decode and prefill rows,
    /// issue #420). Their ops/s leg is gated at
    /// <see cref="GateEvaluator.BandwidthBoundMinOpsFraction"/> instead of the default
    /// stable-row floor; B/op and gen0 stay strict for all rows.
    /// </summary>
    static bool IsBandwidthBound(string name)
        => name.StartsWith("Qwen ", StringComparison.Ordinal);

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
        Console.WriteLine($"No-regression gate vs {Path.GetFileName(baselinePath)} (minOps {minOpsFraction:P0}, bandwidth-bound ops/s floor {GateEvaluator.BandwidthBoundMinOpsFraction:P0}, maxAlloc {(1 - GateEvaluator.MaxAllocationFraction):P0} slack, gen0 +{GateEvaluator.Gen0Tolerance:N2}):");

        int failures = 0;
        foreach (var r in results)
        {
            if (!baselineByName.TryGetValue(r.Name, out var b))
            {
                Console.WriteLine($"  {r.Name,-46}  NEW   (no baseline row; not gated)");
                continue;
            }

            bool bandwidthBound = IsBandwidthBound(r.Name);
            var verdict = GateEvaluator.EvaluateRow(
                new GateRow(r.Name, r.OpsPerSec, r.BytesPerOp, r.Gen0PerOp),
                new GateRow(b.Name, b.OpsPerSec, b.BytesPerOp, b.Gen0PerOp),
                minOpsFraction, bandwidthBound);
            if (!verdict.Pass)
                failures++;

            string mark = verdict.Pass
                ? (bandwidthBound ? "BW-PASS" : "PASS")
                : (bandwidthBound ? "BW-FAIL" : "FAIL");
            Console.WriteLine(
                $"  {mark}  {r.Name,-46}  ops/s {r.OpsPerSec,9:N0} vs {b.OpsPerSec,9:N0} (floor {verdict.OpsFloor:P0})  B/op {r.BytesPerOp,12:N0} vs {b.BytesPerOp,12:N0}  gen0 {r.Gen0PerOp,5:N2} vs {b.Gen0PerOp,5:N2}");
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
    /// Builds the chunked fused-chain scenario: same expression as <see cref="CreateFusedChainScenario"/>
    /// but evaluated through <see cref="FusedExpressionEvaluator.EvaluateChunked"/> in 64k-row batches,
    /// which slices the existing leaf storage instead of copying it (issue #167).
    /// </summary>
    static Action CreateFusedChunkedChainScenario(int length, int chunkSize)
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
            fused.EvaluateChunked(expression, input, chunkSize);
        }

        return () => fused.EvaluateChunked(expression, input, chunkSize);
    }

    /// <summary>
    /// Builds the single-op fused scenario: a null-free single Multiply dispatches to the
    /// TensorPrimitives SIMD kernel in one call (issue #167).
    /// </summary>
    static Action CreateFusedSingleOpScenario(int length)
    {
        var salary = NivaraColumn<double>.Create(Fill(new double[length]));
        var input = new Dictionary<string, IColumn> { ["Salary"] = salary };
        var expression = ColumnExpressions.Col("Salary") * 1.1;

        if (KernelSelector.DetermineKernelType(length, ColumnStorageFactory.IsVectorizable<double>()) != KernelType.Vectorized)
        {
            throw new InvalidOperationException(
                $"Fused-single-op gate requires the vectorized kernel heuristic at length {length} (length >= vectorSize * 4)");
        }

        var fused = new FusedExpressionEvaluator();

        for (int i = 0; i < 3; i++)
        {
            fused.Evaluate(expression, input);
        }

        return () => fused.Evaluate(expression, input);
    }

    /// <summary>
    /// Builds the column-wrapper multiply-scalar scenario (the multi-pass baseline for the fused
    /// single-op TensorPrimitives path).
    /// </summary>
    static Action CreateColumnMulScalarScenario(int length)
    {
        var salary = NivaraColumn<double>.Create(Fill(new double[length]));
        return () => salary.Multiply(1.1);
    }

    /// <summary>
    /// Phase 4 AC2 scenario: cancels a chunk-capable streaming run mid-stream through the
    /// bounded-channel pipeline (<c>StreamingExecutionStrategy.ExecuteCoreAsync</c>, issue #266)
    /// and asserts a clean <see cref="OperationCanceledException"/> — not wrapped in
    /// <c>QueryExecutionException</c> — with prompt unwind. Issue #280 (consumer-side catch
    /// calling <c>channel.Writer.Complete()</c> on an already-completed channel, masking the
    /// OCE with <c>ChannelClosedException</c>) is fixed; the scenario now goes green and B/op
    /// captures any in-flight/channel-buffered chunk frames the cancelled path must dispose.
    /// </summary>
    static Action CreateStreamingCancellationScenario(int totalRows, int chunkSize, int cancelAfterChunks)
    {
        var engine = new ExecutionEngine();
        var operation = new PerfStreamableOperation();

        return () =>
        {
            var source = new PerfChunkedSource(totalRows);
            using var cts = new CancellationTokenSource();
            var plan = new QueryPlan(source, new IQueryOperation[] { operation });
            var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
            {
                CancellationToken = cts.Token,
                ChunkSize = chunkSize,
            };
            source.CancelWhenChunkCountReaches(cts, cancelAfterChunks);

            var task = engine.ExecuteAsync(plan, context);
            try
            {
                task.GetAwaiter().GetResult();
                throw new InvalidOperationException(
                    "Expected OperationCanceledException, but the streaming run completed.");
            }
            catch (OperationCanceledException)
            {
            }
        };
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

    static int[] FillInt(int[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = i;
        return values;
    }
}

/// <summary>
/// In-memory chunk-capable source used by the streaming-cancellation scenario. Cancels the
/// run once <paramref name="cancelAfterChunks"/> chunks have been read so the token fires
/// deterministically mid-stream.
/// </summary>
sealed class PerfChunkedSource : IQuerySource
{
    readonly int totalRowCount;
    int chunksRead;
    int cancelTarget = -1;
    CancellationTokenSource? cancelCts;

    public PerfChunkedSource(int totalRowCount)
    {
        this.totalRowCount = totalRowCount;
    }

    public Schema Schema => new(new[] { ("A", typeof(int)) });

    public bool IsLazy => false;

    public bool CanReadInChunks => true;

    public int? EstimatedRowCount => totalRowCount;

    public void CancelWhenChunkCountReaches(CancellationTokenSource cts, int targetChunk)
    {
        cancelCts = cts;
        cancelTarget = targetChunk;
    }

    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(0, totalRowCount)) };
    }

    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        var start = chunkIndex * chunkSize;
        var length = Math.Min(chunkSize, totalRowCount - start);
        if (length <= 0)
            return new Dictionary<string, IColumn>(0);
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(start, length)) };
    }

    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var n = Interlocked.Increment(ref chunksRead);
        if (cancelCts != null && n >= cancelTarget)
            cancelCts.Cancel();

        await Task.Yield();

        var start = chunkIndex * chunkSize;
        var length = Math.Min(chunkSize, totalRowCount - start);
        if (length <= 0)
            return new Dictionary<string, IColumn>(0);
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(start, length)) };
    }

    static int[] BuildData(int start, int count)
    {
        var data = new int[count];
        for (int i = 0; i < count; i++)
            data[i] = start + i;
        return data;
    }

    public void Dispose()
    {
    }
}

/// <summary>
/// Identity streamable operation (Filter) used by the streaming-cancellation scenario.
/// </summary>
sealed class PerfStreamableOperation : IQueryOperation
{
    public string OperationType => Nivara.Query.OperationType.Filter;

    public Schema TransformSchema(Schema input) => input;

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input) => input;
}

