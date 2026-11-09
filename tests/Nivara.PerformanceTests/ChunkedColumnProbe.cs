using Nivara.Storage;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// <c>--chunked-baseline</c> (aliases <c>--chunked</c>, <c>--chunked-vs-flat</c>):
/// establishes the one-copy cost baseline for chunked-vs-flat column operations
/// (#1076, follow-up to #862 / PR #1073).
///
/// <para>Chunked arithmetic materialises one contiguous copy per operand on first
/// request, then runs the existing flat kernel. This probe measures how that cost
/// scales with chunk count, operand count (one vs. two operands) and operation,
/// using the <c>TransposeKernelProbe</c> discipline: warm both routes, alternate
/// which route is measured first, report a distribution (wins + median ratio),
/// never a single estimate.</para>
/// </summary>
static class ChunkedColumnProbe
{
    const int TotalN = 1_000_000;
    static readonly int[] ChunkCounts = [2, 8, 64];
    const int Rounds = 30;
    const int Warmups = 5;
    const double NoiseBand = 0.03;

    sealed class Cell
    {
        public string Name = "";
        public int ChunkCount;
        public bool Nulls;
        public Action Flat = () => { };
        public Action Chunked = () => { };
    }

    static readonly List<Cell> s_cells = [];

    public static int Run()
    {
        Console.WriteLine("=== #1076 chunked-vs-flat baseline ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Interleaved design: {Rounds} A/B rounds per cell, first route alternates every round,");
        Console.WriteLine($"{Warmups} untimed warmups per route. A ratio > 1 means CHUNKED is SLOWER (one-copy cost).");
        Console.WriteLine($"Noise band: +/-{NoiseBand:P0}.");
        if (IsDebugBuild())
        {
            Console.WriteLine();
            Console.WriteLine("  *** DEBUG BUILD - numbers are NOT evidence. Run with -c Release.");
        }
        Console.WriteLine();

        BuildCells();

        foreach (int k in ChunkCounts)
        {
            Console.WriteLine($"########## {k} chunks (chunk length {TotalN / k:N0}) ##########");
            Console.WriteLine();
            foreach (Cell cell in s_cells.Where(c => c.ChunkCount == k))
            {
                RunCell(cell);
            }
        }

        Console.WriteLine("Legend: ratio = median(chunked ms) / median(flat ms). > 1 means chunked pays the one-copy cost.");
        Console.WriteLine("  uncached = fresh column each timed call (materialises on first read).");
        Console.WriteLine("  cached    = instance reused, materialised copy already present (second call).");
        Console.WriteLine("  slice+op  = Slice() (no copy) followed by op (materialises the slice).");
        return 0;
    }

    static bool IsDebugBuild()
    {
#if DEBUG
        return true;
#else
        return false;
#endif
    }

    static void BuildCells()
    {
        s_cells.Clear();
        foreach (int k in ChunkCounts)
        {
            int chunkLength = TotalN / k;
            ColumnStorage<int>[] chunksNN = MakeChunks(k, nulls: false);
            ColumnStorage<int>[] chunksNull = MakeChunks(k, nulls: true);

            int[] flatDataNN = MakeData(TotalN, nulls: false).data;
            (int[] flatDataNull, bool[] flatMaskNull) = MakeData(TotalN, nulls: true);

            ColumnStorage<int> flatStoreNN = new ColumnStorage<int>(flatDataNN);
            ColumnStorage<int> flatStoreNN2 = new ColumnStorage<int>(flatDataNN);
            ColumnStorage<int> flatStoreNull = new ColumnStorage<int>(new ReadOnlyMemory<int>(flatDataNull), new ReadOnlyMemory<bool>(flatMaskNull));

            // Cached instances: distinct objects so a two-operand op materialises both.
            NivaraColumn<int> flatCachedA = NivaraColumn<int>.Create(flatDataNN);
            NivaraColumn<int> flatCachedB = NivaraColumn<int>.Create(flatDataNN);
            NivaraColumn<int> flatCachedNullA = NivaraColumn<int>.CreateFromSpans(flatDataNull, flatMaskNull);
            NivaraColumn<int> flatCachedNullB = NivaraColumn<int>.CreateFromSpans(flatDataNull, flatMaskNull);

            NivaraColumn<int> chunkedCachedA = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN));
            NivaraColumn<int> chunkedCachedB = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN));
            NivaraColumn<int> chunkedCachedNullA = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNull));
            NivaraColumn<int> chunkedCachedNullB = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNull));

            // Materialise the cached instances once up front.
            _ = chunkedCachedA + 1;
            _ = chunkedCachedB + 1;
            _ = chunkedCachedNullA + 1;
            _ = chunkedCachedNullB + 1;
            _ = flatCachedA + 1;
            _ = flatCachedB + 1;
            _ = flatCachedNullA + 1;
            _ = flatCachedNullB + 1;

            // 1 operand, scalar, uncached (first materialisation each call)
            AddCell($"scalar add (1 op) uncached", k, false,
                flat: () => { var c = new NivaraColumn<int>(flatStoreNN); _ = c + 1; },
                chunked: () => { var c = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = c + 1; });
            AddCell($"scalar add (1 op) cached 2nd", k, false,
                flat: () => { _ = flatCachedA + 1; },
                chunked: () => { _ = chunkedCachedA + 1; });

            // 2 operands, column add
            AddCell($"column add (2 ops) uncached", k, false,
                flat: () => { var a = new NivaraColumn<int>(flatStoreNN); var b = new NivaraColumn<int>(flatStoreNN2); _ = a + b; },
                chunked: () => { var a = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); var b = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = a + b; });
            AddCell($"column add (2 ops) cached 2nd", k, false,
                flat: () => { _ = flatCachedA + flatCachedB; },
                chunked: () => { _ = chunkedCachedA + chunkedCachedB; });

            // equality (scalar 1 op, column 2 ops)
            AddCell($"equals scalar (1 op) uncached", k, false,
                flat: () => { var c = new NivaraColumn<int>(flatStoreNN); _ = c.Equals(1); },
                chunked: () => { var c = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = c.Equals(1); });
            AddCell($"equals scalar (1 op) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.Equals(1); },
                chunked: () => { _ = chunkedCachedA.Equals(1); });
            AddCell($"equals column (2 ops) uncached", k, false,
                flat: () => { var a = new NivaraColumn<int>(flatStoreNN); var b = new NivaraColumn<int>(flatStoreNN2); _ = a.Equals(b); },
                chunked: () => { var a = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); var b = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = a.Equals(b); });
            AddCell($"equals column (2 ops) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.Equals(flatCachedB); },
                chunked: () => { _ = chunkedCachedA.Equals(chunkedCachedB); });

            // greater-than
            AddCell($"greater-than scalar (1 op) uncached", k, false,
                flat: () => { var c = new NivaraColumn<int>(flatStoreNN); _ = c.GreaterThan(1); },
                chunked: () => { var c = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = c.GreaterThan(1); });
            AddCell($"greater-than scalar (1 op) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.GreaterThan(1); },
                chunked: () => { _ = chunkedCachedA.GreaterThan(1); });
            AddCell($"greater-than column (2 ops) uncached", k, false,
                flat: () => { var a = new NivaraColumn<int>(flatStoreNN); var b = new NivaraColumn<int>(flatStoreNN2); _ = a.GreaterThan(b); },
                chunked: () => { var a = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); var b = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = a.GreaterThan(b); });
            AddCell($"greater-than column (2 ops) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.GreaterThan(flatCachedB); },
                chunked: () => { _ = chunkedCachedA.GreaterThan(chunkedCachedB); });

            // less-than
            AddCell($"less-than scalar (1 op) uncached", k, false,
                flat: () => { var c = new NivaraColumn<int>(flatStoreNN); _ = c.LessThan(1); },
                chunked: () => { var c = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = c.LessThan(1); });
            AddCell($"less-than scalar (1 op) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.LessThan(1); },
                chunked: () => { _ = chunkedCachedA.LessThan(1); });
            AddCell($"less-than column (2 ops) uncached", k, false,
                flat: () => { var a = new NivaraColumn<int>(flatStoreNN); var b = new NivaraColumn<int>(flatStoreNN2); _ = a.LessThan(b); },
                chunked: () => { var a = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); var b = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNN)); _ = a.LessThan(b); });
            AddCell($"less-than column (2 ops) cached 2nd", k, false,
                flat: () => { _ = flatCachedA.LessThan(flatCachedB); },
                chunked: () => { _ = chunkedCachedA.LessThan(chunkedCachedB); });

            // Null-bearing operands (mask copy path): scalar + column add.
            AddCell($"scalar add w/ nulls (1 op) uncached", k, true,
                flat: () => { var c = new NivaraColumn<int>(flatStoreNull); _ = c + 1; },
                chunked: () => { var c = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNull)); _ = c + 1; });
            AddCell($"scalar add w/ nulls (1 op) cached 2nd", k, true,
                flat: () => { _ = flatCachedNullA + 1; },
                chunked: () => { _ = chunkedCachedNullA + 1; });
            AddCell($"column add w/ nulls (2 ops) uncached", k, true,
                flat: () => { var a = new NivaraColumn<int>(flatStoreNull); var b = new NivaraColumn<int>(flatStoreNull); _ = a + b; },
                chunked: () => { var a = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNull)); var b = new NivaraColumn<int>(new ChunkedColumn<int>(chunksNull)); _ = a + b; });
            AddCell($"column add w/ nulls (2 ops) cached 2nd", k, true,
                flat: () => { _ = flatCachedNullA + flatCachedNullB; },
                chunked: () => { _ = chunkedCachedNullA + chunkedCachedNullB; });

            // Slice then op (no materialisation yet).
            int sliceStart = TotalN / 4;
            int sliceLength = TotalN / 2;
            AddCell($"slice+scalar add (uncached)", k, false,
                flat: () => { var s = flatCachedA.Slice(sliceStart, sliceLength); _ = s + 1; },
                chunked: () => { var s = chunkedCachedA.Slice(sliceStart, sliceLength); _ = s + 1; });
            AddCell($"slice+scalar add w/ nulls (uncached)", k, true,
                flat: () => { var s = flatCachedNullA.Slice(sliceStart, sliceLength); _ = s + 1; },
                chunked: () => { var s = chunkedCachedNullA.Slice(sliceStart, sliceLength); _ = s + 1; });
        }
    }

    static void AddCell(string name, int k, bool nulls, Action flat, Action chunked)
    {
        s_cells.Add(new Cell { Name = name, ChunkCount = k, Nulls = nulls, Flat = flat, Chunked = chunked });
    }

    static (int[] data, bool[] mask) MakeData(int n, bool nulls)
    {
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = i;
        if (!nulls) return (data, Array.Empty<bool>());
        var mask = new bool[n];
        for (int i = 0; i < n; i++) if (i % 7 == 0) mask[i] = true;
        return (data, mask);
    }

    static ColumnStorage<int>[] MakeChunks(int k, bool nulls)
    {
        int chunkLength = TotalN / k;
        var chunks = new ColumnStorage<int>[k];
        for (int ci = 0; ci < k; ci++)
        {
            var data = new int[chunkLength];
            for (int i = 0; i < chunkLength; i++) data[i] = ci * chunkLength + i;
            if (!nulls)
            {
                chunks[ci] = new ColumnStorage<int>(data);
            }
            else
            {
                var mask = new bool[chunkLength];
                for (int i = 0; i < chunkLength; i++) if ((ci * chunkLength + i) % 7 == 0) mask[i] = true;
                chunks[ci] = new ColumnStorage<int>(new ReadOnlyMemory<int>(data), new ReadOnlyMemory<bool>(mask));
            }
        }
        return chunks;
    }

    static void RunCell(Cell cell)
    {
        string label = $"{cell.Name,-34} {(cell.Nulls ? "null" : "    ")}";
        // Warmup
        for (int w = 0; w < Warmups; w++)
        {
            cell.Flat();
            cell.Chunked();
        }

        var flatTicks = new long[Rounds];
        var chunkedTicks = new long[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            if (r % 2 == 0)
            {
                flatTicks[r] = Time(cell.Flat);
                chunkedTicks[r] = Time(cell.Chunked);
            }
            else
            {
                chunkedTicks[r] = Time(cell.Chunked);
                flatTicks[r] = Time(cell.Flat);
            }
        }

        double[] flatMs = flatTicks.Select(t => t * 1000.0 / Stopwatch.Frequency).ToArray();
        double[] chunkedMs = chunkedTicks.Select(t => t * 1000.0 / Stopwatch.Frequency).ToArray();
        double medianFlat = Median(flatMs);
        double medianChunked = Median(chunkedMs);
        double ratio = medianChunked / medianFlat;
        int chunkedWins = chunkedTicks.Zip(flatTicks, (c, f) => c < f ? 1 : 0).Sum();

        Console.WriteLine($"  {label}  flat {medianFlat,7:F3} ms  chunked {medianChunked,7:F3} ms  ratio {ratio,6:F3}  wins {chunkedWins,2}/{Rounds}  (min/max chunked {chunkedMs.Min():F3}/{chunkedMs.Max():F3})");
    }

    static long Time(Action action)
    {
        long start = Stopwatch.GetTimestamp();
        action();
        return Stopwatch.GetTimestamp() - start;
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return (sorted.Length & 1) == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
}