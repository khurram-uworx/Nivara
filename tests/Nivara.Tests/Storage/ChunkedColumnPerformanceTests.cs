using Nivara.Storage;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.Storage;

/// <summary>
/// Structural guard for the chunk layer (#862): the flat slice hot path still aliases its parent
/// and stays flat, a chunked slice does not copy or materialise, and the interleaved A/B run
/// records the one-copy cost that chunked arithmetic pays by design.
/// </summary>
[TestFixture]
[Category("Performance")]
public class ChunkedColumnPerformanceTests
{
    const int ChunkCount = 8;
    const int ChunkLength = 128;

    static NivaraColumn<int> CreateChunked()
    {
        var chunks = new ColumnStorage<int>[ChunkCount];
        for (int chunkIndex = 0; chunkIndex < ChunkCount; chunkIndex++)
        {
            var values = new int[ChunkLength];
            for (int i = 0; i < ChunkLength; i++)
                values[i] = chunkIndex * ChunkLength + i;
            chunks[chunkIndex] = new ColumnStorage<int>(values);
        }

        return new NivaraColumn<int>(new ChunkedColumn<int>(chunks));
    }

    [Test]
    public void FlatSlice_StillReturnsColumnStorageAndAliasesParent()
    {
        var column = NivaraColumn<int>.Create(Enumerable.Range(0, ChunkCount * ChunkLength).ToArray());

        var slice = column.Slice(100, 10);

        Assert.That(slice.Storage, Is.InstanceOf<ColumnStorage<int>>());
        column.AsWritableSpan()[100] = 42;
        Assert.That(slice[0], Is.EqualTo(42));
    }

    [Test]
    public void ChunkedSlice_DoesNotCopyAndDoesNotMaterialise()
    {
        var column = CreateChunked();

        var slice = column.Slice(ChunkLength / 2, ChunkLength * 2);

        Assert.That(slice.Storage, Is.InstanceOf<ChunkedColumn<int>>());
        var chunked = (ChunkedColumn<int>)slice.Storage;
        Assert.That(chunked.ChunkStorages.Length, Is.EqualTo(3));
        Assert.That(chunked.IsMaterialized, Is.False);

        var firstChunk = chunked.ChunkStorages[0];
        ((IColumnStorage<int>)firstChunk).AsWritableSpan()[0] = 7;
        Assert.That(slice[0], Is.EqualTo(7));
        Assert.That(chunked.IsMaterialized, Is.False);
    }

    [Test]
    public void FlatVersusEightChunk_InterleavedAddition()
    {
        TimingGuards.RequireOptimizedBuildForTiming();

        var flat = NivaraColumn<int>.Create(Enumerable.Range(0, ChunkCount * ChunkLength).ToArray());
        var chunked = CreateChunked();

        Action flatAdd = () => _ = flat + 1;
        Action chunkedAdd = () => _ = chunked + 1;

        flatAdd();
        chunkedAdd();

        const int rounds = 11;
        var flatMs = new double[rounds];
        var chunkedMs = new double[rounds];
        var stopwatch = new Stopwatch();

        for (int round = 0; round < rounds; round++)
        {
            // Alternate the order so load drift cannot systematically favour one route.
            if ((round & 1) == 0)
            {
                flatMs[round] = Time(flatAdd);
                chunkedMs[round] = Time(chunkedAdd);
            }
            else
            {
                chunkedMs[round] = Time(chunkedAdd);
                flatMs[round] = Time(flatAdd);
            }
        }

        var medianFlat = Median(flatMs);
        var medianChunked = Median(chunkedMs);

        // No timing assertion: the structural claims are the guard, and the one contiguous copy
        // per chunked operand is the recorded design consequence, not a regression.
        TestContext.Out.WriteLine(
            $"Scalar add ({ChunkCount * ChunkLength} ints): flat {medianFlat:F4}ms, " +
            $"8-chunk {medianChunked:F4}ms, ratio {medianChunked / medianFlat:F3} " +
            $"({rounds} interleaved rounds, median of each route)");

        double Time(Action action)
        {
            stopwatch.Restart();
            action();
            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        System.Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return (sorted.Length & 1) == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
}
