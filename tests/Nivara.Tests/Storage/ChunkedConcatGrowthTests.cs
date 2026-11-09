using Nivara.Helpers;
using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.Storage;

/// <summary>
/// Growth control for #1077: repeated concat of a chunked accumulator must not add
/// one chunk per call; small adjacent chunks coalesce up to the 10,000-row default.
/// </summary>
[TestFixture]
public class ChunkedConcatGrowthTests
{
    [Test]
    public void SequentialConcat_ChunkedAccumulator_KeepsChunkCountBoundedAndValuesCorrect()
    {
        var first = Enumerable.Range(0, 6_000).ToArray();
        var second = Enumerable.Range(6_000, 6_000).ToArray();
        var accumulator = new NivaraColumn<int>(new ChunkedColumn<int>(new[]
        {
            new ColumnStorage<int>(first),
            new ColumnStorage<int>(second)
        }));

        const int iterations = 100;
        for (int i = 0; i < iterations; i++)
        {
            var slice = NivaraColumn<int>.Create(new[] { 12_000 + i });
            accumulator = (NivaraColumn<int>)ColumnFilterHelper.ConcatenateColumns(new List<IColumn> { accumulator, slice });
        }

        var chunked = accumulator.Storage;
        int chunkCount = chunked is ChunkedColumn<int> c ? c.ChunkStorages.Length : 1;
        Assert.That(chunkCount, Is.LessThanOrEqualTo(4),
            "chunk count must stay bounded under repeated concat");

        Assert.That(accumulator.Length, Is.EqualTo(12_000 + iterations));
        var expected = Enumerable.Range(0, 12_000 + iterations).ToArray();
        Assert.That(accumulator.ToArray(), Is.EqualTo(expected));
    }
}
