using Nivara.Helpers;
using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.Storage;

/// <summary>
/// Concatenation preserves existing chunk boundaries without copying, and keeps the flat
/// copy path byte-for-byte when every input is flat.
/// </summary>
[TestFixture]
public class ChunkedConcatTests
{
    static NivaraColumn<int> Chunked(params int[][] chunkValues)
    {
        var chunks = chunkValues
            .Select(values => new ColumnStorage<int>(values))
            .ToArray();
        return new NivaraColumn<int>(new ChunkedColumn<int>(chunks));
    }

    [Test]
    public void Concatenate_FlatPlusFlat_KeepsFlatStorage()
    {
        var left = NivaraColumn<int>.Create(new[] { 1, 2 });
        var right = NivaraColumn<int>.Create(new[] { 3, 4 });

        var result = ColumnFilterHelper.ConcatenateColumns([left, right]);

        Assert.That(result, Is.InstanceOf<NivaraColumn<int>>());
        Assert.That(((NivaraColumn<int>)result).Storage, Is.InstanceOf<ColumnStorage<int>>());
        Assert.That(((NivaraColumn<int>)result).ToArray(), Is.EqualTo(new[] { 1, 2, 3, 4 }));
    }

    [Test]
    public void Concatenate_ChunkedPlusFlat_PreservesChunkBoundaries()
    {
        var left = Chunked([1, 2], [3, 4]);
        var right = NivaraColumn<int>.Create(new[] { 5, 6 });

        var result = (NivaraColumn<int>)ColumnFilterHelper.ConcatenateColumns([left, right]);

        Assert.That(result.Storage, Is.InstanceOf<ColumnStorage<int>>());
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 1, 2, 3, 4, 5, 6 }));
    }

    [Test]
    public void Concatenate_ChunkedPlusChunked_SumsChunkCounts()
    {
        var left = Chunked([1, 2], [3]);
        var right = Chunked([4], [5, 6]);

        var result = (NivaraColumn<int>)ColumnFilterHelper.ConcatenateColumns([left, right]);

        Assert.That(result.Storage, Is.InstanceOf<ColumnStorage<int>>());
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 1, 2, 3, 4, 5, 6 }));
    }

    [Test]
    public void Concatenate_ChunkedWithNulls_PreservesNullMask()
    {
        var masked = new ColumnStorage<int>(
            new ReadOnlyMemory<int>(new[] { 1, 2 }),
            new ReadOnlyMemory<bool>(new[] { false, true }));
        var chunked = new NivaraColumn<int>(new ChunkedColumn<int>([new ColumnStorage<int>(new[] { 0 }), masked]));
        var flat = NivaraColumn<int>.Create(new[] { 3 });

        var result = (NivaraColumn<int>)ColumnFilterHelper.ConcatenateColumns([chunked, flat]);

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(0), Is.False);
        Assert.That(result.IsNull(1), Is.False);
        Assert.That(result.IsNull(2), Is.True);
        Assert.That(result[0], Is.EqualTo(0));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result[3], Is.EqualTo(3));
    }

    [Test]
    public void Concatenate_ChunkedPlusEmptyFlat_DropsEmptyChunk()
    {
        var left = Chunked([1, 2], [3, 4]);
        var empty = NivaraColumn<int>.Create(Array.Empty<int>());

        var result = (NivaraColumn<int>)ColumnFilterHelper.ConcatenateColumns([left, empty]);

        var chunked = (ColumnStorage<int>)result.Storage;
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 1, 2, 3, 4 }));
    }
}
