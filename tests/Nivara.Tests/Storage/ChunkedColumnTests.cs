using Nivara.Storage;
using NUnit.Framework;
using System.Runtime.InteropServices;

namespace Nivara.Tests.Storage;

/// <summary>
/// Tests for the chunk-structured storage layer: invariant, cross-boundary access,
/// lazy materialisation, span contract, slicing, and disposal semantics.
/// </summary>
[TestFixture]
public class ChunkedColumnTests
{
    static ColumnStorage<int> Flat(params int[] values) => new(values);

    static ColumnStorage<int> Masked(int[] values, bool[] mask)
        => new(new ReadOnlyMemory<int>(values), new ReadOnlyMemory<bool>(mask));

    static ChunkedColumn<int> Chunked(params ColumnStorage<int>[] chunks) => new(chunks);

    [Test]
    public void Constructor_WithFewerThanTwoChunks_Throws()
    {
        Assert.Throws<ArgumentException>(() => _ = new ChunkedColumn<int>(new[] { Flat(1, 2) }));
    }

    [Test]
    public void Constructor_WithEmptyChunk_Throws()
    {
        Assert.Throws<ArgumentException>(() => _ = new ChunkedColumn<int>(new[] { Flat(1, 2), Flat() }));
    }

    [Test]
    public void FromChunkStorages_SingleChunk_CollapsesToColumnStorage()
    {
        var storage = ChunkedColumn<int>.FromChunkStorages(new[] { Flat(1, 2, 3) });

        Assert.That(storage, Is.TypeOf<ColumnStorage<int>>());
        Assert.That(storage.Length, Is.EqualTo(3));
    }

    [Test]
    public void FromChunkStorages_Empty_CollapsesToEmptyColumnStorage()
    {
        var storage = ChunkedColumn<int>.FromChunkStorages(Array.Empty<ColumnStorage<int>>());

        Assert.That(storage, Is.TypeOf<ColumnStorage<int>>());
        Assert.That(storage.Length, Is.EqualTo(0));
    }

    [Test]
    public void FromChunkStorages_TwoChunks_ProducesChunkedColumn()
    {
        var storage = ChunkedColumn<int>.FromChunkStorages(new[] { Flat(1, 2), Flat(3) });

        Assert.That(storage, Is.TypeOf<ChunkedColumn<int>>());
        Assert.That(storage.Length, Is.EqualTo(3));
    }

    [Test]
    public void Indexer_AcrossChunkBoundaries_ReturnsValues()
    {
        var storage = Chunked(Flat(10, 11, 12), Flat(13, 14));

        Assert.That(Enumerable.Range(0, storage.Length).Select(i => storage[i]),
            Is.EqualTo(new[] { 10, 11, 12, 13, 14 }));
    }

    [Test]
    public void Indexer_OutOfRange_Throws()
    {
        var storage = Chunked(Flat(1), Flat(2));

        Assert.Throws<IndexOutOfRangeException>(() => _ = storage[-1]);
        Assert.Throws<IndexOutOfRangeException>(() => _ = storage[2]);
    }

    [Test]
    public void IsNull_AcrossChunkBoundaries_ReturnsMask()
    {
        var storage = Chunked(Flat(1, 2), Masked([3, 4], [true, false]));

        Assert.That(storage.HasNulls, Is.True);
        Assert.That(storage.NullMask.ToArray(), Is.EqualTo(new[] { false, false, true, false }));
    }

    [Test]
    public void IsNull_AcrossChunkBoundaries_ReturnsNullStatus()
    {
        var storage = Chunked(Flat(1, 2), Masked([3, 4], [true, false]));

        Assert.That(storage.IsNull(0), Is.False);
        Assert.That(storage.IsNull(1), Is.False);
        Assert.That(storage.IsNull(2), Is.True);
        Assert.That(storage.IsNull(3), Is.False);
    }

    [Test]
    public void Data_FirstAccessMaterializes_SecondAccessReturnsSameArray()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.That(storage.IsMaterialized, Is.False);

        var first = storage.Data;
        var second = storage.Data;

        Assert.That(storage.IsMaterialized, Is.True);
        Assert.That(MemoryMarshal.TryGetArray(first, out var firstSegment), Is.True);
        Assert.That(MemoryMarshal.TryGetArray(second, out var secondSegment), Is.True);
        Assert.That(ReferenceEquals(firstSegment.Array, secondSegment.Array), Is.True);
        Assert.That(first.ToArray(), Is.EqualTo(new[] { 1, 2, 3, 4 }));
    }

    [Test]
    public void NullMask_WithoutNulls_ReturnsEmptyWithoutMaterializing()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.That(storage.HasNulls, Is.False);
        Assert.That(storage.NullMask.IsEmpty, Is.True);
        Assert.That(((IColumnStorage<int>)storage).NullMaskMemory, Is.Null);
        Assert.That(storage.IsMaterialized, Is.False);
    }

    [Test]
    public void NullMask_WithNulls_MaterializesContiguousMask()
    {
        var storage = Chunked(Flat(1, 2), Masked([3, 4], [false, true]));

        Assert.That(storage.NullMask.ToArray(), Is.EqualTo(new[] { false, false, false, true }));
        Assert.That(((IColumnStorage<int>)storage).NullMaskMemory!.Value.ToArray(),
            Is.EqualTo(new[] { false, false, false, true }));
    }

    [Test]
    public void TryGetSpan_AlwaysReturnsFalse()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        bool result = ((IColumnStorage<int>)storage).TryGetSpan(out var span);

        Assert.That(result, Is.False);
        Assert.That(span.IsEmpty, Is.True);
    }

    [Test]
    public void AsWritableSpan_Throws()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.Throws<NotSupportedException>(() => _ = ((IColumnStorage<int>)storage).AsWritableSpan());
    }

    [Test]
    public void TryGetHashCodes_ReturnsFalse()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.That(((IColumnStorage<int>)storage).TryGetHashCodes(out var hashes), Is.False);
        Assert.That(hashes.IsEmpty, Is.True);
    }

    [Test]
    public void Slice_WithinSingleChunk_CollapsesToColumnStorage()
    {
        var storage = Chunked(Flat(10, 11, 12), Flat(13, 14, 15));

        var slice = storage.Slice(1, 2);

        Assert.That(slice, Is.TypeOf<ColumnStorage<int>>());
        Assert.That(Enumerable.Range(0, slice.Length).Select(i => slice[i]), Is.EqualTo(new[] { 11, 12 }));
    }

    [Test]
    public void Slice_AcrossChunks_PreservesChunkBoundaries()
    {
        var storage = Chunked(Flat(10, 11, 12), Flat(13, 14, 15));

        var slice = storage.Slice(2, 3);

        Assert.That(slice, Is.TypeOf<ChunkedColumn<int>>());
        Assert.That(((ChunkedColumn<int>)slice).ChunkStorages.Length, Is.EqualTo(2));
        Assert.That(Enumerable.Range(0, slice.Length).Select(i => slice[i]), Is.EqualTo(new[] { 12, 13, 14 }));
    }

    [Test]
    public void Slice_ZeroLength_ReturnsEmptyStorage()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        var slice = storage.Slice(1, 0);

        Assert.That(slice.Length, Is.EqualTo(0));
        Assert.That(slice, Is.TypeOf<ColumnStorage<int>>());
    }

    [Test]
    public void Slice_OutOfBounds_Throws()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Slice(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Slice(0, 5));
    }

    [Test]
    public void Dispose_DoesNotDisposeChunks()
    {
        var left = Flat(1, 2);
        var right = Flat(3, 4);
        var storage = new ChunkedColumn<int>(new[] { left, right });

        storage.Dispose();

        Assert.That(left[0], Is.EqualTo(1));
        Assert.That(right[1], Is.EqualTo(4));
        Assert.Throws<ObjectDisposedException>(() => _ = storage.ChunkStorages);
        Assert.Throws<ObjectDisposedException>(() => _ = storage.Length);
    }

    [Test]
    public void Create_FlattensChunkedAndFlatInputs_PreservingBoundaries()
    {
        var first = Chunked(Flat(1), Flat(2));
        var second = Flat(3);
        var third = Chunked(Flat(4), Flat(5));

        var result = ChunkedColumn<int>.Create(new IColumnStorage<int>[] { first, second, third });

        Assert.That(result, Is.TypeOf<ChunkedColumn<int>>());
        Assert.That(((ChunkedColumn<int>)result).ChunkStorages.Length, Is.EqualTo(5));
        Assert.That(Enumerable.Range(0, result.Length).Select(i => result[i]), Is.EqualTo(new[] { 1, 2, 3, 4, 5 }));
    }

    [Test]
    public void IsVectorizable_MatchesElementType()
    {
        var storage = Chunked(Flat(1, 2), Flat(3, 4));

        Assert.That(storage.IsVectorizable, Is.EqualTo(ColumnStorageFactory.IsVectorizable<int>()));
    }
}
