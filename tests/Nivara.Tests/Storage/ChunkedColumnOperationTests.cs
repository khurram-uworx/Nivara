using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.Storage;

/// <summary>
/// Exercises the public <see cref="NivaraColumn{T}"/> surface over a chunked backing, which is
/// the evidence that the retired "Unsupported storage type" guards were blocking only interface
/// members. Every operation materialises one contiguous copy per chunked operand.
/// </summary>
[TestFixture]
public class ChunkedColumnOperationTests
{
    static NivaraColumn<int> ChunkedColumn(params int[][] chunkValues)
    {
        var chunks = chunkValues
            .Select(values => new ColumnStorage<int>(values))
            .ToArray();
        return new NivaraColumn<int>(new ChunkedColumn<int>(chunks));
    }

    static NivaraColumn<int> MaskedChunkedColumn(int[] first, int[] second, bool[] secondMask)
    {
        var chunks = new[]
        {
            new ColumnStorage<int>(first),
            new ColumnStorage<int>(new ReadOnlyMemory<int>(second), new ReadOnlyMemory<bool>(secondMask)),
        };
        return new NivaraColumn<int>(new ChunkedColumn<int>(chunks));
    }

    [Test]
    public void ScalarAddition_OverChunkedColumn_ReturnsValues()
    {
        var column = ChunkedColumn([1, 2], [3, 4]);

        var result = column + 10;

        Assert.That(result.ToArray(), Is.EqualTo(new[] { 11, 12, 13, 14 }));
    }

    [Test]
    public void ColumnAddition_WithChunkedOperand_ReturnsValues()
    {
        var left = ChunkedColumn([1, 2], [3, 4]);
        var right = NivaraColumn<int>.Create(new[] { 10, 20, 30, 40 });

        var result = left + right;

        Assert.That(result.ToArray(), Is.EqualTo(new[] { 11, 22, 33, 44 }));
    }

    [Test]
    public void ScalarAddition_OverChunkedColumn_PropagatesNullMask()
    {
        var column = MaskedChunkedColumn([1, 2], [3, 4], [true, false]);

        var result = column + 1;

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(2), Is.True);
        Assert.That(result.IsNull(3), Is.False);
        Assert.That(result[3], Is.EqualTo(5));
    }

    [Test]
    public void ScalarEquality_OverChunkedColumn_ReturnsValues()
    {
        var column = ChunkedColumn([1, 2], [3, 4]);

        var result = column.Equals(3);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { false, false, true, false }));
    }

    [Test]
    public void ColumnEquality_WithChunkedOperand_ReturnsValues()
    {
        var left = ChunkedColumn([1, 2], [3, 4]);
        var right = NivaraColumn<int>.Create(new[] { 1, 0, 3, 0 });

        var result = left.Equals(right);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { true, false, true, false }));
    }

    [Test]
    public void ScalarGreaterThan_OverChunkedColumn_ReturnsValues()
    {
        var column = ChunkedColumn([1, 2], [3, 4]);

        var result = column.GreaterThan(2);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { false, false, true, true }));
    }

    [Test]
    public void ColumnGreaterThan_WithChunkedOperand_ReturnsValues()
    {
        var left = ChunkedColumn([1, 2], [3, 4]);
        var right = NivaraColumn<int>.Create(new[] { 2, 2, 2, 2 });

        var result = left.GreaterThan(right);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { false, false, true, true }));
    }

    [Test]
    public void ScalarLessThan_OverChunkedColumn_ReturnsValues()
    {
        var column = ChunkedColumn([1, 2], [3, 4]);

        var result = column.LessThan(3);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { true, true, false, false }));
    }

    [Test]
    public void ColumnLessThan_WithChunkedOperand_ReturnsValues()
    {
        var left = ChunkedColumn([1, 2], [3, 4]);
        var right = NivaraColumn<int>.Create(new[] { 2, 2, 2, 2 });

        var result = left.LessThan(right);

        Assert.That(result.ToArray(), Is.EqualTo(new[] { true, false, false, false }));
    }

    [Test]
    public void Slice_OverChunkedColumn_PreservesValues()
    {
        var column = ChunkedColumn([1, 2, 3], [4, 5, 6]);

        var slice = column.Slice(2, 3);

        Assert.That(slice.ToArray(), Is.EqualTo(new[] { 3, 4, 5 }));
    }

    [Test]
    public void AsTensorView_OverChunkedColumn_ThrowsNotSupported()
    {
        var column = ChunkedColumn([1, 2], [3, 4]);

        var exception = Assert.Throws<NotSupportedException>(() => column.AsTensorView());

        Assert.That(exception!.Message, Does.Contain(".Data"));
    }
}
