using Apache.Arrow;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ArrowValidityMaskTests
{
    [Test]
    public void GetBitmapLength_Zero_ReturnsZero()
    {
        Assert.That(ArrowValidityMask.GetBitmapLength(0), Is.EqualTo(0));
    }

    [Test]
    public void GetBitmapLength_ExactMultipleOfEight_ReturnsExactByteCount()
    {
        Assert.That(ArrowValidityMask.GetBitmapLength(8), Is.EqualTo(1));
        Assert.That(ArrowValidityMask.GetBitmapLength(64), Is.EqualTo(8));
    }

    [Test]
    public void GetBitmapLength_NonMultipleOfEight_RoundsUp()
    {
        Assert.That(ArrowValidityMask.GetBitmapLength(1), Is.EqualTo(1));
        Assert.That(ArrowValidityMask.GetBitmapLength(7), Is.EqualTo(1));
        Assert.That(ArrowValidityMask.GetBitmapLength(9), Is.EqualTo(2));
        Assert.That(ArrowValidityMask.GetBitmapLength(15), Is.EqualTo(2));
    }

    [Test]
    public void GetBitmapLength_IntMaxValue_DoesNotOverflow()
    {
        Assert.That(ArrowValidityMask.GetBitmapLength(int.MaxValue), Is.EqualTo(268435456));
    }

    [Test]
    public void GetBitmapLength_Negative_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ArrowValidityMask.GetBitmapLength(-1));
    }

    [Test]
    public void ToValidityBitmap_EmptyMask_ProducesEmptyBitmap()
    {
        Assert.That(ArrowValidityMask.ToValidityBitmap(ReadOnlySpan<bool>.Empty), Is.Empty);
    }

    [Test]
    public void ToValidityBitmap_AllValid_EveryBitSet()
    {
        bool[] mask = [false, false, false, false, false, false, false, false, false, false];

        Assert.That(ArrowValidityMask.ToValidityBitmap(mask), Is.EqualTo(new byte[] { 0xFF, 0x03 }));
    }

    [Test]
    public void ToValidityBitmap_AllNull_EveryBitClear()
    {
        bool[] mask = [true, true, true, true, true, true, true, true, true, true];

        Assert.That(ArrowValidityMask.ToValidityBitmap(mask), Is.EqualTo(new byte[] { 0x00, 0x00 }));
    }

    [Test]
    public void ToValidityBitmap_MixedMask_PacksLsbFirstAndInvertsPolarity()
    {
        bool[] mask = [true, false, true];

        Assert.That(ArrowValidityMask.ToValidityBitmap(mask), Is.EqualTo(new byte[] { 0x02 }));
    }

    [Test]
    public void ToValidityBitmap_SixteenSlots_WritesOneBitPerSlotInLsbOrder()
    {
        bool[] mask = [false, true, false, true, false, true, false, true, true, false, true, false, true, false, true, false];

        Assert.That(ArrowValidityMask.ToValidityBitmap(mask), Is.EqualTo(new byte[] { 0x55, 0xAA }));
    }

    [Test]
    public void ToValidityBitmap_TenSlots_LeavesPaddingBitsClear()
    {
        bool[] mask = [false, false, false, false, false, false, false, false, false, false];

        var bitmap = ArrowValidityMask.ToValidityBitmap(mask);

        Assert.That(bitmap[^1], Is.EqualTo(0x03));
    }

    [Test]
    public void ToValidityBitmap_EmptyMaskWithNonEmptyDestination_ThrowsArgumentException()
    {
        var destination = new byte[2];

        var ex = Assert.Throws<ArgumentException>(
            () => ArrowValidityMask.ToValidityBitmap(ReadOnlySpan<bool>.Empty, destination));

        Assert.That(ex!.Message, Does.Contain("slot count"));
        Assert.That(destination, Is.EqualTo(new byte[2]), "a rejected call must not write");
    }

    [Test]
    public void ToValidityBitmap_DestinationTooSmall_ThrowsArgumentException()
    {
        bool[] mask = [false, false, false, false, false, false, false, false, false];
        var destination = new byte[1];

        var ex = Assert.Throws<ArgumentException>(() => ArrowValidityMask.ToValidityBitmap(mask, destination));

        Assert.That(ex!.Message, Does.Contain("at least 2 bytes"));
    }

    [Test]
    public void ToValidityBitmap_LargerDestination_LeavesTrailingBytesUntouched()
    {
        bool[] mask = [false, true, false, true, false, true, false, true];
        var destination = new byte[] { 0xAA, 0xBB, 0xCC };

        ArrowValidityMask.ToValidityBitmap(mask, destination);

        Assert.That(destination, Is.EqualTo(new byte[] { 0x55, 0xBB, 0xCC }));
    }

    [Test]
    public void ToValidityBitmap_WithLengthAndEmptyMask_WritesAllValidBitmap()
    {
        var destination = new byte[2];

        ArrowValidityMask.ToValidityBitmap(ReadOnlySpan<bool>.Empty, 10, destination);

        Assert.That(destination, Is.EqualTo(new byte[] { 0xFF, 0x03 }));
    }

    [Test]
    public void ToValidityBitmap_WithLengthZeroAndEmptyMask_WritesNothing()
    {
        var destination = new byte[] { 0xAA };

        ArrowValidityMask.ToValidityBitmap(ReadOnlySpan<bool>.Empty, 0, destination);

        Assert.That(destination, Is.EqualTo(new byte[] { 0xAA }));
    }

    [Test]
    public void ToValidityBitmap_WithLengthMismatch_ThrowsArgumentException()
    {
        bool[] mask = [false, true, false];
        var destination = new byte[1];

        var ex = Assert.Throws<ArgumentException>(() => ArrowValidityMask.ToValidityBitmap(mask, 5, destination));

        Assert.That(ex!.Message, Does.Contain("does not match"));
    }

    [Test]
    public void ToValidityBitmap_WithNegativeLength_ThrowsArgumentOutOfRangeException()
    {
        var destination = new byte[1];

        Assert.Throws<ArgumentOutOfRangeException>(
            () => ArrowValidityMask.ToValidityBitmap(ReadOnlySpan<bool>.Empty, -1, destination));
    }

    [Test]
    public void ToNullMask_EmptyBitmap_TreatsEverySlotAsValid()
    {
        var mask = ArrowValidityMask.ToNullMask(ReadOnlySpan<byte>.Empty, 5);

        Assert.That(mask, Is.EqualTo(new bool[5]));
        Assert.That(mask, Has.All.False);
    }

    [Test]
    public void ToNullMask_AllValidBitmap_ProducesAllFalseMask()
    {
        var mask = ArrowValidityMask.ToNullMask(new byte[] { 0xFF, 0x03 }, 10);

        Assert.That(mask, Has.Length.EqualTo(10));
        Assert.That(mask, Has.All.False);
    }

    [Test]
    public void ToNullMask_AllNullBitmap_ProducesAllTrueMask()
    {
        var mask = ArrowValidityMask.ToNullMask(new byte[] { 0x00, 0x00 }, 10);

        Assert.That(mask, Has.Length.EqualTo(10));
        Assert.That(mask, Has.All.True);
    }

    [Test]
    public void ToNullMask_MixedBitmap_UnpacksLsbFirstAndInvertsPolarity()
    {
        var mask = ArrowValidityMask.ToNullMask(new byte[] { 0x02 }, 3);

        Assert.That(mask, Is.EqualTo(new[] { true, false, true }));
    }

    [Test]
    public void ToNullMask_BitmapTooShort_ThrowsArgumentException()
    {
        var destination = new bool[9];

        var ex = Assert.Throws<ArgumentException>(() => ArrowValidityMask.ToNullMask(new byte[] { 0xFF }, destination));

        Assert.That(ex!.Message, Does.Contain("at least 2 bytes"));
    }

    [Test]
    public void ToNullMask_IgnoresBitsPastLength()
    {
        var mask = ArrowValidityMask.ToNullMask(new byte[] { 0xF8 }, 3);

        Assert.That(mask, Is.EqualTo(new[] { true, true, true }));
    }

    [Test]
    public void ToNullMask_WithNegativeLength_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ArrowValidityMask.ToNullMask(new byte[] { 0xFF }, -1));
    }

    [Test]
    public void ToNullMask_ZeroLength_ReturnsEmptyMask()
    {
        Assert.That(ArrowValidityMask.ToNullMask(ReadOnlySpan<byte>.Empty, 0), Is.Empty);
    }

    [Test]
    public void RoundTrip_MixedMask_IsExact()
    {
        bool[] original = [false, true, true, false, false, true, false, true, true, false, true];

        var bitmap = ArrowValidityMask.ToValidityBitmap(original);
        var roundTripped = ArrowValidityMask.ToNullMask(bitmap, original.Length);

        Assert.That(roundTripped, Is.EqualTo(original));
    }

    [Test]
    public void RoundTrip_EveryLengthOneToSixteen_IsExact()
    {
        for (int length = 1; length <= 16; length++)
        {
            var original = new bool[length];
            for (int i = 0; i < length; i++)
                original[i] = i % 3 == 0;

            var bitmap = ArrowValidityMask.ToValidityBitmap(original);
            var roundTripped = ArrowValidityMask.ToNullMask(bitmap, length);

            Assert.That(roundTripped, Is.EqualTo(original), $"length {length}");
        }
    }

    [Test]
    public void RoundTrip_RandomMasksAcrossManyLengths_IsExact()
    {
        var random = new Random(20261103);

        for (int length = 1; length <= 1000; length++)
        {
            var original = new bool[length];
            for (int i = 0; i < length; i++)
                original[i] = random.Next(2) == 0;

            var bitmap = ArrowValidityMask.ToValidityBitmap(original);
            var roundTripped = ArrowValidityMask.ToNullMask(bitmap, length);

            Assert.That(bitmap, Has.Length.EqualTo(ArrowValidityMask.GetBitmapLength(length)));
            Assert.That(roundTripped, Is.EqualTo(original), $"length {length}");
        }
    }

    [Test]
    public void ToValidityBitmap_MatchesApacheArrowBuilderValidityBuffer()
    {
        bool[] mask = [false, true, true, false, false, true, false, false, true, false];

        var builder = new Int32Array.Builder();
        foreach (bool isNull in mask)
        {
            if (isNull)
                builder.AppendNull();
            else
                builder.Append(7);
        }

        using var array = builder.Build();
        var expected = ArrowValidityMask.ToValidityBitmap(mask);
        var actual = array.NullBitmapBuffer.Span[..expected.Length].ToArray();

        Assert.That(array.NullCount, Is.EqualTo(4));
        Assert.That(actual, Is.EqualTo(expected));
    }
}
