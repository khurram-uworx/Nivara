namespace Nivara.IO;

/// <summary>
/// Converts between Nivara's byte-wise null mask and Apache Arrow's bit-packed validity buffer.
/// </summary>
/// <remarks>
/// Nivara's mask is one <see cref="bool"/> per slot, and <see langword="true"/> means null (ADR-007).
/// Arrow's validity buffer is one bit per slot, least-significant-bit first, and a set bit means
/// valid. Conversion therefore packs as well as inverts polarity, and it belongs only to the
/// interchange boundary: core storage stays byte-wise and never sees a bitmap.
/// </remarks>
internal static class ArrowValidityMask
{
    /// <summary>
    /// Gets the number of bytes needed to hold a validity bitmap for the given number of slots.
    /// </summary>
    /// <param name="length">The number of slots.</param>
    /// <returns>The byte count, which is zero for a zero-length bitmap.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="length"/> is negative.</exception>
    internal static int GetBitmapLength(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        return length == 0 ? 0 : ((length - 1) >> 3) + 1;
    }

    /// <summary>
    /// Packs a null mask into an Arrow validity bitmap, inverting polarity as it goes.
    /// </summary>
    /// <param name="nullMask">The mask to pack, where <see langword="true"/> means null.</param>
    /// <param name="destination">
    /// The buffer to write. It must hold at least <see cref="GetBitmapLength(int)"/> bytes for
    /// <c>nullMask.Length</c> slots; bytes past that count are left untouched.
    /// </param>
    /// <remarks>
    /// An empty mask has no slot count to derive, so it packs to an empty bitmap. Callers exporting a
    /// column with no nulls should either omit the buffer (Arrow permits that when the null count is
    /// zero) or use the length-taking overload to materialise an all-valid bitmap. Passing an empty
    /// mask with a non-empty destination is rejected rather than left unwritten, because a zero-filled
    /// bitmap reads as all-null.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="nullMask"/> is empty and <paramref name="destination"/> is not, or
    /// when <paramref name="destination"/> is too small.
    /// </exception>
    internal static void ToValidityBitmap(ReadOnlySpan<bool> nullMask, Span<byte> destination)
    {
        if (nullMask.IsEmpty && !destination.IsEmpty)
        {
            throw new ArgumentException(
                "An empty null mask carries no slot count, so it cannot pack into a non-empty destination. Pass the slot count with the length-taking overload, or omit the validity buffer.",
                nameof(nullMask));
        }

        ToValidityBitmap(nullMask, nullMask.Length, destination);
    }

    /// <summary>
    /// Packs a null mask into an Arrow validity bitmap for an explicit slot count, inverting polarity.
    /// </summary>
    /// <param name="nullMask">
    /// The mask to pack, where <see langword="true"/> means null. An empty mask means "every slot is
    /// valid" and writes a fully set bitmap.
    /// </param>
    /// <param name="length">The number of slots to encode.</param>
    /// <param name="destination">
    /// The buffer to write. It must hold at least <see cref="GetBitmapLength(int)"/> bytes for
    /// <paramref name="length"/> slots; bytes past that count are left untouched.
    /// </param>
    /// <remarks>Bits past <paramref name="length"/> in the final byte are always zero.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="length"/> is negative.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="nullMask"/> is non-empty and its length differs from
    /// <paramref name="length"/>, or when <paramref name="destination"/> is too small.
    /// </exception>
    internal static void ToValidityBitmap(ReadOnlySpan<bool> nullMask, int length, Span<byte> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        if (!nullMask.IsEmpty && nullMask.Length != length)
        {
            throw new ArgumentException(
                $"Null mask length {nullMask.Length} does not match the slot count {length}.",
                nameof(nullMask));
        }

        int required = GetBitmapLength(length);

        if (destination.Length < required)
        {
            throw new ArgumentException(
                $"Destination must hold at least {required} bytes for {length} slots, but was {destination.Length}.",
                nameof(destination));
        }

        if (required == 0)
            return;

        var packed = destination[..required];

        if (nullMask.IsEmpty)
        {
            packed.Fill(0xFF);
            ClearPaddingBits(packed, length);
            return;
        }

        packed.Clear();

        int fullBytes = length >> 3;
        for (int b = 0; b < fullBytes; b++)
        {
            int i = b << 3;
            packed[b] = (byte)(
                (nullMask[i] ? 0 : 1) |
                (nullMask[i + 1] ? 0 : 2) |
                (nullMask[i + 2] ? 0 : 4) |
                (nullMask[i + 3] ? 0 : 8) |
                (nullMask[i + 4] ? 0 : 16) |
                (nullMask[i + 5] ? 0 : 32) |
                (nullMask[i + 6] ? 0 : 64) |
                (nullMask[i + 7] ? 0 : 128));
        }

        for (int i = fullBytes << 3; i < length; i++)
        {
            if (!nullMask[i])
                packed[i >> 3] |= (byte)(1 << (i & 7));
        }
    }

    /// <summary>
    /// Packs a null mask into a new Arrow validity bitmap, inverting polarity as it goes.
    /// </summary>
    /// <param name="nullMask">The mask to pack, where <see langword="true"/> means null.</param>
    /// <returns>A new bitmap, empty when <paramref name="nullMask"/> is empty.</returns>
    internal static byte[] ToValidityBitmap(ReadOnlySpan<bool> nullMask)
    {
        if (nullMask.IsEmpty)
            return Array.Empty<byte>();

        var destination = new byte[GetBitmapLength(nullMask.Length)];
        ToValidityBitmap(nullMask, destination);
        return destination;
    }

    /// <summary>
    /// Unpacks an Arrow validity bitmap into a null mask, inverting polarity as it goes.
    /// </summary>
    /// <param name="validityBitmap">
    /// The bitmap to unpack. An empty bitmap means Arrow elided the buffer, which means every slot is
    /// valid.
    /// </param>
    /// <param name="destination">
    /// The mask to fill; its length is the slot count. Every element is written, where
    /// <see langword="true"/> means null.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="validityBitmap"/> is non-empty but too short for
    /// <paramref name="destination"/>.
    /// </exception>
    internal static void ToNullMask(ReadOnlySpan<byte> validityBitmap, Span<bool> destination)
    {
        if (validityBitmap.IsEmpty)
        {
            destination.Clear();
            return;
        }

        int required = GetBitmapLength(destination.Length);

        if (validityBitmap.Length < required)
        {
            throw new ArgumentException(
                $"Validity bitmap must hold at least {required} bytes for {destination.Length} slots, but was {validityBitmap.Length}.",
                nameof(validityBitmap));
        }

        for (int i = 0; i < destination.Length; i++)
            destination[i] = (validityBitmap[i >> 3] & (1 << (i & 7))) == 0;
    }

    /// <summary>
    /// Unpacks an Arrow validity bitmap into a null mask starting at <paramref name="offset"/>,
    /// inverting polarity as it goes. Used for sliced Arrow arrays whose validity buffer is indexed
    /// by the array's offset rather than zero.
    /// </summary>
    /// <param name="validityBitmap">
    /// The bitmap to unpack. An empty bitmap means Arrow elided the buffer, which means every slot is
    /// valid.
    /// </param>
    /// <param name="offset">The bit offset within <paramref name="validityBitmap"/> where the slice begins.</param>
    /// <param name="destination">
    /// The mask to fill; its length is the slot count. Every element is written, where
    /// <see langword="true"/> means null.
    /// </param>
    /// <remarks>
    /// When <paramref name="validityBitmap"/> is empty (elided by Arrow) every slot is treated as
    /// valid regardless of <paramref name="offset"/>. When the buffer is present but trimmed to the
    /// slice (i.e. its length only covers <paramref name="destination"/>), <paramref name="offset"/>
    /// is ignored and unpacking starts at bit zero.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="validityBitmap"/> is non-empty but too short for the requested
    /// slice.
    /// </exception>
    internal static void ToNullMask(ReadOnlySpan<byte> validityBitmap, int offset, Span<bool> destination)
    {
        if (validityBitmap.IsEmpty)
        {
            destination.Clear();
            return;
        }

        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset));

        int length = destination.Length;
        int requiredWithOffset = GetBitmapLength(offset + length);
        int requiredZeroBased = GetBitmapLength(length);

        int baseOffset;
        if (validityBitmap.Length >= requiredWithOffset)
            baseOffset = offset;
        else if (validityBitmap.Length >= requiredZeroBased)
            baseOffset = 0;
        else
        {
            throw new ArgumentException(
                $"Validity bitmap must hold at least {requiredZeroBased} bytes for {length} slots, but was {validityBitmap.Length}.",
                nameof(validityBitmap));
        }

        for (int i = 0; i < length; i++)
        {
            int bit = baseOffset + i;
            destination[i] = (validityBitmap[bit >> 3] & (1 << (bit & 7))) == 0;
        }
    }

    /// <summary>
    /// Unpacks an Arrow validity bitmap into a new null mask starting at <paramref name="offset"/>.
    /// </summary>
    internal static bool[] ToNullMask(ReadOnlySpan<byte> validityBitmap, int offset, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);

        var mask = new bool[length];
        ToNullMask(validityBitmap, offset, mask);
        return mask;
    }

    /// <summary>
    /// Unpacks an Arrow validity bitmap into a new null mask, inverting polarity as it goes.
    /// </summary>
    /// <param name="validityBitmap">
    /// The bitmap to unpack. An empty bitmap means Arrow elided the buffer, which means every slot is
    /// valid.
    /// </param>
    /// <param name="length">The number of slots to decode.</param>
    /// <returns>A new mask of length <paramref name="length"/>, where <see langword="true"/> means null.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="length"/> is negative.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="validityBitmap"/> is non-empty but too short for
    /// <paramref name="length"/>.
    /// </exception>
    internal static bool[] ToNullMask(ReadOnlySpan<byte> validityBitmap, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        var mask = new bool[length];
        ToNullMask(validityBitmap, mask);
        return mask;
    }

    /// <summary>
    /// Zeroes the bits in the final byte that sit past the slot count.
    /// </summary>
    /// <param name="packed">The bitmap whose tail byte is masked.</param>
    /// <param name="length">The number of slots encoded in <paramref name="packed"/>.</param>
    private static void ClearPaddingBits(Span<byte> packed, int length)
    {
        int usedBits = length & 7;

        if (usedBits != 0 && !packed.IsEmpty)
            packed[^1] &= (byte)((1 << usedBits) - 1);
    }
}
