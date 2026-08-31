using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Nivara.Primitives;

/// <summary>
/// Concrete, non-generic SIMD kernels for the narrow 16-bit float types
/// (<see cref="BFloat16"/> / <see cref="Half"/>).
/// </summary>
/// <remarks>
/// <see cref="Vector{BFloat16}"/> / <see cref="Vector{Half}"/> report
/// <c>IsSupported == false</c> on .NET 11, so the BCL <c>TensorPrimitives</c>
/// paths run scalar loops for these types. These kernels recover SIMD by loading
/// 16-bit values as <see cref="Vector128{ushort}"/>, widening to float lanes,
/// computing in float, then narrowing back to 16-bit — the widen-compute-narrow
/// strategy validated by the Nivara.SimdProbe probe.
///
/// BFloat16 is the top 16 bits of float32, so widen is a pure bit-shift and
/// narrow is a shift-back (truncation, matching the scalar creation path) — no
/// hardware-specific intrinsic. Half needs a real cross-format conversion; .NET
/// 11 does not expose an F16C batch intrinsic, so the Half path uses a portable
/// element-wise conversion in the widen/narrow step while still accumulating in
/// float SIMD.
///
/// Each operation is written as a fully-specialized per-type kernel with the
/// widen/narrow calls inlined directly (no delegate indirection in the hot
/// loop), so the JIT can specialize each (type, op) combination and inline the
/// <see cref="Vector128"/> primitives.
///
/// These are <c>internal</c>: callers reach them through
/// <see cref="WidenPrimitives"/>, which applies the length gate and toggle.
/// </remarks>
internal static class NarrowFloatKernels
{
    // ─────────────────────────────────────────────────────────────────────────
    // BFloat16 <-> float: bit layout (lossless, no conversion)
    // ─────────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (Vector128<float> Lo, Vector128<float> Hi) WidenBf16(Vector128<ushort> v)
    {
        (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(v);
        return (Vector128.ShiftLeft(lo, 16).AsSingle(), Vector128.ShiftLeft(hi, 16).AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<ushort> NarrowBf16(Vector128<float> lo, Vector128<float> hi)
    {
        var loU = Vector128.ShiftRightLogical(lo.AsUInt32(), 16);
        var hiU = Vector128.ShiftRightLogical(hi.AsUInt32(), 16);
        return Vector128.Narrow(loU, hiU);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float WidenBf16Scalar(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ushort NarrowBf16Scalar(float f) => (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);

    // ─────────────────────────────────────────────────────────────────────────
    // Half <-> float: portable conversion (no F16C batch intrinsic in .NET 11)
    // ─────────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (Vector128<float> Lo, Vector128<float> Hi) WidenHalf(Vector128<ushort> v)
        => (WidenHalfPortableLo(v), WidenHalfPortableHi(v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> WidenHalfPortableLo(Vector128<ushort> v)
        => Vector128.Create((float)BitConverter.UInt16BitsToHalf(v.GetElement(0)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(1)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(2)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(3)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> WidenHalfPortableHi(Vector128<ushort> v)
        => Vector128.Create((float)BitConverter.UInt16BitsToHalf(v.GetElement(4)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(5)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(6)),
                            (float)BitConverter.UInt16BitsToHalf(v.GetElement(7)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<ushort> NarrowHalf(Vector128<float> lo, Vector128<float> hi)
    {
        Span<ushort> bits = stackalloc ushort[8];
        for (int i = 0; i < 4; i++) bits[i] = BitConverter.HalfToUInt16Bits((Half)lo.GetElement(i));
        for (int i = 0; i < 4; i++) bits[4 + i] = BitConverter.HalfToUInt16Bits((Half)hi.GetElement(i));
        return Vector128.Create(bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float WidenHalfScalar(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ushort NarrowHalfScalar(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    // ─────────────────────────────────────────────────────────────────────────
    // Dot product (matmul hot path)
    // ─────────────────────────────────────────────────────────────────────────

    internal static BFloat16 Dot(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
        => (BFloat16)(Vector128.IsHardwareAccelerated ? DotBf16Core(a, b) : DotScalar(a, b));

    internal static Half Dot(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
        => (Half)(Vector128.IsHardwareAccelerated ? DotHalfCore(a, b) : DotScalar(a, b));

    private static float DotBf16Core(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        var au = MemoryMarshal.Cast<BFloat16, ushort>(a);
        var bu = MemoryMarshal.Cast<BFloat16, ushort>(b);
        int n = au.Length;
        var accLo = Vector128<float>.Zero;
        var accHi = Vector128<float>.Zero;

        int i = 0, limit = n & ~7;
        for (; i < limit; i += 8)
        {
            var (aLo, aHi) = WidenBf16(Vector128.Create(au.Slice(i, 8)));
            var (bLo, bHi) = WidenBf16(Vector128.Create(bu.Slice(i, 8)));
            accLo += aLo * bLo;
            accHi += aHi * bHi;
        }

        float result = Vector128.Sum(accLo) + Vector128.Sum(accHi);
        for (; i < n; i++)
            result += WidenBf16Scalar(au[i]) * WidenBf16Scalar(bu[i]);
        return result;
    }

    private static float DotHalfCore(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
    {
        var au = MemoryMarshal.Cast<Half, ushort>(a);
        var bu = MemoryMarshal.Cast<Half, ushort>(b);
        int n = au.Length;
        var accLo = Vector128<float>.Zero;
        var accHi = Vector128<float>.Zero;

        int i = 0, limit = n & ~7;
        for (; i < limit; i += 8)
        {
            var (aLo, aHi) = WidenHalf(Vector128.Create(au.Slice(i, 8)));
            var (bLo, bHi) = WidenHalf(Vector128.Create(bu.Slice(i, 8)));
            accLo += aLo * bLo;
            accHi += aHi * bHi;
        }

        float result = Vector128.Sum(accLo) + Vector128.Sum(accHi);
        for (; i < n; i++)
            result += WidenHalfScalar(au[i]) * WidenHalfScalar(bu[i]);
        return result;
    }

    private static float DotScalar(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }

    private static float DotScalar(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Element-wise binary ops (fully specialized per type and op)
    // ─────────────────────────────────────────────────────────────────────────

    internal static void Add(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => BinaryBf16(a, b, dst, static (x, y) => x + y, static (x, y) => x + y);

    internal static void Subtract(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => BinaryBf16(a, b, dst, static (x, y) => x - y, static (x, y) => x - y);

    internal static void Multiply(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => BinaryBf16(a, b, dst, static (x, y) => x * y, static (x, y) => x * y);

    internal static void Divide(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => BinaryBf16(a, b, dst, static (x, y) => x / y, static (x, y) => x / y);

    internal static void Add(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => BinaryHalf(a, b, dst, static (x, y) => x + y, static (x, y) => x + y);

    internal static void Subtract(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => BinaryHalf(a, b, dst, static (x, y) => x - y, static (x, y) => x - y);

    internal static void Multiply(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => BinaryHalf(a, b, dst, static (x, y) => x * y, static (x, y) => x * y);

    internal static void Divide(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => BinaryHalf(a, b, dst, static (x, y) => x / y, static (x, y) => x / y);

    private static void BinaryBf16(
        ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst,
        Func<Vector128<float>, Vector128<float>, Vector128<float>> op,
        Func<float, float, float> scalarOp)
    {
        var au = MemoryMarshal.Cast<BFloat16, ushort>(a);
        var bu = MemoryMarshal.Cast<BFloat16, ushort>(b);
        var du = MemoryMarshal.Cast<BFloat16, ushort>(dst);

        int n = au.Length, i = 0;
        int limit = Vector128.IsHardwareAccelerated ? (n & ~7) : 0;
        for (; i < limit; i += 8)
        {
            var (aLo, aHi) = WidenBf16(Vector128.Create(au.Slice(i, 8)));
            var (bLo, bHi) = WidenBf16(Vector128.Create(bu.Slice(i, 8)));
            NarrowBf16(op(aLo, bLo), op(aHi, bHi)).CopyTo(du.Slice(i, 8));
        }
        for (; i < n; i++)
            du[i] = NarrowBf16Scalar(scalarOp(WidenBf16Scalar(au[i]), WidenBf16Scalar(bu[i])));
    }

    private static void BinaryHalf(
        ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst,
        Func<Vector128<float>, Vector128<float>, Vector128<float>> op,
        Func<float, float, float> scalarOp)
    {
        var au = MemoryMarshal.Cast<Half, ushort>(a);
        var bu = MemoryMarshal.Cast<Half, ushort>(b);
        var du = MemoryMarshal.Cast<Half, ushort>(dst);

        int n = au.Length, i = 0;
        int limit = Vector128.IsHardwareAccelerated ? (n & ~7) : 0;
        for (; i < limit; i += 8)
        {
            var (aLo, aHi) = WidenHalf(Vector128.Create(au.Slice(i, 8)));
            var (bLo, bHi) = WidenHalf(Vector128.Create(bu.Slice(i, 8)));
            NarrowHalf(op(aLo, bLo), op(aHi, bHi)).CopyTo(du.Slice(i, 8));
        }
        for (; i < n; i++)
            du[i] = NarrowHalfScalar(scalarOp(WidenHalfScalar(au[i]), WidenHalfScalar(bu[i])));
    }
}
