using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Nivara.SimdProbe;

/// <summary>
/// SIMD widen-compute-narrow kernels for BFloat16 and Half.
///
/// <see cref="Vector{BFloat16}"/> / <see cref="Vector{Half}"/> report
/// <c>IsSupported == false</c> on .NET 11, so the BCL TensorPrimitives paths run
/// scalar loops and matmul is ~26x slower than F32. These kernels recover SIMD by
/// loading 16-bit values as <see cref="Vector128{ushort}"/>, widening to float
/// lanes, computing in float, then narrowing back to 16-bit.
///
/// BFloat16 is the top 16 bits of float32, so widen is a pure bit-shift and narrow
/// is a shift-back (truncation, matching the scalar T.CreateChecked path). This
/// needs no hardware-specific intrinsic and is the primary target.
///
/// Half requires a real cross-format conversion. .NET 11 does NOT expose an F16C
/// batch intrinsic (verified against the net11 System.Runtime.Intrinsics surface),
/// so the Half path uses a portable element-wise conversion in the widen/narrow
/// step while still accumulating in float SIMD. Its throughput depends on how the
/// conversion dominates; measured separately from BFloat16.
/// </summary>
internal static class NarrowSimdKernels
{
    // -------------------------------------------------------------------------
    // BFloat16 <-> float: bit layout (lossless, no conversion)
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Half <-> float: portable conversion (no F16C batch intrinsic in .NET 11)
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Dot product (matmul hot path)
    // -------------------------------------------------------------------------

    public static float DotBf16(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
        => Vector128.IsHardwareAccelerated
            ? DotCore(MemoryMarshal.Cast<BFloat16, ushort>(a), MemoryMarshal.Cast<BFloat16, ushort>(b), WidenBf16, WidenBf16Scalar)
            : DotScalarBf16(a, b);

    public static float DotHalf(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
        => Vector128.IsHardwareAccelerated
            ? DotCore(MemoryMarshal.Cast<Half, ushort>(a), MemoryMarshal.Cast<Half, ushort>(b), WidenHalf, WidenHalfScalar)
            : DotScalarHalf(a, b);

    private static float DotCore(
        ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b,
        Func<Vector128<ushort>, (Vector128<float> Lo, Vector128<float> Hi)> widen,
        Func<ushort, float> widenScalar)
    {
        int n = a.Length;
        var accLo = Vector128<float>.Zero;
        var accHi = Vector128<float>.Zero;

        int i = 0, limit = n & ~7;
        for (; i < limit; i += 8)
        {
            var va = Vector128.Create(a.Slice(i, 8));
            var vb = Vector128.Create(b.Slice(i, 8));
            var (aLo, aHi) = widen(va);
            var (bLo, bHi) = widen(vb);
            accLo += aLo * bLo;
            accHi += aHi * bHi;
        }

        float result = Vector128.Sum(accLo) + Vector128.Sum(accHi);
        for (; i < n; i++)
            result += widenScalar(a[i]) * widenScalar(b[i]);
        return result;
    }

    private static float DotScalarBf16(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }

    private static float DotScalarHalf(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }

    // -------------------------------------------------------------------------
    // Element-wise binary ops
    // -------------------------------------------------------------------------

    public static void AddBf16(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => Binary(MemoryMarshal.Cast<BFloat16, ushort>(a), MemoryMarshal.Cast<BFloat16, ushort>(b),
                  MemoryMarshal.Cast<BFloat16, ushort>(dst), WidenBf16, NarrowBf16,
                  WidenBf16Scalar, NarrowBf16Scalar, static (x, y) => x + y, static (x, y) => x + y);

    public static void AddHalf(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => Binary(MemoryMarshal.Cast<Half, ushort>(a), MemoryMarshal.Cast<Half, ushort>(b),
                  MemoryMarshal.Cast<Half, ushort>(dst), WidenHalf, NarrowHalf,
                  WidenHalfScalar, NarrowHalfScalar, static (x, y) => x + y, static (x, y) => x + y);

    public static void MultiplyBf16(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b, Span<BFloat16> dst)
        => Binary(MemoryMarshal.Cast<BFloat16, ushort>(a), MemoryMarshal.Cast<BFloat16, ushort>(b),
                  MemoryMarshal.Cast<BFloat16, ushort>(dst), WidenBf16, NarrowBf16,
                  WidenBf16Scalar, NarrowBf16Scalar, static (x, y) => x * y, static (x, y) => x * y);

    public static void MultiplyHalf(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> dst)
        => Binary(MemoryMarshal.Cast<Half, ushort>(a), MemoryMarshal.Cast<Half, ushort>(b),
                  MemoryMarshal.Cast<Half, ushort>(dst), WidenHalf, NarrowHalf,
                  WidenHalfScalar, NarrowHalfScalar, static (x, y) => x * y, static (x, y) => x * y);

    private static void Binary(
        ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b, Span<ushort> dst,
        Func<Vector128<ushort>, (Vector128<float> Lo, Vector128<float> Hi)> widen,
        Func<Vector128<float>, Vector128<float>, Vector128<ushort>> narrow,
        Func<ushort, float> widenScalar,
        Func<float, ushort> narrowScalar,
        Func<Vector128<float>, Vector128<float>, Vector128<float>> op,
        Func<float, float, float> scalarOp)
    {
        int n = a.Length, i = 0;
        int limit = Vector128.IsHardwareAccelerated ? (n & ~7) : 0;
        for (; i < limit; i += 8)
        {
            var va = Vector128.Create(a.Slice(i, 8));
            var vb = Vector128.Create(b.Slice(i, 8));
            var (aLo, aHi) = widen(va);
            var (bLo, bHi) = widen(vb);
            narrow(op(aLo, bLo), op(aHi, bHi)).CopyTo(dst.Slice(i, 8));
        }
        for (; i < n; i++)
        {
            float x = widenScalar(a[i]);
            float y = widenScalar(b[i]);
            dst[i] = narrowScalar(scalarOp(x, y));
        }
    }

    // -------------------------------------------------------------------------
    // RMSNorm per row (validates the normalize path)
    // -------------------------------------------------------------------------

    public static void RmsNormBf16(ReadOnlySpan<BFloat16> src, Span<BFloat16> dst, int rows, int cols, float eps)
    {
        for (int i = 0; i < rows; i++)
        {
            var row = src.Slice(i * cols, cols);
            var drow = dst.Slice(i * cols, cols);
            float sumSq = 0;
            for (int j = 0; j < cols; j++)
            {
                float f = (float)row[j];
                sumSq += f * f;
            }
            float inv = 1f / MathF.Sqrt(sumSq / cols + eps);
            for (int j = 0; j < cols; j++)
                drow[j] = (BFloat16)((float)row[j] * inv);
        }
    }
}
