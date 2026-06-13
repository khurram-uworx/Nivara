using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Extensions;

internal static class TensorPrimitiveExtensions
{
    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, Action<ReadOnlySpan<T>, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        int n = col.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            op(span, buf.AsSpan(0, n));
            return NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> left, NivaraColumn<T> right, Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>> op)
        where T : struct, INumber<T>
    {
        left.TryGetSpan(out var lSpan);
        right.TryGetSpan(out var rSpan);
        int n = left.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            op(lSpan, rSpan, buf.AsSpan(0, n));
            return NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, T scalar, Action<ReadOnlySpan<T>, T, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        int n = col.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            op(span, scalar, buf.AsSpan(0, n));
            return NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, T min, T max, Action<ReadOnlySpan<T>, T, T, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        int n = col.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            op(span, min, max, buf.AsSpan(0, n));
            return NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    public static ReadOnlySpan<T> AsSpan<T>(this NivaraColumn<T> col)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        return span;
    }
}
