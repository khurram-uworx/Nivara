using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Extensions;

internal static class TensorPrimitiveExtensions
{
    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, Action<ReadOnlySpan<T>, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        var result = new T[col.Length];
        op(span, result.AsSpan());
        return NivaraColumn<T>.Create(result);
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> left, NivaraColumn<T> right, Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>> op)
        where T : struct, INumber<T>
    {
        left.TryGetSpan(out var lSpan);
        right.TryGetSpan(out var rSpan);
        var result = new T[left.Length];
        op(lSpan, rSpan, result.AsSpan());
        return NivaraColumn<T>.Create(result);
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, T scalar, Action<ReadOnlySpan<T>, T, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        var result = new T[col.Length];
        op(span, scalar, result.AsSpan());
        return NivaraColumn<T>.Create(result);
    }

    public static NivaraColumn<T> Apply<T>(this NivaraColumn<T> col, T min, T max, Action<ReadOnlySpan<T>, T, T, Span<T>> op)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        var result = new T[col.Length];
        op(span, min, max, result.AsSpan());
        return NivaraColumn<T>.Create(result);
    }

    public static ReadOnlySpan<T> AsSpan<T>(this NivaraColumn<T> col)
        where T : struct, INumber<T>
    {
        col.TryGetSpan(out var span);
        return span;
    }
}
