using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Helpers;

/// <summary>
/// Zero-copy span reinterpretation helpers used to dispatch unconstrained generic column
/// spans to concrete numeric kernels after a runtime <c>typeof(T)</c> check. The source
/// element type is unconstrained, so reinterpretation is only valid when the runtime
/// element type equals <typeparamref name="TTo"/> (callers guarantee this).
/// </summary>
static class SpanReinterpret
{
    public static ReadOnlySpan<TTo> ReadOnly<TFrom, TTo>(ReadOnlySpan<TFrom> source)
        where TTo : unmanaged
        => MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<TFrom, TTo>(ref MemoryMarshal.GetReference(source)), source.Length);

    public static Span<TTo> Writable<TFrom, TTo>(Span<TFrom> destination)
        where TTo : unmanaged
        => MemoryMarshal.CreateSpan(ref Unsafe.As<TFrom, TTo>(ref MemoryMarshal.GetReference(destination)), destination.Length);

    public static TTo Scalar<TFrom, TTo>(TFrom value)
        where TTo : unmanaged
        => Unsafe.As<TFrom, TTo>(ref value);
}
