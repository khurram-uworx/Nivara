namespace Nivara;

/// <summary>
/// Supports C# collection expressions for <see cref="NivaraSeries{T}"/>.
/// </summary>
public static class NivaraSeriesBuilder
{
    /// <summary>
    /// Creates a series from a collection expression.
    /// </summary>
    public static NivaraSeries<T> Create<T>(ReadOnlySpan<T> values)
        => NivaraSeries<T>.Create(values);
}
