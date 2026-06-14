namespace Nivara;

/// <summary>
/// Supports C# collection expressions for <see cref="NivaraColumn{T}"/>.
/// </summary>
public static class NivaraColumnBuilder
{
    /// <summary>
    /// Creates a column from a collection expression.
    /// </summary>
    public static NivaraColumn<T> Create<T>(ReadOnlySpan<T> values)
        => NivaraColumn<T>.Create(values);
}
