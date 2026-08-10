using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Linq;

/// <summary>
/// Entry points for the typed LINQ query layer over <see cref="NivaraFrame"/>.
/// </summary>
public static class NivaraTypedLinqExtensions
{
    /// <summary>
    /// Begins a typed LINQ query over the frame
    /// </summary>
    /// <typeparam name="T">The row type. Must be a non-primitive class whose public properties map
    /// (case-insensitively) to frame columns with exact or nullable-compatible types.</typeparam>
    /// <param name="frame">The source frame</param>
    /// <returns>A lazy typed query</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when the row type does not map to the frame schema</exception>
    public static NivaraQuery<T> Query<T>(this NivaraFrame frame)
        where T : class, new()
    {
        ArgumentNullException.ThrowIfNull(frame);

        TypedLinqMetadata.ValidateRowType(typeof(T), frame.Schema);
        return new NivaraQuery<T>(frame.AsQueryFrame());
    }

    /// <summary>
    /// Creates a typed query over an existing lazy query frame, validating the row type against the
    /// frame's inferred schema. Used by the lazy file-source entries in <c>Nivara.IO.Json</c> and
    /// <c>Nivara.Extensions.IO.Csv</c> (friend-accessible).
    /// </summary>
    internal static NivaraQuery<T> FromFrame<T>(QueryFrame frame)
        where T : class, new()
    {
        ArgumentNullException.ThrowIfNull(frame);

        TypedLinqMetadata.ValidateRowType(typeof(T), frame.Schema);
        return new NivaraQuery<T>(frame);
    }
}
