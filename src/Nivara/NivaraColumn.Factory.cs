using Nivara.Storage;

namespace Nivara;

/// <summary>
/// Non-generic factory entry points for <see cref="NivaraColumn{T}"/>. The generic
/// <see cref="CreateFromNullable{T}(T?[])"/> overload builds columns from nullable value-type
/// arrays without boxing each element.
/// </summary>
public static class NivaraColumn
{
    /// <summary>
    /// Creates a new column from a nullable value-type array without boxing each element.
    /// </summary>
    /// <typeparam name="T">The underlying value type</typeparam>
    /// <param name="values">The nullable value type values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values array is null</exception>
    public static NivaraColumn<T> CreateFromNullable<T>(T?[] values)
        where T : struct
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Length == 0)
        {
            return new NivaraColumn<T>(new ColumnStorage<T>(ReadOnlySpan<T>.Empty));
        }

        var dataArray = new T[values.Length];
        var nullMaskArray = new bool[values.Length];
        bool hasNulls = false;

        for (int i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (value.HasValue)
            {
                dataArray[i] = value.Value;
            }
            else
            {
                dataArray[i] = default;
                nullMaskArray[i] = true;
                hasNulls = true;
            }
        }

        var data = new ReadOnlyMemory<T>(dataArray);
        var nullMask = hasNulls ? new ReadOnlyMemory<bool>(nullMaskArray) : null;
        return new NivaraColumn<T>(new ColumnStorage<T>(data, nullMask));
    }
}
