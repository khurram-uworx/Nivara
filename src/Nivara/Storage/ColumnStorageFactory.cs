namespace Nivara.Storage;

/// <summary>
/// Factory for creating <see cref="ColumnStorage{T}"/> instances. All columns use the
/// single unified storage class; kernel selection is decided separately by
/// <see cref="KernelSelector"/> heuristics, never by the storage implementation.
/// </summary>
static class ColumnStorageFactory
{
    /// <summary>
    /// Creates storage for the given values. For reference types a null mask is detected
    /// and tracked so null positions remain authoritative.
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <returns>A storage instance</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values)
    {
        return new ColumnStorage<T>(values, detectNulls: !typeof(T).IsValueType);
    }

    /// <summary>
    /// Creates storage for the given values with an explicit null mask.
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <param name="nullMask">Optional null mask indicating which positions are null</param>
    /// <returns>A storage instance with the given null mask</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values, ReadOnlyMemory<bool>? nullMask)
    {
        return new ColumnStorage<T>(values.ToArray().AsMemory(), nullMask);
    }

    /// <summary>
    /// Creates storage from an array owned by the caller. The caller must not mutate
    /// the array after this call (zero-copy wrap).
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The owned array of values</param>
    /// <param name="nullMask">Optional null mask indicating which positions are null</param>
    /// <returns>A storage instance wrapping the caller's array</returns>
    internal static IColumnStorage<T> CreateFromOwnedArray<T>(T[] values, ReadOnlyMemory<bool>? nullMask = null)
    {
        ArgumentNullException.ThrowIfNull(values);

        return new ColumnStorage<T>(values, nullMask);
    }

    /// <summary>
    /// Creates storage for nullable value types, preserving null positions as a null mask.
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values to store</param>
    /// <returns>A storage instance with null positions tracked</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T?> values) where T : struct
    {
        if (values.IsEmpty)
        {
            return new ColumnStorage<T>(ReadOnlySpan<T>.Empty);
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
                dataArray[i] = default(T);
                nullMaskArray[i] = true;
                hasNulls = true;
            }
        }

        var nullMask = hasNulls ? new ReadOnlyMemory<bool>(nullMaskArray) : null;

        return new ColumnStorage<T>(dataArray, nullMask);
    }

    /// <summary>
    /// Determines if a type supports vectorized operations
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type supports vectorization, false otherwise</returns>
    public static bool IsVectorizable<T>() => IsVectorizable(typeof(T));

    /// <summary>
    /// Determines if a type supports vectorized operations
    /// </summary>
    /// <param name="type">The type to check</param>
    /// <returns>True if the type supports vectorization, false otherwise</returns>
    public static bool IsVectorizable(Type type)
    {
        // Check for specific vectorizable numeric types
        return type == typeof(int) ||
               type == typeof(float) ||
               type == typeof(double) ||
               type == typeof(long) ||
               type == typeof(short) ||
               type == typeof(byte) ||
               type == typeof(uint) ||
               type == typeof(ulong) ||
               type == typeof(ushort) ||
               type == typeof(sbyte) ||
               type == typeof(char) ||
               type == typeof(bool);
    }
}
