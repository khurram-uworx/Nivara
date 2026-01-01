namespace Nivara.Memory;

/// <summary>
/// Factory for creating appropriate storage implementations based on type characteristics
/// </summary>
internal static class ColumnStorageFactory
{
    /// <summary>
    /// Helper method for creating storage with nullable value types
    /// Creates tensor storage for nullable value types
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values</param>
    /// <returns>A tensor storage instance</returns>
    static TensorStorage<T> createTensorStorage<T>(ReadOnlySpan<T?> values) where T : unmanaged
    {
        if (values.IsEmpty)
        {
            return new TensorStorage<T>(Array.Empty<T>());
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
                nullMaskArray[i] = false;
            }
            else
            {
                dataArray[i] = default(T);
                nullMaskArray[i] = true;
                hasNulls = true;
            }
        }

        var nullMask = hasNulls ? nullMaskArray : null;

        return new TensorStorage<T>(dataArray, nullMask);
    }

    /// <summary>
    /// Helper method for creating storage with nullable value types
    /// Creates memory storage for nullable value types
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values</param>
    /// <returns>A memory storage instance</returns>
    static MemoryStorage<T> createMemoryStorage<T>(ReadOnlySpan<T?> values) where T : struct
    {
        if (values.IsEmpty)
        {
            return new MemoryStorage<T>(ReadOnlySpan<T>.Empty);
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
                nullMaskArray[i] = false;
            }
            else
            {
                dataArray[i] = default(T);
                nullMaskArray[i] = true;
                hasNulls = true;
            }
        }

        var data = new ReadOnlyMemory<T>(dataArray);
        var nullMask = hasNulls ? new ReadOnlyMemory<bool>(nullMaskArray) : null;

        return new MemoryStorage<T>(data, nullMask);
    }

    /// <summary>
    /// Creates storage for the given values, automatically selecting between tensor and memory storage
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values)
    {
        // For now, use memory storage for all types
        // Tensor storage optimization will be implemented in a future iteration
        // when we can properly handle the generic constraints
        return new MemoryStorage<T>(values, detectNulls: !typeof(T).IsValueType);
    }

    /// <summary>
    /// Creates storage for nullable values, automatically selecting between tensor and memory storage
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The nullable values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    static IColumnStorage<T> Create<T>(ReadOnlySpan<T?> values) where T : struct
    {
        return createMemoryStorage(values);
    }

    /// <summary>
    /// Determines if a type supports vectorized operations
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type supports vectorization, false otherwise</returns>
    public static bool IsVectorizable<T>()
    {
        var type = typeof(T);

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
               type == typeof(bool);
    }

    /// <summary>
    /// Creates appropriate storage for the given values based on type characteristics
    /// </summary>
    /// <param name="values">The values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    public static IColumnStorage<T> CreateStorage<T>(ReadOnlySpan<T> values)
    {
        // Use the factory to determine appropriate storage
        if (ColumnStorageFactory.IsVectorizable<T>() && typeof(T).IsValueType)
        {
            // For vectorizable value types, we'll use tensor storage when available
            // For now, the factory returns memory storage, but this will be optimized later
            return ColumnStorageFactory.Create(values);
        }
        else if (typeof(T).IsClass)
        {
            // For reference types, always detect nulls using MemoryStorage directly
            return new MemoryStorage<T>(values, detectNulls: true);
        }
        else
        {
            // For non-vectorizable value types, use memory storage
            return ColumnStorageFactory.Create(values);
        }
    }

    /// <summary>
    /// Creates appropriate storage for nullable value types (T?)
    /// </summary>
    /// <typeparam name="T">The underlying value type</typeparam>
    /// <param name="values">The nullable values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    public static IColumnStorage<T> CreateStorage<T>(ReadOnlySpan<T?> values) where T : struct
    {
        // Currently prefer MemoryStorage for nullable value types so that null masks
        // are correctly created and preserved. This satisfies current tests and ensures
        // correct null tracking. Optimization to route to TensorStorage for unmanaged
        // vectorizable types may be added later when generic constraints allow it.
        return createMemoryStorage(values);
    }
}
