using System.Reflection;

namespace Nivara.Memory;

/// <summary>
/// Factory for creating appropriate storage implementations based on type characteristics
/// </summary>
internal static class ColumnStorageFactory
{
    /// <summary>
    /// Try to create a TensorStorage&lt;T&gt; instance using the provided element type and data array.
    /// Returns null when construction fails.
    /// </summary>
    static object? tryCreateFromArray(Type elementType, Array dataArray, bool[]? nullMask = null)
    {
        if (elementType == null) throw new ArgumentNullException(nameof(elementType));
        if (dataArray == null) throw new ArgumentNullException(nameof(dataArray));

        try
        {
            var tensorType = typeof(TensorStorage<>).MakeGenericType(elementType);

            // Prefer ctor (T[] data, bool[]? nullMask) then fallback to (T[] data)
            var ctor = tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { dataArray.GetType(), typeof(bool[]) }, null)
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { dataArray.GetType() }, null)
                       // Fallback when exact runtime array type differs (e.g., elementType[])
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { elementType.MakeArrayType(), typeof(bool[]) }, null)
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { elementType.MakeArrayType() }, null);

            if (ctor == null)
                return null;

            object? instance;
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2)
                instance = ctor.Invoke(new object?[] { dataArray, nullMask });
            else
                instance = ctor.Invoke(new object?[] { dataArray });

            return instance;
        }
        catch
        {
            return null;
        }
    }

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
        // Prefer tensor-backed storage for vectorizable unmanaged value types.
        if (IsVectorizable<T>() && typeof(T).IsValueType)
        {
            try
            {
                var array = values.ToArray();
                var obj = tryCreateFromArray(typeof(T), array, null);
                if (obj is IColumnStorage<T> storage)
                    return storage;
            }
            catch
            {
                // fall through to MemoryStorage
            }
        }

        // For reference types, non-vectorizable value types, or on failure above use MemoryStorage.
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
        // For nullable value types that are vectorizable create tensor-backed storage when possible
        if (IsVectorizable<T>())
        {
            try
            {
                if (values.IsEmpty)
                {
                    var emptyArray = Array.CreateInstance(typeof(T), 0);
                    var objEmpty = tryCreateFromArray(typeof(T), emptyArray, null);
                    if (objEmpty is IColumnStorage<T> stEmpty)
                        return stEmpty;
                }
                else
                {
                    var dataArray = new T[values.Length];
                    var nullMask = new bool[values.Length];
                    bool hasNulls = false;

                    for (int i = 0; i < values.Length; i++)
                    {
                        var v = values[i];
                        if (v.HasValue)
                        {
                            dataArray[i] = v.GetValueOrDefault();
                            nullMask[i] = false;
                        }
                        else
                        {
                            dataArray[i] = default!;
                            nullMask[i] = true;
                            hasNulls = true;
                        }
                    }

                    bool[]? maybeMask = hasNulls ? nullMask : null;
                    var obj = tryCreateFromArray(typeof(T), dataArray, maybeMask);
                    if (obj is IColumnStorage<T> st)
                        return st;
                }
            }
            catch
            {
                // fall back to memory path
            }
        }

        // Fallback to existing memory-backed helper
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
