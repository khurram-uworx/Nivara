using Nivara.Memory;
using Nivara.Tensors;
using System;

namespace Nivara;

/// <summary>
/// Factory for creating appropriate storage implementations based on type characteristics
/// </summary>
internal static class ColumnStorageFactory
{
    /// <summary>
    /// Creates storage for the given values, automatically selecting between tensor and memory storage
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values)
    {
        // Prefer tensor-backed storage for vectorizable unmanaged value types.
        if (IsVectorizable<T>() && typeof(T).IsValueType)
        {
            try
            {
                var array = values.ToArray();
                var obj = TensorStorageFactory.TryCreateFromArray(typeof(T), array, null);
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
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T?> values) where T : struct
    {
        // For nullable value types that are vectorizable create tensor-backed storage when possible
        if (IsVectorizable<T>())
        {
            try
            {
                if (values.IsEmpty)
                {
                    var emptyArray = Array.CreateInstance(typeof(T), 0);
                    var objEmpty = TensorStorageFactory.TryCreateFromArray(typeof(T), emptyArray, null);
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
                    var obj = TensorStorageFactory.TryCreateFromArray(typeof(T), dataArray, maybeMask);
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
        return NullableStorageHelper.CreateMemoryStorage(values);
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
}
