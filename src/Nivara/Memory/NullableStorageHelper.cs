using Nivara.Tensors;
using System.Numerics.Tensors;

namespace Nivara.Memory;

/// <summary>
/// Helper class for creating storage with nullable value types
/// </summary>
internal static class NullableStorageHelper
{
    /// <summary>
    /// Creates tensor storage for nullable value types
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values</param>
    /// <returns>A tensor storage instance</returns>
    public static TensorStorage<T> CreateTensorStorage<T>(ReadOnlySpan<T?> values) where T : unmanaged
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

        return new TensorStorage<T>(dataArray, hasNulls ? Tensor.Create(nullMaskArray, [values.Length]) : null);
    }

    /// <summary>
    /// Creates tensor storage for nullable value types from array
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values array</param>
    /// <returns>A tensor storage instance</returns>
    public static TensorStorage<T> CreateTensorStorage<T>(T?[] values) where T : unmanaged
    {
        return CreateTensorStorage(values.AsSpan());
    }

    /// <summary>
    /// Creates memory storage for nullable value types
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values</param>
    /// <returns>A memory storage instance</returns>
    public static MemoryStorage<T> CreateMemoryStorage<T>(ReadOnlySpan<T?> values) where T : struct
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
}