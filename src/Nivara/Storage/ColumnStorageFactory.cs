using System.Numerics.Tensors;

namespace Nivara.Storage;

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

        return new TensorStorage<T>(dataArray, hasNulls ? Tensor.Create(nullMaskArray, [values.Length]) : null);
    }

    /// <summary>
    /// Helper method for creating storage with nullable value types
    /// Creates tensor storage for nullable value types from array
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values array</param>
    /// <returns>A tensor storage instance</returns>
    static TensorStorage<T> createTensorStorage<T>(T?[] values) where T : unmanaged
    {
        return createTensorStorage(values.AsSpan());
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
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values)
    {
        // Use TensorStorage for vectorizable types that support the unmanaged constraint
        if (IsVectorizable<T>() && IsUnmanagedType<T>())
        {
            // We need to use runtime type checking since we can't add generic constraints to static methods
            return CreateTensorStorageForType<T>(values);
        }

        // Use MemoryStorage for non-vectorizable types or reference types
        return new MemoryStorage<T>(values, detectNulls: !typeof(T).IsValueType);
    }

    /// <summary>
    /// Creates storage for the given values with an explicit null mask, automatically selecting between tensor and memory storage.
    /// Used by vectorized arithmetic paths in NivaraColumn to preserve null masks when the result is stored as TensorStorage.
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <param name="nullMask">Optional null mask indicating which positions are null</param>
    /// <returns>An appropriate storage implementation with the given null mask</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T> values, ReadOnlyMemory<bool>? nullMask)
    {
        // Use TensorStorage for vectorizable types that support the unmanaged constraint
        if (IsVectorizable<T>() && IsUnmanagedType<T>())
        {
            return CreateTensorStorageForType<T>(values, nullMask);
        }

        // Use MemoryStorage for non-vectorizable types or reference types
        return new MemoryStorage<T>(values.ToArray().AsMemory(), nullMask);
    }

    /// <summary>
    /// Creates storage for nullable values, automatically selecting between tensor and memory storage
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The nullable values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    public static IColumnStorage<T> Create<T>(ReadOnlySpan<T?> values) where T : struct
    {
        // Use TensorStorage for vectorizable value types that support the unmanaged constraint
        if (IsVectorizable<T>() && IsUnmanagedType<T>())
        {
            return CreateTensorStorageForNullableType<T>(values);
        }

        // Use MemoryStorage for non-vectorizable value types
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
    /// Determines if a type satisfies the unmanaged constraint required for TensorStorage
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type is unmanaged, false otherwise</returns>
    private static bool IsUnmanagedType<T>()
    {
        var type = typeof(T);

        // Check for specific unmanaged types that we support in TensorStorage
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
    /// Creates TensorStorage for a specific type using runtime type checking with an optional null mask.
    /// Uses the internal TensorStorage(Tensor&lt;T&gt;, Tensor&lt;bool&gt;?) constructor for proper null-mask propagation.
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <param name="nullMask">Optional null mask to attach to the storage</param>
    /// <returns>A TensorStorage instance cast to IColumnStorage</returns>
    private static IColumnStorage<T> CreateTensorStorageForType<T>(ReadOnlySpan<T> values, ReadOnlyMemory<bool>? nullMask = null)
    {
        var type = typeof(T);

        // Convert to array first since we can't cast spans to object
        var array = values.ToArray();

        // Convert the null mask from ReadOnlyMemory<bool>? to Tensor<bool>?
        Tensor<bool>? nullTensor = null;
        if (nullMask.HasValue && nullMask.Value.Length > 0)
        {
            var nullArray = nullMask.Value.ToArray();
            nullTensor = Tensor.Create(nullArray, new nint[] { nullArray.Length });
        }

        // Use type switching to create the appropriate TensorStorage with its internal constructor
        return type switch
        {
            Type t when t == typeof(int) => (IColumnStorage<T>)(object)new TensorStorage<int>(Tensor.Create((int[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(float) => (IColumnStorage<T>)(object)new TensorStorage<float>(Tensor.Create((float[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(double) => (IColumnStorage<T>)(object)new TensorStorage<double>(Tensor.Create((double[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(long) => (IColumnStorage<T>)(object)new TensorStorage<long>(Tensor.Create((long[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(short) => (IColumnStorage<T>)(object)new TensorStorage<short>(Tensor.Create((short[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(byte) => (IColumnStorage<T>)(object)new TensorStorage<byte>(Tensor.Create((byte[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(uint) => (IColumnStorage<T>)(object)new TensorStorage<uint>(Tensor.Create((uint[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(ulong) => (IColumnStorage<T>)(object)new TensorStorage<ulong>(Tensor.Create((ulong[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(ushort) => (IColumnStorage<T>)(object)new TensorStorage<ushort>(Tensor.Create((ushort[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(sbyte) => (IColumnStorage<T>)(object)new TensorStorage<sbyte>(Tensor.Create((sbyte[])(object)array, new nint[] { array.Length }), nullTensor),
            Type t when t == typeof(bool) => (IColumnStorage<T>)(object)new TensorStorage<bool>(Tensor.Create((bool[])(object)array, new nint[] { array.Length }), nullTensor),
            _ => throw new InvalidOperationException($"Type {type.Name} is not supported for TensorStorage")
        };
    }

    /// <summary>
    /// Creates TensorStorage for nullable value types using runtime type checking
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    /// <param name="values">The nullable values to store</param>
    /// <returns>A TensorStorage instance cast to IColumnStorage</returns>
    private static IColumnStorage<T> CreateTensorStorageForNullableType<T>(ReadOnlySpan<T?> values) where T : struct
    {
        var type = typeof(T);

        // Convert to array first to avoid span casting issues
        var array = values.ToArray();

        // Use type switching to call the appropriate helper method
        return type switch
        {
            Type t when t == typeof(int) => (IColumnStorage<T>)(object)createTensorStorage((int?[])(object)array),
            Type t when t == typeof(float) => (IColumnStorage<T>)(object)createTensorStorage((float?[])(object)array),
            Type t when t == typeof(double) => (IColumnStorage<T>)(object)createTensorStorage((double?[])(object)array),
            Type t when t == typeof(long) => (IColumnStorage<T>)(object)createTensorStorage((long?[])(object)array),
            Type t when t == typeof(short) => (IColumnStorage<T>)(object)createTensorStorage((short?[])(object)array),
            Type t when t == typeof(byte) => (IColumnStorage<T>)(object)createTensorStorage((byte?[])(object)array),
            Type t when t == typeof(uint) => (IColumnStorage<T>)(object)createTensorStorage((uint?[])(object)array),
            Type t when t == typeof(ulong) => (IColumnStorage<T>)(object)createTensorStorage((ulong?[])(object)array),
            Type t when t == typeof(ushort) => (IColumnStorage<T>)(object)createTensorStorage((ushort?[])(object)array),
            Type t when t == typeof(sbyte) => (IColumnStorage<T>)(object)createTensorStorage((sbyte?[])(object)array),
            Type t when t == typeof(bool) => (IColumnStorage<T>)(object)createTensorStorage((bool?[])(object)array),
            _ => throw new InvalidOperationException($"Type {type.Name} is not supported for TensorStorage")
        };
    }
}
