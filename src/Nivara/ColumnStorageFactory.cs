using Nivara.Memory;
using Nivara.Tensors;

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
    /// Creates TensorStorage for a specific type using runtime type checking
    /// </summary>
    /// <typeparam name="T">The type of elements to store</typeparam>
    /// <param name="values">The values to store</param>
    /// <returns>A TensorStorage instance cast to IColumnStorage</returns>
    private static IColumnStorage<T> CreateTensorStorageForType<T>(ReadOnlySpan<T> values)
    {
        var type = typeof(T);

        // Convert to array first since we can't cast spans to object
        var array = values.ToArray();

        // Use type switching to create the appropriate TensorStorage
        return type switch
        {
            Type t when t == typeof(int) => (IColumnStorage<T>)(object)new TensorStorage<int>((int[])(object)array),
            Type t when t == typeof(float) => (IColumnStorage<T>)(object)new TensorStorage<float>((float[])(object)array),
            Type t when t == typeof(double) => (IColumnStorage<T>)(object)new TensorStorage<double>((double[])(object)array),
            Type t when t == typeof(long) => (IColumnStorage<T>)(object)new TensorStorage<long>((long[])(object)array),
            Type t when t == typeof(short) => (IColumnStorage<T>)(object)new TensorStorage<short>((short[])(object)array),
            Type t when t == typeof(byte) => (IColumnStorage<T>)(object)new TensorStorage<byte>((byte[])(object)array),
            Type t when t == typeof(uint) => (IColumnStorage<T>)(object)new TensorStorage<uint>((uint[])(object)array),
            Type t when t == typeof(ulong) => (IColumnStorage<T>)(object)new TensorStorage<ulong>((ulong[])(object)array),
            Type t when t == typeof(ushort) => (IColumnStorage<T>)(object)new TensorStorage<ushort>((ushort[])(object)array),
            Type t when t == typeof(sbyte) => (IColumnStorage<T>)(object)new TensorStorage<sbyte>((sbyte[])(object)array),
            Type t when t == typeof(bool) => (IColumnStorage<T>)(object)new TensorStorage<bool>((bool[])(object)array),
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
            Type t when t == typeof(int) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((int?[])(object)array),
            Type t when t == typeof(float) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((float?[])(object)array),
            Type t when t == typeof(double) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((double?[])(object)array),
            Type t when t == typeof(long) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((long?[])(object)array),
            Type t when t == typeof(short) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((short?[])(object)array),
            Type t when t == typeof(byte) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((byte?[])(object)array),
            Type t when t == typeof(uint) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((uint?[])(object)array),
            Type t when t == typeof(ulong) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((ulong?[])(object)array),
            Type t when t == typeof(ushort) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((ushort?[])(object)array),
            Type t when t == typeof(sbyte) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((sbyte?[])(object)array),
            Type t when t == typeof(bool) => (IColumnStorage<T>)(object)NullableStorageHelper.CreateTensorStorage((bool?[])(object)array),
            _ => throw new InvalidOperationException($"Type {type.Name} is not supported for TensorStorage")
        };
    }
}