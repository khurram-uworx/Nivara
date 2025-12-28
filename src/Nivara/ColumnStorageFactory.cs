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
        // For now, always use memory storage - we'll optimize this later
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
        return NullableStorageHelper.CreateMemoryStorage(values);
    }
    
    /// <summary>
    /// Creates storage for reference type values that may contain nulls
    /// </summary>
    /// <typeparam name="T">The reference type of elements to store</typeparam>
    /// <param name="values">The values to store (may contain nulls)</param>
    /// <returns>A memory storage implementation</returns>
    public static IColumnStorage<T> CreateForReferenceType<T>(ReadOnlySpan<T> values) where T : class
    {
        return new MemoryStorage<T>(values, detectNulls: true);
    }
    
    /// <summary>
    /// Determines if a type supports vectorized operations
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type supports vectorization, false otherwise</returns>
    public static bool IsVectorizable<T>()
    {
        var type = typeof(T);
        
        // Check if it's an unmanaged type that supports SIMD operations
        if (!type.IsValueType || type.IsEnum)
            return false;
            
        // Check for specific vectorizable types
        return type == typeof(byte) ||
               type == typeof(sbyte) ||
               type == typeof(short) ||
               type == typeof(ushort) ||
               type == typeof(int) ||
               type == typeof(uint) ||
               type == typeof(long) ||
               type == typeof(ulong) ||
               type == typeof(float) ||
               type == typeof(double) ||
               type == typeof(bool);
    }
}