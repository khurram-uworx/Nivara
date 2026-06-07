using System.Numerics;
using Nivara;

namespace Nivara.Diagnostics;

/// <summary>
/// Provides diagnostic information about column operations and performance characteristics.
/// Used for performance analysis and optimization decisions.
/// </summary>
public sealed class ColumnDiagnostics
{
    /// <summary>
    /// Initializes a new instance of ColumnDiagnostics
    /// </summary>
    /// <param name="storageType">The type of storage being used</param>
    /// <param name="isVectorizable">Whether the column supports vectorized operations</param>
    /// <param name="elementType">The element type of the column</param>
    /// <param name="length">The number of elements in the column</param>
    /// <param name="hasNulls">Whether the column contains null values</param>
    internal ColumnDiagnostics(
        StorageType storageType,
        bool isVectorizable,
        Type elementType,
        int length,
        bool hasNulls)
    {
        StorageType = storageType;
        IsVectorizable = isVectorizable;
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
        Length = length;
        HasNulls = hasNulls;
        IsHardwareAccelerated = Vector.IsHardwareAccelerated;
        VectorSize = Vector<byte>.Count; // Get the hardware vector size
    }

    /// <summary>
    /// Gets the size in bytes of a single element of the specified type
    /// </summary>
    /// <param name="type">The type to get the size for</param>
    /// <returns>The size in bytes</returns>
    static long GetElementSize(Type type)
    {
        if (type == typeof(bool) || type == typeof(byte) || type == typeof(sbyte))
            return 1;
        if (type == typeof(short) || type == typeof(ushort))
            return 2;
        if (type == typeof(int) || type == typeof(uint) || type == typeof(float))
            return 4;
        if (type == typeof(long) || type == typeof(ulong) || type == typeof(double))
            return 8;
        if (type == typeof(decimal))
            return 16;
        if (type == typeof(Guid))
            return 16;
        if (type == typeof(DateTime) || type == typeof(DateTimeOffset))
            return 8;
        if (type == typeof(TimeSpan))
            return 8;

        // For reference types, estimate pointer size + average object overhead
        if (!type.IsValueType)
            return IntPtr.Size + 32; // Pointer + estimated object overhead

        // For other value types, use a conservative estimate
        return 32;
    }

    /// <summary>
    /// Gets the type of storage implementation being used
    /// </summary>
    public StorageType StorageType { get; }

    /// <summary>
    /// Gets a value indicating whether this column supports vectorized operations
    /// </summary>
    public bool IsVectorizable { get; }

    /// <summary>
    /// Gets the element type of the column
    /// </summary>
    public Type ElementType { get; }

    /// <summary>
    /// Gets the number of elements in the column
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets a value indicating whether the column contains null values
    /// </summary>
    public bool HasNulls { get; }

    /// <summary>
    /// Gets a value indicating whether hardware acceleration (SIMD) is available
    /// </summary>
    public bool IsHardwareAccelerated { get; }

    /// <summary>
    /// Gets the size of hardware vectors in bytes
    /// </summary>
    public int VectorSize { get; }

    /// <summary>
    /// Gets the recommended kernel type for operations on this column
    /// </summary>
    public KernelType RecommendedKernel => KernelSelector.DetermineKernelType(Length, IsVectorizable);

    /// <summary>
    /// Gets the estimated memory usage in bytes
    /// </summary>
    public long EstimatedMemoryUsage
    {
        get
        {
            // Calculate base element size
            long elementSize = GetElementSize(ElementType);
            long dataSize = Length * elementSize;

            // Add null mask overhead if present
            if (HasNulls)
            {
                long nullMaskSize = Length; // 1 byte per boolean
                dataSize += nullMaskSize;
            }

            // Add storage overhead (approximate)
            long storageOverhead = StorageType switch
            {
                StorageType.Memory => 64, // ReadOnlyMemory overhead
                StorageType.Tensor => 128, // Tensor overhead (future)
                _ => 32
            };

            return dataSize + storageOverhead;
        }
    }

    /// <summary>
    /// Gets performance characteristics for this column configuration
    /// </summary>
    public PerformanceCharacteristics Performance
    {
        get
        {
            var throughputMultiplier = RecommendedKernel == KernelType.Vectorized
                ? Math.Min(VectorSize / GetElementSize(ElementType), 8) // Cap at 8x improvement
                : 1;

            var memoryEfficiency = StorageType switch
            {
                StorageType.Memory => HasNulls ? 0.85 : 0.95, // Null mask overhead
                StorageType.Tensor => HasNulls ? 0.80 : 0.90, // Tensor overhead
                _ => 0.75
            };

            return new PerformanceCharacteristics(
                throughputMultiplier,
                memoryEfficiency,
                IsVectorizable && IsHardwareAccelerated);
        }
    }

    /// <summary>
    /// Returns a string representation of the diagnostic information
    /// </summary>
    /// <returns>A formatted string with diagnostic details</returns>
    public override string ToString()
    {
        return $"ColumnDiagnostics {{ " +
               $"Type: {ElementType.Name}, " +
               $"Length: {Length:N0}, " +
               $"Storage: {StorageType}, " +
               $"Vectorizable: {IsVectorizable}, " +
               $"HasNulls: {HasNulls}, " +
               $"RecommendedKernel: {RecommendedKernel}, " +
               $"EstimatedMemory: {EstimatedMemoryUsage:N0} bytes " +
               $"}}";
    }
}

/// <summary>
/// Represents the type of storage implementation being used
/// </summary>
public enum StorageType
{
    /// <summary>
    /// Memory-backed storage using Memory&lt;T&gt;
    /// </summary>
    Memory,

    /// <summary>
    /// Tensor-backed storage using System.Numerics.Tensors (future implementation)
    /// </summary>
    Tensor
}

/// <summary>
/// Represents the type of kernel used for operations
/// </summary>
public enum KernelType
{
    /// <summary>
    /// Scalar operations using standard loops
    /// </summary>
    Scalar,

    /// <summary>
    /// Vectorized operations using SIMD instructions
    /// </summary>
    Vectorized
}

/// <summary>
/// Represents performance characteristics of a column configuration
/// </summary>
public sealed class PerformanceCharacteristics
{
    /// <summary>
    /// Initializes a new instance of PerformanceCharacteristics
    /// </summary>
    /// <param name="throughputMultiplier">The expected throughput multiplier compared to baseline</param>
    /// <param name="memoryEfficiency">The memory efficiency ratio (0.0 to 1.0)</param>
    /// <param name="supportsVectorization">Whether vectorization is supported and beneficial</param>
    internal PerformanceCharacteristics(double throughputMultiplier, double memoryEfficiency, bool supportsVectorization)
    {
        ThroughputMultiplier = throughputMultiplier;
        MemoryEfficiency = memoryEfficiency;
        SupportsVectorization = supportsVectorization;
    }

    /// <summary>
    /// Gets the expected throughput multiplier compared to baseline scalar operations
    /// </summary>
    public double ThroughputMultiplier { get; }

    /// <summary>
    /// Gets the memory efficiency ratio (0.0 to 1.0, where 1.0 is perfect efficiency)
    /// </summary>
    public double MemoryEfficiency { get; }

    /// <summary>
    /// Gets a value indicating whether vectorization is supported and beneficial for this configuration
    /// </summary>
    public bool SupportsVectorization { get; }

    /// <summary>
    /// Returns a string representation of the performance characteristics
    /// </summary>
    /// <returns>A formatted string with performance details</returns>
    public override string ToString()
    {
        return $"PerformanceCharacteristics {{ " +
               $"ThroughputMultiplier: {ThroughputMultiplier:F1}x, " +
               $"MemoryEfficiency: {MemoryEfficiency:P1}, " +
               $"SupportsVectorization: {SupportsVectorization} " +
               $"}}";
    }
}
