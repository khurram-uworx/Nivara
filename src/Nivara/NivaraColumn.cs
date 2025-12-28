using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara;

/// <summary>
/// Strongly-typed, immutable column with automatic storage selection and vectorized operations.
/// Provides the main public API for columnar data processing in Nivara.
/// </summary>
/// <typeparam name="T">The type of elements in the column</typeparam>
public sealed class NivaraColumn<T> : IColumn<T>, IDisposable
{
    private readonly IColumnStorage<T> _storage;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of NivaraColumn with the specified storage
    /// </summary>
    /// <param name="storage">The storage implementation to use</param>
    internal NivaraColumn(IColumnStorage<T> storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <inheritdoc />
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _storage.Length;
        }
    }

    /// <inheritdoc />
    public bool HasNulls
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _storage.HasNulls;
        }
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _storage[index];
        }
    }

    /// <inheritdoc />
    public bool IsNull(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException($"Index {index} is out of range for column of length {Length}");

        var nullMask = _storage.NullMask;
        return !nullMask.IsEmpty && nullMask[index];
    }

    /// <summary>
    /// Gets a value indicating whether this column uses vectorizable storage
    /// </summary>
    public bool IsVectorizable
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _storage.IsVectorizable;
        }
    }

    /// <summary>
    /// Creates a new column from the specified array of values.
    /// Automatically selects appropriate storage based on type characteristics.
    /// </summary>
    /// <param name="values">The values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values is null</exception>
    public static NivaraColumn<T> Create(T[] values)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));

        return Create(values.AsSpan());
    }

    /// <summary>
    /// Creates a new column from the specified span of values.
    /// Automatically selects appropriate storage based on type characteristics.
    /// </summary>
    /// <param name="values">The values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    public static NivaraColumn<T> Create(ReadOnlySpan<T> values)
    {
        var storage = CreateStorage(values);
        return new NivaraColumn<T>(storage);
    }

    /// <summary>
    /// Creates a new column from the specified array of reference type values.
    /// Automatically detects and tracks null references.
    /// </summary>
    /// <param name="values">The reference type values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values array is null</exception>
    public static NivaraColumn<T> CreateForReferenceType(T[] values)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));

        return CreateForReferenceType(values.AsSpan());
    }

    /// <summary>
    /// Creates a new column from the specified span of reference type values.
    /// Automatically detects and tracks null references.
    /// </summary>
    /// <param name="values">The reference type values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    public static NivaraColumn<T> CreateForReferenceType(ReadOnlySpan<T> values)
    {
        // This method should only be called for reference types
        if (!typeof(T).IsClass)
            throw new InvalidOperationException($"CreateForReferenceType can only be used with reference types. Use Create or CreateFromNullable for value types.");

        // For reference types, always detect nulls using MemoryStorage directly
        var storage = new MemoryStorage<T>(values, detectNulls: true);
        return new NivaraColumn<T>(storage);
    }

    /// <summary>
    /// Creates appropriate storage for the given values based on type characteristics
    /// </summary>
    /// <param name="values">The values to store</param>
    /// <returns>An appropriate storage implementation</returns>
    private static IColumnStorage<T> CreateStorage(ReadOnlySpan<T> values)
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
    /// Creates a slice of this column containing elements from the specified range
    /// </summary>
    /// <param name="start">The starting index of the slice</param>
    /// <param name="length">The number of elements in the slice</param>
    /// <returns>A new column representing the slice</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when start or length are invalid</exception>
    public NivaraColumn<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var slicedStorage = _storage.Slice(start, length);
        return new NivaraColumn<T>(slicedStorage);
    }

    /// <summary>
    /// Gets the underlying storage for advanced operations (internal use only)
    /// </summary>
    internal IColumnStorage<T> Storage
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _storage;
        }
    }

    // Arithmetic Operations

    /// <summary>
    /// Multiplies all elements in the column by a scalar value.
    /// Only supported for numeric types that implement INumber&lt;T&gt;.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by</param>
    /// <returns>A new column with all elements multiplied by the scalar</returns>
    /// <exception cref="InvalidOperationException">Thrown when T does not support arithmetic operations</exception>
    public NivaraColumn<T> Multiply(T scalar)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        // Runtime check for numeric types
        if (!IsNumericType(typeof(T)))
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for type {typeof(T).Name}. Only numeric types (int, float, double, long, etc.) support arithmetic operations.");
        }
        
        if (!ColumnStorageFactory.IsVectorizable<T>())
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for non-vectorizable type {typeof(T).Name}. Only numeric primitive types support vectorized arithmetic.");
        }

        return MultiplyVectorized(scalar);
    }

    /// <summary>
    /// Adds corresponding elements of two columns together.
    /// Only supported for numeric types that implement INumber&lt;T&gt;.
    /// </summary>
    /// <param name="other">The column to add to this column</param>
    /// <returns>A new column with element-wise addition results</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support arithmetic operations</exception>
    public NivaraColumn<T> Add(NivaraColumn<T> other)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        
        if (other._disposed)
            throw new ObjectDisposedException(nameof(other));
        
        if (Length != other.Length)
            throw new ArgumentException($"Cannot add columns of different lengths: {Length} vs {other.Length}");
        
        // Runtime check for numeric types
        if (!IsNumericType(typeof(T)))
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for type {typeof(T).Name}. Only numeric types (int, float, double, long, etc.) support arithmetic operations.");
        }
        
        if (!ColumnStorageFactory.IsVectorizable<T>())
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for non-vectorizable type {typeof(T).Name}. Only numeric primitive types support vectorized arithmetic.");
        }

        return AddVectorized(other);
    }

    /// <summary>
    /// Operator overload for scalar multiplication
    /// </summary>
    public static NivaraColumn<T> operator *(NivaraColumn<T> column, T scalar)
    {
        return column.Multiply(scalar);
    }

    /// <summary>
    /// Operator overload for scalar multiplication (commutative)
    /// </summary>
    public static NivaraColumn<T> operator *(T scalar, NivaraColumn<T> column)
    {
        return column.Multiply(scalar);
    }

    /// <summary>
    /// Operator overload for element-wise addition
    /// </summary>
    public static NivaraColumn<T> operator +(NivaraColumn<T> left, NivaraColumn<T> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Performs vectorized scalar multiplication using TensorPrimitives
    /// </summary>
    private NivaraColumn<T> MultiplyVectorized(T scalar)
    {
        // Since we're currently using MemoryStorage for all types, we need to handle it appropriately
        if (_storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new T[data.Length];
            
            // Use our helper method for multiplication
            MultiplyTensorPrimitive(data, scalar, result.AsSpan());
            
            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);
            }
            
            var resultStorage = new MemoryStorage<T>(result, resultNullMask);
            return new NivaraColumn<T>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for vectorized operations");
        }
    }

    /// <summary>
    /// Performs vectorized element-wise addition using TensorPrimitives
    /// </summary>
    private NivaraColumn<T> AddVectorized(NivaraColumn<T> other)
    {
        // Since we're currently using MemoryStorage for all types, we need to handle it appropriately
        if (_storage is MemoryStorage<T> leftMemory && other._storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new T[leftData.Length];
            
            // Use our helper method for addition
            AddTensorPrimitive(leftData, rightData, result.AsSpan());
            
            // Handle null propagation - null + anything = null
            ReadOnlyMemory<bool>? resultNullMask = null;
            var leftNulls = leftMemory.NullMaskMemory;
            var rightNulls = rightMemory.NullMaskMemory;
            
            if ((leftNulls.HasValue && leftNulls.Value.Length > 0) || (rightNulls.HasValue && rightNulls.Value.Length > 0))
            {
                var resultNullMaskArray = new bool[result.Length];
                
                for (int i = 0; i < result.Length; i++)
                {
                    bool leftIsNull = leftNulls.HasValue && leftNulls.Value.Length > 0 && leftNulls.Value.Span[i];
                    bool rightIsNull = rightNulls.HasValue && rightNulls.Value.Length > 0 && rightNulls.Value.Span[i];
                    resultNullMaskArray[i] = leftIsNull || rightIsNull;
                }
                
                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }
            
            var resultStorage = new MemoryStorage<T>(result, resultNullMask);
            return new NivaraColumn<T>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform arithmetic operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Helper method to perform vectorized multiplication using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void MultiplyTensorPrimitive(ReadOnlySpan<T> x, T y, Span<T> destination)
    {
        // For now, use scalar multiplication with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = (T)(object)((dynamic)x[i]! * (dynamic)y!)!;
        }
    }

    /// <summary>
    /// Helper method to perform vectorized addition using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void AddTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        // For now, use scalar addition with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = (T)(object)((dynamic)x[i]! + (dynamic)y[i]!)!;
        }
    }

    /// <summary>
    /// Helper method to check if a type is numeric and supports arithmetic operations
    /// </summary>
    private static bool IsNumericType(Type type)
    {
        return type == typeof(int) ||
               type == typeof(float) ||
               type == typeof(double) ||
               type == typeof(long) ||
               type == typeof(short) ||
               type == typeof(byte) ||
               type == typeof(sbyte) ||
               type == typeof(uint) ||
               type == typeof(ulong) ||
               type == typeof(ushort) ||
               type == typeof(decimal) ||
               type == typeof(bool);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _storage?.Dispose();
            _disposed = true;
        }
    }
}