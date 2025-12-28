using System.Numerics.Tensors;

namespace Nivara;

/// <summary>
/// Strongly-typed, immutable column with automatic storage selection and vectorized operations.
/// Provides the main public API for columnar data processing in Nivara.
/// </summary>
/// <typeparam name="T">The type of elements in the column</typeparam>
public sealed class NivaraColumn<T> : IColumn<T>, IDisposable
{
    private readonly IColumnStorage<T> storage;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of NivaraColumn with the specified storage
    /// </summary>
    /// <param name="storage">The storage implementation to use</param>
    internal NivaraColumn(IColumnStorage<T> storage)
    {
        this.storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <inheritdoc />
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage.Length;
        }
    }

    /// <inheritdoc />
    public bool HasNulls
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage.HasNulls;
        }
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage[index];
        }
    }

    /// <inheritdoc />
    public bool IsNull(int index)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException($"Index {index} is out of range for column of length {Length}");

        var nullMask = storage.NullMask;
        return !nullMask.IsEmpty && nullMask[index];
    }

    /// <summary>
    /// Gets a value indicating whether this column uses vectorizable storage
    /// </summary>
    public bool IsVectorizable
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage.IsVectorizable;
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
        ObjectDisposedException.ThrowIf(disposed, this);

        var slicedStorage = storage.Slice(start, length);
        return new NivaraColumn<T>(slicedStorage);
    }

    /// <summary>
    /// Gets the underlying storage for advanced operations (internal use only)
    /// </summary>
    internal IColumnStorage<T> Storage
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage;
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
        ObjectDisposedException.ThrowIf(disposed, this);

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
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (other.disposed)
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
    /// Multiplies corresponding elements of two columns together.
    /// Only supported for numeric types that implement INumber&lt;T&gt;.
    /// </summary>
    /// <param name="other">The column to multiply with this column</param>
    /// <returns>A new column with element-wise multiplication results</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support arithmetic operations</exception>
    public NivaraColumn<T> Multiply(NivaraColumn<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (other.disposed)
            throw new ObjectDisposedException(nameof(other));

        if (Length != other.Length)
            throw new ArgumentException($"Cannot multiply columns of different lengths: {Length} vs {other.Length}");

        // Runtime check for numeric types
        if (!IsNumericType(typeof(T)))
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for type {typeof(T).Name}. Only numeric types (int, float, double, long, etc.) support arithmetic operations.");
        }

        if (!ColumnStorageFactory.IsVectorizable<T>())
        {
            throw new InvalidOperationException($"Arithmetic operations are not supported for non-vectorizable type {typeof(T).Name}. Only numeric primitive types support vectorized arithmetic.");
        }

        return MultiplyVectorized(other);
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
    /// Operator overload for element-wise multiplication
    /// </summary>
    public static NivaraColumn<T> operator *(NivaraColumn<T> left, NivaraColumn<T> right)
    {
        return left.Multiply(right);
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
        if (storage is MemoryStorage<T> memoryStorage)
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
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
    /// Performs vectorized element-wise multiplication using TensorPrimitives
    /// </summary>
    private NivaraColumn<T> MultiplyVectorized(NivaraColumn<T> other)
    {
        // Since we're currently using MemoryStorage for all types, we need to handle it appropriately
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new T[leftData.Length];

            // Use our helper method for multiplication
            MultiplyTensorPrimitive(leftData, rightData, result.AsSpan());

            // Handle null propagation - null * anything = null
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
    /// Helper method to perform vectorized element-wise multiplication using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void MultiplyTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        // For now, use scalar multiplication with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = (T)(object)((dynamic)x[i]! * (dynamic)y[i]!)!;
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

    // Comparison Operations

    /// <summary>
    /// Compares all elements in the column to a scalar value for equality.
    /// Returns a boolean column indicating comparison results.
    /// </summary>
    /// <param name="value">The scalar value to compare against</param>
    /// <returns>A new boolean column with comparison results</returns>
    public NivaraColumn<bool> Equals(T value)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (storage.IsVectorizable)
        {
            return EqualsVectorized(value);
        }
        else
        {
            return EqualsScalar(value);
        }
    }

    /// <summary>
    /// Compares corresponding elements of two columns for equality.
    /// Returns a boolean column indicating comparison results.
    /// </summary>
    /// <param name="other">The column to compare against</param>
    /// <returns>A new boolean column with element-wise comparison results</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    public NivaraColumn<bool> Equals(NivaraColumn<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (other.disposed)
            throw new ObjectDisposedException(nameof(other));

        if (Length != other.Length)
            throw new ArgumentException($"Cannot compare columns of different lengths: {Length} vs {other.Length}");

        if (storage.IsVectorizable && other.storage.IsVectorizable)
        {
            return EqualsVectorized(other);
        }
        else
        {
            return EqualsScalar(other);
        }
    }

    /// <summary>
    /// Compares all elements in the column to a scalar value for greater than.
    /// Only supported for comparable types.
    /// </summary>
    /// <param name="value">The scalar value to compare against</param>
    /// <returns>A new boolean column with comparison results</returns>
    /// <exception cref="InvalidOperationException">Thrown when T does not support comparison operations</exception>
    public NivaraColumn<bool> GreaterThan(T value)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (!IsComparableType(typeof(T)))
        {
            throw new InvalidOperationException($"Comparison operations are not supported for type {typeof(T).Name}. Only comparable types support comparison operations.");
        }

        if (storage.IsVectorizable)
        {
            return GreaterThanVectorized(value);
        }
        else
        {
            return GreaterThanScalar(value);
        }
    }

    /// <summary>
    /// Compares corresponding elements of two columns for greater than.
    /// Only supported for comparable types.
    /// </summary>
    /// <param name="other">The column to compare against</param>
    /// <returns>A new boolean column with element-wise comparison results</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support comparison operations</exception>
    public NivaraColumn<bool> GreaterThan(NivaraColumn<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (other.disposed)
            throw new ObjectDisposedException(nameof(other));

        if (Length != other.Length)
            throw new ArgumentException($"Cannot compare columns of different lengths: {Length} vs {other.Length}");

        if (!IsComparableType(typeof(T)))
        {
            throw new InvalidOperationException($"Comparison operations are not supported for type {typeof(T).Name}. Only comparable types support comparison operations.");
        }

        if (storage.IsVectorizable && other.storage.IsVectorizable)
        {
            return GreaterThanVectorized(other);
        }
        else
        {
            return GreaterThanScalar(other);
        }
    }

    /// <summary>
    /// Compares all elements in the column to a scalar value for less than.
    /// Only supported for comparable types.
    /// </summary>
    /// <param name="value">The scalar value to compare against</param>
    /// <returns>A new boolean column with comparison results</returns>
    /// <exception cref="InvalidOperationException">Thrown when T does not support comparison operations</exception>
    public NivaraColumn<bool> LessThan(T value)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (!IsComparableType(typeof(T)))
        {
            throw new InvalidOperationException($"Comparison operations are not supported for type {typeof(T).Name}. Only comparable types support comparison operations.");
        }

        if (storage.IsVectorizable)
        {
            return LessThanVectorized(value);
        }
        else
        {
            return LessThanScalar(value);
        }
    }

    /// <summary>
    /// Compares corresponding elements of two columns for less than.
    /// Only supported for comparable types.
    /// </summary>
    /// <param name="other">The column to compare against</param>
    /// <returns>A new boolean column with element-wise comparison results</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support comparison operations</exception>
    public NivaraColumn<bool> LessThan(NivaraColumn<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (other.disposed)
            throw new ObjectDisposedException(nameof(other));

        if (Length != other.Length)
            throw new ArgumentException($"Cannot compare columns of different lengths: {Length} vs {other.Length}");

        if (!IsComparableType(typeof(T)))
        {
            throw new InvalidOperationException($"Comparison operations are not supported for type {typeof(T).Name}. Only comparable types support comparison operations.");
        }

        if (storage.IsVectorizable && other.storage.IsVectorizable)
        {
            return LessThanVectorized(other);
        }
        else
        {
            return LessThanScalar(other);
        }
    }

    /// <summary>
    /// Performs vectorized scalar equality comparison
    /// </summary>
    private NivaraColumn<bool> EqualsVectorized(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];

            // Use our helper method for equality comparison
            EqualsTensorPrimitive(data, scalar, result.AsSpan());

            // Handle null propagation - null compared to anything is null (false in boolean result)
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);

                // Set result to false where nulls exist (null comparisons yield null/false)
                for (int i = 0; i < result.Length; i++)
                {
                    if (nullMaskArray[i])
                        result[i] = false;
                }
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for vectorized operations");
        }
    }

    /// <summary>
    /// Performs vectorized element-wise equality comparison
    /// </summary>
    private NivaraColumn<bool> EqualsVectorized(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];

            // Use our helper method for element-wise equality
            EqualsTensorPrimitive(leftData, rightData, result.AsSpan());

            // Handle null propagation - null compared to anything is null (false in boolean result)
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
                    bool hasNull = leftIsNull || rightIsNull;

                    resultNullMaskArray[i] = hasNull;

                    // Set result to false where nulls exist (null comparisons yield null/false)
                    if (hasNull)
                        result[i] = false;
                }

                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Performs scalar equality comparison using standard equality
    /// </summary>
    private NivaraColumn<bool> EqualsScalar(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];
            var comparer = EqualityComparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;

            for (int i = 0; i < data.Length; i++)
            {
                bool isNull = nullMask.HasValue && nullMask.Value.Length > 0 && nullMask.Value.Span[i];

                if (isNull)
                {
                    result[i] = false; // null compared to anything is false
                }
                else
                {
                    result[i] = comparer.Equals(data[i], scalar);
                }
            }

            // Copy null mask if it exists
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for scalar operations");
        }
    }

    /// <summary>
    /// Performs element-wise equality comparison using standard equality
    /// </summary>
    private NivaraColumn<bool> EqualsScalar(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];
            var comparer = EqualityComparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var leftNulls = leftMemory.NullMaskMemory;
            var rightNulls = rightMemory.NullMaskMemory;

            bool hasAnyNulls = (leftNulls.HasValue && leftNulls.Value.Length > 0) || (rightNulls.HasValue && rightNulls.Value.Length > 0);
            bool[]? resultNullMaskArray = null;

            if (hasAnyNulls)
            {
                resultNullMaskArray = new bool[result.Length];
            }

            for (int i = 0; i < result.Length; i++)
            {
                bool leftIsNull = leftNulls.HasValue && leftNulls.Value.Length > 0 && leftNulls.Value.Span[i];
                bool rightIsNull = rightNulls.HasValue && rightNulls.Value.Length > 0 && rightNulls.Value.Span[i];
                bool hasNull = leftIsNull || rightIsNull;

                if (hasNull)
                {
                    result[i] = false; // null compared to anything is false
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = true;
                }
                else
                {
                    result[i] = comparer.Equals(leftData[i], rightData[i]);
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = false;
                }
            }

            if (resultNullMaskArray != null)
            {
                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Performs vectorized scalar greater than comparison
    /// </summary>
    private NivaraColumn<bool> GreaterThanVectorized(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];

            // Use our helper method for greater than comparison
            GreaterThanTensorPrimitive(data, scalar, result.AsSpan());

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);

                // Set result to false where nulls exist
                for (int i = 0; i < result.Length; i++)
                {
                    if (nullMaskArray[i])
                        result[i] = false;
                }
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for vectorized operations");
        }
    }

    /// <summary>
    /// Performs vectorized element-wise greater than comparison
    /// </summary>
    private NivaraColumn<bool> GreaterThanVectorized(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];

            // Use our helper method for element-wise greater than
            GreaterThanTensorPrimitive(leftData, rightData, result.AsSpan());

            // Handle null propagation
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
                    bool hasNull = leftIsNull || rightIsNull;

                    resultNullMaskArray[i] = hasNull;

                    if (hasNull)
                        result[i] = false;
                }

                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Performs scalar greater than comparison using standard comparison
    /// </summary>
    private NivaraColumn<bool> GreaterThanScalar(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];
            var comparer = Comparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;

            for (int i = 0; i < data.Length; i++)
            {
                bool isNull = nullMask.HasValue && nullMask.Value.Length > 0 && nullMask.Value.Span[i];

                if (isNull)
                {
                    result[i] = false; // null compared to anything is false
                }
                else
                {
                    result[i] = comparer.Compare(data[i], scalar) > 0;
                }
            }

            // Copy null mask if it exists
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for scalar operations");
        }
    }

    /// <summary>
    /// Performs element-wise greater than comparison using standard comparison
    /// </summary>
    private NivaraColumn<bool> GreaterThanScalar(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];
            var comparer = Comparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var leftNulls = leftMemory.NullMaskMemory;
            var rightNulls = rightMemory.NullMaskMemory;

            bool hasAnyNulls = (leftNulls.HasValue && leftNulls.Value.Length > 0) || (rightNulls.HasValue && rightNulls.Value.Length > 0);
            bool[]? resultNullMaskArray = null;

            if (hasAnyNulls)
            {
                resultNullMaskArray = new bool[result.Length];
            }

            for (int i = 0; i < result.Length; i++)
            {
                bool leftIsNull = leftNulls.HasValue && leftNulls.Value.Length > 0 && leftNulls.Value.Span[i];
                bool rightIsNull = rightNulls.HasValue && rightNulls.Value.Length > 0 && rightNulls.Value.Span[i];
                bool hasNull = leftIsNull || rightIsNull;

                if (hasNull)
                {
                    result[i] = false; // null compared to anything is false
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = true;
                }
                else
                {
                    result[i] = comparer.Compare(leftData[i], rightData[i]) > 0;
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = false;
                }
            }

            if (resultNullMaskArray != null)
            {
                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Performs vectorized scalar less than comparison
    /// </summary>
    private NivaraColumn<bool> LessThanVectorized(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];

            // Use our helper method for less than comparison
            LessThanTensorPrimitive(data, scalar, result.AsSpan());

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);

                // Set result to false where nulls exist
                for (int i = 0; i < result.Length; i++)
                {
                    if (nullMaskArray[i])
                        result[i] = false;
                }
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for vectorized operations");
        }
    }

    /// <summary>
    /// Performs vectorized element-wise less than comparison
    /// </summary>
    private NivaraColumn<bool> LessThanVectorized(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];

            // Use our helper method for element-wise less than
            LessThanTensorPrimitive(leftData, rightData, result.AsSpan());

            // Handle null propagation
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
                    bool hasNull = leftIsNull || rightIsNull;

                    resultNullMaskArray[i] = hasNull;

                    if (hasNull)
                        result[i] = false;
                }

                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Performs scalar less than comparison using standard comparison
    /// </summary>
    private NivaraColumn<bool> LessThanScalar(T scalar)
    {
        if (storage is MemoryStorage<T> memoryStorage)
        {
            var data = memoryStorage.Data.Span;
            var result = new bool[data.Length];
            var comparer = Comparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var nullMask = memoryStorage.NullMaskMemory;

            for (int i = 0; i < data.Length; i++)
            {
                bool isNull = nullMask.HasValue && nullMask.Value.Length > 0 && nullMask.Value.Span[i];

                if (isNull)
                {
                    result[i] = false; // null compared to anything is false
                }
                else
                {
                    result[i] = comparer.Compare(data[i], scalar) < 0;
                }
            }

            // Copy null mask if it exists
            if (nullMask.HasValue && nullMask.Value.Length > 0)
            {
                var nullMaskArray = new bool[data.Length];
                nullMask.Value.Span.CopyTo(nullMaskArray);
                resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Unsupported storage type for scalar operations");
        }
    }

    /// <summary>
    /// Performs element-wise less than comparison using standard comparison
    /// </summary>
    private NivaraColumn<bool> LessThanScalar(NivaraColumn<T> other)
    {
        if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
        {
            var leftData = leftMemory.Data.Span;
            var rightData = rightMemory.Data.Span;
            var result = new bool[leftData.Length];
            var comparer = Comparer<T>.Default;

            // Handle null propagation
            ReadOnlyMemory<bool>? resultNullMask = null;
            var leftNulls = leftMemory.NullMaskMemory;
            var rightNulls = rightMemory.NullMaskMemory;

            bool hasAnyNulls = (leftNulls.HasValue && leftNulls.Value.Length > 0) || (rightNulls.HasValue && rightNulls.Value.Length > 0);
            bool[]? resultNullMaskArray = null;

            if (hasAnyNulls)
            {
                resultNullMaskArray = new bool[result.Length];
            }

            for (int i = 0; i < result.Length; i++)
            {
                bool leftIsNull = leftNulls.HasValue && leftNulls.Value.Length > 0 && leftNulls.Value.Span[i];
                bool rightIsNull = rightNulls.HasValue && rightNulls.Value.Length > 0 && rightNulls.Value.Span[i];
                bool hasNull = leftIsNull || rightIsNull;

                if (hasNull)
                {
                    result[i] = false; // null compared to anything is false
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = true;
                }
                else
                {
                    result[i] = comparer.Compare(leftData[i], rightData[i]) < 0;
                    if (resultNullMaskArray != null)
                        resultNullMaskArray[i] = false;
                }
            }

            if (resultNullMaskArray != null)
            {
                resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
            }

            var resultStorage = new MemoryStorage<bool>(result, resultNullMask);
            return new NivaraColumn<bool>(resultStorage);
        }
        else
        {
            throw new InvalidOperationException("Cannot perform comparison operations on columns with different storage types");
        }
    }

    /// <summary>
    /// Helper method to perform vectorized equality comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void EqualsTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = EqualityComparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Equals(x[i], y);
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise equality comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void EqualsTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = EqualityComparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Equals(x[i], y[i]);
        }
    }

    /// <summary>
    /// Helper method to perform vectorized greater than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void GreaterThanTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = Comparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Compare(x[i], y) > 0;
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise greater than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void GreaterThanTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = Comparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Compare(x[i], y[i]) > 0;
        }
    }

    /// <summary>
    /// Helper method to perform vectorized less than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void LessThanTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = Comparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Compare(x[i], y) < 0;
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise less than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void LessThanTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        // For now, use scalar comparison with dynamic dispatch
        // TODO: Optimize with TensorPrimitives when generic constraints are resolved
        var comparer = Comparer<T>.Default;
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = comparer.Compare(x[i], y[i]) < 0;
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

    /// <summary>
    /// Helper method to check if a type supports comparison operations
    /// </summary>
    private static bool IsComparableType(Type type)
    {
        // All numeric types support comparison
        if (IsNumericType(type))
            return true;

        // String supports comparison
        if (type == typeof(string))
            return true;

        // DateTime and other common comparable types
        if (type == typeof(DateTime) || type == typeof(DateTimeOffset) || type == typeof(TimeSpan))
            return true;

        // Guid supports comparison
        if (type == typeof(Guid))
            return true;

        // Check if type implements IComparable<T> or IComparable
        return typeof(IComparable<>).MakeGenericType(type).IsAssignableFrom(type) ||
               typeof(IComparable).IsAssignableFrom(type);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            storage?.Dispose();
            disposed = true;
        }
    }
}
