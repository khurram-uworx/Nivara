using Nivara.Diagnostics;
using Nivara.Helpers;
using Nivara.Storage;
using System.Buffers;
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

        // Track column for resource management
        var estimatedSize = EstimateMemoryUsage();
        NivaraResourceManager.TrackResource(this, $"NivaraColumn<{typeof(T).Name}>", estimatedSize);
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
    public Type ElementType => typeof(T);

    /// <inheritdoc />
    public object? GetValue(int index)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException($"Index {index} is out of range for column of length {Length}. Valid range is 0 to {Length - 1}");

        if (IsNull(index))
            return null;

        return storage[index];
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (index < 0 || index >= Length)
                throw new IndexOutOfRangeException($"Index {index} is out of range for column of length {Length}. Valid range is 0 to {Length - 1}");

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
    /// Gets diagnostic information about this column's storage and performance characteristics.
    /// Provides insights for performance analysis and optimization decisions.
    /// </summary>
    public ColumnDiagnostics Diagnostics
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            // Use the StorageType property from the storage interface
            var storageType = storage.StorageType;

            return new ColumnDiagnostics(
                storageType,
                storage.IsVectorizable,
                typeof(T),
                storage.Length,
                storage.HasNulls);
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
    /// Creates a new column from the specified array of nullable value type values.
    /// Automatically detects and tracks null values using null masks.
    /// </summary>
    /// <param name="values">The nullable value type values to store in the column</param>
    /// <returns>A new NivaraColumn instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values array is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when T is not a value type</exception>
    public static NivaraColumn<T> CreateFromNullable(Array values)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));

        // Ensure this method is only used with value types
        if (!typeof(T).IsValueType)
            throw new InvalidOperationException($"CreateFromNullable can only be used with value types. Use CreateForReferenceType for reference types.");

        // Validate the array type - it should be T?[] where T is a value type
        var actualElementType = values.GetType().GetElementType();
        var expectedNullableType = typeof(Nullable<>).MakeGenericType(typeof(T));

        if (actualElementType != expectedNullableType)
            throw new ArgumentException($"Array element type must be {expectedNullableType.Name}, but was {actualElementType?.Name}");

        // Handle empty array
        if (values.Length == 0)
        {
            return new NivaraColumn<T>(new MemoryStorage<T>(ReadOnlySpan<T>.Empty));
        }

        // Process nullable values manually
        var dataArray = new T[values.Length];
        var nullMaskArray = new bool[values.Length];
        bool hasNulls = false;

        for (int i = 0; i < values.Length; i++)
        {
            var value = values.GetValue(i);
            if (value == null)
            {
                dataArray[i] = default(T)!;
                nullMaskArray[i] = true;
                hasNulls = true;
            }
            else
            {
                dataArray[i] = (T)value;
                nullMaskArray[i] = false;
            }
        }

        var data = new ReadOnlyMemory<T>(dataArray);
        var nullMask = hasNulls ? new ReadOnlyMemory<bool>(nullMaskArray) : null;

        var storage = new MemoryStorage<T>(data, nullMask);
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

        // Enhanced validation with clear error messages
        if (start < 0)
            throw new ArgumentOutOfRangeException(nameof(start), start, $"Start index cannot be negative. Provided: {start}");

        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Length cannot be negative. Provided: {length}");

        if (start > Length)
            throw new ArgumentOutOfRangeException(nameof(start), start, $"Start index {start} is out of range for column of length {Length}. Valid range is 0 to {Length}");

        if (start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Slice range [{start}, {start + length}) exceeds column bounds [0, {Length}). Maximum length from start {start} is {Length - start}");

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

    // Tensor<T> Conversion

    /// <summary>
    /// Converts this column to a System.Numerics.Tensors.Tensor&lt;T&gt;.
    /// The returned tensor owns a copy so column immutability is preserved.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the column contains null values</exception>
    public Tensor<T> ToTensor()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (HasNulls)
            throw new InvalidOperationException("Cannot convert column with null values to Tensor<T>. Use ToTensor(T defaultValue) to provide a null replacement.");

        var rawSpan = storage.AsSpan();
        var result = new T[rawSpan.Length];
        rawSpan.CopyTo(result);
        return Tensor.Create(result, new nint[] { result.Length });
    }

    /// <summary>
    /// Converts this column to a Tensor&lt;T&gt;, replacing null values with the specified default.
    /// </summary>
    /// <param name="defaultValue">Value to use in place of nulls</param>
    public Tensor<T> ToTensor(T defaultValue)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var rawSpan = storage.AsSpan();
        var result = new T[rawSpan.Length];
        rawSpan.CopyTo(result);
        var nullMask = storage.NullMask;
        if (!nullMask.IsEmpty)
        {
            for (int i = 0; i < result.Length; i++)
            {
                if (nullMask[i]) result[i] = defaultValue;
            }
        }
        return Tensor.Create(result, new nint[] { result.Length });
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

        // Use centralized validation
        ValidateTypeSupportsOperation("arithmetic");

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

        // Use centralized validation
        ValidateCompatibleLength(other, "addition");
        ValidateTypeSupportsOperation("arithmetic");

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

        // Use centralized validation
        ValidateCompatibleLength(other, "multiplication");
        ValidateTypeSupportsOperation("arithmetic");

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
        // Determine which kernel will actually be used
        var kernelType = DetermineKernelType();

        // Record diagnostic information
        var diagnostic = new OperationDiagnostics(
            "ScalarMultiplication",
            kernelType,
            Length,
            typeof(T),
            HasNulls);
        DiagnosticsTracker.RecordOperation(diagnostic);

        // Handle both TensorStorage and MemoryStorage
        if (storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledDataBuffer = null;
            try
            {
                var dataBuffer = Length >= 1024
                    ? (pooledDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    dataBuffer[i] = storage[i];
                }
                var result = new T[Length];

                // Use our helper method for multiplication
                MultiplyTensorPrimitive(dataBuffer.AsSpan(0, Length), scalar, result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var nullMask = storage.NullMask;
                if (!nullMask.IsEmpty)
                {
                    var nullMaskArray = nullMask.ToArray();
                    resultNullMask = new ReadOnlyMemory<bool>(nullMaskArray);
                }

                // Create result storage using the factory, passing the null mask
                var resultStorage = ColumnStorageFactory.Create(result.AsSpan(), resultNullMask);
                return new NivaraColumn<T>(resultStorage);
            }
            finally
            {
                if (pooledDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> memoryStorage)
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
        // Determine which kernel will actually be used
        var kernelType = DetermineKernelType();

        // Record diagnostic information
        var diagnostic = new OperationDiagnostics(
            "ElementwiseAddition",
            kernelType,
            Length,
            typeof(T),
            HasNulls || other.HasNulls);
        DiagnosticsTracker.RecordOperation(diagnostic);

        // Handle both TensorStorage and MemoryStorage combinations
        if (storage.StorageType == StorageType.Tensor && other.storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledLeftDataBuffer = null;
            T[]? pooledRightDataBuffer = null;
            try
            {
                var leftDataBuffer = Length >= 1024
                    ? (pooledLeftDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    leftDataBuffer[i] = storage[i];
                }
                var rightDataBuffer = Length >= 1024
                    ? (pooledRightDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < other.storage.Length; i++)
                {
                    rightDataBuffer[i] = other.storage[i];
                }
                var result = new T[Length];

                // Use our helper method for addition
                AddTensorPrimitive(leftDataBuffer.AsSpan(0, Length), rightDataBuffer.AsSpan(0, Length), result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var leftNullMask = storage.NullMask;
                var rightNullMask = other.storage.NullMask;

                if (!leftNullMask.IsEmpty || !rightNullMask.IsEmpty)
                {
                    var resultNullMaskArray = new bool[result.Length];

                    for (int i = 0; i < result.Length; i++)
                    {
                        bool leftIsNull = !leftNullMask.IsEmpty && leftNullMask[i];
                        bool rightIsNull = !rightNullMask.IsEmpty && rightNullMask[i];
                        resultNullMaskArray[i] = leftIsNull || rightIsNull;
                    }

                    resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
                }

                // Create result storage using the factory, passing the null mask
                var resultStorage = ColumnStorageFactory.Create(result.AsSpan(), resultNullMask);
                return new NivaraColumn<T>(resultStorage);
            }
            finally
            {
                if (pooledLeftDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledLeftDataBuffer);
                if (pooledRightDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledRightDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
        // Handle both TensorStorage and MemoryStorage combinations
        if (storage.StorageType == StorageType.Tensor && other.storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledLeftDataBuffer = null;
            T[]? pooledRightDataBuffer = null;
            try
            {
                var leftDataBuffer = Length >= 1024
                    ? (pooledLeftDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    leftDataBuffer[i] = storage[i];
                }
                var rightDataBuffer = Length >= 1024
                    ? (pooledRightDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < other.storage.Length; i++)
                {
                    rightDataBuffer[i] = other.storage[i];
                }
                var result = new T[Length];

                // Use our helper method for multiplication
                MultiplyTensorPrimitive(leftDataBuffer.AsSpan(0, Length), rightDataBuffer.AsSpan(0, Length), result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var leftNullMask = storage.NullMask;
                var rightNullMask = other.storage.NullMask;

                if (!leftNullMask.IsEmpty || !rightNullMask.IsEmpty)
                {
                    var resultNullMaskArray = new bool[result.Length];

                    for (int i = 0; i < result.Length; i++)
                    {
                        bool leftIsNull = !leftNullMask.IsEmpty && leftNullMask[i];
                        bool rightIsNull = !rightNullMask.IsEmpty && rightNullMask[i];
                        resultNullMaskArray[i] = leftIsNull || rightIsNull;
                    }

                    resultNullMask = new ReadOnlyMemory<bool>(resultNullMaskArray);
                }

                // Create result storage using the factory, passing the null mask
                var resultStorage = ColumnStorageFactory.Create(result.AsSpan(), resultNullMask);
                return new NivaraColumn<T>(resultStorage);
            }
            finally
            {
                if (pooledLeftDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledLeftDataBuffer);
                if (pooledRightDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledRightDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
        var type = typeof(T);

        // Use TensorPrimitives for supported types, fall back to scalar for others
        if (type == typeof(float))
        {
            MultiplyFloat(x, y, destination);
        }
        else if (type == typeof(double))
        {
            MultiplyDouble(x, y, destination);
        }
        else
        {
            // Fall back to scalar multiplication with dynamic dispatch for other types
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (T)(object)((dynamic)x[i]! * (dynamic)y!)!;
            }
        }
    }

    private static void MultiplyFloat(ReadOnlySpan<T> x, T y, Span<T> destination)
    {
        var xFloat = new float[x.Length];
        var destFloat = new float[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xFloat[i] = (float)(object)x[i]!;
        }

        var yFloat = (float)(object)y!;
        TensorPrimitives.Multiply(xFloat, yFloat, destFloat);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destFloat[i];
        }
    }

    private static void MultiplyDouble(ReadOnlySpan<T> x, T y, Span<T> destination)
    {
        var xDouble = new double[x.Length];
        var destDouble = new double[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xDouble[i] = (double)(object)x[i]!;
        }

        var yDouble = (double)(object)y!;
        TensorPrimitives.Multiply(xDouble, yDouble, destDouble);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destDouble[i];
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise multiplication using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void MultiplyTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var type = typeof(T);

        // Use TensorPrimitives for supported types, fall back to scalar for others
        if (type == typeof(float))
        {
            MultiplyFloatElementwise(x, y, destination);
        }
        else if (type == typeof(double))
        {
            MultiplyDoubleElementwise(x, y, destination);
        }
        else
        {
            // Fall back to scalar multiplication with dynamic dispatch for other types
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (T)(object)((dynamic)x[i]! * (dynamic)y[i]!)!;
            }
        }
    }

    private static void MultiplyFloatElementwise(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var xFloat = new float[x.Length];
        var yFloat = new float[y.Length];
        var destFloat = new float[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xFloat[i] = (float)(object)x[i]!;
            yFloat[i] = (float)(object)y[i]!;
        }

        TensorPrimitives.Multiply(xFloat, yFloat, destFloat);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destFloat[i];
        }
    }

    private static void MultiplyDoubleElementwise(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var xDouble = new double[x.Length];
        var yDouble = new double[y.Length];
        var destDouble = new double[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xDouble[i] = (double)(object)x[i]!;
            yDouble[i] = (double)(object)y[i]!;
        }

        TensorPrimitives.Multiply(xDouble, yDouble, destDouble);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destDouble[i];
        }
    }

    /// <summary>
    /// Helper method to perform vectorized addition using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void AddTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var type = typeof(T);

        // Use TensorPrimitives for supported types, fall back to scalar for others
        if (type == typeof(float))
        {
            AddFloatElementwise(x, y, destination);
        }
        else if (type == typeof(double))
        {
            AddDoubleElementwise(x, y, destination);
        }
        else
        {
            // Fall back to scalar addition with dynamic dispatch for other types
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (T)(object)((dynamic)x[i]! + (dynamic)y[i]!)!;
            }
        }
    }

    private static void AddFloatElementwise(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var xFloat = new float[x.Length];
        var yFloat = new float[y.Length];
        var destFloat = new float[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xFloat[i] = (float)(object)x[i]!;
            yFloat[i] = (float)(object)y[i]!;
        }

        TensorPrimitives.Add(xFloat, yFloat, destFloat);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destFloat[i];
        }
    }

    private static void AddDoubleElementwise(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        var xDouble = new double[x.Length];
        var yDouble = new double[y.Length];
        var destDouble = new double[destination.Length];

        for (int i = 0; i < x.Length; i++)
        {
            xDouble[i] = (double)(object)x[i]!;
            yDouble[i] = (double)(object)y[i]!;
        }

        TensorPrimitives.Add(xDouble, yDouble, destDouble);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (T)(object)destDouble[i];
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

        // Use centralized validation
        ValidateCompatibleLength(other, "equality comparison");

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

        // Use centralized validation
        ValidateTypeSupportsOperation("comparison");

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

        // Use centralized validation
        ValidateCompatibleLength(other, "greater than comparison");
        ValidateTypeSupportsOperation("comparison");

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

        // Use centralized validation
        ValidateTypeSupportsOperation("comparison");

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

        // Use centralized validation
        ValidateCompatibleLength(other, "less than comparison");
        ValidateTypeSupportsOperation("comparison");

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
        // Determine which kernel will actually be used
        var kernelType = DetermineKernelType();

        // Record diagnostic information
        var diagnostic = new OperationDiagnostics(
            "ScalarEquals",
            kernelType,
            Length,
            typeof(T),
            HasNulls);
        DiagnosticsTracker.RecordOperation(diagnostic);

        // Handle both TensorStorage and MemoryStorage
        if (storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledDataBuffer = null;
            try
            {
                var dataBuffer = Length >= 1024
                    ? (pooledDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    dataBuffer[i] = storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for equality comparison
                EqualsTensorPrimitive(dataBuffer.AsSpan(0, Length), scalar, result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var nullMask = storage.NullMask;
                if (!nullMask.IsEmpty)
                {
                    var nullMaskArray = nullMask.ToArray();
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
            finally
            {
                if (pooledDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> memoryStorage)
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
        // Handle both TensorStorage and MemoryStorage combinations
        if (storage.StorageType == StorageType.Tensor && other.storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledLeftDataBuffer = null;
            T[]? pooledRightDataBuffer = null;
            try
            {
                var leftDataBuffer = Length >= 1024
                    ? (pooledLeftDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    leftDataBuffer[i] = storage[i];
                }
                var rightDataBuffer = Length >= 1024
                    ? (pooledRightDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < other.storage.Length; i++)
                {
                    rightDataBuffer[i] = other.storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for element-wise equality
                EqualsTensorPrimitive(leftDataBuffer.AsSpan(0, Length), rightDataBuffer.AsSpan(0, Length), result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var leftNullMask = storage.NullMask;
                var rightNullMask = other.storage.NullMask;

                if (!leftNullMask.IsEmpty || !rightNullMask.IsEmpty)
                {
                    var resultNullMaskArray = new bool[result.Length];

                    for (int i = 0; i < result.Length; i++)
                    {
                        bool leftIsNull = !leftNullMask.IsEmpty && leftNullMask[i];
                        bool rightIsNull = !rightNullMask.IsEmpty && rightNullMask[i];
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
            finally
            {
                if (pooledLeftDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledLeftDataBuffer);
                if (pooledRightDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledRightDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
        // Handle both TensorStorage and MemoryStorage
        if (storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledDataBuffer = null;
            try
            {
                var dataBuffer = Length >= 1024
                    ? (pooledDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    dataBuffer[i] = storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for greater than comparison
                GreaterThanTensorPrimitive(dataBuffer.AsSpan(0, Length), scalar, result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var nullMask = storage.NullMask;
                if (!nullMask.IsEmpty)
                {
                    var nullMaskArray = nullMask.ToArray();
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
            finally
            {
                if (pooledDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> memoryStorage)
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
        // Handle both TensorStorage and MemoryStorage combinations
        if (storage.StorageType == StorageType.Tensor && other.storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledLeftDataBuffer = null;
            T[]? pooledRightDataBuffer = null;
            try
            {
                var leftDataBuffer = Length >= 1024
                    ? (pooledLeftDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    leftDataBuffer[i] = storage[i];
                }
                var rightDataBuffer = Length >= 1024
                    ? (pooledRightDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < other.storage.Length; i++)
                {
                    rightDataBuffer[i] = other.storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for element-wise greater than
                GreaterThanTensorPrimitive(leftDataBuffer.AsSpan(0, Length), rightDataBuffer.AsSpan(0, Length), result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var leftNullMask = storage.NullMask;
                var rightNullMask = other.storage.NullMask;

                if (!leftNullMask.IsEmpty || !rightNullMask.IsEmpty)
                {
                    var resultNullMaskArray = new bool[result.Length];

                    for (int i = 0; i < result.Length; i++)
                    {
                        bool leftIsNull = !leftNullMask.IsEmpty && leftNullMask[i];
                        bool rightIsNull = !rightNullMask.IsEmpty && rightNullMask[i];
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
            finally
            {
                if (pooledLeftDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledLeftDataBuffer);
                if (pooledRightDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledRightDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
        // Handle both TensorStorage and MemoryStorage
        if (storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledDataBuffer = null;
            try
            {
                var dataBuffer = Length >= 1024
                    ? (pooledDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    dataBuffer[i] = storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for less than comparison
                LessThanTensorPrimitive(dataBuffer.AsSpan(0, Length), scalar, result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var nullMask = storage.NullMask;
                if (!nullMask.IsEmpty)
                {
                    var nullMaskArray = nullMask.ToArray();
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
            finally
            {
                if (pooledDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> memoryStorage)
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
        // Handle both TensorStorage and MemoryStorage combinations
        if (storage.StorageType == StorageType.Tensor && other.storage.StorageType == StorageType.Tensor)
        {
            T[]? pooledLeftDataBuffer = null;
            T[]? pooledRightDataBuffer = null;
            try
            {
                var leftDataBuffer = Length >= 1024
                    ? (pooledLeftDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < storage.Length; i++)
                {
                    leftDataBuffer[i] = storage[i];
                }
                var rightDataBuffer = Length >= 1024
                    ? (pooledRightDataBuffer = ArrayPool<T>.Shared.Rent(Length))
                    : new T[Length];
                for (int i = 0; i < other.storage.Length; i++)
                {
                    rightDataBuffer[i] = other.storage[i];
                }
                var result = new bool[Length];

                // Use our helper method for element-wise less than
                LessThanTensorPrimitive(leftDataBuffer.AsSpan(0, Length), rightDataBuffer.AsSpan(0, Length), result.AsSpan());

                // Handle null propagation for tensor storage
                ReadOnlyMemory<bool>? resultNullMask = null;
                var leftNullMask = storage.NullMask;
                var rightNullMask = other.storage.NullMask;

                if (!leftNullMask.IsEmpty || !rightNullMask.IsEmpty)
                {
                    var resultNullMaskArray = new bool[result.Length];

                    for (int i = 0; i < result.Length; i++)
                    {
                        bool leftIsNull = !leftNullMask.IsEmpty && leftNullMask[i];
                        bool rightIsNull = !rightNullMask.IsEmpty && rightNullMask[i];
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
            finally
            {
                if (pooledLeftDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledLeftDataBuffer);
                if (pooledRightDataBuffer != null)
                    ArrayPool<T>.Shared.Return(pooledRightDataBuffer);
            }
        }
        else if (storage is MemoryStorage<T> leftMemory && other.storage is MemoryStorage<T> rightMemory)
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
    /// Helper method to perform vectorized equality comparison using optimized loops with runtime type dispatch
    /// </summary>
    private static void EqualsTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        var type = typeof(T);

        // Use optimized loops for supported types, fall back to standard comparison for others
        if (type == typeof(float))
        {
            var yFloat = (float)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! == yFloat;
            }
        }
        else if (type == typeof(double))
        {
            var yDouble = (double)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! == yDouble;
            }
        }
        else if (type == typeof(int))
        {
            var yInt = (int)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! == yInt;
            }
        }
        else if (type == typeof(long))
        {
            var yLong = (long)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! == yLong;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = EqualityComparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Equals(x[i], y);
            }
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise equality comparison using optimized loops with runtime type dispatch
    /// </summary>
    private static void EqualsTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        var type = typeof(T);

        // Use optimized loops for supported types, fall back to standard comparison for others
        if (type == typeof(float))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! == (float)(object)y[i]!;
            }
        }
        else if (type == typeof(double))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! == (double)(object)y[i]!;
            }
        }
        else if (type == typeof(int))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! == (int)(object)y[i]!;
            }
        }
        else if (type == typeof(long))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! == (long)(object)y[i]!;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = EqualityComparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Equals(x[i], y[i]);
            }
        }
    }

    /// <summary>
    /// Helper method to perform vectorized greater than comparison using optimized loops with runtime type dispatch
    /// </summary>
    private static void GreaterThanTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        var type = typeof(T);

        // Use optimized loops for supported types, fall back to standard comparison for others
        if (type == typeof(float))
        {
            var yFloat = (float)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! > yFloat;
            }
        }
        else if (type == typeof(double))
        {
            var yDouble = (double)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! > yDouble;
            }
        }
        else if (type == typeof(int))
        {
            var yInt = (int)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! > yInt;
            }
        }
        else if (type == typeof(long))
        {
            var yLong = (long)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! > yLong;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = Comparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Compare(x[i], y) > 0;
            }
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise greater than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void GreaterThanTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        var type = typeof(T);

        // For comparison operations, we use optimized loops since TensorPrimitives doesn't have direct boolean comparison methods
        // However, we can still benefit from vectorization through the JIT compiler
        if (type == typeof(float))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! > (float)(object)y[i]!;
            }
        }
        else if (type == typeof(double))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! > (double)(object)y[i]!;
            }
        }
        else if (type == typeof(int))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! > (int)(object)y[i]!;
            }
        }
        else if (type == typeof(long))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! > (long)(object)y[i]!;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = Comparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Compare(x[i], y[i]) > 0;
            }
        }
    }

    /// <summary>
    /// Helper method to perform vectorized less than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void LessThanTensorPrimitive(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        var type = typeof(T);

        // For comparison operations, we use optimized loops since TensorPrimitives doesn't have direct boolean comparison methods
        // However, we can still benefit from vectorization through the JIT compiler
        if (type == typeof(float))
        {
            var yFloat = (float)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! < yFloat;
            }
        }
        else if (type == typeof(double))
        {
            var yDouble = (double)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! < yDouble;
            }
        }
        else if (type == typeof(int))
        {
            var yInt = (int)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! < yInt;
            }
        }
        else if (type == typeof(long))
        {
            var yLong = (long)(object)y!;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! < yLong;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = Comparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Compare(x[i], y) < 0;
            }
        }
    }

    /// <summary>
    /// Helper method to perform vectorized element-wise less than comparison using TensorPrimitives with runtime type dispatch
    /// </summary>
    private static void LessThanTensorPrimitive(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        var type = typeof(T);

        // For comparison operations, we use optimized loops since TensorPrimitives doesn't have direct boolean comparison methods
        // However, we can still benefit from vectorization through the JIT compiler
        if (type == typeof(float))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (float)(object)x[i]! < (float)(object)y[i]!;
            }
        }
        else if (type == typeof(double))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (double)(object)x[i]! < (double)(object)y[i]!;
            }
        }
        else if (type == typeof(int))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (int)(object)x[i]! < (int)(object)y[i]!;
            }
        }
        else if (type == typeof(long))
        {
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = (long)(object)x[i]! < (long)(object)y[i]!;
            }
        }
        else
        {
            // Fall back to standard comparison for other types
            var comparer = Comparer<T>.Default;
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = comparer.Compare(x[i], y[i]) < 0;
            }
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

    // Null Handling Methods

    /// <summary>
    /// Gets the number of null values in the column
    /// </summary>
    /// <returns>The count of null values</returns>
    public int NullCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (!HasNulls)
                return 0;

            var nullMask = storage.NullMask;
            int count = 0;
            for (int i = 0; i < nullMask.Length; i++)
            {
                if (nullMask[i])
                    count++;
            }
            return count;
        }
    }

    /// <summary>
    /// Gets the indices of all null values in the column
    /// </summary>
    /// <returns>An array of indices where null values are located</returns>
    public int[] GetNullIndices()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (!HasNulls)
            return Array.Empty<int>();

        var nullMask = storage.NullMask;
        var indices = new List<int>();

        for (int i = 0; i < nullMask.Length; i++)
        {
            if (nullMask[i])
                indices.Add(i);
        }

        return indices.ToArray();
    }

    /// <summary>
    /// Creates a new column with all null values replaced by the specified value
    /// </summary>
    /// <param name="fillValue">The value to use for replacing nulls</param>
    /// <returns>A new column with nulls filled</returns>
    /// <exception cref="ArgumentNullException">Thrown when fillValue is null for reference types</exception>
    public NivaraColumn<T> FillNull(T fillValue)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Enhanced validation with clear error messages for reference types
        if (!typeof(T).IsValueType && fillValue == null)
            throw new ArgumentNullException(nameof(fillValue), $"Fill value cannot be null for reference type {typeof(T).Name}. Provide a non-null value to replace null entries.");

        // If no nulls, return a copy of the current column
        if (!HasNulls)
        {
            return Slice(0, Length);
        }

        // Create new data array with nulls filled
        var nullMask = storage.NullMask;
        var newData = new T[Length];

        for (int i = 0; i < Length; i++)
        {
            if (nullMask[i])
            {
                newData[i] = fillValue;
            }
            else
            {
                newData[i] = storage[i];
            }
        }

        // Create storage without null mask since all nulls are filled
        var newStorage = CreateStorage(newData.AsSpan());
        return new NivaraColumn<T>(newStorage);
    }

    // Transformation Operations

    /// <summary>
    /// Applies a transformation function to each element in the column.
    /// Creates a new column with the transformed values while preserving null semantics.
    /// </summary>
    /// <typeparam name="TResult">The type of the result column</typeparam>
    /// <param name="transform">The transformation function to apply to each element</param>
    /// <returns>A new column with transformed values</returns>
    /// <exception cref="ArgumentNullException">Thrown when transform is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when transformation throws an exception</exception>
    public NivaraColumn<TResult> Transform<TResult>(Func<T, TResult> transform)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (transform == null)
            throw new ArgumentNullException(nameof(transform));

        var result = new TResult[Length];
        var resultNullMask = new bool[Length];
        bool hasResultNulls = false;

        // Apply transformation with proper null handling
        for (int i = 0; i < Length; i++)
        {
            if (IsNull(i))
            {
                // Null values remain null in the result
                result[i] = default(TResult)!;
                resultNullMask[i] = true;
                hasResultNulls = true;
            }
            else
            {
                try
                {
                    var transformedValue = transform(storage[i]);

                    // Check if the transformed value is null for reference types
                    if (!typeof(TResult).IsValueType && transformedValue == null)
                    {
                        result[i] = default(TResult)!;
                        resultNullMask[i] = true;
                        hasResultNulls = true;
                    }
                    else
                    {
                        result[i] = transformedValue;
                        resultNullMask[i] = false;
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Transformation function threw an exception at index {i}. " +
                        $"Input value: {storage[i]}, Exception: {ex.Message}", ex);
                }
            }
        }

        // Create result column with appropriate null mask
        ReadOnlyMemory<bool>? finalNullMask = hasResultNulls ? new ReadOnlyMemory<bool>(resultNullMask) : null;

        if (typeof(TResult).IsValueType)
        {
            // For value types, use CreateFromNullable if there are nulls
            if (hasResultNulls)
            {
                var nullableType = typeof(Nullable<>).MakeGenericType(typeof(TResult));
                var nullableArray = System.Array.CreateInstance(nullableType, Length);

                for (int i = 0; i < Length; i++)
                {
                    if (resultNullMask[i])
                    {
                        nullableArray.SetValue(null, i);
                    }
                    else
                    {
                        var nullableInstance = Activator.CreateInstance(nullableType, result[i]);
                        nullableArray.SetValue(nullableInstance, i);
                    }
                }

                return (NivaraColumn<TResult>)typeof(NivaraColumn<>)
                    .MakeGenericType(typeof(TResult))
                    .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                    .Invoke(null, new object[] { nullableArray })!;
            }
            else
            {
                // No nulls, use regular Create
                return NivaraColumn<TResult>.Create(result);
            }
        }
        else
        {
            // For reference types, use CreateForReferenceType
            return NivaraColumn<TResult>.CreateForReferenceType(result);
        }
    }

    /// <summary>
    /// Applies a transformation function to each non-null element in the column.
    /// Null values are preserved as null in the result without applying the transformation.
    /// </summary>
    /// <typeparam name="TResult">The type of the result column</typeparam>
    /// <param name="transform">The transformation function to apply to non-null elements</param>
    /// <returns>A new column with transformed values</returns>
    /// <exception cref="ArgumentNullException">Thrown when transform is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when transformation throws an exception</exception>
    public NivaraColumn<TResult> TransformNonNull<TResult>(Func<T, TResult> transform)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (transform == null)
            throw new ArgumentNullException(nameof(transform));

        var result = new TResult[Length];
        var resultNullMask = new bool[Length];
        bool hasResultNulls = false;

        // Apply transformation only to non-null values
        for (int i = 0; i < Length; i++)
        {
            if (IsNull(i))
            {
                // Preserve null values without transformation
                result[i] = default(TResult)!;
                resultNullMask[i] = true;
                hasResultNulls = true;
            }
            else
            {
                try
                {
                    var transformedValue = transform(storage[i]);

                    // Check if the transformed value is null for reference types
                    if (!typeof(TResult).IsValueType && transformedValue == null)
                    {
                        result[i] = default(TResult)!;
                        resultNullMask[i] = true;
                        hasResultNulls = true;
                    }
                    else
                    {
                        result[i] = transformedValue;
                        resultNullMask[i] = false;
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Transformation function threw an exception at index {i}. " +
                        $"Input value: {storage[i]}, Exception: {ex.Message}", ex);
                }
            }
        }

        // Create result column with appropriate null mask
        if (typeof(TResult).IsValueType)
        {
            // For value types, use CreateFromNullable if there are nulls
            if (hasResultNulls)
            {
                var nullableType = typeof(Nullable<>).MakeGenericType(typeof(TResult));
                var nullableArray = System.Array.CreateInstance(nullableType, Length);

                for (int i = 0; i < Length; i++)
                {
                    if (resultNullMask[i])
                    {
                        nullableArray.SetValue(null, i);
                    }
                    else
                    {
                        var nullableInstance = Activator.CreateInstance(nullableType, result[i]);
                        nullableArray.SetValue(nullableInstance, i);
                    }
                }

                return (NivaraColumn<TResult>)typeof(NivaraColumn<>)
                    .MakeGenericType(typeof(TResult))
                    .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                    .Invoke(null, new object[] { nullableArray })!;
            }
            else
            {
                // No nulls, use regular Create
                return NivaraColumn<TResult>.Create(result);
            }
        }
        else
        {
            // For reference types, use CreateForReferenceType
            return NivaraColumn<TResult>.CreateForReferenceType(result);
        }
    }

    private KernelType DetermineKernelType()
    {
        return KernelSelector.DetermineKernelType(Length, storage.IsVectorizable);
    }

    /// <summary>
    /// Validates that the column is not empty for operations that require data
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty</exception>
    private void ValidateNotEmpty()
    {
        if (Length == 0)
            throw new InvalidOperationException("Cannot perform operation on empty column. Column must contain at least one element.");
    }

    /// <summary>
    /// Validates that two columns have compatible lengths for binary operations
    /// </summary>
    /// <param name="other">The other column to validate against</param>
    /// <param name="operationName">The name of the operation being performed</param>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    private void ValidateCompatibleLength(NivaraColumn<T> other, string operationName)
    {
        if (Length != other.Length)
            throw new ArgumentException($"Cannot perform {operationName} on columns of different lengths: {Length} vs {other.Length}. Both columns must have the same number of elements.");
    }

    /// <summary>
    /// Validates that the type supports the specified operation
    /// </summary>
    /// <param name="operationType">The type of operation being validated</param>
    /// <exception cref="InvalidOperationException">Thrown when the type doesn't support the operation</exception>
    private static void ValidateTypeSupportsOperation(string operationType)
    {
        switch (operationType.ToLowerInvariant())
        {
            case "arithmetic":
                if (!IsNumericType(typeof(T)))
                    throw new InvalidOperationException($"Arithmetic operations are not supported for type {typeof(T).Name}. Only numeric types (int, float, double, long, etc.) support arithmetic operations.");
                if (!ColumnStorageFactory.IsVectorizable<T>())
                    throw new InvalidOperationException($"Arithmetic operations are not supported for non-vectorizable type {typeof(T).Name}. Only numeric primitive types support vectorized arithmetic.");
                break;
            case "comparison":
                if (!IsComparableType(typeof(T)))
                    throw new InvalidOperationException($"Comparison operations are not supported for type {typeof(T).Name}. Only comparable types support comparison operations.");
                break;
            default:
                throw new ArgumentException($"Unknown operation type: {operationType}", nameof(operationType));
        }
    }

    /// <summary>
    /// Creates a new column with null values filled using forward fill strategy.
    /// Each null value is replaced with the last non-null value that appeared before it.
    /// </summary>
    /// <returns>A new column with nulls forward-filled</returns>
    /// <exception cref="InvalidOperationException">Thrown when the first element is null (no value to forward fill)</exception>
    public NivaraColumn<T> FillNullForward()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Validate that the column is not empty
        ValidateNotEmpty();

        // If no nulls, return a copy of the current column
        if (!HasNulls)
        {
            return Slice(0, Length);
        }

        var nullMask = storage.NullMask;
        var newData = new T[Length];
        T? lastValidValue = default;
        bool hasValidValue = false;

        for (int i = 0; i < Length; i++)
        {
            if (nullMask[i])
            {
                if (!hasValidValue)
                    throw new InvalidOperationException($"Cannot forward fill: null value at index {i} has no preceding non-null value. Consider using FillNull() with a default value instead.");

                newData[i] = lastValidValue!;
            }
            else
            {
                var currentValue = storage[i];
                newData[i] = currentValue;
                lastValidValue = currentValue;
                hasValidValue = true;
            }
        }

        // Create storage without null mask since all nulls are filled
        var newStorage = CreateStorage(newData.AsSpan());
        return new NivaraColumn<T>(newStorage);
    }

    /// <summary>
    /// Creates a new column with null values filled using backward fill strategy.
    /// Each null value is replaced with the next non-null value that appears after it.
    /// </summary>
    /// <returns>A new column with nulls backward-filled</returns>
    /// <exception cref="InvalidOperationException">Thrown when the last element is null (no value to backward fill)</exception>
    public NivaraColumn<T> FillNullBackward()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Validate that the column is not empty
        ValidateNotEmpty();

        // If no nulls, return a copy of the current column
        if (!HasNulls)
        {
            return Slice(0, Length);
        }

        var nullMask = storage.NullMask;
        var newData = new T[Length];
        T? nextValidValue = default;
        bool hasValidValue = false;

        // Process backwards to find next valid values
        for (int i = Length - 1; i >= 0; i--)
        {
            if (nullMask[i])
            {
                if (!hasValidValue)
                    throw new InvalidOperationException($"Cannot backward fill: null value at index {i} has no following non-null value. Consider using FillNull() with a default value instead.");

                newData[i] = nextValidValue!;
            }
            else
            {
                var currentValue = storage[i];
                newData[i] = currentValue;
                nextValidValue = currentValue;
                hasValidValue = true;
            }
        }

        // Create storage without null mask since all nulls are filled
        var newStorage = CreateStorage(newData.AsSpan());
        return new NivaraColumn<T>(newStorage);
    }

    /// <summary>
    /// Creates a new column containing only the non-null values from this column
    /// </summary>
    /// <returns>A new column with all null values removed</returns>
    public NivaraColumn<T> DropNulls()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // If no nulls, return a copy of the current column
        if (!HasNulls)
        {
            return Slice(0, Length);
        }

        var nullMask = storage.NullMask;
        var nonNullValues = new List<T>();

        for (int i = 0; i < Length; i++)
        {
            if (!nullMask[i])
            {
                nonNullValues.Add(storage[i]);
            }
        }

        // Create storage without null mask since all nulls are removed
        var newStorage = CreateStorage(nonNullValues.ToArray().AsSpan());
        return new NivaraColumn<T>(newStorage);
    }

    /// <summary>
    /// Converts the column to an array
    /// </summary>
    /// <returns>An array containing all values from the column</returns>
    public T[] ToArray()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var result = new T[Length];
        for (int i = 0; i < Length; i++)
        {
            result[i] = storage[i];
        }
        return result;
    }

    /// <summary>
    /// Gets a read-only span view of the underlying data.
    /// Provides zero-copy access to the column data for high-performance operations.
    /// </summary>
    /// <returns>A read-only span over the column data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the storage doesn't support span access</exception>
    internal ReadOnlySpan<T> AsSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return storage.AsSpan();
    }

    /// <summary>
    /// Gets a writable span view of the underlying data.
    /// Provides zero-copy access for scenarios requiring data mutation.
    /// Note: This may create a copy for immutable storage implementations.
    /// </summary>
    /// <returns>A writable span over the column data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the storage doesn't support writable span access</exception>
    internal Span<T> AsWritableSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return storage.AsWritableSpan();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Untrack from resource manager
            NivaraResourceManager.UntrackResource(this);

            storage?.Dispose();
            disposed = true;
        }
    }

    /// <summary>
    /// Estimates the memory usage of this column
    /// </summary>
    /// <returns>Estimated memory usage in bytes</returns>
    private long EstimateMemoryUsage()
    {
        var elementSize = GetTypeSize(typeof(T));
        var baseMemory = Length * elementSize;

        // Add overhead for null mask if present
        if (HasNulls)
        {
            baseMemory += Length; // 1 byte per boolean in null mask
        }

        // Add some overhead for object structure
        return baseMemory + 64; // 64 bytes overhead estimate
    }

    /// <summary>
    /// Gets the approximate size of a type in bytes
    /// </summary>
    /// <param name="type">The type to get size for</param>
    /// <returns>Size in bytes</returns>
    private static int GetTypeSize(Type type)
    {
        if (type == typeof(bool)) return 1;
        if (type == typeof(byte) || type == typeof(sbyte)) return 1;
        if (type == typeof(short) || type == typeof(ushort)) return 2;
        if (type == typeof(int) || type == typeof(uint)) return 4;
        if (type == typeof(long) || type == typeof(ulong)) return 8;
        if (type == typeof(float)) return 4;
        if (type == typeof(double)) return 8;
        if (type == typeof(decimal)) return 16;
        if (type == typeof(DateTime)) return 8;
        if (type == typeof(Guid)) return 16;

        // For reference types, estimate pointer size + average string length
        if (!type.IsValueType)
        {
            if (type == typeof(string)) return 50; // Average string estimate
            return 8; // Pointer size on 64-bit systems
        }

        return 8; // Default estimate for unknown types
    }
}
