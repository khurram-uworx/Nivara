using System.Numerics;

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