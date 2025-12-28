namespace Nivara;

/// <summary>
/// Tensor-backed storage implementation for vectorizable types.
/// Uses arrays for now, will be optimized with System.Numerics.Tensors later.
/// </summary>
/// <typeparam name="T">The unmanaged type of elements to store</typeparam>
internal sealed class TensorStorage<T> : IColumnStorage<T> where T : unmanaged
{
    private readonly T[] _data;
    private readonly bool[]? _nullMask;
    private bool _disposed;
    
    /// <summary>
    /// Initializes a new instance of TensorStorage with the specified values
    /// </summary>
    /// <param name="values">The values to store</param>
    public TensorStorage(ReadOnlySpan<T> values)
    {
        if (values.IsEmpty)
        {
            _data = Array.Empty<T>();
            _nullMask = null;
        }
        else
        {
            _data = values.ToArray();
            _nullMask = null;
        }
    }
    
    /// <summary>
    /// Initializes a new instance of TensorStorage with existing data and null mask
    /// </summary>
    /// <param name="data">The array containing the data</param>
    /// <param name="nullMask">The optional null mask array</param>
    internal TensorStorage(T[] data, bool[]? nullMask = null)
    {
        _data = data ?? throw new ArgumentNullException(nameof(data));
        _nullMask = nullMask;
    }
    
    /// <inheritdoc />
    public int Length => _data.Length;
    
    /// <inheritdoc />
    public bool IsVectorizable => true;
    
    /// <inheritdoc />
    public bool HasNulls => _nullMask != null;
    
    /// <inheritdoc />
    public ReadOnlySpan<bool> NullMask
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _nullMask != null ? _nullMask.AsSpan() : ReadOnlySpan<bool>.Empty;
        }
    }
    
    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            
            if (index < 0 || index >= Length)
                throw new IndexOutOfRangeException($"Index {index} is out of range for storage of length {Length}");
            
            return _data[index];
        }
    }
    
    /// <inheritdoc />
    public IColumnStorage<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        if (start < 0)
            throw new ArgumentOutOfRangeException(nameof(start), "Start index cannot be negative");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative");
        if (start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), "Start + length exceeds storage bounds");
        
        if (length == 0)
        {
            return new TensorStorage<T>(Array.Empty<T>());
        }
        
        // Create sliced data array
        var slicedData = new T[length];
        Array.Copy(_data, start, slicedData, 0, length);
        
        // Create sliced null mask if it exists
        bool[]? slicedNullMask = null;
        if (_nullMask != null)
        {
            slicedNullMask = new bool[length];
            Array.Copy(_nullMask, start, slicedNullMask, 0, length);
        }
        
        return new TensorStorage<T>(slicedData, slicedNullMask);
    }
    
    /// <summary>
    /// Gets the underlying data array for vectorized operations
    /// </summary>
    internal T[] Data
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data;
        }
    }
    
    /// <summary>
    /// Gets the underlying null mask array for vectorized null operations
    /// </summary>
    internal bool[]? NullMaskArray
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _nullMask;
        }
    }
    
    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            // Arrays don't require explicit disposal
            _disposed = true;
        }
    }
}