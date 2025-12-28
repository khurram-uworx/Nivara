namespace Nivara;

/// <summary>
/// Labeled series built on top of NivaraColumn&lt;T&gt; with optional indexing support.
/// Provides label-based access and maintains index-value relationships during operations.
/// </summary>
/// <typeparam name="T">The type of values in the series</typeparam>
public sealed class NivaraSeries<T> : IDisposable
{
    private readonly NivaraColumn<T> _values;
    private readonly NivaraColumn<object> _index;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of NivaraSeries with the specified values and optional index
    /// </summary>
    /// <param name="values">The column of values</param>
    /// <param name="index">Optional index labels. If null, integer positions (0, 1, 2, ...) will be used</param>
    /// <exception cref="ArgumentNullException">Thrown when values is null</exception>
    /// <exception cref="ArgumentException">Thrown when index length doesn't match values length</exception>
    public NivaraSeries(NivaraColumn<T> values, NivaraColumn<object>? index = null)
    {
        _values = values ?? throw new ArgumentNullException(nameof(values));

        if (index != null)
        {
            if (index.Length != values.Length)
                throw new ArgumentException($"Index length ({index.Length}) must match values length ({values.Length})", nameof(index));

            _index = index;
        }
        else
        {
            _index = CreateDefaultIndex(values.Length);
        }
    }

    /// <summary>
    /// Gets the number of elements in the series
    /// </summary>
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _values.Length;
        }
    }

    /// <summary>
    /// Gets a value indicating whether this series contains any null values
    /// </summary>
    public bool HasNulls
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _values.HasNulls;
        }
    }

    /// <summary>
    /// Gets the underlying values column
    /// </summary>
    public NivaraColumn<T> Values
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _values;
        }
    }

    /// <summary>
    /// Gets the index column
    /// </summary>
    public NivaraColumn<object> Index
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _index;
        }
    }

    /// <summary>
    /// Gets the value at the specified position (zero-based indexing)
    /// </summary>
    /// <param name="position">The zero-based position</param>
    /// <returns>The value at the specified position</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when position is out of bounds</exception>
    public T this[int position]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _values[position];
        }
    }

    /// <summary>
    /// Gets the value associated with the specified label
    /// </summary>
    /// <param name="label">The label to look up</param>
    /// <returns>The value associated with the label</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the label is not found</exception>
    public T this[object label]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return GetByLabel(label);
        }
    }

    /// <summary>
    /// Determines whether the value at the specified position is null
    /// </summary>
    /// <param name="position">The zero-based position to check</param>
    /// <returns>true if the value at the specified position is null; otherwise, false</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when position is out of bounds</exception>
    public bool IsNull(int position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _values.IsNull(position);
    }

    /// <summary>
    /// Gets the value associated with the specified label
    /// </summary>
    /// <param name="label">The label to look up</param>
    /// <returns>The value associated with the label</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the label is not found</exception>
    public T GetByLabel(object label)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var position = FindLabelPosition(label);
        if (position == -1)
        {
            throw new KeyNotFoundException($"Label '{label}' not found in series index");
        }

        return _values[position];
    }

    /// <summary>
    /// Tries to get the value associated with the specified label
    /// </summary>
    /// <param name="label">The label to look up</param>
    /// <param name="value">When this method returns, contains the value associated with the label if found; otherwise, the default value for T</param>
    /// <returns>true if the label was found; otherwise, false</returns>
    public bool TryGetByLabel(object label, out T value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var position = FindLabelPosition(label);
        if (position != -1)
        {
            value = _values[position];
            return true;
        }

        value = default(T)!;
        return false;
    }

    /// <summary>
    /// Determines whether the series contains the specified label
    /// </summary>
    /// <param name="label">The label to check</param>
    /// <returns>true if the series contains the label; otherwise, false</returns>
    public bool ContainsLabel(object label)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return FindLabelPosition(label) != -1;
    }

    /// <summary>
    /// Gets the label at the specified position
    /// </summary>
    /// <param name="position">The zero-based position</param>
    /// <returns>The label at the specified position</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when position is out of bounds</exception>
    public object GetLabel(int position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _index[position];
    }

    /// <summary>
    /// Creates a slice of this series containing elements from the specified range
    /// </summary>
    /// <param name="start">The starting position of the slice</param>
    /// <param name="length">The number of elements in the slice</param>
    /// <returns>A new series representing the slice</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when start or length are invalid</exception>
    public NivaraSeries<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var slicedValues = _values.Slice(start, length);
        var slicedIndex = _index.Slice(start, length);

        return new NivaraSeries<T>(slicedValues, slicedIndex);
    }

    /// <summary>
    /// Creates a new series from the specified values with optional index labels
    /// </summary>
    /// <param name="values">The values for the series</param>
    /// <param name="index">Optional index labels. If null, integer positions will be used</param>
    /// <returns>A new NivaraSeries instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values is null</exception>
    /// <exception cref="ArgumentException">Thrown when index length doesn't match values length</exception>
    public static NivaraSeries<T> Create(T[] values, object[]? index = null)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));

        return Create(values.AsSpan(), index);
    }

    /// <summary>
    /// Creates a new series from the specified values with optional index labels
    /// </summary>
    /// <param name="values">The values for the series</param>
    /// <param name="index">Optional index labels. If null, integer positions will be used</param>
    /// <returns>A new NivaraSeries instance</returns>
    /// <exception cref="ArgumentException">Thrown when index length doesn't match values length</exception>
    public static NivaraSeries<T> Create(ReadOnlySpan<T> values, object[]? index = null)
    {
        var valuesColumn = NivaraColumn<T>.Create(values);

        NivaraColumn<object>? indexColumn = null;
        if (index != null)
        {
            if (index.Length != values.Length)
                throw new ArgumentException($"Index length ({index.Length}) must match values length ({values.Length})", nameof(index));

            indexColumn = NivaraColumn<object>.CreateForReferenceType(index);
        }

        return new NivaraSeries<T>(valuesColumn, indexColumn);
    }

    /// <summary>
    /// Creates a new series from the specified values with string index labels
    /// </summary>
    /// <param name="values">The values for the series</param>
    /// <param name="index">String index labels</param>
    /// <returns>A new NivaraSeries instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when values or index is null</exception>
    /// <exception cref="ArgumentException">Thrown when index length doesn't match values length</exception>
    public static NivaraSeries<T> Create(T[] values, string[] index)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));
        if (index == null)
            throw new ArgumentNullException(nameof(index));

        // Convert string array to object array
        var objectIndex = index.Cast<object>().ToArray();
        return Create(values, objectIndex);
    }

    /// <summary>
    /// Finds the position of the specified label in the index
    /// </summary>
    /// <param name="label">The label to find</param>
    /// <returns>The position of the label, or -1 if not found</returns>
    private int FindLabelPosition(object label)
    {
        var comparer = EqualityComparer<object>.Default;

        for (int i = 0; i < _index.Length; i++)
        {
            if (!_index.IsNull(i) && comparer.Equals(_index[i], label))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// Creates a default integer index for the specified length
    /// </summary>
    /// <param name="length">The length of the index to create</param>
    /// <returns>A column containing integer indices from 0 to length-1</returns>
    private static NivaraColumn<object> CreateDefaultIndex(int length)
    {
        var indexValues = new object[length];
        for (int i = 0; i < length; i++)
        {
            indexValues[i] = i;
        }

        return NivaraColumn<object>.CreateForReferenceType(indexValues);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _values?.Dispose();
            _index?.Dispose();
            _disposed = true;
        }
    }
}