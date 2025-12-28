using System.Numerics;

namespace Nivara;

/// <summary>
/// Labeled series built on top of NivaraColumn&lt;T&gt; with optional indexing support.
/// Provides label-based access and maintains index-value relationships during operations.
/// </summary>
/// <typeparam name="T">The type of values in the series</typeparam>
public sealed class NivaraSeries<T> : IDisposable
{
    private readonly NivaraColumn<T> values;
    private readonly NivaraColumn<object> index;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of NivaraSeries with the specified values and optional index
    /// </summary>
    /// <param name="values">The column of values</param>
    /// <param name="index">Optional index labels. If null, integer positions (0, 1, 2, ...) will be used</param>
    /// <exception cref="ArgumentNullException">Thrown when values is null</exception>
    /// <exception cref="ArgumentException">Thrown when index length doesn't match values length</exception>
    public NivaraSeries(NivaraColumn<T> values, NivaraColumn<object>? index = null)
    {
        this.values = values ?? throw new ArgumentNullException(nameof(values));

        if (index != null)
        {
            if (index.Length != values.Length)
                throw new ArgumentException($"Index length ({index.Length}) must match values length ({values.Length})", nameof(index));

            this.index = index;
        }
        else
        {
            this.index = NivaraSeries<T>.CreateDefaultIndex(values.Length);
        }
    }

    /// <summary>
    /// Gets the number of elements in the series
    /// </summary>
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return values.Length;
        }
    }

    /// <summary>
    /// Gets a value indicating whether this series contains any null values
    /// </summary>
    public bool HasNulls
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return values.HasNulls;
        }
    }

    /// <summary>
    /// Gets the underlying values column
    /// </summary>
    public NivaraColumn<T> Values
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return values;
        }
    }

    /// <summary>
    /// Gets the index column
    /// </summary>
    public NivaraColumn<object> Index
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return index;
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
            ObjectDisposedException.ThrowIf(disposed, this);
            return values[position];
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
            ObjectDisposedException.ThrowIf(disposed, this);
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
        ObjectDisposedException.ThrowIf(disposed, this);
        return values.IsNull(position);
    }

    /// <summary>
    /// Gets the value associated with the specified label
    /// </summary>
    /// <param name="label">The label to look up</param>
    /// <returns>The value associated with the label</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the label is not found</exception>
    public T GetByLabel(object label)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var position = FindLabelPosition(label);
        if (position == -1)
        {
            throw new KeyNotFoundException($"Label '{label}' not found in series index");
        }

        return values[position];
    }

    /// <summary>
    /// Tries to get the value associated with the specified label
    /// </summary>
    /// <param name="label">The label to look up</param>
    /// <param name="value">When this method returns, contains the value associated with the label if found; otherwise, the default value for T</param>
    /// <returns>true if the label was found; otherwise, false</returns>
    public bool TryGetByLabel(object label, out T value)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var position = FindLabelPosition(label);
        if (position != -1)
        {
            value = values[position];
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
        ObjectDisposedException.ThrowIf(disposed, this);
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
        ObjectDisposedException.ThrowIf(disposed, this);
        return index[position];
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
        ObjectDisposedException.ThrowIf(disposed, this);

        var slicedValues = values.Slice(start, length);
        var slicedIndex = index.Slice(start, length);

        return new NivaraSeries<T>(slicedValues, slicedIndex);
    }

    /// <summary>
    /// Aligns this series with another series based on their index labels.
    /// Returns a new series containing only the values where both series have matching index labels.
    /// The resulting series maintains the index-value relationships from this series.
    /// </summary>
    /// <param name="other">The series to align with</param>
    /// <returns>A new aligned series containing values where indices match</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    public NivaraSeries<T> Align(NivaraSeries<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        var alignedPairs = GetAlignedPairs(other);
        
        if (alignedPairs.Count == 0)
        {
            // No matching indices, return empty series
            return NivaraSeries<T>.Create(Array.Empty<T>(), Array.Empty<object>());
        }

        var alignedValues = new T[alignedPairs.Count];
        var alignedIndex = new object[alignedPairs.Count];

        for (int i = 0; i < alignedPairs.Count; i++)
        {
            var (thisPos, _) = alignedPairs[i];
            alignedValues[i] = values[thisPos];
            alignedIndex[i] = index[thisPos];
        }

        return NivaraSeries<T>.Create(alignedValues, alignedIndex);
    }

    /// <summary>
    /// Aligns this series with another series and returns both aligned series.
    /// Both series will have the same index labels in the same order, containing only
    /// values where both original series had matching index labels.
    /// </summary>
    /// <param name="other">The series to align with</param>
    /// <returns>A tuple containing the aligned versions of both series</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    public (NivaraSeries<T> Left, NivaraSeries<T> Right) AlignBoth(NivaraSeries<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        var alignedPairs = GetAlignedPairs(other);
        
        if (alignedPairs.Count == 0)
        {
            // No matching indices, return empty series
            var empty = Array.Empty<T>();
            var emptyIndex = Array.Empty<object>();
            return (NivaraSeries<T>.Create(empty, emptyIndex), 
                    NivaraSeries<T>.Create(empty, emptyIndex));
        }

        var leftValues = new T[alignedPairs.Count];
        var rightValues = new T[alignedPairs.Count];
        var alignedIndex = new object[alignedPairs.Count];

        for (int i = 0; i < alignedPairs.Count; i++)
        {
            var (thisPos, otherPos) = alignedPairs[i];
            leftValues[i] = values[thisPos];
            rightValues[i] = other.values[otherPos];
            alignedIndex[i] = index[thisPos];
        }

        return (NivaraSeries<T>.Create(leftValues, alignedIndex),
                NivaraSeries<T>.Create(rightValues, alignedIndex));
    }

    /// <summary>
    /// Performs element-wise addition with another series after aligning their indices.
    /// Only values with matching index labels are included in the result.
    /// </summary>
    /// <param name="other">The series to add</param>
    /// <returns>A new series containing the element-wise sum of aligned values</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support addition</exception>
    public NivaraSeries<T> Add(NivaraSeries<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        var (alignedLeft, alignedRight) = AlignBoth(other);
        
        if (alignedLeft.Length == 0)
        {
            // No matching indices, return empty series
            return NivaraSeries<T>.Create(Array.Empty<T>(), Array.Empty<object>());
        }

        var resultValues = alignedLeft.values + alignedRight.values;
        return new NivaraSeries<T>(resultValues, alignedLeft.index);
    }

    /// <summary>
    /// Performs element-wise multiplication with another series after aligning their indices.
    /// Only values with matching index labels are included in the result.
    /// </summary>
    /// <param name="other">The series to multiply</param>
    /// <returns>A new series containing the element-wise product of aligned values</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when T does not support multiplication</exception>
    public NivaraSeries<T> Multiply(NivaraSeries<T> other)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        var (alignedLeft, alignedRight) = AlignBoth(other);
        
        if (alignedLeft.Length == 0)
        {
            // No matching indices, return empty series
            return NivaraSeries<T>.Create(Array.Empty<T>(), Array.Empty<object>());
        }

        var resultValues = alignedLeft.values * alignedRight.values;
        return new NivaraSeries<T>(resultValues, alignedLeft.index);
    }

    /// <summary>
    /// Performs scalar multiplication on this series.
    /// The index-value relationships are preserved in the result.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by</param>
    /// <returns>A new series with all values multiplied by the scalar</returns>
    /// <exception cref="InvalidOperationException">Thrown when T does not support multiplication</exception>
    public NivaraSeries<T> Multiply(T scalar)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var resultValues = values * scalar;
        return new NivaraSeries<T>(resultValues, index);
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

        for (int i = 0; i < index.Length; i++)
        {
            if (!index.IsNull(i) && comparer.Equals(index[i], label))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// Gets pairs of positions where both series have matching index labels
    /// </summary>
    /// <param name="other">The other series to align with</param>
    /// <returns>A list of tuples containing (thisPosition, otherPosition) for matching indices</returns>
    private List<(int ThisPos, int OtherPos)> GetAlignedPairs(NivaraSeries<T> other)
    {
        var alignedPairs = new List<(int, int)>();
        var comparer = EqualityComparer<object>.Default;

        // For each index in this series, find matching index in other series
        for (int thisPos = 0; thisPos < index.Length; thisPos++)
        {
            if (index.IsNull(thisPos))
                continue;

            var thisLabel = index[thisPos];

            // Find matching label in other series
            for (int otherPos = 0; otherPos < other.index.Length; otherPos++)
            {
                if (other.index.IsNull(otherPos))
                    continue;

                var otherLabel = other.index[otherPos];

                if (comparer.Equals(thisLabel, otherLabel))
                {
                    alignedPairs.Add((thisPos, otherPos));
                    break; // Found match, move to next index in this series
                }
            }
        }

        return alignedPairs;
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
        if (!disposed)
        {
            values?.Dispose();
            index?.Dispose();
            disposed = true;
        }
    }
}
