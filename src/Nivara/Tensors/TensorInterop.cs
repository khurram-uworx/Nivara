using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.Tensors;

/// <summary>
/// Provides interoperability between Nivara DataFrames/Series and System.Numerics.Tensors.
/// Leverages .NET 10's enhanced Tensor APIs for high-performance tensor operations.
/// </summary>
public static class TensorInterop
{
    /// <summary>
    /// Converts a NivaraSeries to a System.Numerics.Tensors.Tensor&lt;T&gt;.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The NivaraSeries to convert</param>
    /// <returns>A Tensor&lt;T&gt; containing the series data</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="NotSupportedException">Thrown when T does not implement INumber&lt;T&gt;</exception>
    public static Tensor<T> ToTensor<T>(this NivaraSeries<T> series)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
        {
            return Tensor.Create<T>(new T[0], new ReadOnlySpan<nint>(new nint[] { 0 }));
        }

        // Create a 1D tensor with the series data
        var dimensions = new ReadOnlySpan<nint>(new nint[] { series.Length });
        var data = new T[series.Length];

        // Copy valid values to data array, using default(T) for invalid values
        var seriesSpan = series.Values.AsSpan();
        for (int i = 0; i < series.Length; i++)
        {
            data[i] = !series.IsNull(i) ? seriesSpan[i] : default(T);
        }

        return Tensor.Create<T>(data, dimensions);
    }

    /// <summary>
    /// Creates a NivaraSeries from a System.Numerics.Tensors.Tensor&lt;T&gt;.
    /// Only supports 1D tensors. For multi-dimensional tensors, use FlattenFromTensor.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The Tensor&lt;T&gt; to convert</param>
    /// <returns>A NivaraSeries containing the tensor data</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    /// <exception cref="ArgumentException">Thrown when tensor is not 1-dimensional</exception>
    public static NivaraSeries<T> FromTensor<T>(Tensor<T> tensor)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(tensor);

        if (tensor.Rank != 1)
        {
            throw new ArgumentException($"Only 1D tensors are supported. Tensor has {tensor.Rank} dimensions. Use FlattenFromTensor for multi-dimensional tensors.");
        }

        var length = (int)tensor.Lengths[0];
        if (length == 0)
        {
            return new NivaraSeries<T>();
        }

        var data = new T[length];

        // Copy data from tensor
        var tensorSpan = tensor.AsTensorSpan();
        for (int i = 0; i < length; i++)
        {
            data[i] = tensorSpan[i];
        }

        return NivaraSeries<T>.Create(data);
    }

    /// <summary>
    /// Converts a NivaraSeries to a TensorSpan&lt;T&gt; for zero-copy operations.
    /// The returned TensorSpan shares memory with the original series when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The NivaraSeries to convert</param>
    /// <returns>A TensorSpan&lt;T&gt; that may share memory with the series</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    public static TensorSpan<T> ToTensorSpan<T>(this NivaraSeries<T> series)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
        {
            return new TensorSpan<T>(Span<T>.Empty, ReadOnlySpan<nint>.Empty, default);
        }

        // Check if all values are valid for zero-copy operation
        for (int i = 0; i < series.Length; i++)
        {
            if (series.IsNull(i))
            {
                throw new InvalidOperationException("Cannot create TensorSpan from series with null values. Use ToTensor() instead.");
            }
        }

        // Get zero-copy access to the underlying data
        var dataSpan = series.Values.AsSpan();
        var dimensions = new ReadOnlySpan<nint>(new nint[] { series.Length });

        // Convert ReadOnlySpan<T> to Span<T> for TensorSpan
        // This is safe because we're creating a read-only view
        var writableSpan = MemoryMarshal.CreateSpan(
            ref MemoryMarshal.GetReference(dataSpan),
            dataSpan.Length);

        return new TensorSpan<T>(writableSpan, dimensions, default);
    }

    /// <summary>
    /// Creates a NivaraSeries from a TensorSpan&lt;T&gt;.
    /// Only supports 1D tensor spans. For multi-dimensional spans, use FlattenFromTensorSpan.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensorSpan">The TensorSpan&lt;T&gt; to convert</param>
    /// <returns>A NivaraSeries containing the tensor span data</returns>
    /// <exception cref="ArgumentException">Thrown when tensor span is not 1-dimensional</exception>
    public static NivaraSeries<T> FromTensorSpan<T>(TensorSpan<T> tensorSpan)
        where T : struct, INumber<T>
    {
        if (tensorSpan.Rank != 1)
        {
            throw new ArgumentException($"Only 1D tensor spans are supported. TensorSpan has {tensorSpan.Rank} dimensions. Use FlattenFromTensorSpan for multi-dimensional spans.");
        }

        var length = (int)tensorSpan.Lengths[0];
        if (length == 0)
        {
            return new NivaraSeries<T>();
        }

        var data = new T[length];

        // Copy data from tensor span
        for (int i = 0; i < length; i++)
        {
            data[i] = tensorSpan[i];
        }

        return NivaraSeries<T>.Create(data);
    }

    /// <summary>
    /// Converts a NivaraFrame to a 2D Tensor&lt;T&gt;.
    /// All columns must be of the same numeric type T.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <returns>A 2D Tensor&lt;T&gt; with dimensions [rows, columns]</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when frame has no columns or columns have different types</exception>
    /// <exception cref="InvalidOperationException">Thrown when frame contains null values</exception>
    public static Tensor<T> ToTensor<T>(this NivaraFrame frame)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(frame);

        if (frame.ColumnCount == 0)
        {
            throw new ArgumentException("Frame must have at least one column");
        }

        // Verify all columns are of type T
        foreach (var columnType in frame.Schema.ColumnTypes.Values)
        {
            if (columnType != typeof(T))
            {
                throw new ArgumentException($"All columns must be of type {typeof(T).Name}. Found column of type {columnType.Name}");
            }
        }

        if (frame.RowCount == 0)
        {
            return Tensor.Create<T>(new T[0], new ReadOnlySpan<nint>(new nint[] { 0, frame.ColumnCount }));
        }

        // Create 2D tensor [rows, columns]
        var dimensions = new ReadOnlySpan<nint>(new nint[] { frame.RowCount, frame.ColumnCount });
        var data = new T[frame.RowCount * frame.ColumnCount];

        // Copy data from frame to tensor in row-major order
        for (int row = 0; row < frame.RowCount; row++)
        {
            for (int col = 0; col < frame.ColumnCount; col++)
            {
                var columnName = frame.ColumnNames[col];
                var series = frame.GetColumn<T>(columnName);

                // Check for null values
                if (series.IsNull(row))
                {
                    throw new InvalidOperationException($"Cannot create tensor from frame with null values. Found null at row {row}, column '{columnName}'");
                }

                // Copy value in row-major order
                data[row * frame.ColumnCount + col] = series[row];
            }
        }

        return Tensor.Create<T>(data, dimensions);
    }

    /// <summary>
    /// Creates a NivaraFrame from a 2D Tensor&lt;T&gt;.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The 2D Tensor&lt;T&gt; to convert</param>
    /// <param name="columnNames">Optional column names. If null, generates default names</param>
    /// <returns>A NivaraFrame containing the tensor data</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    /// <exception cref="ArgumentException">Thrown when tensor is not 2-dimensional or column names count doesn't match</exception>
    public static NivaraFrame FromTensor<T>(Tensor<T> tensor, string[]? columnNames = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(tensor);

        if (tensor.Rank != 2)
        {
            throw new ArgumentException($"Only 2D tensors are supported for NivaraFrame conversion. Tensor has {tensor.Rank} dimensions.");
        }

        var rows = (int)tensor.Lengths[0];
        var cols = (int)tensor.Lengths[1];

        // Generate default column names if not provided
        columnNames ??= Enumerable.Range(0, cols).Select(i => $"Col_{i}").ToArray();

        if (columnNames.Length != cols)
        {
            throw new ArgumentException($"Column names count ({columnNames.Length}) must match tensor columns ({cols})");
        }

        if (rows == 0 || cols == 0)
        {
            // Create a minimal empty frame with one empty column
            var emptyColumn = NivaraColumn<T>.Create(Array.Empty<T>());
            var emptyColumns = new[] { ("EmptyColumn", (IColumn)emptyColumn) };
            return new NivaraFrame(emptyColumns);
        }

        var columns = new Dictionary<string, IColumn>();
        var tensorSpan = tensor.AsTensorSpan();

        // Create a column for each tensor column
        for (int col = 0; col < cols; col++)
        {
            var columnData = new T[rows];

            for (int row = 0; row < rows; row++)
            {
                columnData[row] = tensorSpan[row, col];
            }

            var series = NivaraSeries<T>.Create(columnData);
            columns[columnNames[col]] = series.Values;
        }

        return new NivaraFrame(columns.Select(kvp => (kvp.Key, kvp.Value)));
    }

    /// <summary>
    /// Flattens a multi-dimensional Tensor&lt;T&gt; into a 1D NivaraSeries.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The multi-dimensional Tensor&lt;T&gt; to flatten</param>
    /// <returns>A NivaraSeries containing the flattened tensor data</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static NivaraSeries<T> FlattenFromTensor<T>(Tensor<T> tensor)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var totalElements = (int)tensor.FlattenedLength;
        if (totalElements == 0)
        {
            return new NivaraSeries<T>();
        }

        var data = new T[totalElements];

        // Copy all values from the tensor in row-major order
        var tensorSpan = tensor.AsTensorSpan();

        // Manually flatten the tensor
        var index = 0;
        flattenTensorRecursive(tensorSpan, new nint[tensor.Rank], 0, data, ref index);

        return NivaraSeries<T>.Create(data);
    }

    /// <summary>
    /// Flattens a multi-dimensional TensorSpan&lt;T&gt; into a 1D NivaraSeries.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensorSpan">The multi-dimensional TensorSpan&lt;T&gt; to flatten</param>
    /// <returns>A NivaraSeries containing the flattened tensor span data</returns>
    public static NivaraSeries<T> FlattenFromTensorSpan<T>(TensorSpan<T> tensorSpan)
        where T : struct, INumber<T>
    {
        var totalElements = (int)tensorSpan.FlattenedLength;
        if (totalElements == 0)
        {
            return new NivaraSeries<T>();
        }

        var data = new T[totalElements];

        // Copy all values from the tensor span in row-major order
        var index = 0;
        flattenTensorRecursive(tensorSpan, new nint[tensorSpan.Rank], 0, data, ref index);

        return NivaraSeries<T>.Create(data);
    }

    /// <summary>
    /// Reshapes a NivaraSeries into a multi-dimensional Tensor&lt;T&gt; with specified dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The NivaraSeries to reshape</param>
    /// <param name="dimensions">The target dimensions for the tensor</param>
    /// <returns>A multi-dimensional Tensor&lt;T&gt; with the specified shape</returns>
    /// <exception cref="ArgumentNullException">Thrown when series or dimensions is null</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions are invalid or total elements don't match</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    public static Tensor<T> ReshapeToTensor<T>(this NivaraSeries<T> series, params int[] dimensions)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);
        ArgumentNullException.ThrowIfNull(dimensions);

        if (dimensions.Length == 0)
        {
            throw new ArgumentException("At least one dimension must be specified", nameof(dimensions));
        }

        var totalElements = dimensions.Aggregate(1, (a, b) => a * b);
        if (totalElements != series.Length)
        {
            throw new ArgumentException($"Total elements in dimensions ({totalElements}) must match series length ({series.Length})");
        }

        // Check for null values
        for (int i = 0; i < series.Length; i++)
        {
            if (series.IsNull(i))
            {
                throw new InvalidOperationException($"Cannot reshape series with null values. Found null at index {i}");
            }
        }

        if (totalElements == 0)
        {
            return Tensor.Create<T>(new T[0], new ReadOnlySpan<nint>(dimensions.Select(d => (nint)d).ToArray()));
        }

        // Create tensor with specified dimensions
        var data = new T[totalElements];

        // Copy data from series using zero-copy span access
        var seriesSpan = series.Values.AsSpan();
        seriesSpan.CopyTo(data);

        var nintDimensions = new ReadOnlySpan<nint>(dimensions.Select(d => (nint)d).ToArray());
        return Tensor.Create<T>(data, nintDimensions);
    }

    /// <summary>
    /// Converts multiple NivaraSeries to a batch of 1D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The collection of NivaraSeries to convert</param>
    /// <returns>An array of 1D Tensor&lt;T&gt; objects</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    public static Tensor<T>[] ToBatchTensors<T>(this IEnumerable<NivaraSeries<T>> series)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        return series.Select(s => s.ToTensor()).ToArray();
    }

    /// <summary>
    /// Creates multiple NivaraSeries from a batch of 1D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensors">The array of 1D Tensor&lt;T&gt; objects</param>
    /// <returns>An array of NivaraSeries</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensors is null</exception>
    public static NivaraSeries<T>[] FromBatchTensors<T>(Tensor<T>[] tensors)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(tensors);

        var result = new NivaraSeries<T>[tensors.Length];

        for (int i = 0; i < tensors.Length; i++)
        {
            result[i] = FromTensor(tensors[i]);
        }

        return result;
    }

    // Helper method to recursively flatten a tensor
    private static void flattenTensorRecursive<T>(TensorSpan<T> tensorSpan, nint[] indices, int dimension, T[] data, ref int index)
        where T : struct
    {
        if (dimension == tensorSpan.Rank)
        {
            // Base case: copy the value
            data[index++] = tensorSpan[indices];
            return;
        }

        // Recursive case: iterate through this dimension
        for (nint i = 0; i < tensorSpan.Lengths[dimension]; i++)
        {
            indices[dimension] = i;
            flattenTensorRecursive(tensorSpan, indices, dimension + 1, data, ref index);
        }
    }
}
