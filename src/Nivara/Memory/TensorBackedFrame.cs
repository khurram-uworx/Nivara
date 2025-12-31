using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Memory;

/// <summary>
/// Internal storage class that uses a 2D Tensor&lt;T&gt; as the backing store for homogeneous numeric DataFrames.
/// Provides zero-copy TensorSpan&lt;T&gt; access when all columns are of the same numeric type and no null values are present.
/// </summary>
/// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
internal sealed class TensorBackedFrame<T> : IDisposable where T : struct, INumber<T>
{
    Tensor<T>? tensor;
    readonly string[] columnNames;
    readonly bool[,]? validityMask; // [row, column] validity mask
    readonly bool ownsTensor;
    bool disposed;

    /// <summary>
    /// Gets the number of rows in this frame.
    /// </summary>
    public int RowCount { get; }

    /// <summary>
    /// Gets the number of columns in this frame.
    /// </summary>
    public int ColumnCount { get; }

    /// <summary>
    /// Gets the column names.
    /// </summary>
    public IReadOnlyList<string> ColumnNames => columnNames;

    /// <summary>
    /// Gets whether this frame uses tensor storage.
    /// </summary>
    public bool IsTensorBacked => tensor != null;

    /// <summary>
    /// Initializes a new instance of the TensorBackedFrame class from a 2D tensor.
    /// </summary>
    /// <param name="tensor">The 2D tensor to use as backing storage.</param>
    /// <param name="columnNames">The names of the columns.</param>
    /// <param name="validityMask">Optional validity mask. If null, all values are considered valid.</param>
    /// <param name="ownsTensor">Whether this instance owns the tensor and should dispose it.</param>
    public TensorBackedFrame(Tensor<T> tensor, string[] columnNames, bool[,]? validityMask = null, bool ownsTensor = true)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(columnNames);

        if (tensor.Rank != 2)
        {
            throw new ArgumentException($"Only 2D tensors are supported. Tensor has {tensor.Rank} dimensions.");
        }

        var rows = (int)tensor.Lengths[0];
        var cols = (int)tensor.Lengths[1];

        if (columnNames.Length != cols)
        {
            throw new ArgumentException($"Column names count ({columnNames.Length}) must match tensor columns ({cols})");
        }

        if (validityMask != null && (validityMask.GetLength(0) != rows || validityMask.GetLength(1) != cols))
        {
            throw new ArgumentException("Validity mask dimensions must match tensor dimensions.", nameof(validityMask));
        }

        this.tensor = tensor;
        this.columnNames = columnNames;
        this.validityMask = validityMask;
        this.ownsTensor = ownsTensor;
        RowCount = rows;
        ColumnCount = cols;
    }

    /// <summary>
    /// Gets a TensorSpan&lt;T&gt; view of this frame for zero-copy operations.
    /// Only available when no null values are present.
    /// </summary>
    /// <returns>A TensorSpan&lt;T&gt; view of the tensor data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when frame contains null values.</exception>
    public TensorSpan<T> GetTensorSpan()
    {
        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        // Check if all values are valid
        if (validityMask != null)
        {
            for (int row = 0; row < RowCount; row++)
            {
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!validityMask[row, col])
                    {
                        throw new InvalidOperationException("Cannot create TensorSpan from frame with null values.");
                    }
                }
            }
        }

        return tensor.AsTensorSpan();
    }

    /// <summary>
    /// Gets the value at the specified row and column.
    /// </summary>
    /// <param name="row">The row index.</param>
    /// <param name="columnIndex">The column index.</param>
    /// <returns>The value at the specified position, or null if the value is not valid.</returns>
    public T? GetValue(int row, int columnIndex)
    {
        if (row < 0 || row >= RowCount)
            throw new ArgumentOutOfRangeException(nameof(row));
        if (columnIndex < 0 || columnIndex >= ColumnCount)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        if (tensor == null)
            return null;

        if (validityMask != null && !validityMask[row, columnIndex])
            return null;

        var tensorSpan = tensor.AsTensorSpan();
        return tensorSpan[row, columnIndex];
    }

    /// <summary>
    /// Gets the raw value at the specified row and column without checking validity.
    /// </summary>
    /// <param name="row">The row index.</param>
    /// <param name="columnIndex">The column index.</param>
    /// <returns>The raw value at the specified position.</returns>
    public T GetRawValue(int row, int columnIndex)
    {
        if (row < 0 || row >= RowCount)
            throw new ArgumentOutOfRangeException(nameof(row));
        if (columnIndex < 0 || columnIndex >= ColumnCount)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        var tensorSpan = tensor.AsTensorSpan();
        return tensorSpan[row, columnIndex];
    }

    /// <summary>
    /// Checks if the value at the specified row and column is valid (not null).
    /// </summary>
    /// <param name="row">The row index.</param>
    /// <param name="columnIndex">The column index.</param>
    /// <returns>True if the value is valid, false otherwise.</returns>
    public bool IsValid(int row, int columnIndex)
    {
        if (row < 0 || row >= RowCount)
            throw new ArgumentOutOfRangeException(nameof(row));
        if (columnIndex < 0 || columnIndex >= ColumnCount)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        if (validityMask == null)
            return true;

        return validityMask[row, columnIndex];
    }

    /// <summary>
    /// Gets the underlying tensor. The caller should not dispose it unless ownsTensor is false.
    /// </summary>
    /// <returns>The underlying tensor.</returns>
    public Tensor<T> GetTensor()
    {
        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        return tensor;
    }

    /// <summary>
    /// Releases all resources used by the TensorBackedFrame.
    /// </summary>
    public void Dispose()
    {
        if (!disposed)
        {
            if (ownsTensor && tensor != null)
            {
                //tensor.Dispose();
                tensor = null;
            }

            disposed = true;
        }
    }
}

