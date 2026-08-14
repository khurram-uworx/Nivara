using Nivara.AutoDiff.Exceptions;
using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

/// <summary>
/// An in-memory dataset over a <see cref="NivaraFrame"/>, materializing feature and label
/// tensors for arbitrary row selections. Requires the selected columns to contain no nulls
/// (the AutoDiff domain is non-nullable).
/// </summary>
public sealed class TensorDataset<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly NivaraFrame frame;
    readonly string[] featureColumns;
    readonly string[] labelColumns;

    /// <summary>The number of rows in the underlying frame.</summary>
    public int Count => frame.RowCount;

    /// <summary>The underlying frame.</summary>
    public NivaraFrame Frame => frame;

    /// <summary>The column names used as model features.</summary>
    public IReadOnlyList<string> FeatureColumns => featureColumns;

    /// <summary>The column names used as labels.</summary>
    public IReadOnlyList<string> LabelColumns => labelColumns;

    /// <summary>
    /// Creates a dataset over a frame with the given feature and label columns.
    /// </summary>
    /// <param name="frame">The source frame (must be non-empty)</param>
    /// <param name="featureColumns">The feature column names (at least one)</param>
    /// <param name="labelColumns">The label column names (at least one)</param>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null</exception>
    /// <exception cref="ArgumentException">Thrown when no feature/label columns are given, the frame is empty, or a named column is missing</exception>
    public TensorDataset(NivaraFrame frame, string[] featureColumns, string[] labelColumns)
    {
        this.frame = frame ?? throw new ArgumentNullException(nameof(frame));
        this.featureColumns = featureColumns ?? throw new ArgumentNullException(nameof(featureColumns));
        this.labelColumns = labelColumns ?? throw new ArgumentNullException(nameof(labelColumns));

        if (featureColumns.Length == 0)
            throw new ArgumentException("At least one feature column is required.", nameof(featureColumns));
        if (labelColumns.Length == 0)
            throw new ArgumentException("At least one label column is required.", nameof(labelColumns));
        if (frame.RowCount == 0)
            throw new ArgumentException("Frame must contain at least one row.", nameof(frame));

        foreach (var name in featureColumns.Concat(labelColumns))
        {
            if (!frame.HasColumn(name))
                throw new ArgumentException($"Column '{name}' not found in frame.", nameof(featureColumns));
        }
    }

    /// <summary>
    /// Creates a dataset over a frame with the given feature columns and a single label column.
    /// </summary>
    /// <param name="frame">The source frame (must be non-empty)</param>
    /// <param name="featureColumns">The feature column names (at least one)</param>
    /// <param name="labelColumn">The single label column name</param>
    public TensorDataset(NivaraFrame frame, string[] featureColumns, string labelColumn)
        : this(frame, featureColumns, [labelColumn])
    {
    }

    /// <summary>
    /// Materializes a batch of features and labels for the given row indices. Features are
    /// created with <see cref="ReverseGradTensor{T}.RequiresGrad"/> set to true so gradients
    /// can flow back to them.
    /// </summary>
    /// <param name="indices">The row indices to include in the batch</param>
    /// <returns>A batch with <c>[batchSize, featureCount]</c> features and labels</returns>
    public Batch<T> GetBatch(ReadOnlySpan<int> indices)
    {
        var features = BuildTensor(featureColumns, indices, requiresGrad: true);
        var labels = BuildTensor(labelColumns, indices, requiresGrad: false);
        return new Batch<T>(features, labels);
    }

    ReverseGradTensor<T> BuildTensor(string[] columnNames, ReadOnlySpan<int> indices, bool requiresGrad)
    {
        int batchSize = indices.Length;
        int numCols = columnNames.Length;
        int totalLength = batchSize * numCols;

        var columns = new NivaraColumn<T>[numCols];
        for (int j = 0; j < numCols; j++)
        {
            columns[j] = frame.GetColumn<T>(columnNames[j]);
            if (columns[j].HasNulls)
                throw new AutoGradException(GradTensor<T>.Adr001Message);
        }

        T[] data = ArrayPool<T>.Shared.Rent(totalLength);

        try
        {
            for (int j = 0; j < numCols; j++)
            {
                columns[j].TryGetSpan(out var colSpan);
                for (int i = 0; i < batchSize; i++)
                    data[i * numCols + j] = colSpan[indices[i]];
            }

            var column = NivaraColumn<T>.Create(data.AsSpan(0, totalLength));
            var tensor = new ReverseGradTensor<T>(column, requiresGrad);
            tensor.Reshape(batchSize, numCols);

            return tensor;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(data);
        }
    }
}
