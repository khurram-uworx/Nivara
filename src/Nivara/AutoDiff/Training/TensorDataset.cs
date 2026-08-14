using Nivara.AutoDiff.Exceptions;
using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class TensorDataset<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly NivaraFrame frame;
    readonly string[] featureColumns;
    readonly string[] labelColumns;

    public int Count => frame.RowCount;
    public NivaraFrame Frame => frame;
    public IReadOnlyList<string> FeatureColumns => featureColumns;
    public IReadOnlyList<string> LabelColumns => labelColumns;

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

    public TensorDataset(NivaraFrame frame, string[] featureColumns, string labelColumn)
        : this(frame, featureColumns, [labelColumn])
    {
    }

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
