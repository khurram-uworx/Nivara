using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class TensorDataset<T> where T : struct, INumber<T>
{
    readonly NivaraFrame _frame;
    readonly string[] _featureColumns;
    readonly string[] _labelColumns;

    public int Count => _frame.RowCount;
    public NivaraFrame Frame => _frame;
    public IReadOnlyList<string> FeatureColumns => _featureColumns;
    public IReadOnlyList<string> LabelColumns => _labelColumns;

    public TensorDataset(NivaraFrame frame, string[] featureColumns, string[] labelColumns)
    {
        _frame = frame ?? throw new ArgumentNullException(nameof(frame));
        _featureColumns = featureColumns ?? throw new ArgumentNullException(nameof(featureColumns));
        _labelColumns = labelColumns ?? throw new ArgumentNullException(nameof(labelColumns));

        if (_featureColumns.Length == 0)
            throw new ArgumentException("At least one feature column is required.", nameof(featureColumns));
        if (_labelColumns.Length == 0)
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
        var features = BuildTensor(_featureColumns, indices, requiresGrad: true);
        var labels = BuildTensor(_labelColumns, indices, requiresGrad: false);
        return new Batch<T>(features, labels);
    }

    ReverseGradTensor<T> BuildTensor(string[] columnNames, ReadOnlySpan<int> indices, bool requiresGrad)
    {
        int batchSize = indices.Length;
        int numCols = columnNames.Length;
        int totalLength = batchSize * numCols;

        var columns = new NivaraColumn<T>[numCols];
        for (int j = 0; j < numCols; j++)
            columns[j] = _frame.GetColumn<T>(columnNames[j]);

        T[] data = ArrayPool<T>.Shared.Rent(totalLength);
        bool[]? nullMask = null;
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < batchSize; i++)
            {
                int rowIdx = indices[i];
                int rowBase = i * numCols;

                for (int j = 0; j < numCols; j++)
                {
                    int destIdx = rowBase + j;
                    data[destIdx] = columns[j][rowIdx];

                    if (columns[j].IsNull(rowIdx))
                    {
                        (nullMask ??= new bool[totalLength])[destIdx] = true;
                        hasNulls = true;
                    }
                }
            }

            NivaraColumn<T> column;
            if (hasNulls)
            {
                column = NivaraColumn<T>.CreateFromSpans(
                    data.AsSpan(0, totalLength),
                    nullMask.AsSpan(0, totalLength));
            }
            else
            {
                column = NivaraColumn<T>.Create(data.AsSpan(0, totalLength));
            }

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
