using Microsoft.ML.Data;
using System.Numerics;

namespace Nivara.MLNet;

/// <summary>
/// Provides specialized tensor conversion utilities between Nivara and ML.NET.
/// </summary>
public static class TensorConversions
{
    /// <summary>
    /// Converts a 2D NivaraFrame to a dense tensor representation for ML.NET.
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <returns>A dense tensor as a 2D array</returns>
    public static float[,] ToDenseTensor(this NivaraFrame frame)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));

        var tensor = new float[frame.RowCount, frame.ColumnCount];

        for (int row = 0; row < frame.RowCount; row++)
        {
            for (int col = 0; col < frame.ColumnCount; col++)
            {
                var columnName = frame.ColumnNames[col];
                var columnValue = frame.GetColumn(columnName).GetValue(row);
                var value = ConvertToFloat(columnValue);
                tensor[row, col] = value;
            }
        }

        return tensor;
    }

    /// <summary>
    /// Creates a NivaraFrame from a dense tensor representation.
    /// </summary>
    /// <param name="tensor">The dense tensor as a 2D array</param>
    /// <param name="columnNames">Optional column names</param>
    /// <returns>A NivaraFrame containing the tensor data</returns>
    public static NivaraFrame FromDenseTensor(float[,] tensor, string[]? columnNames = null)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var rows = tensor.GetLength(0);
        var cols = tensor.GetLength(1);

        // Generate default column names if not provided
        columnNames ??= Enumerable.Range(0, cols).Select(i => $"Col_{i}").ToArray();

        if (columnNames.Length != cols)
            throw new ArgumentException("Column names count must match tensor columns", nameof(columnNames));

        var columns = new List<(string Name, IColumn Column)>();

        // Create a column for each tensor column
        for (int col = 0; col < cols; col++)
        {
            var columnData = new float[rows];

            for (int row = 0; row < rows; row++)
            {
                columnData[row] = tensor[row, col];
            }

            columns.Add((columnNames[col], NivaraColumn<float>.Create(columnData)));
        }

        return new NivaraFrame(columns);
    }

    /// <summary>
    /// Converts multiple NivaraSeries to a batch of ML.NET tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="series">The collection of NivaraSeries</param>
    /// <returns>An array of VBuffer tensors</returns>
    public static VBuffer<T>[] ToMLNetBatchTensors<T>(this IEnumerable<NivaraSeries<T>> series)
        where T : struct, INumber<T>
    {
        if (series == null) throw new ArgumentNullException(nameof(series));

        var seriesList = series.ToList();
        if (seriesList.Count == 0)
            return Array.Empty<VBuffer<T>>();

        var tensors = new VBuffer<T>[seriesList.Count];

        for (int i = 0; i < seriesList.Count; i++)
        {
            var currentSeries = seriesList[i];
            var values = new T[currentSeries.Length];

            // Extract values from the series
            for (int j = 0; j < currentSeries.Length; j++)
            {
                values[j] = currentSeries.Values[j];
            }

            // Create dense VBuffer (all values are significant)
            tensors[i] = new VBuffer<T>(values.Length, values);
        }

        return tensors;
    }

    /// <summary>
    /// Creates multiple NivaraSeries from a batch of ML.NET tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensors">The array of VBuffer tensors</param>
    /// <returns>An array of NivaraSeries</returns>
    public static NivaraSeries<T>[] FromBatchTensors<T>(VBuffer<T>[] tensors)
        where T : struct, INumber<T>
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length == 0)
            return Array.Empty<NivaraSeries<T>>();

        var seriesArray = new NivaraSeries<T>[tensors.Length];

        for (int i = 0; i < tensors.Length; i++)
        {
            var tensor = tensors[i];
            var values = new T[tensor.Length];

            // Extract values from VBuffer (handles both dense and sparse)
            if (tensor.IsDense)
            {
                // Dense tensor - copy all values
                var denseValues = tensor.GetValues();
                for (int j = 0; j < denseValues.Length; j++)
                {
                    values[j] = denseValues[j];
                }
            }
            else
            {
                // Sparse tensor - fill with defaults and set non-zero values
                Array.Fill(values, T.Zero);
                var sparseValues = tensor.GetValues();
                var sparseIndices = tensor.GetIndices();

                for (int j = 0; j < sparseValues.Length; j++)
                {
                    values[sparseIndices[j]] = sparseValues[j];
                }
            }

            // Series default to a virtual positional index; no boxed object array needed
            seriesArray[i] = NivaraSeries<T>.Create(values);
        }

        return seriesArray;
    }

    /// <summary>
    /// Converts a NivaraFrame to sparse tensor format for ML.NET.
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <param name="threshold">Values below this threshold are considered sparse (default: 1e-6)</param>
    /// <returns>An array of sparse VBuffer tensors, one per row</returns>
    public static VBuffer<float>[] ToSparseTensors(this NivaraFrame frame, float threshold = 1e-6f)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));

        var tensors = new VBuffer<float>[frame.RowCount];

        for (int row = 0; row < frame.RowCount; row++)
        {
            var denseValues = new List<float>();
            var indices = new List<int>();

            for (int col = 0; col < frame.ColumnCount; col++)
            {
                var columnName = frame.ColumnNames[col];
                var columnValue = frame.GetColumn(columnName).GetValue(row);
                var value = ConvertToFloat(columnValue);

                if (Math.Abs(value) >= threshold)
                {
                    denseValues.Add(value);
                    indices.Add(col);
                }
            }

            // Create sparse VBuffer
            if (denseValues.Count == 0)
            {
                tensors[row] = new VBuffer<float>(frame.ColumnCount, 0, null, null);
            }
            else
            {
                tensors[row] = new VBuffer<float>(
                    frame.ColumnCount,
                    denseValues.Count,
                    denseValues.ToArray(),
                    indices.ToArray());
            }
        }

        return tensors;
    }

    /// <summary>
    /// Creates a NivaraFrame from sparse tensor format.
    /// </summary>
    /// <param name="sparseTensors">The sparse VBuffer tensors</param>
    /// <param name="columnNames">Optional column names</param>
    /// <returns>A NivaraFrame containing the sparse tensor data</returns>
    public static NivaraFrame FromSparseTensors(VBuffer<float>[] sparseTensors, string[]? columnNames = null)
    {
        if (sparseTensors == null || sparseTensors.Length == 0)
            throw new ArgumentException("Sparse tensors cannot be null or empty", nameof(sparseTensors));

        var columnCount = sparseTensors[0].Length;
        var rowCount = sparseTensors.Length;

        // Generate default column names if not provided
        columnNames ??= Enumerable.Range(0, columnCount).Select(i => $"Col_{i}").ToArray();

        if (columnNames.Length != columnCount)
            throw new ArgumentException("Column names count must match tensor dimension", nameof(columnNames));

        var columns = new List<(string Name, IColumn Column)>();

        // Create a column for each tensor dimension
        for (int col = 0; col < columnCount; col++)
        {
            var columnData = new float[rowCount];

            for (int row = 0; row < rowCount; row++)
            {
                var tensor = sparseTensors[row];
                columnData[row] = tensor.GetItemOrDefault(col);
            }

            columns.Add((columnNames[col], NivaraColumn<float>.Create(columnData)));
        }

        return new NivaraFrame(columns);
    }

    // Private helper methods
    private static float ConvertToFloat(object? value)
    {
        return value switch
        {
            null => 0f,
            float f => f,
            double d => (float)d,
            int i => i,
            long l => l,
            decimal dec => (float)dec,
            byte b => b,
            short s => s,
            _ => throw new InvalidOperationException($"Cannot convert {value?.GetType()} to float")
        };
    }
}
