using Microsoft.ML;
using Microsoft.ML.Data;
using System.Numerics;

namespace Nivara.MLNet;

/// <summary>
/// Provides interoperability between Nivara DataFrames/Series and ML.NET tensors and data structures.
/// </summary>
public static class MLNetInterop
{
    static IEnumerable<TwoColumnData> ConvertToTwoColumnData(NivaraFrame frame)
    {
        for (int row = 0; row < frame.RowCount; row++)
        {
            yield return new TwoColumnData
            {
                Col1 = ConvertToFloat(frame.GetColumn(frame.ColumnNames[0]).GetValue(row)),
                Col2 = ConvertToFloat(frame.GetColumn(frame.ColumnNames[1]).GetValue(row))
            };
        }
    }

    static IEnumerable<TwoColumnFeatureData> ConvertToTwoColumnFeatureData(NivaraFrame frame)
    {
        for (int row = 0; row < frame.RowCount; row++)
        {
            yield return new TwoColumnFeatureData
            {
                Feature1 = ConvertToFloat(frame.GetColumn("Feature1").GetValue(row)),
                Feature2 = ConvertToFloat(frame.GetColumn("Feature2").GetValue(row))
            };
        }
    }

    static IEnumerable<ThreeColumnData> ConvertToThreeColumnData(NivaraFrame frame)
    {
        for (int row = 0; row < frame.RowCount; row++)
        {
            var data = new ThreeColumnData();

            // Map columns by name
            if (frame.HasColumn("Feature1"))
                data.Feature1 = ConvertToFloat(frame.GetColumn("Feature1").GetValue(row));
            else
                data.Feature1 = ConvertToFloat(frame.GetColumn(frame.ColumnNames[0]).GetValue(row));

            if (frame.HasColumn("Feature2"))
                data.Feature2 = ConvertToFloat(frame.GetColumn("Feature2").GetValue(row));
            else if (frame.ColumnCount > 1)
                data.Feature2 = ConvertToFloat(frame.GetColumn(frame.ColumnNames[1]).GetValue(row));

            if (frame.HasColumn("Label"))
                data.Label = ConvertToFloat(frame.GetColumn("Label").GetValue(row));
            else if (frame.ColumnCount > 2)
                data.Label = ConvertToFloat(frame.GetColumn(frame.ColumnNames[2]).GetValue(row));

            yield return data;
        }
    }

    static IEnumerable<GenericData> ConvertToGenericData(NivaraFrame frame)
    {
        for (int row = 0; row < frame.RowCount; row++)
        {
            var values = new float[frame.ColumnCount];
            for (int col = 0; col < frame.ColumnCount; col++)
            {
                values[col] = ConvertToFloat(frame.GetColumn(frame.ColumnNames[col]).GetValue(row));
            }

            yield return new GenericData { Features = values };
        }
    }

    static NivaraFrame ConvertFromGenericData(IDataView dataView, MLContext mlContext, string[] columnNames)
    {
        var columns = new List<(string Name, IColumn Column)>();

        // Process each column individually based on its type
        foreach (var columnName in columnNames)
        {
            var column = dataView.Schema[columnName];
            var columnType = column.Type;

            try
            {
                if (columnType == NumberDataViewType.Single)
                {
                    var values = ExtractFloatColumn(dataView, mlContext, columnName);
                    columns.Add((columnName, NivaraColumn<float>.Create(values)));
                }
                else if (columnType == NumberDataViewType.Double)
                {
                    var values = ExtractDoubleColumn(dataView, mlContext, columnName);
                    columns.Add((columnName, NivaraColumn<double>.Create(values)));
                }
                else if (columnType is VectorDataViewType vectorType && vectorType.ItemType == NumberDataViewType.Single)
                {
                    // Handle VBuffer<float> columns (like Features or Score columns)
                    var values = ExtractVBufferFloatColumn(dataView, mlContext, columnName);
                    columns.Add((columnName, NivaraColumn<float>.Create(values)));
                }
                else
                {
                    // For other types, try to convert to float as fallback
                    var values = ExtractFloatColumn(dataView, mlContext, columnName);
                    columns.Add((columnName, NivaraColumn<float>.Create(values)));
                }
            }
            catch
            {
                // If extraction fails, try VBuffer approach as fallback
                try
                {
                    var values = ExtractVBufferFloatColumn(dataView, mlContext, columnName);
                    columns.Add((columnName, NivaraColumn<float>.Create(values)));
                }
                catch
                {
                    // Skip columns that can't be converted
                    continue;
                }
            }
        }

        return new NivaraFrame(columns);
    }

    static float[] ExtractFloatColumn(IDataView dataView, MLContext mlContext, string columnName)
    {
        var values = new List<float>();
        var column = dataView.Schema[columnName];

        using var cursor = dataView.GetRowCursor(new[] { column });
        var getter = cursor.GetGetter<float>(column);

        while (cursor.MoveNext())
        {
            float value = 0f;
            getter(ref value);
            values.Add(value);
        }

        return values.ToArray();
    }

    static double[] ExtractDoubleColumn(IDataView dataView, MLContext mlContext, string columnName)
    {
        var values = new List<double>();
        var column = dataView.Schema[columnName];

        using var cursor = dataView.GetRowCursor(new[] { column });
        var getter = cursor.GetGetter<double>(column);

        while (cursor.MoveNext())
        {
            double value = 0.0;
            getter(ref value);
            values.Add(value);
        }

        return values.ToArray();
    }

    static float[] ExtractVBufferFloatColumn(IDataView dataView, MLContext mlContext, string columnName)
    {
        var values = new List<float>();
        var column = dataView.Schema[columnName];

        using var cursor = dataView.GetRowCursor(new[] { column });
        var getter = cursor.GetGetter<VBuffer<float>>(column);

        while (cursor.MoveNext())
        {
            VBuffer<float> buffer = default;
            getter(ref buffer);

            // For scalar values (like Score), take the first element
            // For vector values, we might want to handle differently
            if (buffer.Length > 0)
            {
                values.Add(buffer.GetItemOrDefault(0));
            }
            else
            {
                values.Add(0f);
            }
        }

        return values.ToArray();
    }

    static float ConvertToFloat(object? value)
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
            _ => 0f // Default for unsupported types
        };
    }
    /// <summary>
    /// Converts a NivaraFrame to an ML.NET IDataView for use in ML.NET pipelines.
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <param name="mlContext">The ML.NET context</param>
    /// <returns>An IDataView representation of the NivaraFrame</returns>
    public static IDataView ToDataView(this NivaraFrame frame, MLContext mlContext)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));

        // Check if this looks like a training frame (has Feature1, Feature2, Label)
        if (frame.HasColumn("Feature1") && frame.HasColumn("Feature2") && frame.HasColumn("Label"))
        {
            var data = ConvertToThreeColumnData(frame);
            return mlContext.Data.LoadFromEnumerable(data);
        }
        // Check if this looks like a prediction frame (has Feature1, Feature2, no Label)
        else if (frame.HasColumn("Feature1") && frame.HasColumn("Feature2"))
        {
            var data = ConvertToTwoColumnFeatureData(frame);
            return mlContext.Data.LoadFromEnumerable(data);
        }
        // Handle other 2-column cases
        else if (frame.ColumnCount == 2)
        {
            var data = ConvertToTwoColumnData(frame);
            return mlContext.Data.LoadFromEnumerable(data);
        }
        else if (frame.ColumnCount == 3)
        {
            var data = ConvertToThreeColumnData(frame);
            return mlContext.Data.LoadFromEnumerable(data);
        }
        else
        {
            // Fallback for other cases - use a generic approach
            var data = ConvertToGenericData(frame);
            return mlContext.Data.LoadFromEnumerable(data);
        }
    }

    /// <summary>
    /// Creates a NivaraFrame from an ML.NET IDataView.
    /// </summary>
    /// <param name="dataView">The ML.NET IDataView to convert</param>
    /// <param name="mlContext">The ML.NET context</param>
    /// <returns>A NivaraFrame representation of the IDataView</returns>
    public static NivaraFrame ToNivaraFrame(IDataView dataView, MLContext mlContext)
    {
        if (dataView == null) throw new ArgumentNullException(nameof(dataView));
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));

        var schema = dataView.Schema;
        var columnNames = schema.Select(col => col.Name).ToArray();

        // For prediction results, we need to handle all columns dynamically
        // since ML.NET adds prediction columns to the original data
        return ConvertFromGenericData(dataView, mlContext, columnNames);
    }

    /// <summary>
    /// Converts a numeric NivaraSeries to an ML.NET tensor format.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="series">The NivaraSeries to convert</param>
    /// <returns>A tensor representation suitable for ML.NET</returns>
    public static VBuffer<T> ToMLNetTensor<T>(this NivaraSeries<T> series)
        where T : struct, INumber<T>
    {
        if (series == null) throw new ArgumentNullException(nameof(series));

        var column = series.Values;
        var values = new T[column.Length];
        for (int i = 0; i < column.Length; i++)
        {
            values[i] = column[i];
        }
        return new VBuffer<T>(values.Length, values);
    }

    /// <summary>
    /// Creates a NivaraSeries from an ML.NET tensor (VBuffer).
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensor">The ML.NET tensor</param>
    /// <returns>A NivaraSeries containing the tensor data</returns>
    public static NivaraSeries<T> FromMLNetTensor<T>(VBuffer<T> tensor)
        where T : struct, INumber<T>
    {
        var values = new T[tensor.Length];
        tensor.CopyTo(values);

        return NivaraSeries<T>.Create(values);
    }

    /// <summary>
    /// Converts a NivaraFrame to a format suitable for ML.NET feature vectors.
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <param name="featureColumns">Names of columns to include as features</param>
    /// <returns>An array of feature vectors</returns>
    public static VBuffer<float>[] ToFeatureVectors(this NivaraFrame frame, params string[] featureColumns)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (featureColumns == null || featureColumns.Length == 0)
            throw new ArgumentException("At least one feature column must be specified", nameof(featureColumns));

        var vectors = new VBuffer<float>[frame.RowCount];
        var featureCount = featureColumns.Length;

        for (int row = 0; row < frame.RowCount; row++)
        {
            var features = new float[featureCount];

            for (int col = 0; col < featureColumns.Length; col++)
            {
                var columnName = featureColumns[col];
                var columnValue = frame.GetColumn(columnName).GetValue(row);
                features[col] = ConvertToFloat(columnValue);
            }

            vectors[row] = new VBuffer<float>(featureCount, features);
        }

        return vectors;
    }

    /// <summary>
    /// Creates a NivaraFrame from ML.NET feature vectors.
    /// </summary>
    /// <param name="vectors">The feature vectors</param>
    /// <param name="featureNames">Names for the feature columns</param>
    /// <returns>A NivaraFrame containing the feature data</returns>
    public static NivaraFrame FromFeatureVectors(VBuffer<float>[] vectors, string[]? featureNames = null)
    {
        if (vectors == null || vectors.Length == 0)
            throw new ArgumentException("Vectors cannot be null or empty", nameof(vectors));

        var featureCount = vectors[0].Length;
        var rowCount = vectors.Length;

        // Generate default names if not provided
        featureNames ??= Enumerable.Range(0, featureCount).Select(i => $"Feature_{i}").ToArray();

        if (featureNames.Length != featureCount)
            throw new ArgumentException("Feature names count must match vector dimension", nameof(featureNames));

        var columns = new List<(string Name, IColumn Column)>();

        // Create a column for each feature
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
        {
            var columnData = new float[rowCount];

            for (int rowIndex = 0; rowIndex < rowCount; rowIndex++)
            {
                var vector = vectors[rowIndex];
                columnData[rowIndex] = vector.GetItemOrDefault(featureIndex);
            }

            columns.Add((featureNames[featureIndex], NivaraColumn<float>.Create(columnData)));
        }

        return new NivaraFrame(columns);
    }
}

/// <summary>
/// Data classes for ML.NET compatibility.
/// </summary>
public class TwoColumnData
{
    public float Col1 { get; set; }
    public float Col2 { get; set; }
}

public class TwoColumnFeatureData
{
    public float Feature1 { get; set; }
    public float Feature2 { get; set; }
}

public class ThreeColumnData
{
    public float Feature1 { get; set; }
    public float Feature2 { get; set; }
    public float Label { get; set; }
}

public class GenericData
{
    public float[]? Features { get; set; }
}

