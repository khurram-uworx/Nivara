using Microsoft.ML;

namespace Nivara.MLNet;

/// <summary>
/// Extension methods for seamless integration between Nivara and ML.NET.
/// </summary>
public static class MLNetExtensions
{
    /// <summary>
    /// Creates an ML.NET pipeline that can process NivaraFrames directly.
    /// </summary>
    /// <param name="mlContext">The ML.NET context</param>
    /// <param name="frame">The input NivaraFrame</param>
    /// <returns>An IDataView for use in ML.NET pipelines</returns>
    public static IDataView LoadFromNivaraFrame(this MLContext mlContext, NivaraFrame frame)
    {
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));
        if (frame == null) throw new ArgumentNullException(nameof(frame));

        return MLNetInterop.ToDataView(frame, mlContext);
    }

    /// <summary>
    /// Converts ML.NET prediction results back to a NivaraFrame.
    /// </summary>
    /// <param name="mlContext">The ML.NET context</param>
    /// <param name="predictions">The ML.NET predictions</param>
    /// <returns>A NivaraFrame containing the prediction results</returns>
    public static NivaraFrame ToNivaraFrame(this MLContext mlContext, IDataView predictions)
    {
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));

        return MLNetInterop.ToNivaraFrame(predictions, mlContext);
    }

    /// <summary>
    /// Applies an ML.NET transformer to a NivaraFrame and returns the result as a NivaraFrame.
    /// </summary>
    /// <param name="transformer">The ML.NET transformer</param>
    /// <param name="frame">The input NivaraFrame</param>
    /// <param name="mlContext">The ML.NET context</param>
    /// <returns>A NivaraFrame containing the transformed data</returns>
    public static NivaraFrame Transform(this ITransformer transformer, NivaraFrame frame, MLContext mlContext)
    {
        if (transformer == null) throw new ArgumentNullException(nameof(transformer));
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));

        var dataView = MLNetInterop.ToDataView(frame, mlContext);
        var transformedView = transformer.Transform(dataView);
        return MLNetInterop.ToNivaraFrame(transformedView, mlContext);
    }

    /// <summary>
    /// Trains an ML.NET model using a NivaraFrame as training data.
    /// </summary>
    /// <typeparam name="TModel">The model type</typeparam>
    /// <param name="trainer">The ML.NET trainer</param>
    /// <param name="trainingFrame">The training data as a NivaraFrame</param>
    /// <param name="mlContext">The ML.NET context</param>
    /// <returns>The trained model</returns>
    public static TModel Fit<TModel>(this IEstimator<TModel> trainer, NivaraFrame trainingFrame, MLContext mlContext)
        where TModel : class, ITransformer
    {
        if (trainer == null) throw new ArgumentNullException(nameof(trainer));
        if (trainingFrame == null) throw new ArgumentNullException(nameof(trainingFrame));
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));

        var dataView = MLNetInterop.ToDataView(trainingFrame, mlContext);
        return trainer.Fit(dataView);
    }

    /// <summary>
    /// Makes predictions on a NivaraFrame using a trained ML.NET model.
    /// </summary>
    /// <param name="model">The trained ML.NET model</param>
    /// <param name="inputFrame">The input data as a NivaraFrame</param>
    /// <param name="mlContext">The ML.NET context</param>
    /// <returns>A NivaraFrame containing the predictions</returns>
    public static NivaraFrame Predict(this ITransformer model, NivaraFrame inputFrame, MLContext mlContext)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (inputFrame == null) throw new ArgumentNullException(nameof(inputFrame));
        if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));

        var dataView = MLNetInterop.ToDataView(inputFrame, mlContext);
        var predictions = model.Transform(dataView);
        return MLNetInterop.ToNivaraFrame(predictions, mlContext);
    }

    /// <summary>
    /// Creates a feature matrix from selected columns of a NivaraFrame.
    /// </summary>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="featureColumns">The columns to include as features</param>
    /// <returns>A 2D array representing the feature matrix</returns>
    public static float[,] CreateFeatureMatrix(this NivaraFrame frame, params string[] featureColumns)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (featureColumns == null || featureColumns.Length == 0)
            throw new ArgumentException("At least one feature column must be specified", nameof(featureColumns));

        var matrix = new float[frame.RowCount, featureColumns.Length];

        for (int row = 0; row < frame.RowCount; row++)
        {
            for (int col = 0; col < featureColumns.Length; col++)
            {
                var columnName = featureColumns[col];
                var columnValue = frame.GetColumn(columnName).GetValue(row);
                var value = ConvertToFloat(columnValue);
                matrix[row, col] = value;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Extracts labels from a NivaraFrame for supervised learning.
    /// /// </summary>
    /// <typeparam name="T">The label type</typeparam>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="labelColumn">The name of the label column</param>
    /// <returns>An array of labels</returns>
    public static T[] ExtractLabels<T>(this NivaraFrame frame, string labelColumn)
        where T : struct
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (string.IsNullOrEmpty(labelColumn)) throw new ArgumentException("Label column name cannot be null or empty", nameof(labelColumn));

        var column = frame.GetColumn<T>(labelColumn);
        var labels = new T[frame.RowCount];

        for (int row = 0; row < frame.RowCount; row++)
        {
            labels[row] = column[row];
        }

        return labels;
    }

    /// <summary>
    /// Splits a NivaraFrame into training and testing sets.
    /// </summary>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="trainRatio">The ratio of data to use for training (0.0 to 1.0)</param>
    /// <param name="randomSeed">Optional random seed for reproducible splits</param>
    /// <returns>A tuple containing (training data, testing data)</returns>
    public static (NivaraFrame Training, NivaraFrame Testing) TrainTestSplit(
        this NivaraFrame frame,
        double trainRatio = 0.8,
        int? randomSeed = null)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (trainRatio <= 0 || trainRatio >= 1) throw new ArgumentException("Train ratio must be between 0 and 1", nameof(trainRatio));

        var random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
        var indices = Enumerable.Range(0, frame.RowCount).ToArray();

        // Shuffle indices
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var trainCount = (int)(frame.RowCount * trainRatio);
        var trainIndices = indices.Take(trainCount).OrderBy(x => x).ToArray();
        var testIndices = indices.Skip(trainCount).OrderBy(x => x).ToArray();

        var trainingFrame = SelectRows(frame, trainIndices);
        var testingFrame = SelectRows(frame, testIndices);

        return (trainingFrame, testingFrame);
    }

    /// <summary>
    /// Normalizes numeric columns in a NivaraFrame for ML.NET processing.
    /// </summary>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="columns">The columns to normalize (null for all numeric columns)</param>
    /// <returns>A new NivaraFrame with normalized columns</returns>
    public static NivaraFrame Normalize(this NivaraFrame frame, params string[]? columns)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));

        // If no columns specified, normalize all numeric columns
        columns ??= frame.ColumnNames.Where(name => IsNumericColumn(frame, name)).ToArray();

        // Create a set of columns to normalize for quick lookup
        var columnsToNormalize = new HashSet<string>(columns, StringComparer.OrdinalIgnoreCase);

        // Build a new list of columns, normalizing as needed
        var newColumns = new List<(string Name, IColumn Column)>();

        foreach (var columnName in frame.ColumnNames)
        {
            IColumn resultColumn;

            if (columnsToNormalize.Contains(columnName) && IsNumericColumn(frame, columnName))
            {
                resultColumn = NormalizeColumn(frame, columnName);
            }
            else
            {
                resultColumn = frame.GetColumn(columnName);
            }

            newColumns.Add((columnName, resultColumn));
        }

        return new NivaraFrame(newColumns);
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

    private static bool IsNumericColumn(NivaraFrame frame, string columnName)
    {
        var columnType = frame.Schema.GetColumnType(columnName);
        return columnType == typeof(int) || columnType == typeof(long) ||
               columnType == typeof(float) || columnType == typeof(double) ||
               columnType == typeof(decimal) || columnType == typeof(byte) ||
               columnType == typeof(short);
    }

    private static IColumn NormalizeColumn(NivaraFrame frame, string columnName)
    {
        var columnType = frame.Schema.GetColumnType(columnName);

        if (columnType == typeof(float))
        {
            var column = frame.GetColumn<float>(columnName);
            var values = new float[column.Length];
            for (int i = 0; i < column.Length; i++)
            {
                values[i] = column[i];
            }
            var mean = values.Average();
            var stdDev = Math.Sqrt(values.Select(x => Math.Pow(x - mean, 2)).Average());

            if (stdDev > 0)
            {
                var normalized = values.Select(x => (float)((x - mean) / stdDev)).ToArray();
                return NivaraColumn<float>.Create(normalized);
            }
            return NivaraColumn<float>.Create(values);
        }

        if (columnType == typeof(double))
        {
            var column = frame.GetColumn<double>(columnName);
            var values = new double[column.Length];
            for (int i = 0; i < column.Length; i++)
            {
                values[i] = column[i];
            }
            var mean = values.Average();
            var stdDev = Math.Sqrt(values.Select(x => Math.Pow(x - mean, 2)).Average());

            if (stdDev > 0)
            {
                var normalized = values.Select(x => (x - mean) / stdDev).ToArray();
                return NivaraColumn<double>.Create(normalized);
            }
            return NivaraColumn<double>.Create(values);
        }

        throw new NotSupportedException($"Normalization for type {columnType} is not yet implemented");
    }

    /// <summary>
    /// Selects rows from a frame based on the specified indices
    /// </summary>
    private static NivaraFrame SelectRows(NivaraFrame frame, int[] indices)
    {
        if (indices == null || indices.Length == 0)
            throw new ArgumentException("Must specify at least one row index", nameof(indices));

        var selectedColumns = new List<(string Name, IColumn Column)>();

        foreach (var columnName in frame.ColumnNames)
        {
            var column = frame.GetColumn(columnName);
            var selectedValues = SelectRowsForColumn(column, indices);
            selectedColumns.Add((columnName, selectedValues));
        }

        return new NivaraFrame(selectedColumns);
    }

    /// <summary>
    /// Selects specific rows from a column based on indices
    /// </summary>
    private static IColumn SelectRowsForColumn(IColumn column, int[] indices)
    {
        var elementType = column.ElementType;

        return elementType switch
        {
            Type t when t == typeof(int) => SelectRowsTyped<int>(column, indices),
            Type t when t == typeof(double) => SelectRowsTyped<double>(column, indices),
            Type t when t == typeof(float) => SelectRowsTyped<float>(column, indices),
            Type t when t == typeof(long) => SelectRowsTyped<long>(column, indices),
            Type t when t == typeof(string) => SelectRowsTyped<string>(column, indices),
            Type t when t == typeof(bool) => SelectRowsTyped<bool>(column, indices),
            Type t when t == typeof(decimal) => SelectRowsTyped<decimal>(column, indices),
            Type t when t == typeof(byte) => SelectRowsTyped<byte>(column, indices),
            Type t when t == typeof(short) => SelectRowsTyped<short>(column, indices),
            Type t when t == typeof(DateTime) => SelectRowsTyped<DateTime>(column, indices),
            _ => SelectRowsGeneric(column, indices)
        };
    }

    /// <summary>
    /// Selects rows for a typed column
    /// </summary>
    private static IColumn SelectRowsTyped<T>(IColumn column, int[] indices)
    {
        var selectedValues = new T[indices.Length];

        // Try to access as a typed column first for better type preservation
        if (column is NivaraColumn<T> typedColumn)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                selectedValues[i] = typedColumn[indices[i]];
            }
        }
        else
        {
            // Fall back to GetValue for non-typed columns
            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value == null)
                {
                    selectedValues[i] = default(T)!;
                }
                else if (typeof(T) == value.GetType())
                {
                    selectedValues[i] = (T)value;
                }
                else
                {
                    selectedValues[i] = (T)Convert.ChangeType(value, typeof(T))!;
                }
            }
        }

        return NivaraColumn<T>.Create(selectedValues);
    }

    /// <summary>
    /// Selects rows for a generic column (object type)
    /// </summary>
    private static IColumn SelectRowsGeneric(IColumn column, int[] indices)
    {
        var selectedValues = new object[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            selectedValues[i] = column.GetValue(indices[i])!;
        }
        return NivaraColumn<object>.Create(selectedValues);
    }
}
