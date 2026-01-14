using System.Numerics;
using Nivara;
using Nivara.Extensions.AutoDiff.Utilities;

namespace Nivara.Extensions.AutoDiff.Extensions;

/// <summary>
/// Extension methods that provide seamless integration 
/// between Nivara types (Column, Series, Frame) and automatic differentiation.
/// Includes type validation to ensure only supported numeric types are used.
/// </summary>
public static class NivaraAutoGradExtensions
{
    /// <summary>
    /// Converts a NivaraColumn to a GradTensor with type validation.
    /// </summary>
    /// <typeparam name="T">The numeric type (must be float or double)</typeparam>
    /// <param name="column">The column to convert</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor wrapping the column</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type</exception>
    public static GradTensor<T> ToGradTensor<T>(this NivaraColumn<T> column, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        // Type validation is performed in GradTensor constructor
        return GradTensor<T>.FromColumn(column, requiresGrad);
    }

    /// <summary>
    /// Converts a NivaraSeries to a GradTensor with type validation.
    /// </summary>
    /// <typeparam name="T">The numeric type (must be float or double)</typeparam>
    /// <param name="series">The series to convert</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor wrapping the series values</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type</exception>
    public static GradTensor<T> ToGradTensor<T>(this NivaraSeries<T> series, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        // Type validation is performed in GradTensor constructor
        return GradTensor<T>.FromSeries(series, requiresGrad);
    }

    /// <summary>
    /// Checks if a type is supported for automatic differentiation.
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type is supported; otherwise, false</returns>
    public static bool IsAutoGradSupported<T>() where T : struct, INumber<T>
    {
        return TypeValidator.IsSupported<T>();
    }

    /// <summary>
    /// Gets the list of types supported for automatic differentiation.
    /// </summary>
    /// <returns>An array of supported types</returns>
    public static Type[] GetSupportedAutoGradTypes()
    {
        return TypeValidator.GetSupportedTypes();
    }

    /// <summary>
    /// Converts multiple columns from a NivaraFrame to GradTensors.
    /// This enables batch operations for ML workflows.
    /// </summary>
    /// <typeparam name="T">The numeric type (must be float or double)</typeparam>
    /// <param name="frame">The frame containing the columns</param>
    /// <param name="columnNames">The names of the columns to convert</param>
    /// <param name="requiresGrad">Whether the tensors should track gradients</param>
    /// <returns>A dictionary mapping column names to GradTensors</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when columnNames is empty or contains invalid names</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type or column type doesn't match T</exception>
    public static Dictionary<string, GradTensor<T>> ToGradTensors<T>(
        this NivaraFrame frame,
        string[] columnNames,
        bool requiresGrad = false) where T : struct, INumber<T>
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));

        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        if (columnNames.Length == 0)
            throw new ArgumentException("Must specify at least one column name", nameof(columnNames));

        // Validate type is supported for automatic differentiation
        TypeValidator.ValidateNumericType<T>();

        var result = new Dictionary<string, GradTensor<T>>(StringComparer.OrdinalIgnoreCase);

        foreach (var columnName in columnNames)
        {
            if (string.IsNullOrWhiteSpace(columnName))
                throw new ArgumentException($"Column name cannot be null or whitespace", nameof(columnNames));

            // Get the column with type checking
            var column = frame.GetColumn<T>(columnName);

            // Convert to GradTensor
            var gradTensor = column.ToGradTensor(requiresGrad);
            result[columnName] = gradTensor;
        }

        return result;
    }

    /// <summary>
    /// Converts all numeric columns from a NivaraFrame to GradTensors.
    /// Only columns with supported numeric types (float, double) are converted.
    /// </summary>
    /// <param name="frame">The frame containing the columns</param>
    /// <param name="requiresGrad">Whether the tensors should track gradients</param>
    /// <returns>A dictionary mapping column names to GradTensors (as object to support multiple types)</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    public static Dictionary<string, object> ToGradTensorsAuto(
        this NivaraFrame frame,
        bool requiresGrad = false)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));

        var result = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

        foreach (var columnName in frame.ColumnNames)
        {
            var column = frame.GetColumn(columnName);
            var columnType = column.ElementType;

            // Only convert supported numeric types
            if (columnType == typeof(float))
            {
                var typedColumn = frame.GetColumn<float>(columnName);
                result[columnName] = typedColumn.ToGradTensor(requiresGrad);
            }
            else if (columnType == typeof(double))
            {
                var typedColumn = frame.GetColumn<double>(columnName);
                result[columnName] = typedColumn.ToGradTensor(requiresGrad);
            }
            // Skip non-supported types
        }

        return result;
    }

    /// <summary>
    /// Converts a GradTensor back to a NivaraColumn.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="gradTensor">The GradTensor to convert</param>
    /// <returns>The underlying NivaraColumn</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradTensor is null</exception>
    public static NivaraColumn<T> ToColumn<T>(this GradTensor<T> gradTensor)
        where T : struct, INumber<T>
    {
        if (gradTensor == null)
            throw new ArgumentNullException(nameof(gradTensor));

        return gradTensor.ToColumn();
    }

    /// <summary>
    /// Converts a GradTensor back to a NivaraSeries.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="gradTensor">The GradTensor to convert</param>
    /// <returns>A new NivaraSeries wrapping the data</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradTensor is null</exception>
    public static NivaraSeries<T> ToSeries<T>(this GradTensor<T> gradTensor)
        where T : struct, INumber<T>
    {
        if (gradTensor == null)
            throw new ArgumentNullException(nameof(gradTensor));

        return gradTensor.ToSeries();
    }

    /// <summary>
    /// Performs batch gradient computation on multiple GradTensors.
    /// This is useful for computing gradients for all parameters in a model.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensors">The dictionary of named tensors to compute gradients for</param>
    /// <param name="loss">The scalar loss tensor to backpropagate from</param>
    /// <exception cref="ArgumentNullException">Thrown when tensors or loss is null</exception>
    /// <exception cref="ArgumentException">Thrown when tensors is empty</exception>
    /// <exception cref="InvalidOperationException">Thrown when loss is not a scalar or doesn't require gradients</exception>
    public static void BatchBackward<T>(
        this Dictionary<string, GradTensor<T>> tensors,
        GradTensor<T> loss) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        if (tensors.Count == 0)
            throw new ArgumentException("Tensors dictionary cannot be empty", nameof(tensors));

        // Perform backward pass on the loss
        loss.Backward();
    }

    /// <summary>
    /// Zeros gradients for all tensors in a batch.
    /// This is typically called before each training iteration.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensors">The dictionary of named tensors to zero gradients for</param>
    /// <exception cref="ArgumentNullException">Thrown when tensors is null</exception>
    public static void BatchZeroGrad<T>(
        this Dictionary<string, GradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        foreach (var tensor in tensors.Values)
        {
            tensor.ZeroGrad();
        }
    }

    /// <summary>
    /// Converts a batch of GradTensors back to a NivaraFrame.
    /// This is useful for converting model outputs back to a frame for analysis.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensors">The dictionary of named tensors to convert</param>
    /// <returns>A new NivaraFrame containing the tensor data as columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensors is null</exception>
    /// <exception cref="ArgumentException">Thrown when tensors is empty or columns have different lengths</exception>
    public static NivaraFrame ToFrame<T>(
        this Dictionary<string, GradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (tensors.Count == 0)
            throw new ArgumentException("Tensors dictionary cannot be empty", nameof(tensors));

        var namedColumns = tensors.Select(kvp => (kvp.Key, (IColumn)kvp.Value.ToColumn()));
        return NivaraFrame.Create(namedColumns.ToArray());
    }

    /// <summary>
    /// Extracts gradients from a batch of GradTensors and returns them as a NivaraFrame.
    /// This is useful for analyzing gradients during training.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="tensors">The dictionary of named tensors to extract gradients from</param>
    /// <returns>A new NivaraFrame containing the gradient data as columns, or null if no gradients are available</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensors is null</exception>
    /// <exception cref="ArgumentException">Thrown when tensors is empty</exception>
    public static NivaraFrame? ToGradientFrame<T>(
        this Dictionary<string, GradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (tensors.Count == 0)
            throw new ArgumentException("Tensors dictionary cannot be empty", nameof(tensors));

        var namedColumns = new List<(string Name, IColumn Column)>();

        foreach (var kvp in tensors)
        {
            if (kvp.Value.Grad != null)
            {
                namedColumns.Add((kvp.Key, kvp.Value.Grad));
            }
        }

        // Return null if no gradients are available
        if (namedColumns.Count == 0)
            return null;

        return NivaraFrame.Create(namedColumns.ToArray());
    }
}
