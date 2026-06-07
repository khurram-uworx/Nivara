using Nivara;
using Nivara.Extensions.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.Extensions.AutoDiff.Extensions;

/// <summary>
/// Extension methods that provide seamless integration 
/// between Nivara types (Column, Series, Frame) and reverse-mode automatic differentiation.
/// </summary>
public static class NivaraAutoGradExtensions
{
    /// <summary>
    /// Converts a NivaraColumn to a ReverseGradTensor with type validation.
    /// </summary>
    public static ReverseGradTensor<T> ToReverseGradTensor<T>(this NivaraColumn<T> column, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return ReverseGradTensor<T>.FromColumn(column, requiresGrad);
    }

    /// <summary>
    /// Converts a NivaraSeries to a ReverseGradTensor with type validation.
    /// </summary>
    public static ReverseGradTensor<T> ToReverseGradTensor<T>(this NivaraSeries<T> series, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        return ReverseGradTensor<T>.FromSeries(series, requiresGrad);
    }

    /// <summary>
    /// Checks if a type is supported for automatic differentiation.
    /// </summary>
    public static bool IsAutoGradSupported<T>() where T : struct, INumber<T>
    {
        return TypeValidator.IsSupported<T>();
    }

    /// <summary>
    /// Gets the list of types supported for automatic differentiation.
    /// </summary>
    public static Type[] GetSupportedAutoGradTypes()
    {
        return TypeValidator.GetSupportedTypes();
    }

    /// <summary>
    /// Converts multiple columns from a NivaraFrame to ReverseGradTensors.
    /// </summary>
    public static Dictionary<string, ReverseGradTensor<T>> ToReverseGradTensors<T>(
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

        TypeValidator.ValidateNumericType<T>();

        var result = new Dictionary<string, ReverseGradTensor<T>>(StringComparer.OrdinalIgnoreCase);

        foreach (var columnName in columnNames)
        {
            if (string.IsNullOrWhiteSpace(columnName))
                throw new ArgumentException($"Column name cannot be null or whitespace", nameof(columnNames));

            var column = frame.GetColumn<T>(columnName);

            var gradTensor = column.ToReverseGradTensor(requiresGrad);
            result[columnName] = gradTensor;
        }

        return result;
    }

    /// <summary>
    /// Converts all numeric columns from a NivaraFrame to ReverseGradTensors.
    /// Only columns with supported numeric types (float, double) are converted.
    /// </summary>
    public static Dictionary<string, object> ToReverseGradTensorsAuto(
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

            if (columnType == typeof(float))
            {
                var typedColumn = frame.GetColumn<float>(columnName);
                result[columnName] = typedColumn.ToReverseGradTensor(requiresGrad);
            }
            else if (columnType == typeof(double))
            {
                var typedColumn = frame.GetColumn<double>(columnName);
                result[columnName] = typedColumn.ToReverseGradTensor(requiresGrad);
            }
        }

        return result;
    }

    /// <summary>
    /// Converts a ReverseGradTensor back to a NivaraColumn.
    /// </summary>
    public static NivaraColumn<T> ToColumn<T>(this ReverseGradTensor<T> gradTensor)
        where T : struct, INumber<T>
    {
        if (gradTensor == null)
            throw new ArgumentNullException(nameof(gradTensor));

        return gradTensor.ToColumn();
    }

    /// <summary>
    /// Converts a ReverseGradTensor back to a NivaraSeries.
    /// </summary>
    public static NivaraSeries<T> ToSeries<T>(this ReverseGradTensor<T> gradTensor)
        where T : struct, INumber<T>
    {
        if (gradTensor == null)
            throw new ArgumentNullException(nameof(gradTensor));

        return gradTensor.ToSeries();
    }

    /// <summary>
    /// Performs batch gradient computation on multiple ReverseGradTensors.
    /// </summary>
    public static void BatchBackward<T>(
        this Dictionary<string, ReverseGradTensor<T>> tensors,
        ReverseGradTensor<T> loss) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        if (tensors.Count == 0)
            throw new ArgumentException("Tensors dictionary cannot be empty", nameof(tensors));

        loss.Backward();
    }

    /// <summary>
    /// Zeros gradients for all tensors in a batch.
    /// </summary>
    public static void BatchZeroGrad<T>(
        this Dictionary<string, ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        foreach (var tensor in tensors.Values)
        {
            tensor.ZeroGrad();
        }
    }

    /// <summary>
    /// Converts a batch of ReverseGradTensors back to a NivaraFrame.
    /// </summary>
    public static NivaraFrame ToFrame<T>(
        this Dictionary<string, ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (tensors.Count == 0)
            throw new ArgumentException("Tensors dictionary cannot be empty", nameof(tensors));

        var namedColumns = tensors.Select(kvp => (kvp.Key, (IColumn)kvp.Value.ToColumn()));
        return NivaraFrame.Create(namedColumns.ToArray());
    }

    /// <summary>
    /// Extracts gradients from a batch of ReverseGradTensors and returns them as a NivaraFrame.
    /// </summary>
    public static NivaraFrame? ToGradientFrame<T>(
        this Dictionary<string, ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
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

        if (namedColumns.Count == 0)
            return null;

        return NivaraFrame.Create(namedColumns.ToArray());
    }
}
