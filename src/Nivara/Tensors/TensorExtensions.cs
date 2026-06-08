using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Tensors;

/// <summary>
/// Extension methods for integrating Nivara types with System.Numerics.Tensors.
/// </summary>
public static class TensorExtensions
{
    static T?[] toNullableArray<T>(NivaraSeries<T> series)
        where T : unmanaged
    {
        var values = new T?[series.Length];

        for (int i = 0; i < values.Length; i++)
            values[i] = series.IsNull(i) ? null : series[i];

        return values;
    }

    /// <summary>
    /// Performs tensor-aware element-wise addition using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The unmanaged numeric type</typeparam>
    /// <param name="left">The left operand series</param>
    /// <param name="right">The right operand series</param>
    /// <returns>A new NivaraSeries containing the element-wise sum</returns>
    /// <exception cref="ArgumentNullException">Thrown when either series is null</exception>
    /// <exception cref="ArgumentException">Thrown when series have different lengths</exception>
    [Obsolete("Use TensorPrimitives.Add on spans obtained via TryGetSpan", false)]
    public static NivaraSeries<T> AddTensor<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
            throw new ArgumentException("Series must have the same length for tensor addition");

        if (left.Length == 0)
            return new NivaraSeries<T>();

        if (left.HasNulls || right.HasNulls)
        {
            using var leftValues = NivaraColumn<T>.CreateFromNullable(toNullableArray(left));
            using var rightValues = NivaraColumn<T>.CreateFromNullable(toNullableArray(right));
            return new NivaraSeries<T>(leftValues.Add(rightValues), left.Index);
        }

        var leftSpan = left.Values.AsSpan();
        var rightSpan = right.Values.AsSpan();
        var result = new T[left.Length];

        TensorPrimitives.Add(leftSpan, rightSpan, result);

        return NivaraSeries<T>.Create(result);
    }

    /// <summary>
    /// Performs tensor-aware element-wise multiplication using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The unmanaged numeric type</typeparam>
    /// <param name="left">The left operand series</param>
    /// <param name="right">The right operand series</param>
    /// <returns>A new NivaraSeries containing the element-wise product</returns>
    /// <exception cref="ArgumentNullException">Thrown when either series is null</exception>
    /// <exception cref="ArgumentException">Thrown when series have different lengths</exception>
    [Obsolete("Use TensorPrimitives.Multiply on spans obtained via TryGetSpan", false)]
    public static NivaraSeries<T> MultiplyTensor<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
            throw new ArgumentException("Series must have the same length for tensor multiplication");

        if (left.Length == 0)
            return new NivaraSeries<T>();

        if (left.HasNulls || right.HasNulls)
        {
            using var leftValues = NivaraColumn<T>.CreateFromNullable(toNullableArray(left));
            using var rightValues = NivaraColumn<T>.CreateFromNullable(toNullableArray(right));
            return new NivaraSeries<T>(leftValues.Multiply(rightValues), left.Index);
        }

        var leftSpan = left.Values.AsSpan();
        var rightSpan = right.Values.AsSpan();
        var result = new T[left.Length];

        TensorPrimitives.Multiply(leftSpan, rightSpan, result);

        return NivaraSeries<T>.Create(result);
    }

    /// <summary>
    /// Computes the sum of all elements using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The series to sum</param>
    /// <returns>The sum of all valid elements</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    [Obsolete("Use TensorPrimitives.Sum on a span obtained via TryGetSpan", false)]
    public static T SumTensor<T>(this NivaraSeries<T> series)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
            return default(T);

        // Check for null values
        for (int i = 0; i < series.Length; i++)
            if (!series.IsValid(i))
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");

        return TensorPrimitives.Sum(series.Values.AsSpan());
    }

    /// <summary>
    /// Computes the dot product of two series using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="left">The left operand series</param>
    /// <param name="right">The right operand series</param>
    /// <returns>The dot product of the two series</returns>
    /// <exception cref="ArgumentNullException">Thrown when either series is null</exception>
    /// <exception cref="ArgumentException">Thrown when series have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contain null values</exception>
    [Obsolete("Use TensorPrimitives.Dot on spans obtained via TryGetSpan", false)]
    public static T DotProduct<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
            throw new ArgumentException("Series must have the same length for dot product");

        if (left.Length == 0)
            return default(T);

        // Check for null values in both series
        for (int i = 0; i < left.Length; i++)
            if (!left.IsValid(i) || !right.IsValid(i))
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");

        return TensorPrimitives.Dot(left.Values.AsSpan(), right.Values.AsSpan());
    }

    /// <summary>
    /// Computes the Euclidean norm (L2 norm) of the series using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The series to compute the norm for</param>
    /// <returns>The Euclidean norm of the series</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    [Obsolete("Use TensorPrimitives.Norm on a span obtained via TryGetSpan", false)]
    public static T Norm<T>(this NivaraSeries<T> series)
        where T : unmanaged, IRootFunctions<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
            return default(T);

        // Check for null values
        for (int i = 0; i < series.Length; i++)
            if (!series.IsValid(i))
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");

        return TensorPrimitives.Norm(series.Values.AsSpan());
    }

    /// <summary>
    /// Transforms elements of the series using a provided function (tensor-style mapping).
    /// </summary>
    /// <typeparam name="T">The unmanaged numeric type</typeparam>
    /// <param name="series">The series to transform</param>
    /// <param name="function">The transformation function to apply</param>
    /// <returns>A new NivaraSeries with the transformed values</returns>
    /// <exception cref="ArgumentNullException">Thrown when series or function is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    [Obsolete("Use LINQ Select on spans obtained via TryGetSpan", false)]
    public static NivaraSeries<T> TransformTensor<T>(this NivaraSeries<T> series, Func<T, T> function)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(series);
        ArgumentNullException.ThrowIfNull(function);

        if (series.Length == 0)
            return new NivaraSeries<T>();

        // Check for null values
        for (int i = 0; i < series.Length; i++)
            if (!series.IsValid(i))
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");

        var result = new T[series.Length];

        // Apply transformation function to each element
        for (int i = 0; i < series.Length; i++)
            result[i] = function(series[i]);

        return NivaraSeries<T>.Create(result);
    }

    /// <summary>
    /// Performs matrix multiplication between two NivaraFrames treated as 2D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="left">The left matrix (NivaraFrame)</param>
    /// <param name="right">The right matrix (NivaraFrame)</param>
    /// <returns>A new NivaraFrame containing the matrix multiplication result</returns>
    /// <exception cref="ArgumentNullException">Thrown when either frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions are incompatible</exception>
    /// <exception cref="InvalidOperationException">Thrown when frames contain null values or non-numeric columns</exception>
    [Obsolete("Use System.Numerics.Tensors tensor multiply on data obtained via ToTensor", false)]
    public static NivaraFrame MatrixMultiply<T>(this NivaraFrame left, NivaraFrame right)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.ColumnCount != right.RowCount)
            throw new ArgumentException($"Matrix dimensions incompatible: left columns ({left.ColumnCount}) must equal right rows ({right.RowCount})");

        // Convert frames to tensors
        var leftTensor = left.ToTensor<T>();
        var rightTensor = right.ToTensor<T>();

        // Perform matrix multiplication using tensor operations
        var resultDimensions = new ReadOnlySpan<nint>(new nint[] { left.RowCount, right.ColumnCount });
        var resultData = new T[left.RowCount * right.ColumnCount];

        var leftSpan = leftTensor.AsTensorSpan();
        var rightSpan = rightTensor.AsTensorSpan();

        // Manual matrix multiplication (could be optimized with BLAS in the future)
        for (int i = 0; i < left.RowCount; i++)
        {
            for (int j = 0; j < right.ColumnCount; j++)
            {
                T sum = default(T);
                for (int k = 0; k < left.ColumnCount; k++)
                    sum += leftSpan[i, k] * rightSpan[k, j];
                resultData[i * right.ColumnCount + j] = sum;
            }
        }

        var resultTensor = Tensor.Create<T>(resultData, resultDimensions);

        // Convert result tensor back to NivaraFrame
        var columnNames = Enumerable.Range(0, right.ColumnCount).Select(i => $"Result_{i}").ToArray();
        return TensorInteropExtensions.FromTensor(resultTensor, columnNames);
    }
}
