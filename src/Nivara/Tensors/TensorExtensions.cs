using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.Tensors;

/// <summary>
/// Extension methods for integrating Nivara types with System.Numerics.Tensors.
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Performs tensor-aware element-wise addition using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="left">The left operand series</param>
    /// <param name="right">The right operand series</param>
    /// <returns>A new NivaraSeries containing the element-wise sum</returns>
    /// <exception cref="ArgumentNullException">Thrown when either series is null</exception>
    /// <exception cref="ArgumentException">Thrown when series have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contain null values</exception>
    public static NivaraSeries<T> AddTensor<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
        {
            throw new ArgumentException("Series must have the same length for tensor addition");
        }

        if (left.Length == 0)
        {
            return new NivaraSeries<T>();
        }

        // Check for null values in both series
        for (int i = 0; i < left.Length; i++)
        {
            if (!left.IsValid(i) || !right.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var leftSpan = MemoryMarshal.Cast<T, float>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, float>(right.Values.AsSpan());
            var result = new float[left.Length];

            TensorPrimitives.Add(leftSpan, rightSpan, result);

            var resultT = MemoryMarshal.Cast<float, T>(result.AsSpan()).ToArray();
            return NivaraSeries<T>.Create(resultT);
        }
        else if (typeof(T) == typeof(double))
        {
            var leftSpan = MemoryMarshal.Cast<T, double>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, double>(right.Values.AsSpan());
            var result = new double[left.Length];

            TensorPrimitives.Add(leftSpan, rightSpan, result);

            var resultT = MemoryMarshal.Cast<double, T>(result.AsSpan()).ToArray();
            return NivaraSeries<T>.Create(resultT);
        }
        else
        {
            // Fallback to regular SIMD operations for other types
            return left.Add(right);
        }
    }

    /// <summary>
    /// Performs tensor-aware element-wise multiplication using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="left">The left operand series</param>
    /// <param name="right">The right operand series</param>
    /// <returns>A new NivaraSeries containing the element-wise product</returns>
    /// <exception cref="ArgumentNullException">Thrown when either series is null</exception>
    /// <exception cref="ArgumentException">Thrown when series have different lengths</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contain null values</exception>
    public static NivaraSeries<T> MultiplyTensor<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
        {
            throw new ArgumentException("Series must have the same length for tensor multiplication");
        }

        if (left.Length == 0)
        {
            return new NivaraSeries<T>();
        }

        // Check for null values in both series
        for (int i = 0; i < left.Length; i++)
        {
            if (!left.IsValid(i) || !right.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var leftSpan = MemoryMarshal.Cast<T, float>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, float>(right.Values.AsSpan());
            var result = new float[left.Length];

            TensorPrimitives.Multiply(leftSpan, rightSpan, result);

            var resultT = MemoryMarshal.Cast<float, T>(result.AsSpan()).ToArray();
            return NivaraSeries<T>.Create(resultT);
        }
        else if (typeof(T) == typeof(double))
        {
            var leftSpan = MemoryMarshal.Cast<T, double>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, double>(right.Values.AsSpan());
            var result = new double[left.Length];

            TensorPrimitives.Multiply(leftSpan, rightSpan, result);

            var resultT = MemoryMarshal.Cast<double, T>(result.AsSpan()).ToArray();
            return NivaraSeries<T>.Create(resultT);
        }
        else
        {
            // Fallback to regular SIMD operations for other types
            return left.Multiply(right);
        }
    }

    /// <summary>
    /// Computes the sum of all elements using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The series to sum</param>
    /// <returns>The sum of all valid elements</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    public static T SumTensor<T>(this NivaraSeries<T> series)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
        {
            return default(T);
        }

        // Check for null values
        for (int i = 0; i < series.Length; i++)
        {
            if (!series.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var span = MemoryMarshal.Cast<T, float>(series.Values.AsSpan());
            var result = TensorPrimitives.Sum(span);
            return (T)(object)result;
        }
        else if (typeof(T) == typeof(double))
        {
            var span = MemoryMarshal.Cast<T, double>(series.Values.AsSpan());
            var result = TensorPrimitives.Sum(span);
            return (T)(object)result;
        }
        else
        {
            // Fallback to manual sum for other types
            T result = default(T);
            for (int i = 0; i < series.Length; i++)
            {
                result += series[i];
            }
            return result;
        }
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
    public static T DotProduct<T>(this NivaraSeries<T> left, NivaraSeries<T> right)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
        {
            throw new ArgumentException("Series must have the same length for dot product");
        }

        if (left.Length == 0)
        {
            return default(T);
        }

        // Check for null values in both series
        for (int i = 0; i < left.Length; i++)
        {
            if (!left.IsValid(i) || !right.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var leftSpan = MemoryMarshal.Cast<T, float>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, float>(right.Values.AsSpan());
            var result = TensorPrimitives.Dot(leftSpan, rightSpan);
            return (T)(object)result;
        }
        else if (typeof(T) == typeof(double))
        {
            var leftSpan = MemoryMarshal.Cast<T, double>(left.Values.AsSpan());
            var rightSpan = MemoryMarshal.Cast<T, double>(right.Values.AsSpan());
            var result = TensorPrimitives.Dot(leftSpan, rightSpan);
            return (T)(object)result;
        }
        else
        {
            // Fallback to manual dot product calculation for other types
            T result = default(T);
            for (int i = 0; i < left.Length; i++)
            {
                var product = left[i] * right[i];
                result += product;
            }
            return result;
        }
    }

    /// <summary>
    /// Computes the Euclidean norm (L2 norm) of the series using TensorPrimitives for optimized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The series to compute the norm for</param>
    /// <returns>The Euclidean norm of the series</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    public static T Norm<T>(this NivaraSeries<T> series)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);

        if (series.Length == 0)
        {
            return default(T);
        }

        // Check for null values
        for (int i = 0; i < series.Length; i++)
        {
            if (!series.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var span = MemoryMarshal.Cast<T, float>(series.Values.AsSpan());
            var result = TensorPrimitives.Norm(span);
            return (T)(object)result;
        }
        else if (typeof(T) == typeof(double))
        {
            var span = MemoryMarshal.Cast<T, double>(series.Values.AsSpan());
            var result = TensorPrimitives.Norm(span);
            return (T)(object)result;
        }
        else
        {
            // Fallback to manual norm calculation for other types
            T sumOfSquares = default(T);
            for (int i = 0; i < series.Length; i++)
            {
                var value = series[i];
                sumOfSquares += value * value;
            }

            // For integer types, we can't take square root, so return sum of squares
            if (typeof(T) == typeof(int) || typeof(T) == typeof(long))
            {
                return sumOfSquares;
            }

            // For floating point types, take square root
            return T.CreateChecked(Math.Sqrt(double.CreateChecked(sumOfSquares)));
        }
    }

    /// <summary>
    /// Applies a tensor-aware transformation function to each element using TensorPrimitives when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="series">The series to transform</param>
    /// <param name="function">The transformation function to apply</param>
    /// <returns>A new NivaraSeries with the transformed values</returns>
    /// <exception cref="ArgumentNullException">Thrown when series or function is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values</exception>
    public static NivaraSeries<T> TransformTensor<T>(this NivaraSeries<T> series, Func<T, T> function)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(series);
        ArgumentNullException.ThrowIfNull(function);

        if (series.Length == 0)
        {
            return new NivaraSeries<T>();
        }

        // Check for null values
        for (int i = 0; i < series.Length; i++)
        {
            if (!series.IsValid(i))
            {
                throw new InvalidOperationException($"Cannot perform tensor operations on series with null values. Found null at index {i}");
            }
        }

        var result = new T[series.Length];

        // Apply transformation function to each element
        for (int i = 0; i < series.Length; i++)
        {
            result[i] = function(series[i]);
        }

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
    public static NivaraFrame MatrixMultiply<T>(this NivaraFrame left, NivaraFrame right)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.ColumnCount != right.RowCount)
        {
            throw new ArgumentException($"Matrix dimensions incompatible: left columns ({left.ColumnCount}) must equal right rows ({right.RowCount})");
        }

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
                {
                    sum += leftSpan[i, k] * rightSpan[k, j];
                }
                resultData[i * right.ColumnCount + j] = sum;
            }
        }

        var resultTensor = Tensor.Create<T>(resultData, resultDimensions);

        // Convert result tensor back to NivaraFrame
        var columnNames = Enumerable.Range(0, right.ColumnCount).Select(i => $"Result_{i}").ToArray();
        return TensorInterop.FromTensor(resultTensor, columnNames);
    }
}
