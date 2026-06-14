using Nivara.Helpers;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.Tensors;

/// <summary>
/// Extension methods for NivaraColumn&lt;T&gt; and NivaraSeries&lt;T&gt; tensor operations.
/// </summary>
public static class NivaraTensorExtensions
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

    #region Unary Math

    public static NivaraColumn<T> Exp<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            if (typeof(T) == typeof(float))
                TensorPrimitives.Exp(MemoryMarshal.Cast<T, float>(span), MemoryMarshal.Cast<T, float>(result.AsSpan()));
            else if (typeof(T) == typeof(double))
                TensorPrimitives.Exp(MemoryMarshal.Cast<T, double>(span), MemoryMarshal.Cast<T, double>(result.AsSpan()));
            else
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(Math.Exp(double.CreateChecked(span[i])));
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = T.CreateChecked(Math.Exp(double.CreateChecked(inputBuf[i])));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Log<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            if (typeof(T) == typeof(float))
                TensorPrimitives.Log(MemoryMarshal.Cast<T, float>(span), MemoryMarshal.Cast<T, float>(result.AsSpan()));
            else if (typeof(T) == typeof(double))
                TensorPrimitives.Log(MemoryMarshal.Cast<T, double>(span), MemoryMarshal.Cast<T, double>(result.AsSpan()));
            else
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(Math.Log(double.CreateChecked(span[i])));
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = T.CreateChecked(Math.Log(double.CreateChecked(inputBuf[i])));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Abs<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Abs(span, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Abs(inputBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Negate<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Negate(span, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Negate(inputBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Relu<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Max(span, T.Zero, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Max(inputBuf.AsSpan(0, n), T.Zero, resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Clamp<T>(this NivaraColumn<T> column, T min, T max) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Clamp(span, min, max, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Clamp(inputBuf.AsSpan(0, n), min, max, resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Sigmoid<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorsHelper.Sigmoid(span, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorsHelper.Sigmoid(inputBuf.AsSpan(0, n), nullMask.AsSpan(0, n), resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Tanh<T>(this NivaraColumn<T> column) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorsHelper.Tanh(span, result);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorsHelper.Tanh(inputBuf.AsSpan(0, n), nullMask.AsSpan(0, n), resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Softmax<T>(this NivaraColumn<T> column, int classCount) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorsHelper.SoftMax(span, result, classCount);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorsHelper.SoftMax(inputBuf.AsSpan(0, n), nullMask.AsSpan(0, n), resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n), classCount);
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> LogSoftmax<T>(this NivaraColumn<T> column, int classCount) where T : struct, INumber<T>
    {
        int n = column.Length;
        int rows = classCount > 0 ? n / classCount : n;

        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double max = double.NegativeInfinity;
                for (int c = 0; c < classCount; c++)
                {
                    var val = double.CreateChecked(span[rowStart + c]);
                    if (val > max) max = val;
                }
                double sum = 0.0;
                for (int c = 0; c < classCount; c++)
                    sum += Math.Exp(double.CreateChecked(span[rowStart + c]) - max);
                var logSum = Math.Log(sum);
                for (int c = 0; c < classCount; c++)
                    result[rowStart + c] = T.CreateChecked(double.CreateChecked(span[rowStart + c]) - max - logSum);
            }
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));

            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double max = double.NegativeInfinity;
                for (int c = 0; c < classCount; c++)
                {
                    if (!nullMask[rowStart + c])
                    {
                        var val = double.CreateChecked(inputBuf[rowStart + c]);
                        if (val > max) max = val;
                    }
                }
                double sum = 0.0;
                for (int c = 0; c < classCount; c++)
                {
                    if (!nullMask[rowStart + c])
                        sum += Math.Exp(double.CreateChecked(inputBuf[rowStart + c]) - max);
                }
                var logSum = Math.Log(sum);
                for (int c = 0; c < classCount; c++)
                {
                    if (nullMask[rowStart + c])
                        continue;
                    resultBuf[rowStart + c] = T.CreateChecked(double.CreateChecked(inputBuf[rowStart + c]) - max - logSum);
                }
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    #endregion

    #region Binary Math

    public static NivaraColumn<T> Subtract<T>(this NivaraColumn<T> left, NivaraColumn<T> right) where T : struct, INumber<T>
    {
        int n = left.Length;
        if (!left.HasNulls && !right.HasNulls)
        {
            left.TryGetSpan(out var leftSpan);
            right.TryGetSpan(out var rightSpan);
            var result = new T[n];
            TensorPrimitives.Subtract(leftSpan, rightSpan, result);
            return NivaraColumn<T>.Create(result);
        }
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var leftBuf = ArrayPool<T>.Shared.Rent(n);
        var rightBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            left.CopyTo(leftBuf.AsSpan(0, n), T.Zero);
            right.CopyTo(rightBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(left, right, nullMask.AsSpan(0, n));
            TensorPrimitives.Subtract(leftBuf.AsSpan(0, n), rightBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(leftBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(rightBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Divide<T>(this NivaraColumn<T> left, NivaraColumn<T> right) where T : struct, INumber<T>
    {
        int n = left.Length;
        if (!left.HasNulls && !right.HasNulls)
        {
            left.TryGetSpan(out var leftSpan);
            right.TryGetSpan(out var rightSpan);
            var result = new T[n];
            TensorPrimitives.Divide(leftSpan, rightSpan, result);
            return NivaraColumn<T>.Create(result);
        }
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var leftBuf = ArrayPool<T>.Shared.Rent(n);
        var rightBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            left.CopyTo(leftBuf.AsSpan(0, n), T.Zero);
            right.CopyTo(rightBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(left, right, nullMask.AsSpan(0, n));
            TensorPrimitives.Divide(leftBuf.AsSpan(0, n), rightBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(leftBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(rightBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> Divide<T>(this NivaraColumn<T> column, T divisor) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Divide(span, divisor, result);
            return NivaraColumn<T>.Create(result);
        }
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(resultBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Divide(resultBuf.AsSpan(0, n), divisor, resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    #endregion

    #region Matrix Operations

    public static NivaraColumn<T> Transpose<T>(this NivaraColumn<T> column, int rows, int cols) where T : struct, INumber<T>
    {
        int n = rows * cols;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorsHelper.Transpose(span, result, rows, cols);
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMaskBuf = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMaskBuf.AsSpan(0, n));
            TensorsHelper.Transpose(inputBuf.AsSpan(0, n), nullMaskBuf.AsSpan(0, n), resultBuf.AsSpan(0, n), nullMaskBuf.AsSpan(0, n), rows, cols);
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMaskBuf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMaskBuf, clearArray: true);
        }
    }

    public static NivaraColumn<T> MatMul<T>(this NivaraColumn<T> a, NivaraColumn<T> b, int aRows, int aCols, int bCols) where T : struct, INumber<T>
    {
        int n = aRows * bCols;
        if (!a.HasNulls && !b.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            b.TryGetSpan(out var bSpan);
            var result = new T[n];
            TensorsHelper.MultiplyCore(aSpan, bSpan, result, aRows, aCols, bCols);
            return NivaraColumn<T>.Create(result);
        }
        var aBuf = ArrayPool<T>.Shared.Rent(a.Length);
        var bBuf = ArrayPool<T>.Shared.Rent(b.Length);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            a.CopyTo(aBuf.AsSpan(0, a.Length), T.Zero);
            b.CopyTo(bBuf.AsSpan(0, b.Length), T.Zero);
            a.TryGetNullMask(out var aMask);
            b.TryGetNullMask(out var bMask);
            TensorsHelper.Multiply(aBuf.AsSpan(0, a.Length), aMask, bBuf.AsSpan(0, b.Length), bMask, resultBuf, nullMask.AsSpan(0, n), aRows, aCols, bCols);
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    #endregion

    #region Gradient Helpers

    public static NivaraColumn<T> SigmoidGradient<T>(this NivaraColumn<T> sigmoidOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = sigmoidOutput.Length;
        if (!sigmoidOutput.HasNulls && !gradOutput.HasNulls)
        {
            sigmoidOutput.TryGetSpan(out var sigSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var sig = sigSpan[i];
                result[i] = sig * (T.One - sig) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }
        var sigBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            sigmoidOutput.CopyTo(sigBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(sigmoidOutput, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var sig = sigBuf[i];
                resultBuf[i] = sig * (T.One - sig) * gradBuf[i];
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(sigBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> TanhGradient<T>(this NivaraColumn<T> tanhOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = tanhOutput.Length;
        if (!tanhOutput.HasNulls && !gradOutput.HasNulls)
        {
            tanhOutput.TryGetSpan(out var tanhSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var tanh = tanhSpan[i];
                result[i] = (T.One - tanh * tanh) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }
        var tanhBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            tanhOutput.CopyTo(tanhBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(tanhOutput, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var tanh = tanhBuf[i];
                resultBuf[i] = (T.One - tanh * tanh) * gradBuf[i];
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(tanhBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> ReluGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;
        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = inSpan[i] > T.Zero ? gradSpan[i] : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = inputBuf[i] > T.Zero ? gradBuf[i] : T.Zero;
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> AbsGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;
        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = T.CreateChecked(T.Sign(inSpan[i])) * gradSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = T.CreateChecked(T.Sign(inputBuf[i])) * gradBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> ClipGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput, T min, T max) where T : struct, INumber<T>
    {
        int n = input.Length;
        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = (inSpan[i] >= min && inSpan[i] <= max) ? gradSpan[i] : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = (!nullMask[i] && inputBuf[i] >= min && inputBuf[i] <= max) ? gradBuf[i] : T.Zero;
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> LeakyRelu<T>(this NivaraColumn<T> column, T negativeSlope) where T : struct, INumber<T>
    {
        int n = column.Length;
        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = span[i] > T.Zero ? span[i] : negativeSlope * span[i];
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = inputBuf[i] > T.Zero ? inputBuf[i] : negativeSlope * inputBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> LeakyReluGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput, T negativeSlope) where T : struct, INumber<T>
    {
        int n = input.Length;
        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = inSpan[i] > T.Zero ? gradSpan[i] : negativeSlope * gradSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = inputBuf[i] > T.Zero ? gradBuf[i] : negativeSlope * gradBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> LogGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;
        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = gradSpan[i] / inSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = gradBuf[i] / inputBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> SoftmaxGradient<T>(this NivaraColumn<T> softmaxOutput, NivaraColumn<T> gradOutput, int classCount) where T : struct, INumber<T>
    {
        int n = softmaxOutput.Length;
        int rows = classCount > 0 ? n / classCount : n;

        if (!softmaxOutput.HasNulls && !gradOutput.HasNulls)
        {
            softmaxOutput.TryGetSpan(out var softSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double dot = 0.0;
                for (int c = 0; c < classCount; c++)
                    dot += double.CreateChecked(softSpan[rowStart + c]) * double.CreateChecked(gradSpan[rowStart + c]);
                for (int c = 0; c < classCount; c++)
                {
                    var s = double.CreateChecked(softSpan[rowStart + c]);
                    var dy = double.CreateChecked(gradSpan[rowStart + c]);
                    result[rowStart + c] = T.CreateChecked(s * (dy - dot));
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var softBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            softmaxOutput.CopyTo(softBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(softmaxOutput, gradOutput, nullMask.AsSpan(0, n));

            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double dot = 0.0;
                for (int c = 0; c < classCount; c++)
                {
                    if (!nullMask[rowStart + c])
                        dot += double.CreateChecked(softBuf[rowStart + c]) * double.CreateChecked(gradBuf[rowStart + c]);
                }
                for (int c = 0; c < classCount; c++)
                {
                    if (nullMask[rowStart + c])
                        continue;
                    var s = double.CreateChecked(softBuf[rowStart + c]);
                    var dy = double.CreateChecked(gradBuf[rowStart + c]);
                    resultBuf[rowStart + c] = T.CreateChecked(s * (dy - dot));
                }
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(softBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static NivaraColumn<T> LogSoftmaxGradient<T>(this NivaraColumn<T> input, NivaraColumn<T> gradOutput, int classCount) where T : struct, INumber<T>
    {
        int n = input.Length;
        int rows = classCount > 0 ? n / classCount : n;

        var softmax = input.Softmax(classCount);

        if (!softmax.HasNulls && !gradOutput.HasNulls)
        {
            softmax.TryGetSpan(out var softSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double sumGrad = 0.0;
                for (int c = 0; c < classCount; c++)
                    sumGrad += double.CreateChecked(gradSpan[rowStart + c]);
                for (int c = 0; c < classCount; c++)
                {
                    var dy = double.CreateChecked(gradSpan[rowStart + c]);
                    var s = double.CreateChecked(softSpan[rowStart + c]);
                    result[rowStart + c] = T.CreateChecked(dy - s * sumGrad);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var softBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            softmax.CopyTo(softBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(softmax, gradOutput, nullMask.AsSpan(0, n));

            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double sumGrad = 0.0;
                for (int c = 0; c < classCount; c++)
                {
                    if (!nullMask[rowStart + c])
                        sumGrad += double.CreateChecked(gradBuf[rowStart + c]);
                }
                for (int c = 0; c < classCount; c++)
                {
                    if (nullMask[rowStart + c])
                        continue;
                    var dy = double.CreateChecked(gradBuf[rowStart + c]);
                    var s = double.CreateChecked(softBuf[rowStart + c]);
                    resultBuf[rowStart + c] = T.CreateChecked(dy - s * sumGrad);
                }
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(softBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    #endregion

    // ── Column-level terminal reductions ──

    /// <summary>
    /// Computes the sum of all non-null elements in the column.
    /// Returns T.Zero if all elements are null.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to sum</param>
    /// <returns>The sum of non-null elements</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty</exception>
    public static T Sum<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Sum on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Sum(span);

            T sum = T.Zero;
            for (int i = 0; i < column.Length; i++)
                sum += column[i];
            return sum;
        }

        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
                result += column[i];
        }
        return result;
    }

    /// <summary>
    /// Computes the mean (average) of all non-null elements in the column.
    /// Returns NaN if all elements are null.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to average</param>
    /// <returns>The mean of non-null elements as a double</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty</exception>
    public static double Mean<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Mean on an empty column");

        T sum = T.Zero;
        int count = 0;

        if (!column.HasNulls)
        {
            count = column.Length;
            if (column.TryGetSpan(out var span))
                return double.CreateChecked(TensorPrimitives.Sum(span)) / count;

            for (int i = 0; i < column.Length; i++)
                sum += column[i];
        }
        else
        {
            for (int i = 0; i < column.Length; i++)
            {
                if (!column.IsNull(i))
                {
                    sum += column[i];
                    count++;
                }
            }
        }

        return count > 0
            ? double.CreateChecked(sum) / count
            : double.NaN;
    }

    /// <summary>
    /// Returns the minimum non-null value in the column.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to find the minimum of</param>
    /// <returns>The minimum non-null value</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty or all values are null</exception>
    public static T Min<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Min on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Min(span);

            T min = column[0];
            for (int i = 1; i < column.Length; i++)
                min = T.Min(min, column[i]);
            return min;
        }

        bool found = false;
        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                if (!found)
                {
                    result = column[i];
                    found = true;
                }
                else
                {
                    result = T.Min(result, column[i]);
                }
            }
        }

        if (!found)
            throw new InvalidOperationException("Cannot compute Min on a column where all values are null");
        return result;
    }

    /// <summary>
    /// Returns the maximum non-null value in the column.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to find the maximum of</param>
    /// <returns>The maximum non-null value</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty or all values are null</exception>
    public static T Max<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Max on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Max(span);

            T max = column[0];
            for (int i = 1; i < column.Length; i++)
                max = T.Max(max, column[i]);
            return max;
        }

        bool found = false;
        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                if (!found)
                {
                    result = column[i];
                    found = true;
                }
                else
                {
                    result = T.Max(result, column[i]);
                }
            }
        }

        if (!found)
            throw new InvalidOperationException("Cannot compute Max on a column where all values are null");
        return result;
    }
}
