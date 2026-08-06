using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Pure span-in/span-out numerical kernels for the AutoDiff domain.
/// ADR-001: the AutoDiff domain is non-nullable, so these kernels perform no
/// null checks, no mask propagation, and reference no columnar types.
/// Kernels are vectorized through generic <see cref="TensorPrimitives"/>
/// overloads; scalar loops are used only where no platform primitive maps
/// (erf-based GELU, row-reduction gradients).
/// </summary>
public static class GradKernels
{
    // ═══════════════════════════════════════════════════════════════
    //  Activations
    // ═══════════════════════════════════════════════════════════════

    public static void Sigmoid<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Sigmoid(input, output);
    }

    public static void SigmoidGradient<T>(ReadOnlySpan<T> sigmoidOutput, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (sigmoidOutput.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        TensorPrimitives.Subtract(T.One, sigmoidOutput, output);
        TensorPrimitives.Multiply(output, sigmoidOutput, output);
        TensorPrimitives.Multiply(output, gradOutput, output);
    }

    public static void Tanh<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Tanh(input, output);
    }

    public static void TanhGradient<T>(ReadOnlySpan<T> tanhOutput, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (tanhOutput.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        TensorPrimitives.Multiply(tanhOutput, tanhOutput, output);
        TensorPrimitives.Subtract(T.One, output, output);
        TensorPrimitives.Multiply(output, gradOutput, output);
    }

    public static void Relu<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Max(input, T.Zero, output);
    }

    public static void ReluGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] > T.Zero ? gradOutput[i] : T.Zero;
    }

    public static void LeakyRelu<T>(ReadOnlySpan<T> input, T negativeSlope, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] > T.Zero ? input[i] : negativeSlope * input[i];
    }

    public static void LeakyReluGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, T negativeSlope, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] > T.Zero ? gradOutput[i] : negativeSlope * gradOutput[i];
    }

    public static void Gelu<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        T sqrt2OverPi = T.CreateChecked(0.7978845608028654);
        T coeff = T.CreateChecked(0.044715);
        T half = T.CreateChecked(0.5);
        int n = input.Length;
        var x3Arr = ArrayPool<T>.Shared.Rent(n);
        var innerArr = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var x3 = x3Arr.AsSpan(0, n);
            var inner = innerArr.AsSpan(0, n);
            TensorPrimitives.Multiply(input, input, x3);
            TensorPrimitives.Multiply(x3, input, x3);
            TensorPrimitives.MultiplyAdd(x3, coeff, input, inner);
            TensorPrimitives.Multiply(inner, sqrt2OverPi, inner);
            TensorPrimitives.Tanh(inner, output);
            TensorPrimitives.Add(output, T.One, output);
            TensorPrimitives.Multiply(input, half, inner);
            TensorPrimitives.Multiply(inner, output, output);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(x3Arr);
            ArrayPool<T>.Shared.Return(innerArr);
        }
    }

    public static void GeluGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        T sqrt2OverPi = T.CreateChecked(0.7978845608028654);
        T coeff = T.CreateChecked(0.044715);
        T half = T.CreateChecked(0.5);
        T one = T.One;
        T threeCoeff = T.CreateChecked(3) * coeff;
        int n = input.Length;
        var buf1Arr = ArrayPool<T>.Shared.Rent(n);
        var buf2Arr = ArrayPool<T>.Shared.Rent(n);
        var buf3Arr = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var buf1 = buf1Arr.AsSpan(0, n);
            var buf2 = buf2Arr.AsSpan(0, n);
            var buf3 = buf3Arr.AsSpan(0, n);

            TensorPrimitives.Multiply(input, input, buf1);
            TensorPrimitives.Multiply(buf1, input, buf2);

            TensorPrimitives.MultiplyAdd(buf2, coeff, input, buf3);
            TensorPrimitives.Multiply(buf3, sqrt2OverPi, buf3);

            TensorPrimitives.Tanh(buf3, output);

            TensorPrimitives.Multiply(output, output, buf3);
            TensorPrimitives.Negate(buf3, buf3);
            TensorPrimitives.Add(buf3, one, buf3);

            TensorPrimitives.Multiply(buf1, threeCoeff, buf1);
            TensorPrimitives.Add(buf1, one, buf1);
            TensorPrimitives.Multiply(buf1, sqrt2OverPi, buf1);

            TensorPrimitives.Multiply(input, buf3, buf2);
            TensorPrimitives.Multiply(buf2, buf1, buf2);
            TensorPrimitives.Multiply(buf2, half, buf2);

            TensorPrimitives.Add(output, one, buf3);
            TensorPrimitives.Multiply(buf3, half, buf3);

            TensorPrimitives.Add(buf3, buf2, buf1);
            TensorPrimitives.Multiply(buf1, gradOutput, output);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf1Arr);
            ArrayPool<T>.Shared.Return(buf2Arr);
            ArrayPool<T>.Shared.Return(buf3Arr);
        }
    }

    public static void GeluExact<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        T invSqrt2 = T.CreateChecked(0.7071067811865475);
        for (int i = 0; i < input.Length; i++)
        {
            T v = input[i];
            output[i] = T.CreateChecked(0.5) * v * (T.One + Erf(v * invSqrt2));
        }
    }

    public static void GeluExactGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        T invSqrt2 = T.CreateChecked(0.7071067811865475);
        T invSqrt2Pi = T.CreateChecked(0.3989422804014327);
        for (int i = 0; i < input.Length; i++)
        {
            T v = input[i];
            T cdf = T.CreateChecked(0.5) * (T.One + Erf(v * invSqrt2));
            T pdf = T.Exp(-T.CreateChecked(0.5) * v * v) * invSqrt2Pi;
            output[i] = (cdf + v * pdf) * gradOutput[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Element-wise math
    // ═══════════════════════════════════════════════════════════════

    public static void Exp<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Exp(input, output);
    }

    public static void Log<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Log(input, output);
    }

    public static void LogGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        TensorPrimitives.Divide(gradOutput, input, output);
    }

    public static void Abs<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Abs(input, output);
    }

    public static void AbsGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        for (int i = 0; i < input.Length; i++)
            output[i] = T.CreateChecked(T.Sign(input[i])) * gradOutput[i];
    }

    public static void Clamp<T>(ReadOnlySpan<T> input, T min, T max, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Clamp(input, min, max, output);
    }

    public static void ClipGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, T min, T max, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] >= min && input[i] <= max ? gradOutput[i] : T.Zero;
    }

    public static void Negate<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        TensorPrimitives.Negate(input, output);
    }

    public static void Divide<T>(ReadOnlySpan<T> numerator, ReadOnlySpan<T> denominator, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (numerator.Length != denominator.Length || output.Length < denominator.Length)
            throw new ArgumentException("All spans must have the same length.");
        TensorPrimitives.Divide(numerator, denominator, output);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Softmax / LogSoftmax
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Row-wise softmax over a flat row-major span with <paramref name="classCount"/>
    /// elements per row. When <paramref name="classCount"/> is not a valid row
    /// width (≤ 0 or ≥ input length), the whole input is treated as one row.
    /// </summary>
    public static void Softmax<T>(ReadOnlySpan<T> input, Span<T> output, int classCount)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        if (classCount <= 0 || classCount >= input.Length)
        {
            SoftmaxSingle(input, output);
            return;
        }
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
            SoftmaxSingle(input.Slice(r * classCount, classCount), output.Slice(r * classCount, classCount));
    }

    public static void SoftmaxGradient<T>(ReadOnlySpan<T> softmaxOutput, ReadOnlySpan<T> gradOutput, Span<T> output, int classCount)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (softmaxOutput.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        if (classCount <= 0 || classCount >= softmaxOutput.Length)
        {
            SoftmaxGradientSingle(softmaxOutput, gradOutput, output);
            return;
        }
        int rows = softmaxOutput.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            int start = r * classCount;
            SoftmaxGradientSingle(softmaxOutput.Slice(start, classCount), gradOutput.Slice(start, classCount), output.Slice(start, classCount));
        }
    }

    /// <summary>
    /// Row-wise log-softmax: log(exp(x - max) / sum(exp(x - max))) computed with
    /// double intermediates for numerical stability, matching the historic
    /// column-extension numerics.
    /// </summary>
    public static void LogSoftmax<T>(ReadOnlySpan<T> input, Span<T> output, int classCount)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span length ({output.Length}) must be at least the input length ({input.Length}).", nameof(output));
        if (classCount <= 0 || classCount >= input.Length)
        {
            LogSoftmaxSingle(input, output);
            return;
        }
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
            LogSoftmaxSingle(input.Slice(r * classCount, classCount), output.Slice(r * classCount, classCount));
    }

    /// <summary>
    /// Gradient of log-softmax given the original (pre-softmax) input:
    /// dy - softmax(x) * sum(dy), computed per row.
    /// </summary>
    public static void LogSoftmaxGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output, int classCount)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input.Length != gradOutput.Length || output.Length < gradOutput.Length)
            throw new ArgumentException("All spans must have the same length.");
        if (classCount <= 0 || classCount >= input.Length)
        {
            LogSoftmaxGradientSingle(input, gradOutput, output);
            return;
        }
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            int start = r * classCount;
            LogSoftmaxGradientSingle(input.Slice(start, classCount), gradOutput.Slice(start, classCount), output.Slice(start, classCount));
        }
    }

    static void SoftmaxSingle<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        T max = input[0];
        for (int i = 1; i < input.Length; i++)
            if (input[i] > max) max = input[i];
        TensorPrimitives.Subtract(input, max, output);
        TensorPrimitives.Exp(output, output);
        TensorPrimitives.Divide(output, TensorPrimitives.Sum(output), output);
    }

    static void SoftmaxGradientSingle<T>(ReadOnlySpan<T> softmaxOutput, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        double dot = 0.0;
        for (int i = 0; i < softmaxOutput.Length; i++)
            dot += double.CreateChecked(softmaxOutput[i]) * double.CreateChecked(gradOutput[i]);
        for (int i = 0; i < softmaxOutput.Length; i++)
        {
            double s = double.CreateChecked(softmaxOutput[i]);
            double dy = double.CreateChecked(gradOutput[i]);
            output[i] = T.CreateChecked(s * (dy - dot));
        }
    }

    static void LogSoftmaxSingle<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            double val = double.CreateChecked(input[i]);
            if (val > max) max = val;
        }
        double sum = 0.0;
        for (int i = 0; i < input.Length; i++)
            sum += Math.Exp(double.CreateChecked(input[i]) - max);
        double logSum = Math.Log(sum);
        for (int i = 0; i < input.Length; i++)
            output[i] = T.CreateChecked(double.CreateChecked(input[i]) - max - logSum);
    }

    static void LogSoftmaxGradientSingle<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output)
        where T : struct, IFloatingPointIeee754<T>
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            double val = double.CreateChecked(input[i]);
            if (val > max) max = val;
        }
        double sumExp = 0.0;
        double sumGrad = 0.0;
        for (int i = 0; i < input.Length; i++)
        {
            sumExp += Math.Exp(double.CreateChecked(input[i]) - max);
            sumGrad += double.CreateChecked(gradOutput[i]);
        }
        for (int i = 0; i < input.Length; i++)
        {
            double soft = Math.Exp(double.CreateChecked(input[i]) - max) / sumExp;
            output[i] = T.CreateChecked(double.CreateChecked(gradOutput[i]) - soft * sumGrad);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Matrix operations
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Dense matmul of flat row-major matrices. The output buffer is an array
    /// because the underlying parallel tiled kernel requires a captured array
    /// (spans cannot cross <see cref="Parallel"/> lambda boundaries).
    /// </summary>
    public static void MatMul<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] output, int aRows, int aCols, int bCols)
        where T : struct, IFloatingPointIeee754<T>
        => TensorsHelper.MultiplyCore(a, b, output, aRows, aCols, bCols);

    public static void MatMulTransposedB<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, T[] output, int aRows, int aCols, int bCols)
        where T : struct, IFloatingPointIeee754<T>
        => TensorsHelper.MultiplyCore(a, b, output, aRows, aCols, bCols, bTransposed: true);

    public static void Transpose<T>(ReadOnlySpan<T> src, Span<T> dst, int rows, int cols)
        where T : struct, IFloatingPointIeee754<T>
        => TensorsHelper.Transpose(src, dst, rows, cols);

    // ═══════════════════════════════════════════════════════════════
    //  Erf (Abramowitz–Stegun 7.1.26 approximation)
    // ═══════════════════════════════════════════════════════════════

    static T Erf<T>(T x)
        where T : struct, IFloatingPointIeee754<T>
    {
        T az = T.Abs(x);
        T t = T.One / (T.One + T.CreateChecked(0.3275911) * az);
        T p = T.FusedMultiplyAdd(T.CreateChecked(1.061405429), t, T.CreateChecked(-1.453152027));
        p = T.FusedMultiplyAdd(p, t, T.CreateChecked(1.421413741));
        p = T.FusedMultiplyAdd(p, t, T.CreateChecked(-0.284496736));
        p = T.FusedMultiplyAdd(p, t, T.CreateChecked(0.254829592));
        T erfAbs = T.One - p * t * T.Exp(-az * az);
        return x < T.Zero ? -erfAbs : erfAbs;
    }
}
