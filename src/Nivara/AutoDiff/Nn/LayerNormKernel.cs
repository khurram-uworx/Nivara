using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Span-based LayerNorm kernel with TensorPrimitives.
/// Normalizes over the last dimension (features) per instance.
/// </summary>
internal static class LayerNormKernel<T> where T : struct, IFloatingPointIeee754<T>
{
    internal readonly struct ForwardResult
    {
        public readonly T[] Output;
        public readonly T[] Mean;
        public readonly T[] InvStd;
        public readonly T[] XHat;

        public ForwardResult(T[] output, T[] mean, T[] invStd, T[] xHat)
        {
            Output = output;
            Mean = mean;
            InvStd = invStd;
            XHat = xHat;
        }
    }

    /// <summary>
    /// Forward pass: y = (x - mean) / sqrt(var + eps) * gamma + beta
    /// input: [rows, normalizedShape] flattened row-major
    /// gamma/beta: [normalizedShape] (affine parameters)
    /// </summary>
    internal static ForwardResult Forward(
        ReadOnlySpan<T> input,
        int rows, int normalizedShape,
        ReadOnlySpan<T> gamma, ReadOnlySpan<T> beta,
        T eps, bool affine)
    {
        var output = new T[input.Length];
        var mean = new T[rows];
        var invStd = new T[rows];
        var xHat = new T[input.Length];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var row = input.Slice(offset, normalizedShape);

            T sum = T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(row)));
            mean[r] = sum / T.CreateChecked(normalizedShape);
        }

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var row = input.Slice(offset, normalizedShape);
            T m = mean[r];

            var diff = ArrayPool<T>.Shared.Rent(normalizedShape);
            try
            {
                var diffSpan = diff.AsSpan(0, normalizedShape);
                TensorPrimitives.Add(row, -m, diffSpan);

                T sumSq = T.CreateChecked(double.CreateChecked(TensorPrimitives.Dot(diffSpan, diffSpan)));

                T variance = sumSq / T.CreateChecked(normalizedShape);
                invStd[r] = T.One / T.CreateChecked(Math.Sqrt(double.CreateChecked(variance + eps)));

                T inv = invStd[r];
                var outputSlice = output.AsSpan(offset, normalizedShape);
                var xHatSlice = xHat.AsSpan(offset, normalizedShape);
                if (affine)
                {
                    TensorPrimitives.Multiply(diffSpan, inv, diffSpan);
                    TensorPrimitives.Multiply(diffSpan, gamma, outputSlice);
                    TensorPrimitives.Add(outputSlice, beta, outputSlice);
                    diffSpan.CopyTo(xHatSlice);
                }
                else
                {
                    TensorPrimitives.Multiply(diffSpan, inv, outputSlice);
                    outputSlice.CopyTo(xHatSlice);
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(diff, clearArray: true);
            }
        }

        return new ForwardResult(output, mean, invStd, xHat);
    }

    /// <summary>
    /// Inference-only forward: y = (x - mean) / sqrt(var + eps) * gamma + beta.
    /// No mean/invStd/xHat saved state; the output array doubles as the diff workspace.
    /// </summary>
    internal static T[] ForwardInference(
        ReadOnlySpan<T> input,
        int rows, int normalizedShape,
        ReadOnlySpan<T> gamma, ReadOnlySpan<T> beta,
        T eps, bool affine)
    {
        var output = new T[input.Length];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var row = input.Slice(offset, normalizedShape);

            T mean = T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(row))) / T.CreateChecked(normalizedShape);

            var outputSlice = output.AsSpan(offset, normalizedShape);
            TensorPrimitives.Add(row, -mean, outputSlice);

            T sumSq = T.CreateChecked(double.CreateChecked(TensorPrimitives.Dot(outputSlice, outputSlice)));
            T invStd = T.One / T.CreateChecked(Math.Sqrt(double.CreateChecked((sumSq / T.CreateChecked(normalizedShape)) + eps)));

            TensorPrimitives.Multiply(outputSlice, invStd, outputSlice);
            if (affine)
            {
                TensorPrimitives.Multiply(outputSlice, gamma, outputSlice);
                TensorPrimitives.Add(outputSlice, beta, outputSlice);
            }
        }

        return output;
    }

    /// <summary>
    /// Backward: dx = gamma * invStd * (dy - mean(dy) - xHat * mean(dy * xHat))
    /// </summary>
    internal static T[] BackwardInput(
        ReadOnlySpan<T> gradOutput,
        ReadOnlySpan<T> xHat,
        ReadOnlySpan<T> gamma,
        ReadOnlySpan<T> invStd,
        int rows, int normalizedShape,
        bool affine)
    {
        var gradInput = new T[gradOutput.Length];
        T scale = T.One / T.CreateChecked(normalizedShape);

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var dy = gradOutput.Slice(offset, normalizedShape);
            var xh = xHat.Slice(offset, normalizedShape);
            var dx = gradInput.AsSpan(offset, normalizedShape);
            T inv = invStd[r];

            T sumDY = T.Zero;
            T sumDYXHat = T.Zero;

            if (normalizedShape >= 4 && (typeof(T) == typeof(float) || typeof(T) == typeof(double)))
            {
                sumDY = T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(dy)));
                var product = ArrayPool<T>.Shared.Rent(normalizedShape);
                try
                {
                    TensorPrimitives.Multiply(dy, xh, product.AsSpan(0, normalizedShape));
                    sumDYXHat = T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(product.AsSpan(0, normalizedShape))));
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(product, clearArray: true);
                }
            }
            else
            {
                for (int j = 0; j < normalizedShape; j++)
                {
                    sumDY += dy[j];
                    sumDYXHat += dy[j] * xh[j];
                }
            }

            T sumDYScaled = sumDY * scale;
            T sumDYXHatScaled = sumDYXHat * scale;

            if (affine)
            {
                for (int j = 0; j < normalizedShape; j++)
                {
                    T gInv = gamma[j] * inv;
                    dx[j] = gInv * (dy[j] - sumDYScaled - xh[j] * sumDYXHatScaled);
                }
            }
            else
            {
                for (int j = 0; j < normalizedShape; j++)
                    dx[j] = inv * (dy[j] - sumDYScaled - xh[j] * sumDYXHatScaled);
            }
        }

        return gradInput;
    }

    internal static T[] BackwardWeight(
        ReadOnlySpan<T> gradOutput,
        ReadOnlySpan<T> xHat,
        int rows, int normalizedShape)
    {
        var gradGamma = new T[normalizedShape];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var dy = gradOutput.Slice(offset, normalizedShape);
            var xh = xHat.Slice(offset, normalizedShape);

            if (normalizedShape >= 4 && (typeof(T) == typeof(float) || typeof(T) == typeof(double)))
            {
                var product = ArrayPool<T>.Shared.Rent(normalizedShape);
                try
                {
                    TensorPrimitives.Multiply(dy, xh, product.AsSpan(0, normalizedShape));
                    for (int j = 0; j < normalizedShape; j++)
                        gradGamma[j] += product[j];
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(product, clearArray: true);
                }
            }
            else
            {
                for (int j = 0; j < normalizedShape; j++)
                    gradGamma[j] += dy[j] * xh[j];
            }
        }

        return gradGamma;
    }

    internal static T[] BackwardBias(
        ReadOnlySpan<T> gradOutput,
        int rows, int normalizedShape)
    {
        var gradBeta = new T[normalizedShape];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * normalizedShape;
            var dy = gradOutput.Slice(offset, normalizedShape);

            for (int j = 0; j < normalizedShape; j++)
                gradBeta[j] += dy[j];
        }

        return gradBeta;
    }
}
