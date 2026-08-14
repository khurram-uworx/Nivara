using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Fused batch-normalization kernels over NCHW-style layouts: forward (train and eval),
/// plus backward passes for the input, gamma, and beta. Uses <see cref="TensorPrimitives"/>
/// SIMD paths for float planes of size at least 4 and scalar fallbacks otherwise.
/// </summary>
internal static class BatchNormKernel<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Result of a forward batch-normalization pass: the normalized output, per-channel
    /// batch mean, inverse standard deviation, and the normalized values (x-hat).
    /// </summary>
    internal readonly struct ForwardResult
    {
        /// <summary>The normalized output.</summary>
        public readonly T[] Output;
        /// <summary>Per-channel batch mean.</summary>
        public readonly T[] Mean;
        /// <summary>Per-channel inverse standard deviation (<c>1 / sqrt(var + eps)</c>).</summary>
        public readonly T[] InvStd;
        /// <summary>Per-channel normalized values before the affine transform.</summary>
        public readonly T[] XHat;

        /// <summary>
        /// Creates a forward result.
        /// </summary>
        /// <param name="output">The normalized output</param>
        /// <param name="mean">Per-channel batch mean</param>
        /// <param name="invStd">Per-channel inverse standard deviation</param>
        /// <param name="xHat">Per-channel normalized values</param>
        public ForwardResult(T[] output, T[] mean, T[] invStd, T[] xHat)
        {
            Output = output;
            Mean = mean;
            InvStd = invStd;
            XHat = xHat;
        }
    }

    /// <summary>
    /// Computes batch normalization using the current batch statistics, optionally updating
    /// the running statistics, and returns the values needed for backward.
    /// </summary>
    /// <param name="input">The raw input</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Number of channels</param>
    /// <param name="planeSize">Spatial size per channel</param>
    /// <param name="gamma">Learnable scale, or empty when not affine</param>
    /// <param name="beta">Learnable shift, or empty when not affine</param>
    /// <param name="eps">Stability term added to the variance</param>
    /// <param name="affine">Whether the gamma/beta transform is applied</param>
    /// <returns>The forward result</returns>
    internal static ForwardResult Forward(
        ReadOnlySpan<T> input, int n, int c, int planeSize,
        ReadOnlySpan<T> gamma, ReadOnlySpan<T> beta,
        T eps, bool affine)
    {
        int channelTotal = n * planeSize;
        var output = new T[input.Length];
        var mean = new T[c];
        var invStd = new T[c];
        var xHat = new T[input.Length];

        for (int ch = 0; ch < c; ch++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var plane = input.Slice(offset, planeSize);
                sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(plane)));
            }
            mean[ch] = sum / T.CreateChecked(channelTotal);
        }

        for (int ch = 0; ch < c; ch++)
        {
            T sumSq = T.Zero;
            T m = mean[ch];
            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var plane = input.Slice(offset, planeSize);
                var diff = ArrayPool<T>.Shared.Rent(planeSize);
                try
                {
                    TensorPrimitives.Add(plane, -m, diff.AsSpan(0, planeSize));
                    for (int j = 0; j < planeSize; j++)
                        sumSq += diff[j] * diff[j];
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(diff, clearArray: true);
                }
            }
            T variance = sumSq / T.CreateChecked(channelTotal);
            invStd[ch] = T.One / T.CreateChecked(Math.Sqrt(double.CreateChecked(variance + eps)));
        }

        for (int ch = 0; ch < c; ch++)
        {
            T m = mean[ch];
            T inv = invStd[ch];
            T g = affine ? gamma[ch] : T.One;
            T b = affine ? beta[ch] : T.Zero;

            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var inPlane = input.Slice(offset, planeSize);
                var outPlane = output.AsSpan(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                {
                    var normalized = ArrayPool<T>.Shared.Rent(planeSize);
                    try
                    {
                        TensorPrimitives.Add(inPlane, -m, normalized.AsSpan(0, planeSize));
                        TensorPrimitives.Multiply(normalized.AsSpan(0, planeSize), inv, normalized.AsSpan(0, planeSize));
                        if (affine)
                        {
                            TensorPrimitives.Multiply(normalized.AsSpan(0, planeSize), g, normalized.AsSpan(0, planeSize));
                            TensorPrimitives.Add(normalized.AsSpan(0, planeSize), b, outPlane);
                        }
                        else
                        {
                            normalized.AsSpan(0, planeSize).CopyTo(outPlane);
                        }
                        if (affine)
                            normalized.AsSpan(0, planeSize).CopyTo(xHat.AsSpan(offset, planeSize));
                        else
                            outPlane.CopyTo(xHat.AsSpan(offset, planeSize));
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(normalized, clearArray: true);
                    }
                }
                else
                {
                    for (int j = 0; j < planeSize; j++)
                    {
                        T normalized = (inPlane[j] - m) * inv;
                        xHat[offset + j] = normalized;
                        outPlane[j] = normalized * g + b;
                    }
                }
            }
        }

        return new ForwardResult(output, mean, invStd, xHat);
    }


    /// <summary>
    /// Computes batch normalization using pre-computed running statistics (evaluation mode).
    /// </summary>
    /// <param name="input">The raw input</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Number of channels</param>
    /// <param name="planeSize">Spatial size per channel</param>
    /// <param name="gamma">Learnable scale, or empty when not affine</param>
    /// <param name="beta">Learnable shift, or empty when not affine</param>
    /// <param name="runningMean">Per-channel running mean</param>
    /// <param name="runningVar">Per-channel running variance</param>
    /// <param name="eps">Stability term added to the variance</param>
    /// <param name="affine">Whether the gamma/beta transform is applied</param>
    /// <returns>The forward result</returns>
    internal static ForwardResult ForwardEval(
        ReadOnlySpan<T> input, int n, int c, int planeSize,
        ReadOnlySpan<T> gamma, ReadOnlySpan<T> beta,
        ReadOnlySpan<T> runningMean, ReadOnlySpan<T> runningVar,
        T eps, bool affine)
    {
        var output = new T[input.Length];
        var mean = runningMean.Length >= c ? runningMean[..c].ToArray() : new T[c];
        var invStd = new T[c];
        var xHat = new T[input.Length];

        for (int ch = 0; ch < c; ch++)
            invStd[ch] = T.One / T.CreateChecked(Math.Sqrt(double.CreateChecked(runningVar[ch] + eps)));

        for (int ch = 0; ch < c; ch++)
        {
            T m = mean[ch];
            T inv = invStd[ch];
            T g = affine ? gamma[ch] : T.One;
            T b = affine ? beta[ch] : T.Zero;

            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var inPlane = input.Slice(offset, planeSize);
                var outPlane = output.AsSpan(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                {
                    var normalized = ArrayPool<T>.Shared.Rent(planeSize);
                    try
                    {
                        TensorPrimitives.Add(inPlane, -m, normalized.AsSpan(0, planeSize));
                        TensorPrimitives.Multiply(normalized.AsSpan(0, planeSize), inv, normalized.AsSpan(0, planeSize));
                        if (affine)
                        {
                            TensorPrimitives.Multiply(normalized.AsSpan(0, planeSize), g, normalized.AsSpan(0, planeSize));
                            TensorPrimitives.Add(normalized.AsSpan(0, planeSize), b, outPlane);
                        }
                        else
                        {
                            normalized.AsSpan(0, planeSize).CopyTo(outPlane);
                        }
                        if (affine)
                            normalized.AsSpan(0, planeSize).CopyTo(xHat.AsSpan(offset, planeSize));
                        else
                            outPlane.CopyTo(xHat.AsSpan(offset, planeSize));
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(normalized, clearArray: true);
                    }
                }
                else
                {
                    for (int j = 0; j < planeSize; j++)
                    {
                        T normalized = (inPlane[j] - m) * inv;
                        xHat[offset + j] = normalized;
                        outPlane[j] = normalized * g + b;
                    }
                }
            }
        }

        return new ForwardResult(output, mean, invStd, xHat);
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the batch-normalization input.
    /// </summary>
    /// <param name="gradOutput">Gradient of the loss w.r.t. the output</param>
    /// <param name="xHat">Normalized values saved during forward</param>
    /// <param name="gamma">Learnable scale, or empty when not affine</param>
    /// <param name="invStd">Inverse standard deviation saved during forward</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Number of channels</param>
    /// <param name="planeSize">Spatial size per channel</param>
    /// <param name="affine">Whether the gamma/beta transform is applied</param>
    /// <returns>The input gradient</returns>
    internal static T[] BackwardInput(
        ReadOnlySpan<T> gradOutput, ReadOnlySpan<T> xHat,
        ReadOnlySpan<T> gamma, ReadOnlySpan<T> invStd,
        int n, int c, int planeSize, bool affine)
    {
        int channelTotal = n * planeSize;
        T scale = T.One / T.CreateChecked(channelTotal);
        var gradInput = new T[gradOutput.Length];

        for (int ch = 0; ch < c; ch++)
        {
            T g = affine ? gamma[ch] : T.One;
            T inv = invStd[ch];
            T gInv = g * inv;

            T sumDY = T.Zero;
            T sumDYXHat = T.Zero;

            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var dyPlane = gradOutput.Slice(offset, planeSize);
                var xhPlane = xHat.Slice(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                {
                    var product = ArrayPool<T>.Shared.Rent(planeSize);
                    try
                    {
                        TensorPrimitives.Multiply(dyPlane, xhPlane, product.AsSpan(0, planeSize));
                        sumDY += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(dyPlane)));
                        sumDYXHat += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(product.AsSpan(0, planeSize))));
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(product, clearArray: true);
                    }
                }
                else
                {
                    for (int j = 0; j < planeSize; j++)
                    {
                        sumDY += dyPlane[j];
                        sumDYXHat += dyPlane[j] * xhPlane[j];
                    }
                }
            }

            T sumDYScaled = sumDY * scale;
            T sumDYXHatScaled = sumDYXHat * scale;

            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var dyPlane = gradOutput.Slice(offset, planeSize);
                var xhPlane = xHat.Slice(offset, planeSize);
                var dxPlane = gradInput.AsSpan(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                {
                    var tmp = ArrayPool<T>.Shared.Rent(planeSize);
                    try
                    {
                        TensorPrimitives.Multiply(xhPlane, sumDYXHatScaled, tmp.AsSpan(0, planeSize));
                        TensorPrimitives.Add(dyPlane, -sumDYScaled, dxPlane);
                        TensorPrimitives.Subtract(dxPlane, tmp.AsSpan(0, planeSize), dxPlane);
                        TensorPrimitives.Multiply(dxPlane, gInv, dxPlane);
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(tmp, clearArray: true);
                    }
                }
                else
                {
                    for (int j = 0; j < planeSize; j++)
                    {
                        dxPlane[j] = gInv * (dyPlane[j] - sumDYScaled - xhPlane[j] * sumDYXHatScaled);
                    }
                }
            }
        }

        return gradInput;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the gamma parameter.
    /// </summary>
    /// <param name="gradOutput">Gradient of the loss w.r.t. the output</param>
    /// <param name="xHat">Normalized values saved during forward</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Number of channels</param>
    /// <param name="planeSize">Spatial size per channel</param>
    /// <returns>The per-channel gamma gradient</returns>
    internal static T[] BackwardWeight(
        ReadOnlySpan<T> gradOutput, ReadOnlySpan<T> xHat,
        int n, int c, int planeSize)
    {
        var gradGamma = new T[c];

        for (int ch = 0; ch < c; ch++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var dyPlane = gradOutput.Slice(offset, planeSize);
                var xhPlane = xHat.Slice(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                {
                    var product = ArrayPool<T>.Shared.Rent(planeSize);
                    try
                    {
                        TensorPrimitives.Multiply(dyPlane, xhPlane, product.AsSpan(0, planeSize));
                        sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(product.AsSpan(0, planeSize))));
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(product, clearArray: true);
                    }
                }
                else
                {
                    for (int j = 0; j < planeSize; j++)
                        sum += dyPlane[j] * xhPlane[j];
                }
            }
            gradGamma[ch] = sum;
        }

        return gradGamma;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the beta parameter.
    /// </summary>
    /// <param name="gradOutput">Gradient of the loss w.r.t. the output</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Number of channels</param>
    /// <param name="planeSize">Spatial size per channel</param>
    /// <returns>The per-channel beta gradient</returns>
    internal static T[] BackwardBias(
        ReadOnlySpan<T> gradOutput,
        int n, int c, int planeSize)
    {
        var gradBeta = new T[c];

        for (int ch = 0; ch < c; ch++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                int offset = i * c * planeSize + ch * planeSize;
                var dyPlane = gradOutput.Slice(offset, planeSize);

                if (planeSize >= 4 && typeof(T) == typeof(float))
                    sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(dyPlane)));
                else
                    for (int j = 0; j < planeSize; j++)
                        sum += dyPlane[j];
            }
            gradBeta[ch] = sum;
        }

        return gradBeta;
    }
}
