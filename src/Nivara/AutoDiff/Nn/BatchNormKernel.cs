using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

internal static class BatchNormKernel<T> where T : struct, INumber<T>
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
