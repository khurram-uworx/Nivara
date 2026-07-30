using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class MaxPool2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int _kernelSize;
    readonly int _stride;
    readonly int _padding;

    public int KernelSize => _kernelSize;
    public int Stride => _stride;
    public int Padding => _padding;

    public MaxPool2d(int kernelSize, int stride = 0, int padding = 0)
    {
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _kernelSize = kernelSize;
        _stride = stride <= 0 ? kernelSize : stride;
        _padding = padding;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Shape.Length != 4)
            throw new InvalidOperationException(
                $"MaxPool2d expects 4D input [N, C, H, W], got {input.Shape.Length}D.");

        int n = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        int oH = (h + 2 * _padding - _kernelSize) / _stride + 1;
        int oW = (w + 2 * _padding - _kernelSize) / _stride + 1;

        if (oH <= 0 || oW <= 0)
            throw new InvalidOperationException(
                $"MaxPool2d output dimensions are non-positive: oH={oH}, oW={oW}. " +
                $"Input: [{n},{c},{h},{w}], kernel={_kernelSize}, stride={_stride}, padding={_padding}.");

        var outputData = new T[n * c * oH * oW];
        var argmaxData = new int[n * c * oH * oW];

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : ModuleHelpers<T>.CopyToTemp(input.Data, n * c * h * w);

        MaxPoolForwardKernel(inputSpan, outputData, argmaxData,
            n, c, h, w, oH, oW, _kernelSize, _stride, _padding);

        var outShape = new[] { n, c, oH, oW };
        var result = NivaraColumn<T>.Create(outputData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(input);
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var capturedArgmax = argmaxData;

            var gradFn = new OpNode<T>("MaxPool2d", [input], (gradOutput) =>
            {
                var inputGrad = new T[n * c * h * w];
                var gradOutData = new T[n * c * oH * oW];
                gradOutput.CopyTo(gradOutData, T.Zero);

                MaxPoolBackwardKernel(gradOutData, inputGrad, capturedArgmax,
                    n, c, h, w, oH, oW, _kernelSize, _stride, _padding);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }



    static void MaxPoolForwardKernel(
        ReadOnlySpan<T> input, Span<T> output, Span<int> argmax,
        int n, int c, int h, int w, int oH, int oW,
        int kernelSize, int stride, int padding)
    {
        int spatialIn = h * w;
        int spatialOut = oH * oW;
        T minVal = T.CreateChecked(double.MinValue);

        for (int batch = 0; batch < n; batch++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int inBase = (batch * c + ch) * spatialIn;
                int outBase = (batch * c + ch) * spatialOut;

                for (int oh = 0; oh < oH; oh++)
                {
                    for (int ow = 0; ow < oW; ow++)
                    {
                        int ihStart = oh * stride - padding;
                        int iwStart = ow * stride - padding;
                        int ihEnd = Math.Min(ihStart + kernelSize, h);
                        int iwEnd = Math.Min(iwStart + kernelSize, w);
                        int ihClamped = Math.Max(ihStart, 0);
                        int iwClamped = Math.Max(iwStart, 0);

                        T maxVal = minVal;
                        int maxIdx = -1;

                        for (int ih = ihClamped; ih < ihEnd; ih++)
                        {
                            for (int iw = iwClamped; iw < iwEnd; iw++)
                            {
                                T val = input[inBase + ih * w + iw];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = inBase + ih * w + iw;
                                }
                            }
                        }

                        int outIdx = outBase + oh * oW + ow;
                        output[outIdx] = maxVal;
                        argmax[outIdx] = maxIdx;
                    }
                }
            }
        }
    }

    static void MaxPoolBackwardKernel(
        ReadOnlySpan<T> gradOut, Span<T> inputGrad, ReadOnlySpan<int> argmax,
        int n, int c, int h, int w, int oH, int oW,
        int kernelSize, int stride, int padding)
    {
        int spatialOut = oH * oW;

        for (int i = 0; i < n * c * spatialOut; i++)
        {
            if (argmax[i] >= 0)
                inputGrad[argmax[i]] += gradOut[i];
        }
    }
}
