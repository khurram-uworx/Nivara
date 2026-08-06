using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class AdaptiveAvgPool2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int _outputSize;

    public int OutputSize => _outputSize;

    public AdaptiveAvgPool2d(int outputSize)
    {
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        _outputSize = outputSize;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Shape.Length != 4)
            throw new InvalidOperationException(
                $"AdaptiveAvgPool2d expects 4D input [N, C, H, W], got {input.Shape.Length}D.");

        int n = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        int outH = _outputSize, outW = _outputSize;
        int outputLen = n * c * outH * outW;
        var outputData = new T[outputLen];

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : ModuleHelpers<T>.CopyToTemp(input.Data, n * c * h * w);

        AdaptiveAvgForwardKernel(inputSpan, outputData, n, c, h, w, outH, outW);

        var outShape = new[] { n, c, outH, outW };
        var result = NivaraColumn<T>.CreateFromOwnedArray(outputData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(input);
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var gradFn = new OpNode<T>("AdaptiveAvgPool2d", [input], (gradOutput) =>
            {
                var inputGrad = new T[n * c * h * w];
                var gradOutData = new T[outputLen];
                gradOutput.CopyTo(gradOutData, T.Zero);

                AdaptiveAvgBackwardKernel(gradOutData, inputGrad, n, c, h, w, outH, outW);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.CreateFromOwnedArray(inputGrad));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }



    static void AdaptiveAvgForwardKernel(
        ReadOnlySpan<T> input, Span<T> output,
        int n, int c, int h, int w, int outH, int outW)
    {
        int spatialIn = h * w;
        int spatialOut = outH * outW;
        T scale = T.CreateChecked(1.0) / T.CreateChecked(spatialIn);

        for (int batch = 0; batch < n; batch++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int inBase = (batch * c + ch) * spatialIn;
                int outBase = (batch * c + ch) * spatialOut;

                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        T sum = T.Zero;
                        for (int ih = 0; ih < h; ih++)
                        {
                            for (int iw = 0; iw < w; iw++)
                                sum += input[inBase + ih * w + iw];
                        }
                        output[outBase + oh * outW + ow] = sum * scale;
                    }
                }
            }
        }
    }

    static void AdaptiveAvgBackwardKernel(
        ReadOnlySpan<T> gradOut, Span<T> inputGrad,
        int n, int c, int h, int w, int outH, int outW)
    {
        int spatialIn = h * w;
        int spatialOut = outH * outW;
        T scale = T.CreateChecked(1.0) / T.CreateChecked(spatialIn);

        for (int batch = 0; batch < n; batch++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int inBase = (batch * c + ch) * spatialIn;
                int outBase = (batch * c + ch) * spatialOut;

                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        T gradVal = gradOut[outBase + oh * outW + ow] * scale;
                        for (int ih = 0; ih < h; ih++)
                        {
                            for (int iw = 0; iw < w; iw++)
                                inputGrad[inBase + ih * w + iw] += gradVal;
                        }
                    }
                }
            }
        }
    }
}
