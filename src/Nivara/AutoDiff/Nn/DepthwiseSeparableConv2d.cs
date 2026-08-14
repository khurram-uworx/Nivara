using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class DepthwiseSeparableConv2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv2d<T> depthwise;
    readonly Conv2d<T> pointwise;

    public Conv2d<T> DepthwiseConv => depthwise;
    public Conv2d<T> PointwiseConv => pointwise;

    public DepthwiseSeparableConv2d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        bool useBias = true)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));

        depthwise = new Conv2d<T>(
            inChannels, inChannels, kernelSize,
            stride: stride, padding: padding,
            bias: false, groups: inChannels);

        pointwise = new Conv2d<T>(
            inChannels, outChannels, 1,
            stride: 1, padding: 0,
            bias: useBias);

        RegisterModules(depthwise, pointwise);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var h = depthwise.Forward(input);
        h = Activation.Relu(h);
        return pointwise.Forward(h);
    }
}
