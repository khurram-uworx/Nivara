using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class DepthwiseSeparableConv2d<T> : Module<T> where T : struct, INumber<T>
{
    readonly Conv2d<T> _depthwise;
    readonly Conv2d<T> _pointwise;

    public Conv2d<T> DepthwiseConv => _depthwise;
    public Conv2d<T> PointwiseConv => _pointwise;

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

        _depthwise = new Conv2d<T>(
            inChannels, inChannels, kernelSize,
            stride: stride, padding: padding,
            bias: false, groups: inChannels);

        _pointwise = new Conv2d<T>(
            inChannels, outChannels, 1,
            stride: 1, padding: 0,
            bias: useBias);

        RegisterModules(_depthwise, _pointwise);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var h = _depthwise.Forward(input);
        h = Activation.Relu(h);
        return _pointwise.Forward(h);
    }
}
