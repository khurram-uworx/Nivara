using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// MobileNet-style depthwise separable convolution: a per-channel depthwise convolution
/// followed by a 1×1 pointwise convolution, with a ReLU between the two stages.
/// </summary>
public sealed class DepthwiseSeparableConv2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv2d<T> depthwise;
    readonly Conv2d<T> pointwise;

    /// <summary>Gets the depthwise convolution sub-module.</summary>
    public Conv2d<T> DepthwiseConv => depthwise;
    /// <summary>Gets the pointwise (1×1) convolution sub-module.</summary>
    public Conv2d<T> PointwiseConv => pointwise;

    /// <summary>
    /// Creates a depthwise separable convolution.
    /// </summary>
    /// <param name="inChannels">Number of input channels (must be positive)</param>
    /// <param name="outChannels">Number of output channels (must be positive)</param>
    /// <param name="kernelSize">Spatial kernel size of the depthwise stage (must be positive)</param>
    /// <param name="stride">Stride of the depthwise convolution</param>
    /// <param name="padding">Zero padding of the depthwise convolution</param>
    /// <param name="useBias">Whether the pointwise stage includes a bias</param>
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

    /// <summary>
    /// Runs depthwise convolution, ReLU, then pointwise convolution.
    /// </summary>
    /// <param name="input">The input tensor (rank 4)</param>
    /// <returns>The output tensor</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var h = depthwise.Forward(input);
        h = Activation.Relu(h);
        return pointwise.Forward(h);
    }
}
