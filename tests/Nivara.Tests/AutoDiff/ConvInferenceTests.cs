using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Inference-default regression tests for conv modules with trainable bias
/// parameters. A bias parameter created with requiresGrad: true must NOT force
/// graph construction when forward is called outside a Grad() scope. See the
/// Conv2d/Conv1d/ConvTranspose2d bias-track gate bug (bias.RequiresGrad was
/// OR-ed into shouldTrack without checking GradientUtils.IsGradEnabled).
/// </summary>
[TestFixture]
public class ConvInferenceTests
{
    [Test]
    public void Conv2d_Forward_OutsideGrad_WithBias_ProducesLeafTensor()
    {
        using var conv = new Conv2d<float>(2, 4, kernelSize: 3, padding: 1, bias: true);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 2, 4, 4);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True,
            "A trainable bias parameter must not create graph nodes outside Grad()");
        Assert.That(output.RequiresGrad, Is.False);
    }

    [Test]
    public void Conv2d_Forward_OutsideGrad_NoBias_ProducesLeafTensor()
    {
        using var conv = new Conv2d<float>(2, 4, kernelSize: 3, padding: 1, bias: false);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 2, 4, 4);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True);
    }

    [Test]
    public void Conv1d_Forward_OutsideGrad_WithBias_ProducesLeafTensor()
    {
        using var conv = new Conv1d<float>(2, 4, kernelSize: 3, padding: 1, bias: true);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 8]),
            requiresGrad: false);
        input.Reshape(1, 2, 8);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True,
            "A trainable bias parameter must not create graph nodes outside Grad()");
        Assert.That(output.RequiresGrad, Is.False);
    }

    [Test]
    public void Conv1d_Forward_OutsideGrad_NoBias_ProducesLeafTensor()
    {
        using var conv = new Conv1d<float>(2, 4, kernelSize: 3, padding: 1, bias: false);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 8]),
            requiresGrad: false);
        input.Reshape(1, 2, 8);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True);
    }

    [Test]
    public void ConvTranspose2d_Forward_OutsideGrad_WithBias_ProducesLeafTensor()
    {
        using var conv = new ConvTranspose2d<float>(2, 4, kernelSize: 3, padding: 1, bias: true);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 2, 4, 4);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True,
            "A trainable bias parameter must not create graph nodes outside Grad()");
        Assert.That(output.RequiresGrad, Is.False);
    }

    [Test]
    public void ConvTranspose2d_Forward_OutsideGrad_NoBias_ProducesLeafTensor()
    {
        using var conv = new ConvTranspose2d<float>(2, 4, kernelSize: 3, padding: 1, bias: false);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 2, 4, 4);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True);
    }

    [Test]
    public void Conv2d_Forward_InsideGrad_WithBias_BuildsGraphNode()
    {
        using var conv = new Conv2d<float>(2, 4, kernelSize: 3, padding: 1, bias: true);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 2, 4, 4);

        ReverseGradTensor<float> output;
        using (GradientUtils.Grad())
        {
            output = conv.Forward(input);
        }

        Assert.That(output.IsLeaf, Is.False,
            "Training forward inside Grad() with trainable parameters must record graph nodes");
    }

    [Test]
    public void Conv2d_Forward_OutsideGrad_WithBias_AppliesBiasValues()
    {
        using var conv = new Conv2d<float>(1, 1, kernelSize: 1, padding: 0, bias: true);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(new float[] { 5f }, requiresGrad: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(new float[] { 1f }, 1, 1, requiresGrad: true);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 3f }),
            requiresGrad: false);
        input.Reshape(1, 1, 1, 1);

        var output = conv.Forward(input);

        Assert.That(output.IsLeaf, Is.True);
        Assert.That(output[0], Is.EqualTo(8f).Within(1e-6f),
            "Inference conv with bias must still apply the bias values");
    }
}
