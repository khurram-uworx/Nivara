using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class InferenceGraphTests
{
    [Test]
    public void Forward_OutsideGrad_ProducesNoGraphNodes()
    {
        var model = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 1, cols: 4, requiresGrad: false);

        var output = model.Forward(input);

        Assert.That(output.IsLeaf, Is.True,
            "Output tensor should have no GradFn when forward is called outside Grad() scope");
    }

    [Test]
    public void Forward_InsideGrad_ProducesGraphNodes()
    {
        var model = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 1, cols: 4, requiresGrad: false);

        ReverseGradTensor<float> output;
        using (GradientUtils.Grad())
        {
            output = model.Forward(input);
        }

        Assert.That(output.IsLeaf, Is.False,
            "Output tensor should have GradFn when forward is called inside Grad() scope");
    }

    [Test]
    public void DivideScalar_OutsideGrad_ProducesNoGraphNode()
    {
        var a = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);

        var result = ReverseGradOperations.DivideScalar(a, 2f);

        Assert.That(result.IsLeaf, Is.True,
            "DivideScalar should create no graph node outside Grad() scope");
        Assert.That(result[0], Is.EqualTo(1f));
    }

    [Test]
    public void DivideScalar_InsideGrad_ProducesGraphNode()
    {
        var a = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);

        ReverseGradTensor<float> result;
        using (GradientUtils.Grad())
        {
            result = ReverseGradOperations.DivideScalar(a, 2f);
        }

        Assert.That(result.IsLeaf, Is.False,
            "DivideScalar should create a graph node inside Grad() scope");
    }
}
