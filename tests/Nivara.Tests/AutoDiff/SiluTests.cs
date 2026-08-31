using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class SiluTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static float SiluScalar(float x) => x / (1f + MathF.Exp(-x));

    static float SiluDerivative(float x)
    {
        float s = 1f / (1f + MathF.Exp(-x));
        return s * (1f + x * (1f - s));
    }

    [Test]
    public void Forward_MatchesScalarReference()
    {
        var values = new[] { -3f, -1.5f, 0f, 1f, 2f, 4f };
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(values), requiresGrad: false);

        var output = Activation.Silu(input);

        for (int i = 0; i < values.Length; i++)
            Assert.That(output[i], Is.EqualTo(SiluScalar(values[i])).Within(1e-5f));
    }

    [Test]
    public void Backward_MatchesAnalyticGradient()
    {
        var values = new[] { -2f, -0.5f, 0f, 0.5f, 3f };
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(values), requiresGrad: true);

        var output = Activation.Silu(input);
        var sum = ReverseGradOperations.Sum(output);
        sum.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(values.Length));
        for (int i = 0; i < values.Length; i++)
            Assert.That(input.Grad![i], Is.EqualTo(SiluDerivative(values[i])).Within(1e-5f));
    }

    [Test]
    public void InferencePath_BuildsNoGraphNodes()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new[] { 1f, 2f, 3f }), requiresGrad: false);

        ReverseGradTensor<float> outside;
        using (GradientUtils.Grad())
            outside = Activation.Silu(input);

        Assert.That(outside.RequiresGrad, Is.False);
        Assert.That(outside.IsLeaf, Is.True);
    }
}
