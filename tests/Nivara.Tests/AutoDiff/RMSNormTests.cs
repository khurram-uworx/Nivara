using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class RMSNormTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static float RmsNormSingleValue(float x, ReadOnlySpan<float> row, float eps)
    {
        float sumSq = 0;
        foreach (var v in row)
            sumSq += v * v;
        float rms = MathF.Sqrt(sumSq / row.Length + eps);
        return x / rms;
    }

    [Test]
    public void Forward_MatchesScalarReference_WithUnitGamma()
    {
        var rows = new[] { new[] { 1f, 2f, 3f, 4f }, new[] { 5f, 6f, 7f, 8f } };
        const float eps = 1e-5f;
        using var rmsnorm = new RMSNorm<float>(4, eps);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(2, 4);

        var output = rmsnorm.Forward(input);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
            {
                var expected = RmsNormSingleValue(rows[i][j], rows[i], eps);
                Assert.That(output[i * 4 + j], Is.EqualTo(expected).Within(1e-5f));
            }
    }

    [Test]
    public void Forward_AppliesGammaPerDimension()
    {
        const float eps = 1e-5f;
        var gamma = new float[] { 2f, 0.5f, 1f, 3f };
        using var rmsnorm = new RMSNorm<float>(4, eps);
        rmsnorm.Weight!.Tensor = Module<float>.CloneTensor(
            ReverseGradTensor<float>.FromArray(gamma, requiresGrad: true));

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(2, 4);

        var output = rmsnorm.Forward(input);

        var row0 = new[] { 1f, 2f, 3f, 4f };
        var row1 = new[] { 5f, 6f, 7f, 8f };
        for (int j = 0; j < 4; j++)
        {
            Assert.That(output[j], Is.EqualTo(RmsNormSingleValue(row0[j], row0, eps) * gamma[j]).Within(1e-5f));
            Assert.That(output[4 + j], Is.EqualTo(RmsNormSingleValue(row1[j], row1, eps) * gamma[j]).Within(1e-5f));
        }
    }

    [Test]
    public void InferencePath_BuildsNoGraphNodes()
    {
        const float eps = 1e-5f;
        using var rmsnorm = new RMSNorm<float>(4, eps);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(2, 4);

        ReverseGradTensor<float> outside;
        using (GradientUtils.Grad())
        {
            // Even inside Grad() scope, a requiresGrad:false input must not build a graph.
            outside = rmsnorm.Forward(input);
        }

        Assert.That(outside.RequiresGrad, Is.False);
        Assert.That(outside.IsLeaf, Is.True,
            "RMSNorm with requiresGrad:false input should create no graph node even inside Grad() scope");
    }

    [Test]
    public void Backward_AccumulatesGradientsOnInputAndWeight()
    {
        const float eps = 1e-5f;
        using var rmsnorm = new RMSNorm<float>(4, eps);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: true);
        input.Reshape(2, 4);

        var output = rmsnorm.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(8));
        Assert.That(rmsnorm.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(rmsnorm.Weight!.Tensor.Grad!.Length, Is.EqualTo(4));

        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False);
        foreach (var g in rmsnorm.Weight!.Tensor.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False);
    }
}
