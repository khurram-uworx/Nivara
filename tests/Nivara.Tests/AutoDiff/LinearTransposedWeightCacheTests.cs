using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for the version-stamped transposed-weight cache on the training path
/// of <see cref="Linear{T}"/> (issue #87). The cache eliminates the per-forward
/// transpose allocation/copy across a Grad() training loop while preserving
/// identical forward values and gradient flow into the weight parameter.
/// </summary>
[TestFixture]
public class LinearTransposedWeightCacheTests
{
    static ReverseGradTensor<float> InputMatrix() => ReverseGradTensor<float>.FromMatrix(
        new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 1.5f, 2.5f, -0.5f, 0.25f }, rows: 2, cols: 4, requiresGrad: false);

    static float[] Extract(ReverseGradTensor<float> t)
    {
        var arr = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
            arr[i] = t[i];
        return arr;
    }

    static float[] ExtractColumn(NivaraColumn<float> c)
    {
        var arr = new float[c.Length];
        for (int i = 0; i < c.Length; i++)
            arr[i] = c[i];
        return arr;
    }

    static float Sum(ReverseGradTensor<float> t)
    {
        var sum = 0f;
        for (int i = 0; i < t.Length; i++)
            sum += t[i];
        return sum;
    }

    static float[] RunForwardBackward(Linear<float> linear, ReverseGradTensor<float> input, ReverseGradTensor<float> seed)
    {
        using (GradientUtils.Grad())
        {
            var output = linear.Forward(input);
            output.Backward(seed);
        }

        return ExtractColumn(linear.WeightParam.Tensor.Grad!);
    }

    static float[] RunExplicitForwardBackward(Linear<float> linear, ReverseGradTensor<float> input, ReverseGradTensor<float> seed)
    {
        using (GradientUtils.Grad())
        {
            var output = ReverseGradOperations.AddBias(
                ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(linear.Weight)),
                linear.Bias!);
            output.Backward(seed);
        }

        return ExtractColumn(linear.WeightParam.Tensor.Grad!);
    }

    [Test]
    public void TransposedWeightCache_TrainingForward_MatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();

        using (GradientUtils.Grad())
        {
            // Act
            var actual = linear.Forward(input);
            var expected = ReverseGradOperations.AddBias(
                ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(linear.Weight)),
                linear.Bias!);

            // Assert
            Assert.That(actual.Length, Is.EqualTo(expected.Length));
            for (int i = 0; i < actual.Length; i++)
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f));
        }
    }

    [Test]
    public void TransposedWeightCache_MultipleForwards_ReturnStableResults()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();

        using (GradientUtils.Grad())
        {
            // Act
            var first = Extract(linear.Forward(input));
            var second = Extract(linear.Forward(input));

            // Assert - identical outputs from the reused cache
            Assert.That(second, Is.EqualTo(first));
        }
    }

    [Test]
    public void TransposedWeightCache_WeightReplacement_InvalidatesCache()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();

        using (GradientUtils.Grad())
        {
            var before = Extract(linear.Forward(input));

            // Act - replace the weight tensor (as SGD/Adam/AdamW Step does)
            linear.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(
                new float[12], rows: 3, cols: 4, requiresGrad: true);
            var after = Extract(linear.Forward(input));

            // Assert - zero weights => output collapses to the bias broadcast
            var bias = linear.Bias!;
            Assert.That(before, Is.Not.All.EqualTo(0f));
            for (int j = 0; j < after.Length; j++)
                Assert.That(after[j], Is.EqualTo(bias[j % 3]).Within(1e-6f));
        }
    }

    [Test]
    public void TransposedWeightCache_Backward_MatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();
        var seed = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: false);

        // Act - cached path, then explicit path on the same unchanged weights
        var wGradCached = RunForwardBackward(linear, input, seed);
        linear.WeightParam.Tensor.ZeroGrad();
        linear.Bias!.ZeroGrad();
        var wGradExplicit = RunExplicitForwardBackward(linear, input, seed);

        // Assert - identical gradients into the weight parameter
        Assert.That(wGradCached.Length, Is.EqualTo(wGradExplicit.Length));
        for (int i = 0; i < wGradCached.Length; i++)
            Assert.That(wGradCached[i], Is.EqualTo(wGradExplicit[i]).Within(1e-5f));
    }

    [Test]
    public void Parameter_Version_IncrementsOnReplacementAndTouch()
    {
        // Arrange
        var p = new Parameter<float>("w", new float[12]);

        // Act
        var v0 = p.Version;
        p.Touch();
        p.Tensor = ReverseGradTensor<float>.FromMatrix(new float[12], rows: 3, cols: 4);

        // Assert
        Assert.That(v0, Is.EqualTo(0));
        Assert.That(p.Version, Is.EqualTo(v0 + 2));
    }

    [Test]
    public void TransposedWeightCache_OptimizerStep_InvalidatesCacheAndTrains()
    {
        // Arrange - one full training step through SGD
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var optimizer = new Nivara.AutoDiff.Optimizer.SGD<float>(0.1f);
        optimizer.AddParameterGroup(linear.GetParameters().Values);
        var input = InputMatrix();

        float before;
        float after;
        using (GradientUtils.Grad())
        {
            // Act
            var output = linear.Forward(input);
            before = Sum(output);
            var loss = ReverseGradOperations.Sum(output);
            loss.Backward();
            optimizer.Step();
            var output2 = linear.Forward(input);
            after = Sum(output2);
        }

        // Assert - optimizer step mutated the weight and the cache picked it up
        Assert.That(after, Is.Not.EqualTo(before).Within(1e-5f));
    }
}
