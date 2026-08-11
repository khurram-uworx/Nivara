using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for the inference-only transpose-free linear forward path
/// (MatMulTransposedB), which passes the raw weight in the kernel's
/// transposed-B layout so no transposes are performed per forward.
/// </summary>
[TestFixture]
public class LinearInferenceTests
{
    [Test]
    public void Linear_Forward_OutsideGrad_MatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 1.5f, 2.5f, -0.5f, 0.25f }, rows: 2, cols: 4, requiresGrad: false);

        // Reference: old path MatMul(input, Transpose(w)) + bias
        var w = linear.Weight!.Tensor;
        var b = linear.Bias!.Tensor;
        var matMul = ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(w));
        var expected = ReverseGradOperations.AddBias(matMul, b);

        // Act
        var actual = linear.Forward(input);

        // Assert
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        Assert.That(actual.IsLeaf, Is.True, "Inference linear forward must not create graph nodes");
        for (int i = 0; i < actual.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f));
    }

    [Test]
    public void Linear_Forward_NoBias_OutsideGrad_MatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3, bias: false);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 1.5f, 2.5f, -0.5f, 0.25f }, rows: 2, cols: 4, requiresGrad: false);

        var w = linear.Weight!.Tensor;
        var expected = ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(w));

        // Act
        var actual = linear.Forward(input);

        // Assert
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < actual.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f));
    }

    [Test]
    public void Linear_Forward_InsideGrad_StillMatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 1.5f, 2.5f, -0.5f, 0.25f }, rows: 2, cols: 4, requiresGrad: false);

        ReverseGradTensor<float> expected;
        using (GradientUtils.Grad())
        {
            var w = linear.Weight!.Tensor;
            var matMul = ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(w));
            expected = ReverseGradOperations.AddBias(matMul, linear.Bias!.Tensor);
        }

        ReverseGradTensor<float> actual;
        using (GradientUtils.Grad())
        {
            actual = linear.Forward(input);
        }

        // Assert - values identical on the training path too
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < actual.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f));
        Assert.That(actual.IsLeaf, Is.False, "Training forward must still record graph nodes");
    }

    [Test]
    public void Linear_Forward_InsideGrad_ComputesCorrectBackward()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 1, cols: 4, requiresGrad: false);

        ReverseGradTensor<float> output;
        using (GradientUtils.Grad())
        {
            output = linear.Forward(input);
            var seed = ReverseGradTensor<float>.FromMatrix(
                new float[] { 1, 1, 1 }, rows: 1, cols: 3, requiresGrad: false);
            output.Backward(seed);
        }

        // Assert - gradient reaches the weight and bias parameters
        Assert.That(linear.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(linear.Bias!.Tensor.Grad, Is.Not.Null);
        Assert.That(linear.Bias!.Tensor.Grad!.Length, Is.EqualTo(3));
    }

    [Test]
    public void Linear_Forward_AfterWeightReplacement_UsesUpdatedWeights()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 1, 1, 1 }, rows: 1, cols: 4, requiresGrad: false);

        var before = linear.Forward(input);

        var newWeight = ReverseGradTensor<float>.FromMatrix(
            new float[12], rows: 3, cols: 4, requiresGrad: true);
        linear.Weight!.Tensor = newWeight;

        var after = linear.Forward(input);

        // Assert - zero weights => output collapses to the bias broadcast
        var bias = linear.Bias!.Tensor;
        Assert.That(before[0], Is.Not.EqualTo(0f));
        for (int j = 0; j < 3; j++)
            Assert.That(after[j], Is.EqualTo(bias[j]).Within(1e-6f));
    }
}
