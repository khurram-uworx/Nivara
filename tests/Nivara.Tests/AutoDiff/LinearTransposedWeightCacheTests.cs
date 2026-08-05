using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for the transpose-free Linear training path: <see cref="Linear{T}"/>
/// feeds the raw weight in the kernel's transposed-B layout to
/// <see cref="ReverseGradOperations.MatMulTransposedB{T}"/>, which records a
/// single VJP (dA = g @ b, dB = g^T @ a) so no weight transpose happens on
/// forward or backward. Replaces the former version-stamped transposed-weight
/// cache (issue #87) eliminated in the P2 perf work (issue #66).
/// </summary>
[TestFixture]
public class LinearTransposedBTrainingTests
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
    public void TransposedB_TrainingForward_MatchesExplicitTransposeMatMul()
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
    public void TransposedB_MultipleForwards_ReturnStableResults()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();

        using (GradientUtils.Grad())
        {
            // Act
            var first = Extract(linear.Forward(input));
            var second = Extract(linear.Forward(input));

            // Assert - identical outputs, no state carried between forwards
            Assert.That(second, Is.EqualTo(first));
        }
    }

    [Test]
    public void TransposedB_WeightReplacement_ForwardUsesUpdatedWeights()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();

        using (GradientUtils.Grad())
        {
            var before = Extract(linear.Forward(input));

            // Act - replace the weight tensor (as the allocate-and-replace SGD/Adam path did)
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
    public void TransposedB_Backward_WeightGrad_MatchesExplicitTransposeMatMul()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = InputMatrix();
        var seed = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: false);

        // Act - transposed-B path, then explicit path on the same unchanged weights
        var wGradNew = RunForwardBackward(linear, input, seed);
        linear.WeightParam.Tensor.ZeroGrad();
        linear.Bias!.ZeroGrad();
        var wGradExplicit = RunExplicitForwardBackward(linear, input, seed);

        // Assert - identical gradients into the weight parameter (dB = g^T @ a)
        Assert.That(wGradNew.Length, Is.EqualTo(wGradExplicit.Length));
        for (int i = 0; i < wGradNew.Length; i++)
            Assert.That(wGradNew[i], Is.EqualTo(wGradExplicit[i]).Within(1e-5f));
    }

    [Test]
    public void TransposedB_Backward_InputGrad_MatchesExplicitTransposeMatMul()
    {
        // Arrange - input tracks gradients so the VJP's dA = g @ b is exercised
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 1.5f, 2.5f, -0.5f, 0.25f }, rows: 2, cols: 4, requiresGrad: true);
        var seed = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: false);

        float[] dInputNew;
        using (GradientUtils.Grad())
        {
            var output = linear.Forward(input);
            output.Backward(seed);
            dInputNew = ExtractColumn(input.Grad!);
        }
        input.ZeroGrad();

        float[] dInputExplicit;
        using (GradientUtils.Grad())
        {
            var output = ReverseGradOperations.AddBias(
                ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(linear.Weight)),
                linear.Bias!);
            output.Backward(seed);
            dInputExplicit = ExtractColumn(input.Grad!);
        }

        // Assert - identical gradients into the input (dA = g @ b, no weight transpose)
        Assert.That(dInputNew, Is.EqualTo(dInputExplicit).Within(1e-5f));
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
    public void TransposedB_OptimizerStep_ForwardTracksUpdatedWeights()
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

        // Assert - optimizer step mutated the weight and the next forward reflects it
        Assert.That(after, Is.Not.EqualTo(before).Within(1e-5f));
    }
}
