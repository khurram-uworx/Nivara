using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for the row-broadcast AddBias operation, added to replace the
/// Ones+MatMul broadcast used for linear layer biases.
/// </summary>
[TestFixture]
public class AddBiasTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void AddBias_Forward_AddsBiasVectorToEachRow()
    {
        // Arrange
        // a = [[1, 2, 3], [4, 5, 6]], bias = [10, 20, 30]
        var a = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(
            new float[] { 10, 20, 30 }, rows: 1, cols: 3, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.AddBias(a, bias);

        // Assert
        Assert.That(result.Length, Is.EqualTo(6));
        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(11f));
        Assert.That(result[1], Is.EqualTo(22f));
        Assert.That(result[2], Is.EqualTo(33f));
        Assert.That(result[3], Is.EqualTo(14f));
        Assert.That(result[4], Is.EqualTo(25f));
        Assert.That(result[5], Is.EqualTo(36f));
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void AddBias_Backward_InputGradEqualsOutputGrad()
    {
        // Arrange
        var a = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(
            new float[] { 10, 20, 30 }, rows: 1, cols: 3, requiresGrad: true);
        var seed = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: false);

        // Act
        var result = ReverseGradOperations.AddBias(a, bias);
        result.Backward(seed);

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(6));
        for (int i = 0; i < 6; i++)
            Assert.That(a.Grad[i], Is.EqualTo(i + 1f));
    }

    [Test]
    public void AddBias_Backward_BiasGradEqualsColumnSumOfOutputGrad()
    {
        // Arrange
        var a = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(
            new float[] { 10, 20, 30 }, rows: 1, cols: 3, requiresGrad: true);
        var seed = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6 }, rows: 2, cols: 3, requiresGrad: false);

        // Act
        var result = ReverseGradOperations.AddBias(a, bias);
        result.Backward(seed);

        // Assert - column sums: [1+4, 2+5, 3+6] = [5, 7, 9]
        Assert.That(bias.Grad, Is.Not.Null);
        Assert.That(bias.Grad!.Length, Is.EqualTo(3));
        Assert.That(bias.Grad[0], Is.EqualTo(5f));
        Assert.That(bias.Grad[1], Is.EqualTo(7f));
        Assert.That(bias.Grad[2], Is.EqualTo(9f));
    }

    [Test]
    public void AddBias_Parity_Forward_MatchesOnesMatMulBroadcast()
    {
        // Arrange
        var aData = new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 4.0f, -2.5f };
        var biasData = new float[] { 0.1f, -0.2f, 0.3f };

        var a = ReverseGradTensor<float>.FromMatrix(aData, rows: 2, cols: 3, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(biasData, rows: 1, cols: 3, requiresGrad: true);

        var newOut = ReverseGradOperations.AddBias(a, bias);

        // Old path: MatMul(ones[rows,1], bias) then Add
        var aOld = ReverseGradTensor<float>.FromMatrix(aData, rows: 2, cols: 3, requiresGrad: true);
        var biasOld = ReverseGradTensor<float>.FromMatrix(biasData, rows: 1, cols: 3, requiresGrad: true);
        var ones = GradientUtils.Ones<float>(2);
        ones.Reshape(2, 1);
        var biasBroadcast = ReverseGradOperations.MatMul(ones, biasOld);
        var oldOut = ReverseGradOperations.Add(aOld, biasBroadcast);

        // Assert
        Assert.That(newOut.Length, Is.EqualTo(oldOut.Length));
        for (int i = 0; i < newOut.Length; i++)
            Assert.That(newOut[i], Is.EqualTo(oldOut[i]).Within(1e-5f));
    }

    [Test]
    public void AddBias_Parity_Backward_MatchesOnesMatMulBroadcast()
    {
        // Arrange
        var aData = new float[] { 0.5f, -1.0f, 2.0f, 3.0f, 4.0f, -2.5f };
        var biasData = new float[] { 0.1f, -0.2f, 0.3f };
        var seedData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var seed = ReverseGradTensor<float>.FromMatrix(seedData, rows: 2, cols: 3, requiresGrad: false);

        var a = ReverseGradTensor<float>.FromMatrix(aData, rows: 2, cols: 3, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(biasData, rows: 1, cols: 3, requiresGrad: true);
        ReverseGradOperations.AddBias(a, bias).Backward(seed);

        var aOld = ReverseGradTensor<float>.FromMatrix(aData, rows: 2, cols: 3, requiresGrad: true);
        var biasOld = ReverseGradTensor<float>.FromMatrix(biasData, rows: 1, cols: 3, requiresGrad: true);
        var ones = GradientUtils.Ones<float>(2);
        ones.Reshape(2, 1);
        var biasBroadcast = ReverseGradOperations.MatMul(ones, biasOld);
        ReverseGradOperations.Add(aOld, biasBroadcast).Backward(seed);

        // Assert - input grads match
        for (int i = 0; i < a.Length; i++)
            Assert.That(a.Grad![i], Is.EqualTo(aOld.Grad![i]).Within(1e-5f));

        // Assert - bias grads match (column sums)
        for (int i = 0; i < bias.Length; i++)
            Assert.That(bias.Grad![i], Is.EqualTo(biasOld.Grad![i]).Within(1e-5f));
    }

    [Test]
    public void AddBias_NonMatrixInput_Throws()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 1, 2, 3 }), requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(new float[] { 1, 2, 3 }, rows: 1, cols: 3, requiresGrad: true);

        Assert.That(() => ReverseGradOperations.AddBias(a, bias), Throws.ArgumentException);
    }

    [Test]
    public void AddBias_BiasLengthMismatch_Throws()
    {
        var a = ReverseGradTensor<float>.FromMatrix(new float[] { 1, 2, 3, 4 }, rows: 2, cols: 2, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(new float[] { 1, 2, 3 }, rows: 1, cols: 3, requiresGrad: true);

        Assert.That(() => ReverseGradOperations.AddBias(a, bias), Throws.ArgumentException);
    }

    [Test]
    public void Linear_WithBias_Forward_MatchesAddBiasResult()
    {
        // Arrange
        var linear = new Linear<float>(inFeatures: 4, outFeatures: 3);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 1, cols: 4, requiresGrad: false);

        var w = linear.Weight!.Tensor;
        var b = linear.Bias!.Tensor;
        var matMul = ReverseGradOperations.MatMul(input, ReverseGradOperations.Transpose(w));
        var expected = ReverseGradOperations.AddBias(matMul, b);

        // Act
        var actual = linear.Forward(input);

        // Assert
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < actual.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f));
    }

    [Test]
    public void AddBias_OutsideGrad_ProducesNoGraphNode()
    {
        gradScope?.Dispose();
        gradScope = null;

        var a = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 2, cols: 2, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2 }, rows: 1, cols: 2, requiresGrad: true);

        var result = ReverseGradOperations.AddBias(a, bias);

        Assert.That(result.IsLeaf, Is.True,
            "AddBias should not create a graph node outside Grad() scope");
    }
}
