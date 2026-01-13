using NUnit.Framework;
using Nivara;
using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for gradient-aware operations in automatic differentiation.
/// Verifies that element-wise and matrix operations work correctly with gradient computation.
/// </summary>
[TestFixture]
public class GradOperationsTests
{
    [Test]
    public void Add_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.Add(a, b);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(5.0f));
        Assert.That(result[1], Is.EqualTo(7.0f));
        Assert.That(result[2], Is.EqualTo(9.0f));
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Multiply_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f, 3.0f, 4.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 5.0f, 6.0f, 7.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.Multiply(a, b);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(10.0f));
        Assert.That(result[1], Is.EqualTo(18.0f));
        Assert.That(result[2], Is.EqualTo(28.0f));
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Subtract_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 10.0f, 8.0f, 6.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 2.0f, 1.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.Subtract(a, b);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(7.0f));
        Assert.That(result[1], Is.EqualTo(6.0f));
        Assert.That(result[2], Is.EqualTo(5.0f));
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Divide_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 12.0f, 15.0f, 18.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 5.0f, 6.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.Divide(a, b);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(4.0f));
        Assert.That(result[1], Is.EqualTo(3.0f));
        Assert.That(result[2], Is.EqualTo(3.0f));
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void MatMul_SimpleCase_ComputesCorrectResult()
    {
        // Arrange: 2x2 matrix multiplication
        // A = [[1, 2], [3, 4]] (flattened: [1, 2, 3, 4])
        // B = [[5, 6], [7, 8]] (flattened: [5, 6, 7, 8])
        // Expected result: [[19, 22], [43, 50]] (flattened: [19, 22, 43, 50])
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.MatMul(a, b, aRows: 2, aCols: 2, bCols: 2);

        // Assert
        Assert.That(result.Length, Is.EqualTo(4));
        Assert.That(result[0], Is.EqualTo(19.0f)); // 1*5 + 2*7 = 5 + 14 = 19
        Assert.That(result[1], Is.EqualTo(22.0f)); // 1*6 + 2*8 = 6 + 16 = 22
        Assert.That(result[2], Is.EqualTo(43.0f)); // 3*5 + 4*7 = 15 + 28 = 43
        Assert.That(result[3], Is.EqualTo(50.0f)); // 3*6 + 4*8 = 18 + 32 = 50
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Transpose_SimpleCase_ComputesCorrectResult()
    {
        // Arrange: 2x3 matrix transpose
        // A = [[1, 2, 3], [4, 5, 6]] (flattened: [1, 2, 3, 4, 5, 6])
        // Expected result: [[1, 4], [2, 5], [3, 6]] (flattened: [1, 4, 2, 5, 3, 6])
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Transpose(a, rows: 2, cols: 3);

        // Assert
        Assert.That(result.Length, Is.EqualTo(6));
        Assert.That(result[0], Is.EqualTo(1.0f)); // [0,0] -> [0,0]
        Assert.That(result[1], Is.EqualTo(4.0f)); // [1,0] -> [0,1]
        Assert.That(result[2], Is.EqualTo(2.0f)); // [0,1] -> [1,0]
        Assert.That(result[3], Is.EqualTo(5.0f)); // [1,1] -> [1,1]
        Assert.That(result[4], Is.EqualTo(3.0f)); // [0,2] -> [2,0]
        Assert.That(result[5], Is.EqualTo(6.0f)); // [1,2] -> [2,1]
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Operations_WithoutGradients_DoNotRequireGrad()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });
        var a = new GradTensor<float>(aData, requiresGrad: false);
        var b = new GradTensor<float>(bData, requiresGrad: false);

        // Act
        var result = GradOperations.Add(a, b);

        // Assert
        Assert.That(result.RequiresGrad, Is.False);
    }

    [Test]
    public void Divide_ByZero_ThrowsException()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 0.0f, 1.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Act & Assert
        Assert.Throws<DivideByZeroException>(() => GradOperations.Divide(a, b));
    }
}