using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using NUnit.Framework;

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

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

    #region Shape-aware Overloads Tests

    [Test]
    public void MatMul_ShapeAwareOverload_ComputesCorrectResult()
    {
        // Arrange: 2x2 matrix multiplication using shape inference
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);
        a.Reshape(2, 2);
        b.Reshape(2, 2);

        // Act
        var result = GradOperations.MatMul(a, b);

        // Assert
        Assert.That(result.Length, Is.EqualTo(4));
        Assert.That(result[0], Is.EqualTo(19.0f));
        Assert.That(result[3], Is.EqualTo(50.0f));
    }

    [Test]
    public void MatMul_ShapeAware_IncorrectRank_Throws()
    {
        // Arrange: 1D shape (should fail — MatMul requires rank 2)
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true); // shape [2]
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true); // shape [2]

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => GradOperations.MatMul(a, b));
        Assert.That(ex.Message, Does.Contain("rank 2"));
    }

    [Test]
    public void Transpose_ShapeAwareOverload_ComputesCorrectResult()
    {
        // Arrange: 2x3 matrix transpose using shape inference
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        a.Reshape(2, 3);

        // Act
        var result = GradOperations.Transpose(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(6));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(4.0f));
        Assert.That(result[2], Is.EqualTo(2.0f));
        Assert.That(result[5], Is.EqualTo(6.0f));
    }

    [Test]
    public void ReverseGradTensor_DefaultShape_IsOneDimensional()
    {
        // Arrange
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });

        // Act
        var tensor = new ReverseGradTensor<float>(data, requiresGrad: true);

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(1), "Default shape should be 1D");
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 3 }), "Shape should be [Length]");
    }

    [Test]
    public void ReverseGradTensor_Reshape_ChangesShape()
    {
        // Arrange
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
        var tensor = new ReverseGradTensor<float>(data, requiresGrad: true);

        // Act
        tensor.Reshape(2, 3);

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 2, 3 }));
    }

    [Test]
    public void ReverseGradTensor_Reshape_WrongSize_Throws()
    {
        // Arrange
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var tensor = new ReverseGradTensor<float>(data, requiresGrad: true);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => tensor.Reshape(3, 3));
        Assert.That(ex.Message, Does.Contain("9 elements"));
        Assert.That(ex.Message, Does.Contain("4 elements"));
    }

    [Test]
    public void Add_PreservesInputShape()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var result = GradOperations.Add(a, b);

        // Assert
        Assert.That(result.Shape, Is.EqualTo(new[] { 3 }), "Add should preserve shape");
    }

    [Test]
    public void MatMul_ResultHasCorrectShape()
    {
        // Arrange: 2x3 @ 3x4 = 2x4
        var aData = NivaraColumn<float>.Create(new float[6]);
        var bData = NivaraColumn<float>.Create(new float[12]);
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);
        a.Reshape(2, 3);
        b.Reshape(3, 4);

        // Act
        var result = GradOperations.MatMul(a, b);

        // Assert
        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 4 }), "MatMul result shape should be [aRows, bCols]");
    }

    [Test]
    public void Transpose_ResultHasCorrectShape()
    {
        // Arrange: 2x3 -> 3x2
        var aData = NivaraColumn<float>.Create(new float[6]);
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        a.Reshape(2, 3);

        // Act
        var result = GradOperations.Transpose(a);

        // Assert
        Assert.That(result.Shape, Is.EqualTo(new[] { 3, 2 }), "Transpose result shape should be [cols, rows]");
    }

    #endregion

    [Test]
    public void Operations_WithoutGradients_DoNotRequireGrad()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: false);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: false);

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
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act & Assert
        Assert.Throws<DivideByZeroException>(() => GradOperations.Divide(a, b));
    }

    #region Reduction Operations Tests

    [Test]
    public void Sum_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Sum(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(1));
        Assert.That(result[0], Is.EqualTo(10.0f)); // 1 + 2 + 3 + 4 = 10
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Sum_EmptyTensor_ThrowsException()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(Array.Empty<float>());
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => GradOperations.Sum(a));
    }

    [Test]
    public void Mean_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f, 4.0f, 6.0f, 8.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Mean(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(1));
        Assert.That(result[0], Is.EqualTo(5.0f)); // (2 + 4 + 6 + 8) / 4 = 5
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Mean_EmptyTensor_ThrowsException()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(Array.Empty<float>());
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => GradOperations.Mean(a));
    }

    [Test]
    public void Sum_WithBackward_ComputesCorrectGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Sum(a);
        result.Backward();

        // Assert - gradient of sum is ones for all inputs
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(3));
        Assert.That(a.Grad[0], Is.EqualTo(1.0f));
        Assert.That(a.Grad[1], Is.EqualTo(1.0f));
        Assert.That(a.Grad[2], Is.EqualTo(1.0f));
    }

    [Test]
    public void Mean_WithBackward_ComputesCorrectGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f, 4.0f, 6.0f, 8.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Mean(a);
        result.Backward();

        // Assert - gradient of mean is 1/n for all inputs
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(4));
        Assert.That(a.Grad[0], Is.EqualTo(0.25f)); // 1/4
        Assert.That(a.Grad[1], Is.EqualTo(0.25f));
        Assert.That(a.Grad[2], Is.EqualTo(0.25f));
        Assert.That(a.Grad[3], Is.EqualTo(0.25f));
    }

    #endregion

    #region Activation Functions Tests

    [Test]
    public void Relu_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Relu(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(5));
        Assert.That(result[0], Is.EqualTo(0.0f)); // max(0, -2) = 0
        Assert.That(result[1], Is.EqualTo(0.0f)); // max(0, -1) = 0
        Assert.That(result[2], Is.EqualTo(0.0f)); // max(0, 0) = 0
        Assert.That(result[3], Is.EqualTo(1.0f)); // max(0, 1) = 1
        Assert.That(result[4], Is.EqualTo(2.0f)); // max(0, 2) = 2
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Relu_WithBackward_ComputesCorrectGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f, 2.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Relu(a);

        // Create gradient output (all ones)
        var gradData = NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f, 1.0f, 1.0f });
        var grad = new ReverseGradTensor<float>(gradData, requiresGrad: false);
        result.Backward(grad);

        // Assert - gradient is 1 where input > 0, else 0
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(4));
        Assert.That(a.Grad[0], Is.EqualTo(0.0f)); // input -1 <= 0
        Assert.That(a.Grad[1], Is.EqualTo(0.0f)); // input 0 <= 0
        Assert.That(a.Grad[2], Is.EqualTo(1.0f)); // input 1 > 0
        Assert.That(a.Grad[3], Is.EqualTo(1.0f)); // input 2 > 0
    }

    [Test]
    public void Sigmoid_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 0.0f, 1.0f, -1.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Sigmoid(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(0.5f).Within(0.0001f)); // sigmoid(0) = 0.5
        Assert.That(result[1], Is.EqualTo(0.7311f).Within(0.001f)); // sigmoid(1) ≈ 0.7311
        Assert.That(result[2], Is.EqualTo(0.2689f).Within(0.001f)); // sigmoid(-1) ≈ 0.2689
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Sigmoid_WithBackward_ComputesCorrectGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 0.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Sigmoid(a);
        result.Backward();

        // Assert - gradient at x=0 is sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(1));
        Assert.That(a.Grad[0], Is.EqualTo(0.25f).Within(0.0001f));
    }

    [Test]
    public void Tanh_SimpleCase_ComputesCorrectResult()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 0.0f, 1.0f, -1.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Tanh(a);

        // Assert
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(0.0f).Within(0.0001f)); // tanh(0) = 0
        Assert.That(result[1], Is.EqualTo(0.7616f).Within(0.001f)); // tanh(1) ≈ 0.7616
        Assert.That(result[2], Is.EqualTo(-0.7616f).Within(0.001f)); // tanh(-1) ≈ -0.7616
        Assert.That(result.RequiresGrad, Is.True);
    }

    [Test]
    public void Tanh_WithBackward_ComputesCorrectGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 0.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var result = GradOperations.Tanh(a);
        result.Backward();

        // Assert - gradient at x=0 is 1 - tanh(0)^2 = 1 - 0 = 1
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad!.Length, Is.EqualTo(1));
        Assert.That(a.Grad[0], Is.EqualTo(1.0f).Within(0.0001f));
    }

    [Test]
    public void ActivationFunctions_WithoutGradients_DoNotRequireGrad()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: false);

        // Act
        var reluResult = GradOperations.Relu(a);
        var sigmoidResult = GradOperations.Sigmoid(a);
        var tanhResult = GradOperations.Tanh(a);

        // Assert
        Assert.That(reluResult.RequiresGrad, Is.False);
        Assert.That(sigmoidResult.RequiresGrad, Is.False);
        Assert.That(tanhResult.RequiresGrad, Is.False);
    }

    #endregion

    #region Integration Tests

    [Test]
    public void ComplexExpression_WithReductionAndActivation_ComputesCorrectGradients()
    {
        // Arrange: Test a simple neural network-like computation
        // y = mean(relu(x * w + b))
        var xData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var wData = NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f });
        var bData = NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f });

        var x = new ReverseGradTensor<float>(xData, requiresGrad: false);
        var w = new ReverseGradTensor<float>(wData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act: Forward pass
        var mul = GradOperations.Multiply(x, w);  // [0.5, 1.0, 1.5]
        var add = GradOperations.Add(mul, b);     // [-0.5, 1.0, 2.5]
        var relu = GradOperations.Relu(add);      // [0.0, 1.0, 2.5]
        var mean = GradOperations.Mean(relu);     // (0 + 1 + 2.5) / 3 = 1.1667

        // Backward pass
        mean.Backward();

        // Assert: Check forward pass result
        Assert.That(mean[0], Is.EqualTo(1.1667f).Within(0.001f));

        // Assert: Check gradients exist
        Assert.That(w.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);

        // The gradients should be non-zero for elements where relu was active
        // For w: gradient flows through where relu was active (indices 1 and 2)
        Assert.That(w.Grad![0], Is.EqualTo(0.0f).Within(0.001f)); // relu blocked gradient
        Assert.That(w.Grad[1], Is.GreaterThan(0.0f)); // gradient flows through
        Assert.That(w.Grad[2], Is.GreaterThan(0.0f)); // gradient flows through
    }

    [Test]
    public void SigmoidAndSum_ComputesCorrectGradients()
    {
        // Arrange: Test sigmoid followed by sum
        var xData = NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f });
        var x = new ReverseGradTensor<float>(xData, requiresGrad: true);

        // Act: Forward pass
        var sigmoid = GradOperations.Sigmoid(x);
        var sum = GradOperations.Sum(sigmoid);

        // Backward pass
        sum.Backward();

        // Assert: Check that gradients exist and are reasonable
        Assert.That(x.Grad, Is.Not.Null);
        Assert.That(x.Grad!.Length, Is.EqualTo(3));

        // Gradient at x=0 should be sigmoid(0) * (1 - sigmoid(0)) = 0.25
        Assert.That(x.Grad[1], Is.EqualTo(0.25f).Within(0.001f));

        // Gradients should be positive for all inputs (sigmoid derivative is always positive)
        Assert.That(x.Grad[0], Is.GreaterThan(0.0f));
        Assert.That(x.Grad[1], Is.GreaterThan(0.0f));
        Assert.That(x.Grad[2], Is.GreaterThan(0.0f));
    }

    #endregion

    #region Negate

    [Test]
    public void Negate_SimpleValues_NegatesCorrectly()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f, -2.0f, 3.0f }), requiresGrad: true);
        var result = GradOperations.Negate(a);

        Assert.That(result[0], Is.EqualTo(-1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[2], Is.EqualTo(-3.0f));
    }

    [Test]
    public void Negate_Backward_CorrectGradient()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: true);
        var n = GradOperations.Negate(a);
        var sum = GradOperations.Sum(n);
        sum.Backward();

        // Gradient of -x is -1, so d(loss)/dx = -1
        Assert.That(a.Grad![0], Is.EqualTo(-1.0f));
        Assert.That(a.Grad[1], Is.EqualTo(-1.0f));
        Assert.That(a.Grad[2], Is.EqualTo(-1.0f));
    }

    [Test]
    public void Negate_RequiresGradFalse_ReturnsNoGrad()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f }), requiresGrad: false);
        var result = GradOperations.Negate(a);

        Assert.That(result.RequiresGrad, Is.False);
    }

    #endregion

    #region Abs

    [Test]
    public void Abs_SimpleValues_ComputesCorrectly()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f }), requiresGrad: true);
        var result = GradOperations.Abs(a);

        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(1.0f));
        Assert.That(result[2], Is.EqualTo(0.0f));
        Assert.That(result[3], Is.EqualTo(1.0f));
        Assert.That(result[4], Is.EqualTo(2.0f));
    }

    [Test]
    public void Abs_Backward_CorrectGradient()
    {
        // d/dx |x| = sign(x), sign(0) = 0
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { -2.0f, 0.0f, 3.0f }), requiresGrad: true);
        var abs = GradOperations.Abs(a);
        var sum = GradOperations.Sum(abs);
        sum.Backward();

        Assert.That(a.Grad![0], Is.EqualTo(-1.0f));
        Assert.That(a.Grad[1], Is.EqualTo(0.0f));
        Assert.That(a.Grad[2], Is.EqualTo(1.0f));
    }

    [Test]
    public void Abs_DoubleType_ComputesCorrectly()
    {
        var a = new ReverseGradTensor<double>(NivaraColumn<double>.Create(new double[] { -3.0, 0.0, 5.0 }), requiresGrad: true);
        var result = GradOperations.Abs(a);

        Assert.That(result[0], Is.EqualTo(3.0));
        Assert.That(result[1], Is.EqualTo(0.0));
        Assert.That(result[2], Is.EqualTo(5.0));

        var sum = GradOperations.Sum(result);
        sum.Backward();
        Assert.That(a.Grad![0], Is.EqualTo(-1.0));
        Assert.That(a.Grad[1], Is.EqualTo(0.0));
    }

    [Test]
    public void Abs_RequiresGradFalse_ReturnsNoGrad()
    {
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { -1.0f, 2.0f }), requiresGrad: false);
        var result = GradOperations.Abs(a);

        Assert.That(result.RequiresGrad, Is.False);
    }

    #endregion
}