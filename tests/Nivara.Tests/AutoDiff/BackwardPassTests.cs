using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for backward pass and gradient computation in automatic differentiation.
/// Verifies that gradients are computed correctly through the computation graph.
/// </summary>
[TestFixture]
public class BackwardPassTests
{
    [Test]
    public void Backward_SimpleAddition_ComputesCorrectGradients()
    {
        // Arrange: y = a + b, where a = [2.0], b = [3.0]
        // Expected: dy/da = 1.0, dy/db = 1.0
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var y = GradOperations.Add(a, b);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(1.0f).Within(1e-6f));
    }

    [Test]
    public void Backward_SimpleMultiplication_ComputesCorrectGradients()
    {
        // Arrange: y = a * b, where a = [2.0], b = [3.0]
        // Expected: dy/da = b = 3.0, dy/db = a = 2.0
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var y = GradOperations.Multiply(a, b);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(2.0f).Within(1e-6f));
    }

    [Test]
    public void Backward_ChainedOperations_ComputesCorrectGradients()
    {
        // Arrange: y = (a + b) * c, where a = [1.0], b = [2.0], c = [3.0]
        // Expected: dy/da = c = 3.0, dy/db = c = 3.0, dy/dc = (a + b) = 3.0
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var cData = NivaraColumn<float>.Create(new float[] { 3.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);
        var c = new ReverseGradTensor<float>(cData, requiresGrad: true);

        // Act
        var sum = GradOperations.Add(a, b);
        var y = GradOperations.Multiply(sum, c);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(c.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(c.Grad![0], Is.EqualTo(3.0f).Within(1e-6f));
    }

    [Test]
    public void Backward_GradientAccumulation_AccumulatesCorrectly()
    {
        // Arrange: y = a + a (tensor used twice)
        // Expected: dy/da = 2.0 (gradient accumulates)
        var aData = NivaraColumn<float>.Create(new float[] { 5.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);

        // Act
        var y = GradOperations.Add(a, a);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(2.0f).Within(1e-6f));
    }

    [Test]
    public void Backward_NonScalarWithoutGradient_ThrowsException()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var y = GradOperations.Add(a, b);

        // Assert
        var ex = Assert.Throws<InvalidOperationException>(() => y.Backward());
        Assert.That(ex!.Message, Does.Contain("scalar"));
    }

    [Test]
    public void Backward_ExplicitGradientWithSameLengthDifferentShape_ThrowsException()
    {
        var a = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }),
            requiresGrad: true);
        var b = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 5.0f, 6.0f, 7.0f, 8.0f }),
            requiresGrad: true);
        a.Reshape(2, 2);
        b.Reshape(2, 2);
        var y = GradOperations.MatMul(a, b);

        var gradient = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }),
            requiresGrad: false);
        gradient.Reshape(4);

        var ex = Assert.Throws<ArgumentException>(() => y.Backward(gradient));
        Assert.That(ex!.Message, Does.Contain("Gradient shape mismatch"));
    }

    [Test]
    public void Backward_TensorWithoutRequiresGrad_ThrowsException()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: false);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => a.Backward());
        Assert.That(ex!.Message, Does.Contain("require"));
    }

    [Test]
    public void Backward_SubtractionOperation_ComputesCorrectGradients()
    {
        // Arrange: y = a - b, where a = [5.0], b = [3.0]
        // Expected: dy/da = 1.0, dy/db = -1.0
        var aData = NivaraColumn<float>.Create(new float[] { 5.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var y = GradOperations.Subtract(a, b);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(-1.0f).Within(1e-6f));
    }

    [Test]
    public void Backward_DivisionOperation_ComputesCorrectGradients()
    {
        // Arrange: y = a / b, where a = [6.0], b = [2.0]
        // Expected: dy/da = 1/b = 0.5, dy/db = -a/(b^2) = -6/4 = -1.5
        var aData = NivaraColumn<float>.Create(new float[] { 6.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);

        // Act
        var y = GradOperations.Divide(a, b);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(0.5f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(-1.5f).Within(1e-6f));
    }

    [Test]
    public void Backward_ComplexExpression_ComputesCorrectGradients()
    {
        // Arrange: y = (a * b) + (c / d), where a = [2.0], b = [3.0], c = [8.0], d = [2.0]
        // y = 6.0 + 4.0 = 10.0
        // dy/da = b = 3.0, dy/db = a = 2.0, dy/dc = 1/d = 0.5, dy/dd = -c/(d^2) = -2.0
        var aData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f });
        var cData = NivaraColumn<float>.Create(new float[] { 8.0f });
        var dData = NivaraColumn<float>.Create(new float[] { 2.0f });
        var a = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bData, requiresGrad: true);
        var c = new ReverseGradTensor<float>(cData, requiresGrad: true);
        var d = new ReverseGradTensor<float>(dData, requiresGrad: true);

        // Act
        var product = GradOperations.Multiply(a, b);
        var quotient = GradOperations.Divide(c, d);
        var y = GradOperations.Add(product, quotient);
        y.Backward();

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(c.Grad, Is.Not.Null);
        Assert.That(d.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(b.Grad![0], Is.EqualTo(2.0f).Within(1e-6f));
        Assert.That(c.Grad![0], Is.EqualTo(0.5f).Within(1e-6f));
        Assert.That(d.Grad![0], Is.EqualTo(-2.0f).Within(1e-6f));
    }
}
