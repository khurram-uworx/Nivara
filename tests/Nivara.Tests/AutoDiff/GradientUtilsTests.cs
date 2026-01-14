using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using Nivara.Extensions.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for gradient utility functions in automatic differentiation.
/// Verifies gradient management, clipping, constant creation, and graph inspection utilities.
/// </summary>
[TestFixture]
public class GradientUtilsTests
{
    #region Gradient Management Tests

    [Test]
    public void ZeroGrad_SingleTensor_ClearsGradient()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Compute some gradients
        var result = GradOperations.Sum(a);
        result.Backward();

        Assert.That(a.Grad, Is.Not.Null, "Gradient should exist before clearing");

        // Act
        GradientUtils.ZeroGrad(a);

        // Assert
        Assert.That(a.Grad, Is.Null, "Gradient should be null after clearing");
    }

    [Test]
    public void ZeroGrad_MultipleTensors_ClearsAllGradients()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var bData = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);
        var b = new GradTensor<float>(bData, requiresGrad: true);

        // Compute some gradients
        var result = GradOperations.Add(a, b);
        var sum = GradOperations.Sum(result);
        sum.Backward();

        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(b.Grad, Is.Not.Null);

        // Act
        GradientUtils.ZeroGrad(new[] { a, b });

        // Assert
        Assert.That(a.Grad, Is.Null);
        Assert.That(b.Grad, Is.Null);
    }

    [Test]
    public void Detach_SingleTensor_RemovesGradientTracking()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act
        var detached = GradientUtils.Detach(a);

        // Assert
        Assert.That(detached.RequiresGrad, Is.False);
        Assert.That(detached.Length, Is.EqualTo(a.Length));
        Assert.That(detached[0], Is.EqualTo(a[0]));
        Assert.That(detached[1], Is.EqualTo(a[1]));
        Assert.That(detached[2], Is.EqualTo(a[2]));
    }

    [Test]
    public void Detach_MultipleTensors_RemovesGradientTrackingFromAll()
    {
        // Arrange
        var tensors = new[]
        {
            new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true),
            new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 2.0f }), requiresGrad: true),
            new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 3.0f }), requiresGrad: true)
        };

        // Act
        var detached = GradientUtils.Detach(tensors).ToList();

        // Assert
        Assert.That(detached.Count, Is.EqualTo(3));
        Assert.That(detached.All(t => !t.RequiresGrad), Is.True);
    }

    #endregion

    #region Gradient Clipping Tests

    [Test]
    public void ClipGradValue_ClipsLargeValues()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Set gradient manually
        var gradData = NivaraColumn<float>.Create(new float[] { -5.0f, 2.0f, 10.0f });
        a.Grad = gradData;

        // Act
        GradientUtils.ClipGradValue(a, 3.0f);

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(-3.0f)); // Clipped from -5
        Assert.That(a.Grad[1], Is.EqualTo(2.0f));   // Unchanged
        Assert.That(a.Grad[2], Is.EqualTo(3.0f));   // Clipped from 10
    }

    [Test]
    public void ClipGradValue_NoGradient_DoesNothing()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act & Assert - should not throw
        Assert.DoesNotThrow(() => GradientUtils.ClipGradValue(a, 1.0f));
    }

    [Test]
    public void ClipGradNorm_SingleTensor_ScalesGradient()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Set gradient with norm = sqrt(3^2 + 4^2) = 5
        var gradData = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });
        a.Grad = gradData;

        // Act - clip to norm of 2.5 (should scale by 0.5)
        GradientUtils.ClipGradNorm(a, 2.5);

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(1.5f).Within(0.001f)); // 3 * 0.5
        Assert.That(a.Grad[1], Is.EqualTo(2.0f).Within(0.001f));  // 4 * 0.5
    }

    [Test]
    public void ClipGradNorm_NormBelowMax_DoesNotClip()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Set gradient with norm = sqrt(1 + 1) = sqrt(2) ≈ 1.414
        var gradData = NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f });
        a.Grad = gradData;

        // Act - clip to norm of 10 (should not change anything)
        GradientUtils.ClipGradNorm(a, 10.0);

        // Assert
        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad![0], Is.EqualTo(1.0f));
        Assert.That(a.Grad[1], Is.EqualTo(1.0f));
    }

    [Test]
    public void ClipGradNorm_MultipleTensors_ClipsGlobalNorm()
    {
        // Arrange
        var a = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true);
        var b = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true);

        // Set gradients: a.grad = [3], b.grad = [4]
        // Global norm = sqrt(9 + 16) = 5
        a.Grad = NivaraColumn<float>.Create(new float[] { 3.0f });
        b.Grad = NivaraColumn<float>.Create(new float[] { 4.0f });

        // Act - clip to global norm of 2.5 (should scale by 0.5)
        GradientUtils.ClipGradNorm(new[] { a, b }, 2.5);

        // Assert
        Assert.That(a.Grad![0], Is.EqualTo(1.5f).Within(0.001f)); // 3 * 0.5
        Assert.That(b.Grad![0], Is.EqualTo(2.0f).Within(0.001f));  // 4 * 0.5
    }

    #endregion

    #region Constant Tensor Creation Tests

    [Test]
    public void Constant_FromArray_CreatesNonGradTensor()
    {
        // Arrange & Act
        var tensor = GradientUtils.Constant(new float[] { 1.0f, 2.0f, 3.0f });

        // Assert
        Assert.That(tensor.RequiresGrad, Is.False);
        Assert.That(tensor.Length, Is.EqualTo(3));
        Assert.That(tensor[0], Is.EqualTo(1.0f));
        Assert.That(tensor[1], Is.EqualTo(2.0f));
        Assert.That(tensor[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void Constant_FromColumn_CreatesNonGradTensor()
    {
        // Arrange
        var column = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f });

        // Act
        var tensor = GradientUtils.Constant(column);

        // Assert
        Assert.That(tensor.RequiresGrad, Is.False);
        Assert.That(tensor.Length, Is.EqualTo(2));
        Assert.That(tensor[0], Is.EqualTo(4.0f));
        Assert.That(tensor[1], Is.EqualTo(5.0f));
    }

    [Test]
    public void Zeros_CreatesZeroFilledTensor()
    {
        // Act
        var tensor = GradientUtils.Zeros<float>(5);

        // Assert
        Assert.That(tensor.RequiresGrad, Is.False);
        Assert.That(tensor.Length, Is.EqualTo(5));
        for (int i = 0; i < 5; i++)
        {
            Assert.That(tensor[i], Is.EqualTo(0.0f));
        }
    }

    [Test]
    public void Ones_CreatesOneFilledTensor()
    {
        // Act
        var tensor = GradientUtils.Ones<float>(4);

        // Assert
        Assert.That(tensor.RequiresGrad, Is.False);
        Assert.That(tensor.Length, Is.EqualTo(4));
        for (int i = 0; i < 4; i++)
        {
            Assert.That(tensor[i], Is.EqualTo(1.0f));
        }
    }

    [Test]
    public void Full_CreatesValueFilledTensor()
    {
        // Act
        var tensor = GradientUtils.Full(3, 7.5f);

        // Assert
        Assert.That(tensor.RequiresGrad, Is.False);
        Assert.That(tensor.Length, Is.EqualTo(3));
        for (int i = 0; i < 3; i++)
        {
            Assert.That(tensor[i], Is.EqualTo(7.5f));
        }
    }

    #endregion

    #region Computation Graph Inspection Tests

    [Test]
    public void GetGraphInfo_LeafTensor_ReturnsCorrectInfo()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act
        var info = GradientUtils.GetGraphInfo(a);

        // Assert
        Assert.That(info["TotalNodes"], Is.EqualTo(0));
        Assert.That(info["IsLeaf"], Is.True);
        Assert.That(info["RequiresGrad"], Is.True);
    }

    [Test]
    public void GetGraphInfo_ComplexGraph_ReturnsCorrectInfo()
    {
        // Arrange
        var a = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f }), requiresGrad: true);
        var b = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f }), requiresGrad: true);

        var add = GradOperations.Add(a, b);
        var mul = GradOperations.Multiply(add, a);
        var sum = GradOperations.Sum(mul);

        // Act
        var info = GradientUtils.GetGraphInfo(sum);

        // Assert
        Assert.That(info["TotalNodes"], Is.EqualTo(3)); // Add, Multiply, Sum
        Assert.That(info["IsLeaf"], Is.False);
        Assert.That(info["RequiresGrad"], Is.True);

        var opCounts = (Dictionary<string, int>)info["OperationCounts"];
        Assert.That(opCounts["Add"], Is.EqualTo(1));
        Assert.That(opCounts["Multiply"], Is.EqualTo(1));
        Assert.That(opCounts["Sum"], Is.EqualTo(1));
    }

    [Test]
    public void PrintGraphSummary_ReturnsFormattedString()
    {
        // Arrange
        var a = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true);
        var b = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 2.0f }), requiresGrad: true);
        var result = GradOperations.Add(a, b);

        // Act
        var summary = GradientUtils.PrintGraphSummary(result);

        // Assert
        Assert.That(summary, Does.Contain("Computation Graph Summary"));
        Assert.That(summary, Does.Contain("Total Nodes"));
        Assert.That(summary, Does.Contain("Add"));
    }

    [Test]
    public void HasGradient_WithGradient_ReturnsTrue()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        var result = GradOperations.Sum(a);
        result.Backward();

        // Act & Assert
        Assert.That(GradientUtils.HasGradient(a), Is.True);
    }

    [Test]
    public void HasGradient_WithoutGradient_ReturnsFalse()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act & Assert
        Assert.That(GradientUtils.HasGradient(a), Is.False);
    }

    [Test]
    public void GetGradientNorm_ComputesCorrectNorm()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Set gradient [3, 4] with norm = 5
        a.Grad = NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f });

        // Act
        var norm = GradientUtils.GetGradientNorm(a);

        // Assert
        Assert.That(norm, Is.EqualTo(5.0).Within(0.001));
    }

    [Test]
    public void GetGradientNorm_NoGradient_ReturnsZero()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act
        var norm = GradientUtils.GetGradientNorm(a);

        // Assert
        Assert.That(norm, Is.EqualTo(0.0));
    }

    [Test]
    public void GetGlobalGradientNorm_ComputesCorrectNorm()
    {
        // Arrange
        var a = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true);
        var b = new GradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f }), requiresGrad: true);

        // Set gradients: a.grad = [3], b.grad = [4]
        // Global norm = sqrt(9 + 16) = 5
        a.Grad = NivaraColumn<float>.Create(new float[] { 3.0f });
        b.Grad = NivaraColumn<float>.Create(new float[] { 4.0f });

        // Act
        var norm = GradientUtils.GetGlobalGradientNorm(new[] { a, b });

        // Assert
        Assert.That(norm, Is.EqualTo(5.0).Within(0.001));
    }

    [Test]
    public void CanBackward_ScalarTensorWithGrad_ReturnsTrue()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act & Assert
        Assert.That(GradientUtils.CanBackward(a), Is.True);
    }

    [Test]
    public void CanBackward_NonScalarTensor_ReturnsFalse()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act & Assert
        Assert.That(GradientUtils.CanBackward(a), Is.False);
    }

    [Test]
    public void CanBackward_TensorWithoutGrad_ReturnsFalse()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f });
        var a = new GradTensor<float>(aData, requiresGrad: false);

        // Act & Assert
        Assert.That(GradientUtils.CanBackward(a), Is.False);
    }

    [Test]
    public void DescribeTensor_ReturnsFormattedDescription()
    {
        // Arrange
        var aData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var a = new GradTensor<float>(aData, requiresGrad: true);

        // Act
        var description = GradientUtils.DescribeTensor(a);

        // Assert
        Assert.That(description, Does.Contain("GradTensor<Single>"));
        Assert.That(description, Does.Contain("Length: 3"));
        Assert.That(description, Does.Contain("Requires Grad: True"));
        Assert.That(description, Does.Contain("Is Leaf: True"));
    }

    #endregion

    #region Integration Tests

    [Test]
    public void TrainingLoop_Simulation_WorksCorrectly()
    {
        // Arrange - simulate a simple training loop
        var weights = new GradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f }),
            requiresGrad: true);

        var inputs = GradientUtils.Constant(new float[] { 1.0f, 2.0f });
        var targets = GradientUtils.Constant(new float[] { 3.0f });

        // Act - simulate one training step
        var predictions = GradOperations.Multiply(inputs, weights);
        var sum = GradOperations.Sum(predictions);
        var loss = GradOperations.Subtract(sum, targets);

        // Backward pass
        loss.Backward();

        // Check gradient exists
        Assert.That(GradientUtils.HasGradient(weights), Is.True);
        var gradNorm = GradientUtils.GetGradientNorm(weights);
        Assert.That(gradNorm, Is.GreaterThan(0.0));

        // Clear gradients for next iteration
        GradientUtils.ZeroGrad(weights);

        // Assert
        Assert.That(GradientUtils.HasGradient(weights), Is.False);
    }

    [Test]
    public void GradientClipping_PreventsExplodingGradients()
    {
        // Arrange
        var weights = new GradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f }),
            requiresGrad: true);

        // Set very large gradients
        weights.Grad = NivaraColumn<float>.Create(new float[] { 100.0f, 100.0f });

        var initialNorm = GradientUtils.GetGradientNorm(weights);
        Assert.That(initialNorm, Is.GreaterThan(100.0));

        // Act - clip to reasonable value
        GradientUtils.ClipGradNorm(weights, 1.0);

        // Assert
        var clippedNorm = GradientUtils.GetGradientNorm(weights);
        Assert.That(clippedNorm, Is.EqualTo(1.0).Within(0.001));
    }

    #endregion
}
