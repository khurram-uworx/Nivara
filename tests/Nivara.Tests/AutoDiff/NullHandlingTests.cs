using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for null handling in automatic differentiation operations.
/// Verifies that GradTensor operations preserve Nivara's explicit null semantics.
/// Tests Requirements 5.5 and 9.4.
/// </summary>
[TestFixture]
public class NullHandlingTests
{
    #region Element-wise Operations Null Handling

    [Test]
    public void Add_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange: Create columns with null values using Nivara's explicit null semantics
        var aValues = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var bValues = new float?[] { 2.0f, 4.0f, null, null, 6.0f };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new GradTensor<float>(aColumn, requiresGrad: true);
        var b = new GradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = GradOperations.Add(a, b);

        // Assert: Verify null propagation (null + anything = null)
        Assert.That(result.IsNull(0), Is.False, "Position 0: 1 + 2 should not be null");
        Assert.That(result[0], Is.EqualTo(3.0f), "Position 0: 1 + 2 = 3");

        Assert.That(result.IsNull(1), Is.True, "Position 1: null + 4 should be null");
        Assert.That(result.IsNull(2), Is.True, "Position 2: 3 + null should be null");
        Assert.That(result.IsNull(3), Is.True, "Position 3: null + null should be null");

        Assert.That(result.IsNull(4), Is.False, "Position 4: 5 + 6 should not be null");
        Assert.That(result[4], Is.EqualTo(11.0f), "Position 4: 5 + 6 = 11");
    }

    [Test]
    public void Multiply_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var aValues = new float?[] { 2.0f, null, 4.0f, null };
        var bValues = new float?[] { 3.0f, 5.0f, null, null };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new GradTensor<float>(aColumn, requiresGrad: true);
        var b = new GradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = GradOperations.Multiply(a, b);

        // Assert: Verify null propagation (null * anything = null)
        Assert.That(result.IsNull(0), Is.False, "Position 0: 2 * 3 should not be null");
        Assert.That(result[0], Is.EqualTo(6.0f), "Position 0: 2 * 3 = 6");

        Assert.That(result.IsNull(1), Is.True, "Position 1: null * 5 should be null");
        Assert.That(result.IsNull(2), Is.True, "Position 2: 4 * null should be null");
        Assert.That(result.IsNull(3), Is.True, "Position 3: null * null should be null");
    }

    [Test]
    public void Subtract_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var aValues = new double?[] { 10.0, null, 6.0 };
        var bValues = new double?[] { 3.0, 2.0, null };

        var aColumn = NivaraColumn<double>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<double>.CreateFromNullable(bValues);

        var a = new GradTensor<double>(aColumn, requiresGrad: true);
        var b = new GradTensor<double>(bColumn, requiresGrad: true);

        // Act
        var result = GradOperations.Subtract(a, b);

        // Assert: Verify null propagation (null - anything = null)
        Assert.That(result.IsNull(0), Is.False, "Position 0: 10 - 3 should not be null");
        Assert.That(result[0], Is.EqualTo(7.0), "Position 0: 10 - 3 = 7");

        Assert.That(result.IsNull(1), Is.True, "Position 1: null - 2 should be null");
        Assert.That(result.IsNull(2), Is.True, "Position 2: 6 - null should be null");
    }

    [Test]
    public void Divide_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var aValues = new float?[] { 12.0f, null, 18.0f };
        var bValues = new float?[] { 3.0f, 5.0f, null };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new GradTensor<float>(aColumn, requiresGrad: true);
        var b = new GradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = GradOperations.Divide(a, b);

        // Assert: Verify null propagation (null / anything = null)
        Assert.That(result.IsNull(0), Is.False, "Position 0: 12 / 3 should not be null");
        Assert.That(result[0], Is.EqualTo(4.0f), "Position 0: 12 / 3 = 4");

        Assert.That(result.IsNull(1), Is.True, "Position 1: null / 5 should be null");
        Assert.That(result.IsNull(2), Is.True, "Position 2: 18 / null should be null");
    }

    #endregion

    #region Activation Functions Null Handling

    [Test]
    public void Relu_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var values = new float?[] { -2.0f, null, 0.0f, null, 2.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Act
        var result = GradOperations.Relu(tensor);

        // Assert: Verify null propagation
        Assert.That(result.IsNull(0), Is.False, "Position 0: relu(-2) should not be null");
        Assert.That(result[0], Is.EqualTo(0.0f), "Position 0: relu(-2) = 0");

        Assert.That(result.IsNull(1), Is.True, "Position 1: relu(null) should be null");

        Assert.That(result.IsNull(2), Is.False, "Position 2: relu(0) should not be null");
        Assert.That(result[2], Is.EqualTo(0.0f), "Position 2: relu(0) = 0");

        Assert.That(result.IsNull(3), Is.True, "Position 3: relu(null) should be null");

        Assert.That(result.IsNull(4), Is.False, "Position 4: relu(2) should not be null");
        Assert.That(result[4], Is.EqualTo(2.0f), "Position 4: relu(2) = 2");
    }

    [Test]
    public void Sigmoid_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var values = new float?[] { 0.0f, null, 1.0f, null };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Act
        var result = GradOperations.Sigmoid(tensor);

        // Assert: Verify null propagation
        Assert.That(result.IsNull(0), Is.False, "Position 0: sigmoid(0) should not be null");
        Assert.That(result[0], Is.EqualTo(0.5f).Within(0.0001f), "Position 0: sigmoid(0) = 0.5");

        Assert.That(result.IsNull(1), Is.True, "Position 1: sigmoid(null) should be null");

        Assert.That(result.IsNull(2), Is.False, "Position 2: sigmoid(1) should not be null");
        Assert.That(result[2], Is.EqualTo(0.7311f).Within(0.001f), "Position 2: sigmoid(1) ≈ 0.7311");

        Assert.That(result.IsNull(3), Is.True, "Position 3: sigmoid(null) should be null");
    }

    [Test]
    public void Tanh_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var values = new double?[] { 0.0, null, 1.0 };
        var column = NivaraColumn<double>.CreateFromNullable(values);
        var tensor = new GradTensor<double>(column, requiresGrad: true);

        // Act
        var result = GradOperations.Tanh(tensor);

        // Assert: Verify null propagation
        Assert.That(result.IsNull(0), Is.False, "Position 0: tanh(0) should not be null");
        Assert.That(result[0], Is.EqualTo(0.0).Within(0.0001), "Position 0: tanh(0) = 0");

        Assert.That(result.IsNull(1), Is.True, "Position 1: tanh(null) should be null");

        Assert.That(result.IsNull(2), Is.False, "Position 2: tanh(1) should not be null");
        Assert.That(result[2], Is.EqualTo(0.7616).Within(0.001), "Position 2: tanh(1) ≈ 0.7616");
    }

    #endregion

    #region Gradient Computation with Nulls

    [Test]
    public void Add_WithNullValues_ComputesGradientsCorrectly()
    {
        // Arrange
        var aValues = new float?[] { 1.0f, null, 3.0f };
        var bValues = new float?[] { 2.0f, 4.0f, null };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new GradTensor<float>(aColumn, requiresGrad: true);
        var b = new GradTensor<float>(bColumn, requiresGrad: true);

        // Act: Forward pass
        var result = GradOperations.Add(a, b);

        // Create gradient output (all ones for non-null positions)
        var gradValues = new float?[] { 1.0f, 1.0f, 1.0f };
        var gradColumn = NivaraColumn<float>.CreateFromNullable(gradValues);
        var grad = new GradTensor<float>(gradColumn, requiresGrad: false);

        result.Backward(grad);

        // Assert: Gradients should be computed for non-null positions
        Assert.That(a.Grad, Is.Not.Null, "Gradient for a should exist");
        Assert.That(b.Grad, Is.Not.Null, "Gradient for b should exist");

        // Position 0: both non-null, gradient should flow
        Assert.That(a.Grad![0], Is.EqualTo(1.0f), "Gradient for a[0] should be 1");
        Assert.That(b.Grad![0], Is.EqualTo(1.0f), "Gradient for b[0] should be 1");

        // Position 1: a is null, gradient should still accumulate
        Assert.That(a.Grad[1], Is.EqualTo(1.0f), "Gradient for a[1] should be 1");
        Assert.That(b.Grad[1], Is.EqualTo(1.0f), "Gradient for b[1] should be 1");

        // Position 2: b is null, gradient should still accumulate
        Assert.That(a.Grad[2], Is.EqualTo(1.0f), "Gradient for a[2] should be 1");
        Assert.That(b.Grad[2], Is.EqualTo(1.0f), "Gradient for b[2] should be 1");
    }

    [Test]
    public void Relu_WithNullValues_ComputesGradientsCorrectly()
    {
        // Arrange
        var values = new float?[] { -1.0f, null, 1.0f, 2.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Act: Forward pass
        var result = GradOperations.Relu(tensor);

        // Create gradient output (all ones)
        var gradValues = new float?[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var gradColumn = NivaraColumn<float>.CreateFromNullable(gradValues);
        var grad = new GradTensor<float>(gradColumn, requiresGrad: false);

        result.Backward(grad);

        // Assert: Gradients should be computed correctly
        Assert.That(tensor.Grad, Is.Not.Null, "Gradient should exist");

        // Position 0: input -1 <= 0, gradient should be 0
        Assert.That(tensor.Grad![0], Is.EqualTo(0.0f), "Gradient for negative input should be 0");

        // Position 1: input is null, gradient should be 0 (null handling)
        Assert.That(tensor.Grad[1], Is.EqualTo(0.0f), "Gradient for null input should be 0");

        // Position 2: input 1 > 0, gradient should be 1
        Assert.That(tensor.Grad[2], Is.EqualTo(1.0f), "Gradient for positive input should be 1");

        // Position 3: input 2 > 0, gradient should be 1
        Assert.That(tensor.Grad[3], Is.EqualTo(1.0f), "Gradient for positive input should be 1");
    }

    #endregion

    #region Null Mask Preservation

    [Test]
    public void GradTensor_PreservesNullMaskFromNivaraColumn()
    {
        // Arrange
        var values = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);

        // Act
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Assert: Verify null mask is preserved
        Assert.That(tensor.HasNulls, Is.True, "Tensor should indicate it has nulls");
        Assert.That(tensor.Length, Is.EqualTo(5), "Tensor should have correct length");

        for (int i = 0; i < values.Length; i++)
        {
            bool expectedIsNull = values[i] == null;
            Assert.That(tensor.IsNull(i), Is.EqualTo(expectedIsNull),
                $"Position {i} null status should match original column");
        }
    }

    [Test]
    public void Operations_PreserveExplicitNullSemantics()
    {
        // Arrange: Create tensors with explicit null masks
        var aValues = new float?[] { 1.0f, null, 3.0f };
        var bValues = new float?[] { 2.0f, 4.0f, null };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new GradTensor<float>(aColumn, requiresGrad: true);
        var b = new GradTensor<float>(bColumn, requiresGrad: true);

        // Act: Perform multiple operations
        var add = GradOperations.Add(a, b);
        var mul = GradOperations.Multiply(a, b);

        // Assert: Verify explicit null semantics are preserved (no sentinel values)
        // Addition result
        Assert.That(add.HasNulls, Is.True, "Addition result should have nulls");
        Assert.That(add.IsNull(1), Is.True, "Addition: null + 4 should be null");
        Assert.That(add.IsNull(2), Is.True, "Addition: 3 + null should be null");

        // Multiplication result
        Assert.That(mul.HasNulls, Is.True, "Multiplication result should have nulls");
        Assert.That(mul.IsNull(1), Is.True, "Multiplication: null * 4 should be null");
        Assert.That(mul.IsNull(2), Is.True, "Multiplication: 3 * null should be null");

        // Verify no sentinel values are used (explicit null mask approach)
        // This is implicit in Nivara's design - nulls are tracked via null masks, not special values
    }

    #endregion

    #region Reduction Operations with Nulls

    [Test]
    public void Sum_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange: NivaraSeries.Sum() skips null values by default
        var values = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Act
        var result = GradOperations.Sum(tensor);

        // Assert: Sum should skip nulls (1 + 3 + 5 = 9)
        Assert.That(result.Length, Is.EqualTo(1), "Result should be scalar");
        Assert.That(result[0], Is.EqualTo(9.0f), "Sum should skip null values: 1 + 3 + 5 = 9");
        Assert.That(result.RequiresGrad, Is.True, "Result should require gradients");
    }

    [Test]
    public void Mean_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange: NivaraSeries.Average() skips null values by default
        var values = new float?[] { 2.0f, null, 4.0f, null, 6.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Act
        var result = GradOperations.Mean(tensor);

        // Assert: Mean should skip nulls (2 + 4 + 6) / 3 = 4
        Assert.That(result.Length, Is.EqualTo(1), "Result should be scalar");
        Assert.That(result[0], Is.EqualTo(4.0f), "Mean should skip null values: (2 + 4 + 6) / 3 = 4");
        Assert.That(result.RequiresGrad, Is.True, "Result should require gradients");
    }

    #endregion

    #region Integration Tests

    [Test]
    public void ComplexExpression_WithNulls_PreservesNullSemantics()
    {
        // Arrange: Test a complex expression with nulls
        var xValues = new float?[] { 1.0f, null, 3.0f };
        var wValues = new float?[] { 0.5f, 0.5f, null };

        var xColumn = NivaraColumn<float>.CreateFromNullable(xValues);
        var wColumn = NivaraColumn<float>.CreateFromNullable(wValues);

        var x = new GradTensor<float>(xColumn, requiresGrad: false);
        var w = new GradTensor<float>(wColumn, requiresGrad: true);

        // Act: Forward pass with nulls
        var mul = GradOperations.Multiply(x, w);  // [0.5, null, null]
        var relu = GradOperations.Relu(mul);      // [0.5, null, null]

        // Assert: Verify null propagation through operations
        Assert.That(mul.IsNull(0), Is.False, "Position 0: 1 * 0.5 should not be null");
        Assert.That(mul[0], Is.EqualTo(0.5f), "Position 0: 1 * 0.5 = 0.5");

        Assert.That(mul.IsNull(1), Is.True, "Position 1: null * 0.5 should be null");
        Assert.That(mul.IsNull(2), Is.True, "Position 2: 3 * null should be null");

        // ReLU should preserve nulls
        Assert.That(relu.IsNull(0), Is.False, "Position 0: relu(0.5) should not be null");
        Assert.That(relu[0], Is.EqualTo(0.5f), "Position 0: relu(0.5) = 0.5");

        Assert.That(relu.IsNull(1), Is.True, "Position 1: relu(null) should be null");
        Assert.That(relu.IsNull(2), Is.True, "Position 2: relu(null) should be null");
    }

    [Test]
    public void NoSentinelValues_OnlyExplicitNullMasks()
    {
        // This test verifies that Nivara's explicit null semantics are preserved
        // and no sentinel values (like NaN, -1, etc.) are used to represent nulls

        // Arrange
        var values = new float?[] { 0.0f, null, float.NaN, null, -1.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new GradTensor<float>(column, requiresGrad: true);

        // Assert: Verify explicit null tracking
        Assert.That(tensor.IsNull(0), Is.False, "Position 0: 0.0 should not be null");
        Assert.That(tensor[0], Is.EqualTo(0.0f), "Position 0: value should be 0.0");

        Assert.That(tensor.IsNull(1), Is.True, "Position 1: should be null (explicit)");

        // NaN is a valid value, not a null sentinel
        Assert.That(tensor.IsNull(2), Is.False, "Position 2: NaN should not be treated as null");
        Assert.That(float.IsNaN(tensor[2]), Is.True, "Position 2: value should be NaN");

        Assert.That(tensor.IsNull(3), Is.True, "Position 3: should be null (explicit)");

        Assert.That(tensor.IsNull(4), Is.False, "Position 4: -1.0 should not be null");
        Assert.That(tensor[4], Is.EqualTo(-1.0f), "Position 4: value should be -1.0");
    }

    #endregion
}
