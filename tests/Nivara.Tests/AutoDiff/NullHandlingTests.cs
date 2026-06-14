using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for null handling in automatic differentiation operations.
/// Verifies that ReverseGradTensor operations preserve Nivara's explicit null semantics.
/// Tests Requirements 5.5 and 9.4.
/// </summary>
[TestFixture]
public class NullHandlingTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    #region Element-wise Operations Null Handling

    [Test]
    public void Add_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange: Create columns with null values using Nivara's explicit null semantics
        var aValues = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var bValues = new float?[] { 2.0f, 4.0f, null, null, 6.0f };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Add(a, b);

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

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Multiply(a, b);

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

        var a = new ReverseGradTensor<double>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<double>(bColumn, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Subtract(a, b);

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

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Divide(a, b);

        // Assert: Verify null propagation (null / anything = null)
        Assert.That(result.IsNull(0), Is.False, "Position 0: 12 / 3 should not be null");
        Assert.That(result[0], Is.EqualTo(4.0f), "Position 0: 12 / 3 = 4");

        Assert.That(result.IsNull(1), Is.True, "Position 1: null / 5 should be null");
        Assert.That(result.IsNull(2), Is.True, "Position 2: 18 / null should be null");
    }

    #endregion

    #region Matrix Operations Null Handling

    [Test]
    public void MatMul_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange: 2x2 matrix multiplication with nulls
        // A = [[1, null], [3, 4]] (flattened: [1, null, 3, 4])
        // B = [[5, 6], [null, 8]] (flattened: [5, 6, null, 8])
        // Expected: result[i,j] is null if any contributing a[i,k] or b[k,j] is null
        var aValues = new float?[] { 1.0f, null, 3.0f, 4.0f };
        var bValues = new float?[] { 5.0f, 6.0f, null, 8.0f };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);
        a.Reshape(2, 2);
        b.Reshape(2, 2);

        // Act
        var result = ReverseGradOperations.MatMul(a, b);

        // Assert: result[0] = 1*5 + null*7 = null (k=1 has null in a)
        Assert.That(result.IsNull(0), Is.True, "Position 0: a[1]=null -> result should be null");

        // result[1] = 1*6 + null*8 = null (k=1 has null in a)
        Assert.That(result.IsNull(1), Is.True, "Position 1: a[1]=null -> result should be null");

        // result[2] = 3*5 + 4*null = null (k=1 has null in b)
        Assert.That(result.IsNull(2), Is.True, "Position 2: b[2]=null -> result should be null");

        // result[3] = 3*6 + 4*8 = 18 + 32 = 50 (both non-null)
        Assert.That(result.IsNull(3), Is.False, "Position 3: all non-null -> result should not be null");
        Assert.That(result[3], Is.EqualTo(50.0f), "Position 3: 3*6 + 4*8 = 50");
    }

    [Test]
    public void MatMul_AllNullValues_ProducesAllNullResult()
    {
        // Arrange: Both matrices fully null
        var aValues = new float?[] { null, null, null, null };
        var bValues = new float?[] { null, null, null, null };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);
        a.Reshape(2, 2);
        b.Reshape(2, 2);

        // Act
        var result = ReverseGradOperations.MatMul(a, b);

        // Assert: All positions should be null
        for (int i = 0; i < result.Length; i++)
        {
            Assert.That(result.IsNull(i), Is.True, $"Position {i} should be null (all inputs null)");
        }
    }

    [Test]
    public void Transpose_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange: 2x3 matrix with nulls
        // A = [[1, null, 3], [4, 5, null]] (flattened: [1, null, 3, 4, 5, null])
        // Transpose -> [[1, 4], [null, 5], [3, null]] (flattened: [1, 4, null, 5, 3, null])
        var values = new float?[] { 1.0f, null, 3.0f, 4.0f, 5.0f, null };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var a = new ReverseGradTensor<float>(column, requiresGrad: true);
        a.Reshape(2, 3);

        // Act
        var result = ReverseGradOperations.Transpose(a);

        // Assert: result[j * rows + i] = a[i * cols + j]
        Assert.That(result.IsNull(0), Is.False, "Position 0: from a[0]=1 should not be null");
        Assert.That(result[0], Is.EqualTo(1.0f), "Position 0: transpose[0,0] = a[0,0] = 1");

        Assert.That(result.IsNull(1), Is.False, "Position 1: from a[3]=4 should not be null");
        Assert.That(result[1], Is.EqualTo(4.0f), "Position 1: transpose[0,1] = a[1,0] = 4");

        Assert.That(result.IsNull(2), Is.True, "Position 2: from a[1]=null should be null");
        Assert.That(result.IsNull(3), Is.False, "Position 3: from a[4]=5 should not be null");
        Assert.That(result[3], Is.EqualTo(5.0f), "Position 3: transpose[1,1] = a[1,1] = 5");

        Assert.That(result.IsNull(4), Is.False, "Position 4: from a[2]=3 should not be null");
        Assert.That(result[4], Is.EqualTo(3.0f), "Position 4: transpose[2,0] = a[0,2] = 3");

        Assert.That(result.IsNull(5), Is.True, "Position 5: from a[5]=null should be null");
    }

    [Test]
    public void MatMul_WithNulls_BackwardComputesGradientsCorrectly()
    {
        // Arrange: 2x2 matrix multiplication with nulls
        var aValues = new float?[] { 1.0f, null, 3.0f, 4.0f };
        var bValues = new float?[] { 5.0f, 6.0f, 7.0f, 8.0f };

        var aColumn = NivaraColumn<float>.CreateFromNullable(aValues);
        var bColumn = NivaraColumn<float>.CreateFromNullable(bValues);

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);
        a.Reshape(2, 2);
        b.Reshape(2, 2);

        // Act
        var result = ReverseGradOperations.MatMul(a, b);

        // Create gradient output (all ones)
        var gradValues = new float?[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var gradColumn = NivaraColumn<float>.CreateFromNullable(gradValues);
        var grad = new ReverseGradTensor<float>(gradColumn, requiresGrad: false);
        grad.Reshape(2, 2);
        result.Backward(grad);

        // Assert: Gradients should exist
        Assert.That(a.Grad, Is.Not.Null, "Gradient for a should exist");
        Assert.That(b.Grad, Is.Not.Null, "Gradient for b should exist");
    }

    #endregion

    #region Activation Functions Null Handling

    [Test]
    public void Relu_WithNullValues_PropagatesNullsCorrectly()
    {
        // Arrange
        var values = new float?[] { -2.0f, null, 0.0f, null, 2.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Relu(tensor);

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
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Sigmoid(tensor);

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
        var tensor = new ReverseGradTensor<double>(column, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Tanh(tensor);

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

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);

        // Act: Forward pass
        var result = ReverseGradOperations.Add(a, b);

        // Create gradient output (all ones for non-null positions)
        var gradValues = new float?[] { 1.0f, 1.0f, 1.0f };
        var gradColumn = NivaraColumn<float>.CreateFromNullable(gradValues);
        var grad = new ReverseGradTensor<float>(gradColumn, requiresGrad: false);

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
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act: Forward pass
        var result = ReverseGradOperations.Relu(tensor);

        // Create gradient output (all ones)
        var gradValues = new float?[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var gradColumn = NivaraColumn<float>.CreateFromNullable(gradValues);
        var grad = new ReverseGradTensor<float>(gradColumn, requiresGrad: false);

        result.Backward(grad);

        // Assert: Gradients should be computed correctly
        Assert.That(tensor.Grad, Is.Not.Null, "Gradient should exist");

        // Position 0: input -1 <= 0, gradient should be 0
        Assert.That(tensor.Grad![0], Is.EqualTo(0.0f), "Gradient for negative input should be 0");

        // Position 1: input is null, gradient should be 0 (nulls stripped by default)
        Assert.That(tensor.Grad![1], Is.EqualTo(0.0f), "Gradient for null input should be 0");

        // Position 2: input 1 > 0, gradient should be 1
        Assert.That(tensor.Grad[2], Is.EqualTo(1.0f), "Gradient for positive input should be 1");

        // Position 3: input 2 > 0, gradient should be 1
        Assert.That(tensor.Grad[3], Is.EqualTo(1.0f), "Gradient for positive input should be 1");
    }

    [Test]
    public void ActivationBackward_WithNullGradientOutput_StripsNullMask()
    {
        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f }),
            requiresGrad: true);
        var result = ReverseGradOperations.Sigmoid(tensor);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 1.0f }),
            requiresGrad: false);

        result.Backward(grad);

        Assert.That(tensor.Grad, Is.Not.Null);
        Assert.That(tensor.Grad!.IsNull(0), Is.False);
        Assert.That(tensor.Grad[1], Is.EqualTo(0.0f), "Null in upstream gradient should become 0");
        Assert.That(tensor.Grad.IsNull(2), Is.False);
    }

    [Test]
    public void SumBackward_WithNullGradientOutput_BroadcastsAsZeros()
    {
        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }),
            requiresGrad: true);
        var sum = ReverseGradOperations.Sum(tensor);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { null }),
            requiresGrad: false);

        sum.Backward(grad);

        Assert.That(tensor.Grad, Is.Not.Null);
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.That(tensor.Grad![i], Is.EqualTo(0.0f), $"Null gradient broadcast should produce 0 at position {i}");
        }
    }

    [Test]
    public void Backward_WithStripGradientNullsFalse_PreservesOriginalNullBehavior()
    {
        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f }),
            requiresGrad: true);
        var result = ReverseGradOperations.Sigmoid(tensor);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 1.0f }),
            requiresGrad: false);

        result.Backward(grad, stripGradientNulls: false);

        Assert.That(tensor.Grad, Is.Not.Null);
        Assert.That(tensor.Grad!.IsNull(0), Is.False);
        Assert.That(tensor.Grad.IsNull(1), Is.True, "With stripGradientNulls=false, null should propagate");
        Assert.That(tensor.Grad.IsNull(2), Is.False);
    }

    #endregion

    #region Null Mask Preservation

    [Test]
    public void ReverseGradTensor_PreservesNullMaskFromNivaraColumn()
    {
        // Arrange
        var values = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);

        // Act
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

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

        var a = new ReverseGradTensor<float>(aColumn, requiresGrad: true);
        var b = new ReverseGradTensor<float>(bColumn, requiresGrad: true);

        // Act: Perform multiple operations
        var add = ReverseGradOperations.Add(a, b);
        var mul = ReverseGradOperations.Multiply(a, b);

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
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Sum(tensor);

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
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var result = ReverseGradOperations.Mean(tensor);

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

        var x = new ReverseGradTensor<float>(xColumn, requiresGrad: false);
        var w = new ReverseGradTensor<float>(wColumn, requiresGrad: true);

        // Act: Forward pass with nulls
        var mul = ReverseGradOperations.Multiply(x, w);  // [0.5, null, null]
        var relu = ReverseGradOperations.Relu(mul);      // [0.5, null, null]

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
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);

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

    [Test]
    public void NullSeedGradient_ProducesNullFreeAccumulatedGradients()
    {
        // All non-null inputs avoid mixed-storage issues in backward intermediates.
        // Nulls enter only via the seed gradient, testing AccumulateGradient's stripping.
        var a = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }),
            requiresGrad: true);
        var b = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 4f, 5f, 6f }),
            requiresGrad: true);

        var result = ReverseGradOperations.Add(a, b);

        // Seed gradient with null at position 1
        var seedGrad = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 1f }),
            requiresGrad: false);

        result.Backward(seedGrad);

        Assert.That(a.Grad, Is.Not.Null);
        Assert.That(a.Grad.HasNulls, Is.False,
            "Gradient for a should have no nulls (stripped by default)");
        Assert.That(b.Grad, Is.Not.Null);
        Assert.That(b.Grad.HasNulls, Is.False,
            "Gradient for b should have no nulls (stripped by default)");
    }

    #endregion
}
