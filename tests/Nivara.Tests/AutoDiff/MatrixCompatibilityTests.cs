using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class MatrixCompatibilityTests
{
#pragma warning disable CS0618
    [Test]
    public void MatMul_ShapeAwareAndLegacyOverloads_ProduceEquivalentValuesAndShapes()
    {
        var cases = new[]
        {
            new MatrixCase(
                [1f, 2f, 3f, 4f],
                [5f, 6f, 7f, 8f],
                2,
                2,
                2),
            new MatrixCase(
                [1f, 2f, 3f, 4f, 5f, 6f],
                [7f, 8f, 9f, 10f, 11f, 12f],
                2,
                3,
                2),
            new MatrixCase(
                [1f, 2f, 3f],
                [4f, 5f, 6f],
                1,
                3,
                1)
        };

        foreach (var testCase in cases)
        {
            var shapeAware = ReverseGradOperations.MatMul(
                Matrix(testCase.Left, testCase.LeftRows, testCase.Inner, requiresGrad: true),
                Matrix(testCase.Right, testCase.Inner, testCase.RightCols, requiresGrad: true));

            var legacy = ReverseGradOperations.MatMul(
                Vector(testCase.Left, requiresGrad: true),
                Vector(testCase.Right, requiresGrad: true),
                testCase.LeftRows,
                testCase.Inner,
                testCase.RightCols);

            AssertTensorEquivalent(shapeAware, legacy);
        }
    }

    [Test]
    public void MatMul_ShapeAwareAndLegacyOverloads_ProduceEquivalentNullMasks()
    {
        var left = new float?[] { 1f, null, 3f, 4f };
        var right = new float?[] { 5f, 6f, null, 8f };

        var shapeAware = ReverseGradOperations.MatMul(
            Matrix(left, 2, 2, requiresGrad: true),
            Matrix(right, 2, 2, requiresGrad: true));

        var legacy = ReverseGradOperations.MatMul(
            Vector(left, requiresGrad: true),
            Vector(right, requiresGrad: true),
            2,
            2,
            2);

        AssertTensorEquivalent(shapeAware, legacy);
    }

    [Test]
    public void MatMul_ShapeAwareAndLegacyOverloads_ProduceEquivalentGradients()
    {
        var left = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var right = new float[] { 7f, 8f, 9f, 10f, 11f, 12f };

        var shapeAwareLeft = Matrix(left, 2, 3, requiresGrad: true);
        var shapeAwareRight = Matrix(right, 3, 2, requiresGrad: true);
        var shapeAware = ReverseGradOperations.MatMul(shapeAwareLeft, shapeAwareRight);

        var legacyLeft = Vector(left, requiresGrad: true);
        var legacyRight = Vector(right, requiresGrad: true);
        var legacy = ReverseGradOperations.MatMul(legacyLeft, legacyRight, 2, 3, 2);

        var shapeAwareGrad = Matrix([1f, 1f, 1f, 1f], 2, 2, requiresGrad: false);
        var legacyGrad = Matrix([1f, 1f, 1f, 1f], 2, 2, requiresGrad: false);

        shapeAware.Backward(shapeAwareGrad);
        legacy.Backward(legacyGrad);

        AssertColumnEquivalent(shapeAwareLeft.Grad, legacyLeft.Grad);
        AssertColumnEquivalent(shapeAwareRight.Grad, legacyRight.Grad);
    }

    [Test]
    public void Transpose_ShapeAwareAndLegacyOverloads_ProduceEquivalentValuesMasksAndShapes()
    {
        var cases = new[]
        {
            new TransposeCase(new float?[] { 1f, null, 3f, 4f, 5f, 6f }, 2, 3),
            new TransposeCase(new float?[] { 1f, 2f, 3f, null, 5f, 6f }, 3, 2)
        };

        foreach (var testCase in cases)
        {
            var shapeAware = ReverseGradOperations.Transpose(
                Matrix(testCase.Values, testCase.Rows, testCase.Cols, requiresGrad: true));

            var legacy = ReverseGradOperations.Transpose(
                Vector(testCase.Values, requiresGrad: true),
                testCase.Rows,
                testCase.Cols);

            AssertTensorEquivalent(shapeAware, legacy);
        }
    }

    [Test]
    public void Transpose_ShapeAwareAndLegacyOverloads_ProduceEquivalentGradients()
    {
        var values = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var shapeAwareInput = Matrix(values, 2, 3, requiresGrad: true);
        var legacyInput = Vector(values, requiresGrad: true);

        var shapeAware = ReverseGradOperations.Transpose(shapeAwareInput);
        var legacy = ReverseGradOperations.Transpose(legacyInput, 2, 3);

        var shapeAwareGrad = Matrix([1f, 2f, 3f, 4f, 5f, 6f], 3, 2, requiresGrad: false);
        var legacyGrad = Matrix([1f, 2f, 3f, 4f, 5f, 6f], 3, 2, requiresGrad: false);

        shapeAware.Backward(shapeAwareGrad);
        legacy.Backward(legacyGrad);

        AssertColumnEquivalent(shapeAwareInput.Grad, legacyInput.Grad);
    }
#pragma warning restore CS0618

    static ReverseGradTensor<float> Vector(float[] values, bool requiresGrad)
        => new(NivaraColumn<float>.Create(values), requiresGrad);

    static ReverseGradTensor<float> Vector(float?[] values, bool requiresGrad)
        => new(NivaraColumn<float>.CreateFromNullable(values), requiresGrad);

    static ReverseGradTensor<float> Matrix(float[] values, int rows, int cols, bool requiresGrad)
    {
        var tensor = Vector(values, requiresGrad);
        tensor.Reshape(rows, cols);
        return tensor;
    }

    static ReverseGradTensor<float> Matrix(float?[] values, int rows, int cols, bool requiresGrad)
    {
        var tensor = Vector(values, requiresGrad);
        tensor.Reshape(rows, cols);
        return tensor;
    }

    static void AssertTensorEquivalent(ReverseGradTensor<float> expected, ReverseGradTensor<float> actual)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        Assert.That(actual.Shape, Is.EqualTo(expected.Shape));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.That(actual.IsNull(i), Is.EqualTo(expected.IsNull(i)), $"Null mask differs at index {i}.");
            if (!expected.IsNull(i))
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"Value differs at index {i}.");
        }
    }

    static void AssertColumnEquivalent(NivaraColumn<float>? expected, NivaraColumn<float>? actual)
    {
        Assert.That(actual, Is.Not.Null);
        Assert.That(expected, Is.Not.Null);
        Assert.That(actual!.Length, Is.EqualTo(expected!.Length));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.That(actual.IsNull(i), Is.EqualTo(expected.IsNull(i)), $"Gradient null mask differs at index {i}.");
            if (!expected.IsNull(i))
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"Gradient value differs at index {i}.");
        }
    }

    readonly record struct MatrixCase(float[] Left, float[] Right, int LeftRows, int Inner, int RightCols);

    readonly record struct TransposeCase(float?[] Values, int Rows, int Cols);
}
