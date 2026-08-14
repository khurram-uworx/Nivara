using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class ForwardGradOperationsTests
{
    #region Factory Method Tests

    [Test]
    public void FromArray_Basic_CreatesCorrectValues()
    {
        var tensor = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        Assert.That(tensor.Length, Is.EqualTo(3));
        Assert.That(tensor[0], Is.EqualTo(1.0f));
        Assert.That(tensor[1], Is.EqualTo(2.0f));
        Assert.That(tensor[2], Is.EqualTo(3.0f));
        Assert.That(tensor.RequiresTangent, Is.False);
        Assert.That(tensor.Tangent, Is.Null);
    }

    [Test]
    public void FromArray_WithTangent_CreatesCorrectTangent()
    {
        var tensor = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f });

        Assert.That(tensor.RequiresTangent, Is.True);
        Assert.That(tensor.Tangent, Is.Not.Null);
        Assert.That(tensor.Tangent![0], Is.EqualTo(3.0f));
        Assert.That(tensor.Tangent[1], Is.EqualTo(4.0f));
    }

    [Test]
    public void FromArray_NullData_Throws()
    {
        Assert.That(() => ForwardGradTensor<float>.FromArray(null!),
            Throws.ArgumentNullException);
    }

    [Test]
    public void FromArray_LengthMismatch_Throws()
    {
        Assert.That(() => ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f }, new float[] { 3f }),
            Throws.InstanceOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void FromMatrix_CreatesCorrectShape()
    {
        var tensor = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);

        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(tensor.Length, Is.EqualTo(6));
        Assert.That(tensor.RequiresTangent, Is.False);
    }

    [Test]
    public void FromMatrix_WithTangent_SetsRequiresTangent()
    {
        var tensor = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f }, 2, 2, new float[] { 0f, 1f, 0f, 1f });

        Assert.That(tensor.RequiresTangent, Is.True);
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void FromMatrix_DataLengthMismatch_Throws()
    {
        Assert.That(() => ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f }, 2, 2),
            Throws.ArgumentException);
    }

    [Test]
    public void FromMatrix_NullData_Throws()
    {
        Assert.That(() => ForwardGradTensor<float>.FromMatrix(null!, 2, 2),
            Throws.ArgumentNullException);
    }

    [Test]
    public void DefaultShape_IsOneDimensional()
    {
        var tensor = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        Assert.That(tensor.Rank, Is.EqualTo(1));
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 3 }));
    }

    [Test]
    public void Reshape_ChangesShape()
    {
        var tensor = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f });

        tensor.Reshape(2, 3);

        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 2, 3 }));
    }

    #endregion

    #region Element-wise Operations

    [Test]
    public void Add_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f }, new float[] { 1.0f, 1.0f, 1.0f });
        var b = ForwardGradTensor<float>.FromArray(
            new float[] { 4.0f, 5.0f, 6.0f }, new float[] { 2.0f, 2.0f, 2.0f });

        var result = ForwardGradOperations.Add(a, b);

        Assert.That(result[0], Is.EqualTo(5.0f));
        Assert.That(result[1], Is.EqualTo(7.0f));
        Assert.That(result[2], Is.EqualTo(9.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(3.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(3.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void Add_WithoutTangent_DoesNotPropagateTangent()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 3.0f, 4.0f });

        var result = ForwardGradOperations.Add(a, b);

        Assert.That(result[0], Is.EqualTo(4.0f));
        Assert.That(result[1], Is.EqualTo(6.0f));
        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result.Tangent, Is.Null);
    }

    [Test]
    public void Add_OneTangentOnly_PropagatesCorrectly()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f }, new float[] { 5.0f, 5.0f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 3.0f, 4.0f });

        var result = ForwardGradOperations.Add(a, b);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(5.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(5.0f));
    }

    [Test]
    public void Subtract_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 10.0f, 8.0f, 6.0f }, new float[] { 1.0f, 1.0f, 1.0f });
        var b = ForwardGradTensor<float>.FromArray(
            new float[] { 3.0f, 2.0f, 1.0f }, new float[] { 2.0f, 2.0f, 2.0f });

        var result = ForwardGradOperations.Subtract(a, b);

        Assert.That(result[0], Is.EqualTo(7.0f));
        Assert.That(result[1], Is.EqualTo(6.0f));
        Assert.That(result[2], Is.EqualTo(5.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(-1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(-1.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(-1.0f));
    }

    [Test]
    public void Multiply_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: t_out = t_a * b + a * t_b
        // a=[2,3,4], b=[5,6,7], t_a=[1,1,1], t_b=[1,1,1]
        // tangent = [1*5+2*1, 1*6+3*1, 1*7+4*1] = [7, 9, 11]
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 2.0f, 3.0f, 4.0f }, new float[] { 1.0f, 1.0f, 1.0f });
        var b = ForwardGradTensor<float>.FromArray(
            new float[] { 5.0f, 6.0f, 7.0f }, new float[] { 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.Multiply(a, b);

        Assert.That(result[0], Is.EqualTo(10.0f));
        Assert.That(result[1], Is.EqualTo(18.0f));
        Assert.That(result[2], Is.EqualTo(28.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(7.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(9.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(11.0f));
    }

    [Test]
    public void Divide_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: t_out = (t_a - result * t_b) / b
        // a=[12,15], b=[3,5], t_a=[1,1], t_b=[2,2]
        // result=[4,3]
        // tangent[0] = (1 - 4*2) / 3 = -7/3 ≈ -2.3333
        // tangent[1] = (1 - 3*2) / 5 = -5/5 = -1.0
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 12.0f, 15.0f }, new float[] { 1.0f, 1.0f });
        var b = ForwardGradTensor<float>.FromArray(
            new float[] { 3.0f, 5.0f }, new float[] { 2.0f, 2.0f });

        var result = ForwardGradOperations.Divide(a, b);

        Assert.That(result[0], Is.EqualTo(4.0f));
        Assert.That(result[1], Is.EqualTo(3.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(-7.0f / 3.0f).Within(1e-6f));
        Assert.That(result.Tangent[1], Is.EqualTo(-1.0f));
    }

    [Test]
    public void Divide_ByZero_ThrowsException()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 0.0f, 1.0f });

        Assert.Throws<DivideByZeroException>(() => ForwardGradOperations.Divide(a, b));
    }

    [Test]
    public void DivideScalar_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: t_out = t_a / scalar
        // a=[12,15], scalar=3, t_a=[1,1]
        // result=[4,5], tangent=[1/3,1/3]
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 12.0f, 15.0f }, new float[] { 1.0f, 1.0f });

        var result = ForwardGradOperations.DivideScalar(a, 3f);

        Assert.That(result[0], Is.EqualTo(4.0f));
        Assert.That(result[1], Is.EqualTo(5.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f / 3.0f).Within(1e-6f));
        Assert.That(result.Tangent[1], Is.EqualTo(1.0f / 3.0f).Within(1e-6f));
    }

    [Test]
    public void DivideScalar_ByZero_ThrowsException()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f });

        Assert.Throws<DivideByZeroException>(() => ForwardGradOperations.DivideScalar(a, 0f));
    }

    [Test]
    public void ElementWiseOperation_LengthMismatch_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 3.0f });

        Assert.That(() => ForwardGradOperations.Add(a, b), Throws.ArgumentException);
        Assert.That(() => ForwardGradOperations.Multiply(a, b), Throws.ArgumentException);
        Assert.That(() => ForwardGradOperations.Divide(a, b), Throws.ArgumentException);
    }

    #endregion

    #region Matrix Operations

    [Test]
    public void MatMul_Simple_ComputesCorrectValuesAndTangents()
    {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A @ B = [[19, 22], [43, 50]]
        // JVP: t_a @ B + A @ t_b
        // t_a = [[1, 0], [0, 0]], t_b = [[0, 0], [0, 0]]
        // tangent = t_a @ B = [[1*5+0*7, 1*6+0*8], [0*5+0*7, 0*6+0*8]] = [[5, 6], [0, 0]]
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f }, 2, 2,
            new float[] { 1f, 0f, 0f, 0f });
        var b = ForwardGradTensor<float>.FromMatrix(
            new float[] { 5f, 6f, 7f, 8f }, 2, 2);

        var result = ForwardGradOperations.MatMul(a, b);

        Assert.That(result[0], Is.EqualTo(19.0f));
        Assert.That(result[1], Is.EqualTo(22.0f));
        Assert.That(result[2], Is.EqualTo(43.0f));
        Assert.That(result[3], Is.EqualTo(50.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(5.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(6.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
    }

    [Test]
    public void MatMul_ResultHasCorrectShape()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);
        var b = ForwardGradTensor<float>.FromMatrix(new float[12], 3, 4);

        var result = ForwardGradOperations.MatMul(a, b);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 4 }));
    }

    [Test]
    public void MatMul_IncorrectRank_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 3f, 4f });

        var ex = Assert.Throws<ArgumentException>(() => ForwardGradOperations.MatMul(a, b));
        Assert.That(ex.Message, Does.Contain("rank 2"));
    }

    [Test]
    public void Transpose_Simple_ComputesCorrectValuesAndTangents()
    {
        // A = [[1, 2, 3], [4, 5, 6]], t_a = [[1, 0, 0], [0, 0, 0]]
        // Transpose(A) = [[1, 4], [2, 5], [3, 6]]
        // Transpose(t_a) = [[1, 0], [0, 0], [0, 0]]
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3,
            new float[] { 1f, 0f, 0f, 0f, 0f, 0f });

        var result = ForwardGradOperations.Transpose(a);

        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(4.0f));
        Assert.That(result[2], Is.EqualTo(2.0f));
        Assert.That(result[3], Is.EqualTo(5.0f));
        Assert.That(result[4], Is.EqualTo(3.0f));
        Assert.That(result[5], Is.EqualTo(6.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[4], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[5], Is.EqualTo(0.0f));
    }

    [Test]
    public void Transpose_ResultHasCorrectShape()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);

        var result = ForwardGradOperations.Transpose(a);

        Assert.That(result.Shape, Is.EqualTo(new[] { 3, 2 }));
    }

    #endregion

    #region Reduction Operations

    [Test]
    public void Sum_Simple_ComputesCorrectResult()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

        var result = ForwardGradOperations.Sum(a);

        Assert.That(result.Length, Is.EqualTo(1));
        Assert.That(result[0], Is.EqualTo(10.0f));
    }

    [Test]
    public void Sum_WithTangent_ComputesCorrectTangent()
    {
        // sum([1,2,3]) = 6
        // JVP: sum(t_a) = sum([2,3,4]) = 9
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f }, new float[] { 2.0f, 3.0f, 4.0f });

        var result = ForwardGradOperations.Sum(a);

        Assert.That(result[0], Is.EqualTo(6.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(9.0f));
    }

    [Test]
    public void Sum_EmptyTensor_ThrowsException()
    {
        var a = ForwardGradTensor<float>.FromArray(Array.Empty<float>());

        Assert.Throws<InvalidOperationException>(() => ForwardGradOperations.Sum(a));
    }

    [Test]
    public void Mean_Simple_ComputesCorrectResult()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 2.0f, 4.0f, 6.0f, 8.0f });

        var result = ForwardGradOperations.Mean(a);

        Assert.That(result[0], Is.EqualTo(5.0f));
    }

    [Test]
    public void Mean_WithTangent_ComputesCorrectTangent()
    {
        // mean([2,4]) = 3, t_a = [2,6]
        // JVP: sum(t_a) / n = (2+6) / 2 = 4
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 2.0f, 4.0f }, new float[] { 2.0f, 6.0f });

        var result = ForwardGradOperations.Mean(a);

        Assert.That(result[0], Is.EqualTo(3.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(4.0f));
    }

    [Test]
    public void Mean_EmptyTensor_ThrowsException()
    {
        var a = ForwardGradTensor<float>.FromArray(Array.Empty<float>());

        Assert.Throws<InvalidOperationException>(() => ForwardGradOperations.Mean(a));
    }

    #endregion

    #region Activation Functions

    [Test]
    public void Relu_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f },
            new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.Relu(a);

        Assert.That(result[0], Is.EqualTo(0.0f));
        Assert.That(result[1], Is.EqualTo(0.0f));
        Assert.That(result[2], Is.EqualTo(0.0f));
        Assert.That(result[3], Is.EqualTo(1.0f));
        Assert.That(result[4], Is.EqualTo(2.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[4], Is.EqualTo(1.0f));
    }

    [Test]
    public void Sigmoid_Simple_ComputesCorrectValuesAndTangents()
    {
        // σ(0) = 0.5, σ'(0) = 0.25
        // JVP: σ(0) * (1 - σ(0)) * t_a = 0.25 * 2 = 0.5
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 0.0f }, new float[] { 2.0f });

        var result = ForwardGradOperations.Sigmoid(a);

        Assert.That(result[0], Is.EqualTo(0.5f).Within(0.0001f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.5f).Within(0.0001f));
    }

    [Test]
    public void Tanh_Simple_ComputesCorrectValuesAndTangents()
    {
        // tanh(0) = 0, tanh'(0) = 1 - 0² = 1
        // JVP: (1 - 0²) * 3 = 3
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 0.0f }, new float[] { 3.0f });

        var result = ForwardGradOperations.Tanh(a);

        Assert.That(result[0], Is.EqualTo(0.0f).Within(0.0001f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(3.0f).Within(0.0001f));
    }

    [Test]
    public void Negate_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, -2.0f, 3.0f },
            new float[] { 2.0f, 3.0f, 4.0f });

        var result = ForwardGradOperations.Negate(a);

        Assert.That(result[0], Is.EqualTo(-1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[2], Is.EqualTo(-3.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(-2.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(-3.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(-4.0f));
    }

    [Test]
    public void Abs_Simple_ComputesCorrectValuesAndTangents()
    {
        // |x|' = sign(x)
        // JVP: sign(a) * t_a
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { -2.0f, 0.0f, 3.0f },
            new float[] { 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.Abs(a);

        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(0.0f));
        Assert.That(result[2], Is.EqualTo(3.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(-1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0f));
    }

    [Test]
    public void Abs_DoubleType_ComputesCorrectly()
    {
        var a = ForwardGradTensor<double>.FromArray(
            new double[] { -3.0, 0.0, 5.0 },
            new double[] { 1.0, 1.0, 1.0 });

        var result = ForwardGradOperations.Abs(a);

        Assert.That(result[0], Is.EqualTo(3.0));
        Assert.That(result[1], Is.EqualTo(0.0));
        Assert.That(result[2], Is.EqualTo(5.0));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(-1.0));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0));
    }

    [Test]
    public void Clip_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: (a in [min,max]) ? t_a : 0
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { -2.0f, 0.5f, 1.0f, 3.0f },
            new float[] { 1.0f, 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.Clip(a, -1.0f, 2.0f);

        Assert.That(result[0], Is.EqualTo(-1.0f));
        Assert.That(result[1], Is.EqualTo(0.5f));
        Assert.That(result[2], Is.EqualTo(1.0f));
        Assert.That(result[3], Is.EqualTo(2.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
    }

    [Test]
    public void LeakyRelu_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: (a > 0) ? t_a : α * t_a
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { -1.0f, 0.0f, 1.0f },
            new float[] { 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.LeakyRelu(a, 0.01f);

        Assert.That(result[0], Is.EqualTo(-0.01f).Within(1e-6f));
        Assert.That(result[1], Is.EqualTo(0.0f));
        Assert.That(result[2], Is.EqualTo(1.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.01f).Within(1e-6f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.01f).Within(1e-6f));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0f));
    }

    [Test]
    public void Exp_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: e^a * t_a = result * t_a
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 0.0f, 1.0f },
            new float[] { 2.0f, 3.0f });

        var result = ForwardGradOperations.Exp(a);

        Assert.That(result[0], Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(result[1], Is.EqualTo(2.71828f).Within(0.001f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(2.0f).Within(1e-6f));
        Assert.That(result.Tangent[1], Is.EqualTo(3.0f * 2.71828f).Within(0.01f));
    }

    [Test]
    public void Log_Simple_ComputesCorrectValuesAndTangents()
    {
        // JVP: t_a / a
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.71828f },
            new float[] { 2.0f, 3.0f });

        var result = ForwardGradOperations.Log(a);

        Assert.That(result[0], Is.EqualTo(0.0f).Within(1e-6f));
        Assert.That(result[1], Is.EqualTo(1.0f).Within(0.001f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(2.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(3.0f / 2.71828f).Within(0.01f));
    }

    [Test]
    public void Softmax_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.Softmax(a);

        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
        var sum = result[0] + result[1] + result[2];
        Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f));
    }

    [Test]
    public void LogSoftmax_Simple_ComputesCorrectValuesAndTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 1.0f, 1.0f, 1.0f });

        var result = ForwardGradOperations.LogSoftmax(a);

        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
    }

    [Test]
    public void Operations_WithoutTangents_DoNotRequireTangent()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 3.0f, 4.0f });

        Assert.That(ForwardGradOperations.Relu(a).RequiresTangent, Is.False);
        Assert.That(ForwardGradOperations.Sigmoid(a).RequiresTangent, Is.False);
        Assert.That(ForwardGradOperations.Tanh(a).RequiresTangent, Is.False);
        Assert.That(ForwardGradOperations.Negate(a).RequiresTangent, Is.False);
        Assert.That(ForwardGradOperations.Add(a, b).RequiresTangent, Is.False);
        Assert.That(ForwardGradOperations.Multiply(a, b).RequiresTangent, Is.False);
    }

    #endregion

    #region Dropout

    [Test]
    public void DropoutWithMask_SimpleCase_AppliesMask()
    {
        var input = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f, 4.0f },
            new float[] { 10.0f, 20.0f, 30.0f, 40.0f });
        var keepMask = new bool[] { true, false, true, false };
        float scale = 2.0f;

        var result = ForwardGradOperations.DropoutWithMask(input, keepMask, scale);

        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(0.0f));
        Assert.That(result[2], Is.EqualTo(6.0f));
        Assert.That(result[3], Is.EqualTo(0.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(20.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(60.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
    }

    [Test]
    public void Dropout_ProbabilityZero_ReturnsInput()
    {
        var input = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f },
            new float[] { 3.0f, 4.0f });

        var result = ForwardGradOperations.Dropout(input, 0.0, isTraining: true);

        Assert.That(result, Is.SameAs(input));
    }

    #endregion

    #region VAE Operations

    [Test]
    public void KlDivergence_ZeroMeanUnitVar_ReturnsZero()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 0f, 0f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f, 0f });

        var kl = ForwardGradOperations.KlDivergence(mean, logVar);

        Assert.That(kl.Length, Is.EqualTo(1));
        Assert.That(kl[0], Is.EqualTo(0f).Within(1e-6f));
        Assert.That(kl.RequiresTangent, Is.False);
    }

    [Test]
    public void KlDivergence_NonZeroMean_ComputesCorrectValue()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 1f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f });

        var kl = ForwardGradOperations.KlDivergence(mean, logVar);

        Assert.That(kl[0], Is.EqualTo(0.5f).Within(1e-6f));
    }

    [Test]
    public void KlDivergence_WithTangent_ComputesCorrectTangent()
    {
        // KL = -0.5 * Σ(1 + logVar - μ² - exp(logVar))
        // μ=[1,2], logVar=[0,1], t_μ=[1,1], t_logVar=[1,1]
        // ∂KL/∂μ = μ,  ∂KL/∂logVar = -0.5*(1 - exp(logVar))
        // JVP = sum(μ * t_μ) + sum(0.5 * (exp(logVar) - 1) * t_logVar)
        //     = (1*1 + 2*1) + 0.5*((exp(0)-1)*1 + (exp(1)-1)*1)
        //     = 3 + 0.5*((1-1) + (2.718-1))
        //     = 3 + 0.5*1.718 = 3 + 0.859 = 3.859
        var mean = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f }, new float[] { 1f, 1f });
        var logVar = ForwardGradTensor<float>.FromArray(
            new float[] { 0f, 1f }, new float[] { 1f, 1f });

        var kl = ForwardGradOperations.KlDivergence(mean, logVar);

        Assert.That(kl.Length, Is.EqualTo(1));
        Assert.That(kl.RequiresTangent, Is.True);
        var expected = 3.0f + 0.5f * (0.0f + 1.71828f);
        Assert.That(kl.Tangent![0], Is.EqualTo(expected).Within(0.01f));
    }

    [Test]
    public void KlDivergence_DifferentLengths_Throws()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f });

        Assert.That(() => ForwardGradOperations.KlDivergence(mean, logVar), Throws.ArgumentException);
    }

    [Test]
    public void KlDivergence_DoubleType_ComputesCorrectly()
    {
        var mean = ForwardGradTensor<double>.FromArray(new double[] { 1.0 });
        var logVar = ForwardGradTensor<double>.FromArray(new double[] { 0.0 });

        var kl = ForwardGradOperations.KlDivergence(mean, logVar);

        Assert.That(kl[0], Is.EqualTo(0.5).Within(1e-12));
    }

    [Test]
    public void SampleNormal_Forward_ProducesCorrectShape()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f, 0f, 0f });

        var z = ForwardGradOperations.SampleNormal(mean, logVar, seed: 42);

        Assert.That(z.Length, Is.EqualTo(3));
        Assert.That(z.RequiresTangent, Is.False);
    }

    [Test]
    public void SampleNormal_WithTangent_ComputesCorrectTangent()
    {
        // JVP: t_z = t_mean + 0.5 * exp(0.5 * logVar) * ε * t_logVar
        // logVar=[0]: σ=1,  JVP = t_mean + 0.5 * ε * t_logVar
        var mean = ForwardGradTensor<float>.FromArray(
            new float[] { 0f }, new float[] { 2f });
        var logVar = ForwardGradTensor<float>.FromArray(
            new float[] { 0f }, new float[] { 3f });

        var z = ForwardGradOperations.SampleNormal(mean, logVar, seed: 42);

        Assert.That(z.RequiresTangent, Is.True);
        Assert.That(z.Tangent, Is.Not.Null);
        Assert.That(float.IsNaN(z.Tangent![0]), Is.False);
        Assert.That(float.IsInfinity(z.Tangent[0]), Is.False);
    }

    [Test]
    public void SampleNormal_DifferentSeeds_DifferentResults()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 0f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f });

        var z1 = ForwardGradOperations.SampleNormal(mean, logVar, seed: 42);
        var z2 = ForwardGradOperations.SampleNormal(mean, logVar, seed: 99);

        Assert.That(z1[0], Is.Not.EqualTo(z2[0]).Within(1e-6f));
    }

    [Test]
    public void SampleNormal_SameSeed_Deterministic()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 0f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f });

        var z1 = ForwardGradOperations.SampleNormal(mean, logVar, seed: 42);
        var z2 = ForwardGradOperations.SampleNormal(mean, logVar, seed: 42);

        Assert.That(z1[0], Is.EqualTo(z2[0]).Within(1e-6f));
    }

    [Test]
    public void SampleNormal_DifferentLengths_Throws()
    {
        var mean = ForwardGradTensor<float>.FromArray(new float[] { 1f });
        var logVar = ForwardGradTensor<float>.FromArray(new float[] { 0f, 1f });

        Assert.That(() => ForwardGradOperations.SampleNormal(mean, logVar), Throws.ArgumentException);
    }

    #endregion

    #region MatMulTransposedB

    [Test]
    public void MatMulTransposedB_Simple_ComputesCorrectValuesAndShape()
    {
        // a = [[1,2,3],[4,5,6]], b = [[7,8,9],[10,11,12]]
        // result = a @ b^T = [[50,68],[122,167]]
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(result[0], Is.EqualTo(50.0f));
        Assert.That(result[1], Is.EqualTo(68.0f));
        Assert.That(result[2], Is.EqualTo(122.0f));
        Assert.That(result[3], Is.EqualTo(167.0f));
    }

    [Test]
    public void MatMulTransposedB_OnlyATangent_ComputesCorrectTangent()
    {
        // JVP: t_out = t_a @ b^T. t_a = ones → each column sums to sum(b column).
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3, new float[] { 1f, 1f, 1f, 1f, 1f, 1f });
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
        Assert.That(result.Tangent![0], Is.EqualTo(24.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(33.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(24.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(33.0f));
    }

    [Test]
    public void MatMulTransposedB_OnlyBTangent_ComputesCorrectTangent()
    {
        // JVP: t_out = a @ t_b^T. t_b = ones → rows sum a's row sums.
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = ForwardGradTensor<float>.FromMatrix(
            new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, 2, 3, new float[] { 1f, 1f, 1f, 1f, 1f, 1f });

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
        Assert.That(result.Tangent![0], Is.EqualTo(6.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(6.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(15.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(15.0f));
    }

    [Test]
    public void MatMulTransposedB_BothTangents_SumsContributions()
    {
        // t_out = t_a @ b^T + a @ t_b^T = [[30,39],[39,48]]
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3, new float[] { 1f, 1f, 1f, 1f, 1f, 1f });
        var b = ForwardGradTensor<float>.FromMatrix(
            new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, 2, 3, new float[] { 1f, 1f, 1f, 1f, 1f, 1f });

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.Tangent![0], Is.EqualTo(30.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(39.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(39.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(48.0f));
    }

    [Test]
    public void MatMulTransposedB_NoTangents_DoesNotTrackTangent()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result.Tangent, Is.Null);
    }

    [Test]
    public void MatMulTransposedB_DimensionMismatch_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 2, 2);

        var ex = Assert.Throws<ArgumentException>(() => ForwardGradOperations.MatMulTransposedB(a, b));
        Assert.That(ex.Message, Does.Contain("must equal"));
    }

    [Test]
    public void MatMulTransposedB_WrongRank_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 4f, 5f, 6f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.MatMulTransposedB(a, b));
    }

    #endregion

    #region TransposeAxes

    [Test]
    public void TransposeAxes_2D_SwapsRowsAndColumns()
    {
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3,
            new float[] { 1f, 0f, 0f, 0f, 0f, 0f });

        var result = ForwardGradOperations.TransposeAxes(a, 0, 1);

        Assert.That(result.Shape, Is.EqualTo(new[] { 3, 2 }));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(4.0f));
        Assert.That(result[2], Is.EqualTo(2.0f));
        Assert.That(result[3], Is.EqualTo(5.0f));
        Assert.That(result[4], Is.EqualTo(3.0f));
        Assert.That(result[5], Is.EqualTo(6.0f));
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[5], Is.EqualTo(0.0f));
    }

    [Test]
    public void TransposeAxes_3D_SwapsFirstAndLastAxes()
    {
        // a flat [1..8] with shape [2,2,2], swap axes 0 and 2
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        a.Reshape(2, 2, 2);

        var result = ForwardGradOperations.TransposeAxes(a, 0, 2);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2, 2 }));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(5.0f));
        Assert.That(result[2], Is.EqualTo(3.0f));
        Assert.That(result[3], Is.EqualTo(7.0f));
        Assert.That(result[4], Is.EqualTo(2.0f));
        Assert.That(result[5], Is.EqualTo(6.0f));
        Assert.That(result[6], Is.EqualTo(4.0f));
        Assert.That(result[7], Is.EqualTo(8.0f));
    }

    [Test]
    public void TransposeAxes_3DWithTangent_TransposesTangent()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f },
            new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f });
        a.Reshape(2, 2, 2);

        var result = ForwardGradOperations.TransposeAxes(a, 0, 2);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[4], Is.EqualTo(0.0f));
    }

    [Test]
    public void TransposeAxes_InvalidAxis_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardGradOperations.TransposeAxes(a, 0, 2));
    }

    [Test]
    public void TransposeAxes_SameAxis_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.TransposeAxes(a, 1, 1));
    }

    [Test]
    public void TransposeAxes_WrongRank_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.TransposeAxes(a, 0, 1));
    }

    #endregion

    #region Slice

    [Test]
    public void Slice_RowVector_ExtractsSubrange()
    {
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 1, 6,
            new float[] { 6f, 5f, 4f, 3f, 2f, 1f });

        var result = ForwardGradOperations.Slice(a, 1, 3);

        Assert.That(result.Shape, Is.EqualTo(new[] { 3 }));
        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(3.0f));
        Assert.That(result[2], Is.EqualTo(4.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(5.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(4.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void Slice_Matrix_SlicesEveryRow()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);

        var result = ForwardGradOperations.Slice(a, 1, 2);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(3.0f));
        Assert.That(result[2], Is.EqualTo(5.0f));
        Assert.That(result[3], Is.EqualTo(6.0f));
    }

    [Test]
    public void Slice_OutOfRange_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 1, 4);

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.Slice(a, 3, 2));
    }

    [Test]
    public void Slice_NegativeStart_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 1, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardGradOperations.Slice(a, -1, 2));
    }

    #endregion

    #region Concat

    [Test]
    public void Concat_1D_JoinsValuesAndZeroFillsMissingTangents()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f, 3f }, new float[] { 1f, 1f, 1f });
        var b = ForwardGradTensor<float>.FromArray(new float[] { 4f, 5f });

        var result = ForwardGradOperations.Concat(new[] { a, b });

        Assert.That(result.Shape, Is.EqualTo(new[] { 5 }));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[2], Is.EqualTo(3.0f));
        Assert.That(result[3], Is.EqualTo(4.0f));
        Assert.That(result[4], Is.EqualTo(5.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[4], Is.EqualTo(0.0f));
    }

    [Test]
    public void Concat_2D_Axis1_JoinsColumns()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 5f, 6f }, 2, 1);

        var result = ForwardGradOperations.Concat(new[] { a, b }, axis: 1);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[2], Is.EqualTo(5.0f));
        Assert.That(result[3], Is.EqualTo(3.0f));
        Assert.That(result[4], Is.EqualTo(4.0f));
        Assert.That(result[5], Is.EqualTo(6.0f));
    }

    [Test]
    public void Concat_2D_Axis0_StacksRows()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = ForwardGradTensor<float>.FromMatrix(new float[] { 7f, 8f }, 1, 2);

        var result = ForwardGradOperations.Concat(new[] { a, b }, axis: 0);

        Assert.That(result.Shape, Is.EqualTo(new[] { 3, 2 }));
        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[4], Is.EqualTo(7.0f));
        Assert.That(result[5], Is.EqualTo(8.0f));
    }

    [Test]
    public void Concat_Axis1_RowMismatch_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);
        var b = ForwardGradTensor<float>.FromMatrix(new float[3], 1, 3);

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.Concat(new[] { a, b }, axis: 1));
    }

    [Test]
    public void Concat_SingleTensor_ReturnsTensorUnchanged()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });

        var result = ForwardGradOperations.Concat(new[] { a });

        Assert.That(ReferenceEquals(result, a), Is.True);
    }

    [Test]
    public void Concat_Empty_Throws()
    {
        Assert.Throws<ArgumentException>(() => ForwardGradOperations.Concat<float>(Array.Empty<ForwardGradTensor<float>>()));
    }

    #endregion

    #region Gather

    [Test]
    public void Gather_Rows_SelectsByIndex()
    {
        var source = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 3, 2,
            new float[] { 1f, 0f, 0f, 1f, 1f, 0f });

        var result = ForwardGradOperations.Gather(source, new[] { 2, 0 });

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(result[0], Is.EqualTo(5.0f));
        Assert.That(result[1], Is.EqualTo(6.0f));
        Assert.That(result[2], Is.EqualTo(1.0f));
        Assert.That(result[3], Is.EqualTo(2.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(1.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(0.0f));
    }

    [Test]
    public void Gather_OutOfRange_Throws()
    {
        var source = ForwardGradTensor<float>.FromMatrix(new float[6], 3, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardGradOperations.Gather(source, new[] { 3 }));
    }

    [Test]
    public void Gather_UnsupportedAxis_Throws()
    {
        var source = ForwardGradTensor<float>.FromMatrix(new float[6], 3, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardGradOperations.Gather(source, new[] { 0 }, axis: 1));
    }

    #endregion

    #region SparseEmbeddingBag

    [Test]
    public void SparseEmbeddingBag_SumsSelectedRows()
    {
        var weight = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f }, 4, 3);
        var indices = ForwardGradTensor<float>.FromMatrix(new float[] { 0f, 2f, 3f, 1f }, 2, 2);

        var result = ForwardGradOperations.SparseEmbeddingBag(weight, indices);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(8.0f));
        Assert.That(result[1], Is.EqualTo(10.0f));
        Assert.That(result[2], Is.EqualTo(12.0f));
        Assert.That(result[3], Is.EqualTo(14.0f));
        Assert.That(result[4], Is.EqualTo(16.0f));
        Assert.That(result[5], Is.EqualTo(18.0f));
    }

    [Test]
    public void SparseEmbeddingBag_PaddingIndex_SkipsRows()
    {
        var weight = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f }, 4, 3);
        var indices = ForwardGradTensor<float>.FromMatrix(new float[] { 0f, -1f, 3f, 1f }, 2, 2);

        var result = ForwardGradOperations.SparseEmbeddingBag(weight, indices, paddingIndex: -1);

        Assert.That(result[0], Is.EqualTo(1.0f));
        Assert.That(result[1], Is.EqualTo(2.0f));
        Assert.That(result[2], Is.EqualTo(3.0f));
        Assert.That(result[3], Is.EqualTo(14.0f));
        Assert.That(result[4], Is.EqualTo(16.0f));
        Assert.That(result[5], Is.EqualTo(18.0f));
    }

    [Test]
    public void SparseEmbeddingBag_WeightTangent_SumsSelectedTangents()
    {
        var weight = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f }, 4, 3,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var indices = ForwardGradTensor<float>.FromMatrix(new float[] { 0f, 2f, 3f, 1f }, 2, 2);

        var result = ForwardGradOperations.SparseEmbeddingBag(weight, indices);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
        for (int i = 0; i < 6; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(2.0f));
    }

    [Test]
    public void SparseEmbeddingBag_OutOfRange_Throws()
    {
        var weight = ForwardGradTensor<float>.FromMatrix(new float[12], 4, 3);
        var indices = ForwardGradTensor<float>.FromMatrix(new float[] { 0f, 4f }, 1, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardGradOperations.SparseEmbeddingBag(weight, indices));
    }

    #endregion

    #region MeanPool

    [Test]
    public void MeanPool_AveragesNonOverlappingWindows()
    {
        // batch=2, poolSize=2, embedDim=3
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f });

        var result = ForwardGradOperations.MeanPool(a, poolSize: 2, embedDim: 3);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(2.5f));
        Assert.That(result[1], Is.EqualTo(3.5f));
        Assert.That(result[2], Is.EqualTo(4.5f));
        Assert.That(result[3], Is.EqualTo(8.5f));
        Assert.That(result[4], Is.EqualTo(9.5f));
        Assert.That(result[5], Is.EqualTo(10.5f));
    }

    [Test]
    public void MeanPool_WithTangent_AveragesTangent()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f },
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });

        var result = ForwardGradOperations.MeanPool(a, poolSize: 2, embedDim: 3);

        Assert.That(result.RequiresTangent, Is.True);
        for (int i = 0; i < 6; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(1.0f));
    }

    [Test]
    public void MeanPool_NonDivisibleLength_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f, 4f, 5f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.MeanPool(a, poolSize: 2, embedDim: 3));
    }

    #endregion

    #region GeluExact

    [Test]
    public void GeluExact_AtZero_ComputesValueAndTangent()
    {
        // gelu(0) = 0, gelu'(0) = 0.5, JVP = 0.5 * 2 = 1
        var a = ForwardGradTensor<float>.FromArray(new float[] { 0.0f }, new float[] { 2.0f });

        var result = ForwardGradOperations.GeluExact(a);

        Assert.That(result[0], Is.EqualTo(0.0f).Within(1e-4f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(1.0f).Within(1e-4f));
    }

    [Test]
    public void GeluExact_NoTangent_DoesNotTrack()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = ForwardGradOperations.GeluExact(a);

        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result.Tangent, Is.Null);
        Assert.That(result.Length, Is.EqualTo(3));
    }

    #endregion

    #region Pow

    [Test]
    public void Pow_ComputesValuesAndTangent()
    {
        // a=[2,3], exp=2 → [4,9]; JVP = 2 * a^(1) * t_a = [4,6]
        var a = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f }, new float[] { 1f, 1f });

        var result = ForwardGradOperations.Pow(a, 2.0);

        Assert.That(result[0], Is.EqualTo(4.0f));
        Assert.That(result[1], Is.EqualTo(9.0f));
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(4.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(6.0f));
    }

    [Test]
    public void Pow_FractionalExponent_ComputesTangent()
    {
        // a=[4], exp=0.5 → 2; JVP = 0.5 * 4^(-0.5) * t_a = 0.5 * 0.5 * 8 = 2
        var a = ForwardGradTensor<float>.FromArray(new float[] { 4f }, new float[] { 8f });

        var result = ForwardGradOperations.Pow(a, 0.5);

        Assert.That(result[0], Is.EqualTo(2.0f).Within(1e-4f));
        Assert.That(result.Tangent![0], Is.EqualTo(2.0f).Within(1e-4f));
    }

    #endregion

    #region RMSNorm

    [Test]
    public void RMSNorm_NormalizesByRowNorm()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });

        var result = ForwardGradOperations.RMSNorm(a);

        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(0.46291f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(0.92582f).Within(1e-4f));
        Assert.That(result[2], Is.EqualTo(1.38873f).Within(1e-4f));
    }

    [Test]
    public void RMSNorm_WithTangent_ComputesSymmetricJvp()
    {
        var a = ForwardGradTensor<float>.FromArray(
            new float[] { 1f, 2f, 3f }, new float[] { 1f, 1f, 1f });

        var result = ForwardGradOperations.RMSNorm(a);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.26460f).Within(1e-4f));
        Assert.That(result.Tangent[1], Is.EqualTo(0.06619f).Within(1e-4f));
        Assert.That(result.Tangent[2], Is.EqualTo(-0.13217f).Within(1e-4f));
    }

    #endregion

    #region PerRowRMSNorm

    [Test]
    public void PerRowRMSNorm_NormalizesEachRow()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 2, 2);

        var result = ForwardGradOperations.PerRowRMSNorm(a, rows: 2, cols: 2);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(result[0], Is.EqualTo(0.632455f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(1.264911f).Within(1e-4f));
        Assert.That(result[2], Is.EqualTo(0.848528f).Within(1e-4f));
        Assert.That(result[3], Is.EqualTo(1.131371f).Within(1e-4f));
    }

    [Test]
    public void PerRowRMSNorm_WithTangent_ComputesPerRowJvp()
    {
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f }, 2, 2,
            new float[] { 1f, 1f, 1f, 1f });

        var result = ForwardGradOperations.PerRowRMSNorm(a, rows: 2, cols: 2);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(0.252982f).Within(1e-4f));
        Assert.That(result.Tangent[1], Is.EqualTo(-0.126491f).Within(1e-4f));
        Assert.That(result.Tangent[2], Is.EqualTo(0.045251f).Within(1e-4f));
        Assert.That(result.Tangent[3], Is.EqualTo(-0.033946f).Within(1e-4f));
    }

    #endregion

    #region AddBias

    [Test]
    public void AddBias_AddsBiasToEachRow()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 10f, 20f, 30f });

        var result = ForwardGradOperations.AddBias(a, bias);

        Assert.That(result[0], Is.EqualTo(11.0f));
        Assert.That(result[1], Is.EqualTo(22.0f));
        Assert.That(result[2], Is.EqualTo(33.0f));
        Assert.That(result[3], Is.EqualTo(14.0f));
        Assert.That(result[4], Is.EqualTo(25.0f));
        Assert.That(result[5], Is.EqualTo(36.0f));
        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
    }

    [Test]
    public void AddBias_OnlyBiasTangent_BroadcastsToAllRows()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 10f, 20f, 30f }, new float[] { 1f, 1f, 1f });

        var result = ForwardGradOperations.AddBias(a, bias);

        Assert.That(result.RequiresTangent, Is.True);
        for (int i = 0; i < 6; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(1.0f));
    }

    [Test]
    public void AddBias_BothTangents_SumsContributions()
    {
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f });
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 10f, 20f, 30f }, new float[] { 1f, 1f, 1f });

        var result = ForwardGradOperations.AddBias(a, bias);

        for (int i = 0; i < 6; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(2.0f));
    }

    [Test]
    public void AddBias_LengthMismatch_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.AddBias(a, bias));
    }

    [Test]
    public void AddBias_WrongRank_Throws()
    {
        var a = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.AddBias(a, bias));
    }

    #endregion

    #region Broadcasting

    static ForwardGradTensor<float> From3D(float[] data, int b, int l, int d, float[]? tangent = null)
    {
        var dataCol = NivaraColumn<float>.CreateFromOwnedArray(data);
        NivaraColumn<float>? tanCol = tangent != null ? NivaraColumn<float>.CreateFromOwnedArray(tangent) : null;
        return new ForwardGradTensor<float>(dataCol, tanCol, new[] { b, l, d });
    }

    [Test]
    public void BroadcastMultiply_ScalesByChannel()
    {
        // input [2,2,2], scale [2,3]; channel = dim 1
        var input = From3D(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2);
        var scale = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f });

        var result = ForwardGradOperations.BroadcastMultiply(input, scale);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2, 2 }));
        Assert.That(result[0], Is.EqualTo(2.0f));
        Assert.That(result[1], Is.EqualTo(4.0f));
        Assert.That(result[2], Is.EqualTo(9.0f));
        Assert.That(result[3], Is.EqualTo(12.0f));
        Assert.That(result[4], Is.EqualTo(10.0f));
        Assert.That(result[5], Is.EqualTo(12.0f));
        Assert.That(result[6], Is.EqualTo(21.0f));
        Assert.That(result[7], Is.EqualTo(24.0f));
    }

    [Test]
    public void BroadcastMultiply_InputTangent_ScalesTangent()
    {
        var input = From3D(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var scale = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f });

        var result = ForwardGradOperations.BroadcastMultiply(input, scale);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(2.0f));
        Assert.That(result.Tangent[1], Is.EqualTo(2.0f));
        Assert.That(result.Tangent[2], Is.EqualTo(3.0f));
        Assert.That(result.Tangent[3], Is.EqualTo(3.0f));
        Assert.That(result.Tangent[4], Is.EqualTo(2.0f));
        Assert.That(result.Tangent[7], Is.EqualTo(3.0f));
    }

    [Test]
    public void BroadcastMultiply_BothTangents_SumsContributions()
    {
        var input = From3D(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var scale = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f }, new float[] { 1f, 1f });

        var result = ForwardGradOperations.BroadcastMultiply(input, scale);

        Assert.That(result.Tangent![0], Is.EqualTo(3.0f));
        Assert.That(result.Tangent![1], Is.EqualTo(4.0f));
        Assert.That(result.Tangent![2], Is.EqualTo(6.0f));
        Assert.That(result.Tangent![3], Is.EqualTo(7.0f));
    }

    [Test]
    public void BroadcastAdd_AddsChannelBias()
    {
        var input = From3D(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2);
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f });

        var result = ForwardGradOperations.BroadcastAdd(input, bias);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2, 2 }));
        Assert.That(result[0], Is.EqualTo(3.0f));
        Assert.That(result[1], Is.EqualTo(4.0f));
        Assert.That(result[2], Is.EqualTo(6.0f));
        Assert.That(result[3], Is.EqualTo(7.0f));
        Assert.That(result[4], Is.EqualTo(7.0f));
        Assert.That(result[5], Is.EqualTo(8.0f));
        Assert.That(result[6], Is.EqualTo(10.0f));
        Assert.That(result[7], Is.EqualTo(11.0f));
    }

    [Test]
    public void BroadcastAdd_BiasTangent_Broadcasts()
    {
        var input = From3D(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var bias = ForwardGradTensor<float>.FromArray(new float[] { 2f, 3f }, new float[] { 1f, 1f });

        var result = ForwardGradOperations.BroadcastAdd(input, bias);

        for (int i = 0; i < 8; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(2.0f));
    }

    [Test]
    public void BroadcastMultiply_ChannelMismatch_Throws()
    {
        var input = From3D(new float[8], 2, 2, 2);
        var scale = ForwardGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f });

        Assert.Throws<ArgumentException>(() => ForwardGradOperations.BroadcastMultiply(input, scale));
    }

    #endregion

    #region MultiHeadAttention

    [Test]
    public void MultiHeadAttention_ComputesOutputShape()
    {
        var query = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 4);
        var key = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f }, 2, 4);
        var value = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f }, 2, 4);

        var result = ForwardGradOperations.MultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 4 }));
        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result[0], Is.EqualTo(1.73106f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(1.73106f).Within(1e-4f));
        Assert.That(result[4], Is.EqualTo(1.73106f).Within(1e-4f));
        Assert.That(result[7], Is.EqualTo(1.73106f).Within(1e-4f));
    }

    [Test]
    public void MultiHeadAttention_WithMask_ProducesFiniteOutput()
    {
        var query = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 4);
        var key = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f }, 2, 4);
        var value = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f }, 2, 4);
        var mask = ForwardGradTensor<float>.FromMatrix(
            new float[] { float.NegativeInfinity, 0f, 0f, 0f }, 2, 2);

        var result = ForwardGradOperations.MultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f, mask);

        Assert.That(result.Length, Is.EqualTo(8));
        foreach (var v in result.AsSpan())
        {
            Assert.That(float.IsNaN(v), Is.False);
            Assert.That(float.IsInfinity(v), Is.False);
        }
    }

    [Test]
    public void MultiHeadAttention_QueryTangentSummedSoftmax_ZeroJvp()
    {
        // With t_Q = ones and t_K = t_V = 0, t_scores = ones; each softmax row sums to 1,
        // so the softmax JVP is zero and t_out = 0.
        var query = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 4,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var key = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f }, 2, 4);
        var value = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f }, 2, 4);

        var result = ForwardGradOperations.MultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
        for (int i = 0; i < 8; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(0.0f).Within(1e-4f));
    }

    [Test]
    public void MultiHeadAttention_InvalidMaskShape_Throws()
    {
        var query = ForwardGradTensor<float>.FromMatrix(new float[8], 2, 4);
        var key = ForwardGradTensor<float>.FromMatrix(new float[8], 2, 4);
        var value = ForwardGradTensor<float>.FromMatrix(new float[8], 2, 4);
        var mask = ForwardGradTensor<float>.FromMatrix(new float[2], 1, 2);

        Assert.Throws<ArgumentException>(() =>
            ForwardGradOperations.MultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f, mask));
    }

    [Test]
    public void MultiHeadAttention_WidthNotDivisible_Throws()
    {
        var query = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);
        var key = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);
        var value = ForwardGradTensor<float>.FromMatrix(new float[6], 2, 3);

        Assert.Throws<ArgumentException>(() =>
            ForwardGradOperations.MultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f));
    }

    #endregion

    #region BatchedMultiHeadAttention

    [Test]
    public void BatchedMultiHeadAttention_ComputesOutputShape()
    {
        // D=2, numHeads=2 → headDim=1; softmax of scores gives the expected values below.
        var query = From3D(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2);
        var key = From3D(new float[] { 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f }, 2, 2, 2);
        var value = From3D(new float[] { 1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f }, 2, 2, 2);

        var result = ForwardGradOperations.BatchedMultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 2, 2 }));
        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result[0], Is.EqualTo(1.26894f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(1.88080f).Within(1e-4f));
        Assert.That(result[2], Is.EqualTo(1.04743f).Within(1e-4f));
        Assert.That(result[3], Is.EqualTo(1.98201f).Within(1e-4f));
        Assert.That(result[7], Is.EqualTo(1.99966f).Within(1e-4f));
    }

    [Test]
    public void BatchedMultiHeadAttention_QueryTangent_ZeroJvp()
    {
        // With K = ones, t_scores = scale * t_Q @ K^T = ones; ds is constant per key
        // position, so the softmax JVP is zero and t_out = 0.
        var query = From3D(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 2, 2, 2,
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var key = From3D(
            new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f }, 2, 2, 2);
        var value = From3D(new float[] { 1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f }, 2, 2, 2);

        var result = ForwardGradOperations.BatchedMultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f);

        Assert.That(result.RequiresTangent, Is.True);
        for (int i = 0; i < 8; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(0.0f).Within(1e-4f));
    }

    [Test]
    public void BatchedMultiHeadAttention_WrongRank_Throws()
    {
        var query = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);
        var key = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);
        var value = ForwardGradTensor<float>.FromMatrix(new float[4], 2, 2);

        Assert.Throws<ArgumentException>(() =>
            ForwardGradOperations.BatchedMultiHeadAttention(query, key, value, numHeads: 2, scale: 1.0f));
    }

    #endregion
}
