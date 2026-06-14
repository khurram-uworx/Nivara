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
}
