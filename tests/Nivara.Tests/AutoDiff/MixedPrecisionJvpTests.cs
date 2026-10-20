using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class MixedPrecisionJvpTests
{
    static ForwardGradTensor<BFloat16> ToBf(ForwardGradTensor<float> t) =>
        throw new NotSupportedException();

    [Test]
    public void TransposedB_Bf16_Simple_ComputesCorrectValues()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result[0], Is.EqualTo(50.0f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(68.0f).Within(1e-4f));
        Assert.That(result[2], Is.EqualTo(122.0f).Within(1e-4f));
        Assert.That(result[3], Is.EqualTo(167.0f).Within(1e-4f));
    }

    [Test]
    public void TransposedB_Bf16_OnlyATangent_ComputesCorrectTangent()
    {
        var a = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3, new float[] { 1f, 1f, 1f, 1f, 1f, 1f });
        var bData = Array.ConvertAll(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(24.0f).Within(1e-4f));
        Assert.That(result.Tangent[1], Is.EqualTo(33.0f).Within(1e-4f));
        Assert.That(result.Tangent[2], Is.EqualTo(24.0f).Within(1e-4f));
        Assert.That(result.Tangent[3], Is.EqualTo(33.0f).Within(1e-4f));
    }

    [Test]
    public void TransposedB_Bf16_OnlyBTangent_ComputesCorrectTangent()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (BFloat16)x);
        var bTan = Array.ConvertAll(new float[] { 1f, 1f, 1f, 1f, 1f, 1f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 2, 3, bTan);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent![0], Is.EqualTo(6.0f).Within(1e-3f));
        Assert.That(result.Tangent[1], Is.EqualTo(6.0f).Within(1e-3f));
        Assert.That(result.Tangent[2], Is.EqualTo(15.0f).Within(1e-3f));
        Assert.That(result.Tangent[3], Is.EqualTo(15.0f).Within(1e-3f));
    }

    [Test]
    public void TransposedB_Bf16_NoTangents_DoesNotTrack()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);
        Assert.That(result.RequiresTangent, Is.False);
        Assert.That(result.Tangent, Is.Null);
    }

    [Test]
    public void TransposedB_Half_Simple_ComputesCorrectValues()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (Half)x);
        var b = ForwardGradTensor<Half>.FromMatrix(bData, 2, 3);

        var result = ForwardGradOperations.MatMulTransposedB(a, b);

        Assert.That(result[0], Is.EqualTo(50.0f).Within(1e-2f));
        Assert.That(result[3], Is.EqualTo(167.0f).Within(1e-2f));
    }

    [Test]
    public void TransposedB_DimensionMismatch_Throws()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 1f, 2f, 3f, 4f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 2, 2);

        var ex = Assert.Throws<ArgumentException>(() => ForwardGradOperations.MatMulTransposedB(a, b));
        Assert.That(ex!.Message, Does.Contain("must equal"));
    }

    [Test]
    public void MatMul_Bf16_Simple_ComputesCorrectValues()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 1f, 2f, 7f, 8f, 13f, 14f }, x => (BFloat16)x); // 3x2
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 3, 2);

        var result = ForwardGradOperations.MatMul(a, b);
        // row0: 1*1+2*7+3*13=54; 1*2+2*8+3*14=60
        Assert.That(result[0], Is.EqualTo(54.0f).Within(1e-4f));
        Assert.That(result[1], Is.EqualTo(60.0f).Within(1e-4f));
    }

    [Test]
    public void MatMul_Bf16_OnlyBTangent_ComputesCorrectTangent()
    {
        var a = ForwardGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var bData = Array.ConvertAll(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, x => (BFloat16)x);
        var bTan = Array.ConvertAll(new float[] { 1f, 1f, 1f, 1f, 1f, 1f }, x => (BFloat16)x);
        var b = ForwardGradTensor<BFloat16>.FromMatrix(bData, 3, 2, bTan);

        var result = ForwardGradOperations.MatMul(a, b);
        Assert.That(result.RequiresTangent, Is.True);
        Assert.That(result.Tangent, Is.Not.Null);
    }
}
