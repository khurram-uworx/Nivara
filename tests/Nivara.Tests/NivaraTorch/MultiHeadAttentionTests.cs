using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class MultiHeadAttentionTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void MultiHeadAttention_SelfAttention_MatchesPyTorch()
    {
        var q = Matrix("attn_self_q.bin", 4, 16, requiresGrad: false);
        var k = Matrix("attn_self_k.bin", 4, 16, requiresGrad: false);
        var v = Matrix("attn_self_v.bin", 4, 16, requiresGrad: false);
        var expected = TestHelpers.LoadBin("attn_self_output.bin");

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, 4, 0.5f);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "attn_self");
    }

    [Test]
    public void MultiHeadAttention_SelfAttentionCausalMask_MatchesPyTorch()
    {
        var q = Matrix("attn_self_causal_q.bin", 4, 16, requiresGrad: false);
        var k = Matrix("attn_self_causal_k.bin", 4, 16, requiresGrad: false);
        var v = Matrix("attn_self_causal_v.bin", 4, 16, requiresGrad: false);
        var mask = Matrix("attn_self_causal_mask.bin", 4, 4, requiresGrad: false);
        var expected = TestHelpers.LoadBin("attn_self_causal_output.bin");

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, 4, 0.5f, mask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "attn_self_causal");
    }

    [Test]
    public void MultiHeadAttention_Backward_MatchesPyTorch()
    {
        var q = Matrix("attn_self_causal_q.bin", 4, 16, requiresGrad: true);
        var k = Matrix("attn_self_causal_k.bin", 4, 16, requiresGrad: true);
        var v = Matrix("attn_self_causal_v.bin", 4, 16, requiresGrad: true);
        var mask = Matrix("attn_self_causal_mask.bin", 4, 4, requiresGrad: false);
        var dout = Matrix("attn_self_causal_dout.bin", 4, 16, requiresGrad: false);

        var expectedDq = TestHelpers.LoadBin("attn_self_causal_dq.bin");
        var expectedDk = TestHelpers.LoadBin("attn_self_causal_dk.bin");
        var expectedDv = TestHelpers.LoadBin("attn_self_causal_dv.bin");

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, 4, 0.5f, mask);
        output.Backward(dout);

        TestHelpers.AssertTensorEqual(expectedDq, GradArray(q), label: "dQ");
        TestHelpers.AssertTensorEqual(expectedDk, GradArray(k), label: "dK");
        TestHelpers.AssertTensorEqual(expectedDv, GradArray(v), label: "dV");
    }

    [Test]
    public void MultiHeadAttention_CrossAttention_MatchesPyTorch()
    {
        var q = Matrix("attn_cross_q.bin", 3, 8, requiresGrad: true);
        var k = Matrix("attn_cross_k.bin", 5, 8, requiresGrad: true);
        var v = Matrix("attn_cross_v.bin", 5, 8, requiresGrad: true);
        var mask = Matrix("attn_cross_mask.bin", 3, 5, requiresGrad: false);
        var dout = Matrix("attn_cross_dout.bin", 3, 8, requiresGrad: false);

        var expected = TestHelpers.LoadBin("attn_cross_output.bin");
        var expectedDq = TestHelpers.LoadBin("attn_cross_dq.bin");
        var expectedDk = TestHelpers.LoadBin("attn_cross_dk.bin");
        var expectedDv = TestHelpers.LoadBin("attn_cross_dv.bin");

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, 2, 0.5f, mask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 3, 8 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "attn_cross");

        output.Backward(dout);

        TestHelpers.AssertTensorEqual(expectedDq, GradArray(q), label: "cross dQ");
        TestHelpers.AssertTensorEqual(expectedDk, GradArray(k), label: "cross dK");
        TestHelpers.AssertTensorEqual(expectedDv, GradArray(v), label: "cross dV");
    }

    [Test]
    public void MultiHeadAttention_InvalidArguments_Throw()
    {
        var q = Matrix("attn_self_q.bin", 4, 16, requiresGrad: false);
        var k = Matrix("attn_self_k.bin", 4, 16, requiresGrad: false);
        var v = Matrix("attn_self_v.bin", 4, 16, requiresGrad: false);

        Assert.Throws<ArgumentException>(() => ReverseGradOperations.MultiHeadAttention(q, k, v, 3, 0.5f));
        Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.MultiHeadAttention(q, Matrix("attn_cross_k.bin", 5, 8, requiresGrad: false), v, 4, 0.5f));
    }

    static ReverseGradTensor<float> Matrix(string name, int rows, int cols, bool requiresGrad)
    {
        var data = TestHelpers.LoadBin(name);
        Assert.That(data.Length, Is.EqualTo(rows * cols), $"Unexpected {name} length {data.Length}");
        return ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);
    }

    static float[] GradArray(ReverseGradTensor<float> tensor)
    {
        Assert.That(tensor.Grad, Is.Not.Null, $"{nameof(tensor)}.Grad should be populated after backward");
        var grad = tensor.Grad!;
        var result = new float[grad.Length];
        for (int i = 0; i < grad.Length; i++)
            result[i] = grad[i];
        return result;
    }
}
