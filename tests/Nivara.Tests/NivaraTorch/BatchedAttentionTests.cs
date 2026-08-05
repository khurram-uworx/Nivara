using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class BatchedAttentionTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BatchedMultiHeadAttention_CausalSelfAttention_MatchesPyTorch()
    {
        var q = Tensor3D("batched_attn_causal_q.bin", 2, 4, 16, requiresGrad: false);
        var k = Tensor3D("batched_attn_causal_k.bin", 2, 4, 16, requiresGrad: false);
        var v = Tensor3D("batched_attn_causal_v.bin", 2, 4, 16, requiresGrad: false);
        var mask = Tensor3D("batched_attn_causal_mask.bin", 2, 4, 4, requiresGrad: false);
        var expected = TestHelpers.LoadBin("batched_attn_causal_output.bin");

        var output = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 4, 0.5f, mask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "batched_attn_causal");
    }

    [Test]
    public void BatchedMultiHeadAttention_CausalSelfAttention_Backward_MatchesPyTorch()
    {
        var q = Tensor3D("batched_attn_causal_q.bin", 2, 4, 16, requiresGrad: true);
        var k = Tensor3D("batched_attn_causal_k.bin", 2, 4, 16, requiresGrad: true);
        var v = Tensor3D("batched_attn_causal_v.bin", 2, 4, 16, requiresGrad: true);
        var mask = Tensor3D("batched_attn_causal_mask.bin", 2, 4, 4, requiresGrad: false);
        var dout = Tensor3D("batched_attn_causal_dout.bin", 2, 4, 16, requiresGrad: false);

        var expectedDq = TestHelpers.LoadBin("batched_attn_causal_dq.bin");
        var expectedDk = TestHelpers.LoadBin("batched_attn_causal_dk.bin");
        var expectedDv = TestHelpers.LoadBin("batched_attn_causal_dv.bin");

        var output = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 4, 0.5f, mask);
        output.Backward(dout);

        TestHelpers.AssertTensorEqual(expectedDq, GradArray(q), label: "dQ");
        TestHelpers.AssertTensorEqual(expectedDk, GradArray(k), label: "dK");
        TestHelpers.AssertTensorEqual(expectedDv, GradArray(v), label: "dV");
    }

    [Test]
    public void BatchedMultiHeadAttention_CrossAttention_MatchesPyTorch()
    {
        var q = Tensor3D("batched_attn_cross_q.bin", 2, 3, 8, requiresGrad: true);
        var k = Tensor3D("batched_attn_cross_k.bin", 2, 5, 8, requiresGrad: true);
        var v = Tensor3D("batched_attn_cross_v.bin", 2, 5, 8, requiresGrad: true);
        var mask = Tensor3D("batched_attn_cross_mask.bin", 2, 3, 5, requiresGrad: false);
        var dout = Tensor3D("batched_attn_cross_dout.bin", 2, 3, 8, requiresGrad: false);

        var expected = TestHelpers.LoadBin("batched_attn_cross_output.bin");
        var expectedDq = TestHelpers.LoadBin("batched_attn_cross_dq.bin");
        var expectedDk = TestHelpers.LoadBin("batched_attn_cross_dk.bin");
        var expectedDv = TestHelpers.LoadBin("batched_attn_cross_dv.bin");

        var output = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 2, 0.5f, mask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 3, 8 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "batched_attn_cross");

        output.Backward(dout);

        TestHelpers.AssertTensorEqual(expectedDq, GradArray(q), label: "cross dQ");
        TestHelpers.AssertTensorEqual(expectedDk, GradArray(k), label: "cross dK");
        TestHelpers.AssertTensorEqual(expectedDv, GradArray(v), label: "cross dV");
    }

    static ReverseGradTensor<float> Tensor3D(string name, int b, int l, int d, bool requiresGrad)
    {
        var data = TestHelpers.LoadBin(name);
        Assert.That(data.Length, Is.EqualTo(b * l * d), $"Unexpected {name} length {data.Length}");
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(b, l, d);
        return tensor;
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
