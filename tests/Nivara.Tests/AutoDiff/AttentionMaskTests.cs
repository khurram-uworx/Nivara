using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// The attention mask is a non-differentiable constant. <c>ApplyMask</c> assigns
/// -inf to a suppressed cell rather than adding to it, so the mask is only
/// piecewise differentiable and no gradient is produced for it.
/// See <c>AttentionKernels.ApplyMask</c> and issue #481.
/// </summary>
[TestFixture]
public class AttentionMaskTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void MultiHeadAttention_MaskRequiresGrad_Throws()
    {
        var q = Mat2D(Rand(16, 1), 4, 4, requiresGrad: true);
        var k = Mat2D(Rand(16, 2), 4, 4, requiresGrad: true);
        var v = Mat2D(Rand(16, 3), 4, 4, requiresGrad: true);
        var mask = Mat2D(Rand(16, 4), 4, 4, requiresGrad: true);

        var ex = Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.MultiHeadAttention(q, k, v, 2, 0.5f, mask));

        Assert.That(ex!.Message, Does.Contain("non-differentiable"));
        Assert.That(ex.ParamName, Is.EqualTo("mask"));
    }

    [Test]
    public void BatchedMultiHeadAttention_MaskRequiresGrad_Throws()
    {
        var q = Mat3D(Rand(32, 5), 2, 4, 4, requiresGrad: true);
        var k = Mat3D(Rand(32, 6), 2, 4, 4, requiresGrad: true);
        var v = Mat3D(Rand(32, 7), 2, 4, 4, requiresGrad: true);
        var mask = Mat3D(Rand(32, 8), 2, 4, 4, requiresGrad: true);

        var ex = Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 2, 0.5f, mask));

        Assert.That(ex!.Message, Does.Contain("non-differentiable"));
        Assert.That(ex.ParamName, Is.EqualTo("mask"));
    }

    [Test]
    public void MultiHeadAttention_MaskOnlyRequiresGrad_ThrowsRatherThanSilentlySkippingTheGraph()
    {
        // shouldTrack excludes the mask, so before the guard this built no node at
        // all and the failure surfaced later as Backward's generic
        // "tensor that doesn't require gradients", pointing at the wrong tensor.
        var q = Mat2D(Rand(16, 9), 4, 4, requiresGrad: false);
        var k = Mat2D(Rand(16, 10), 4, 4, requiresGrad: false);
        var v = Mat2D(Rand(16, 11), 4, 4, requiresGrad: false);
        var mask = Mat2D(Rand(16, 12), 4, 4, requiresGrad: true);

        var ex = Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.MultiHeadAttention(q, k, v, 2, 0.5f, mask));

        Assert.That(ex!.Message, Does.Contain("non-differentiable"));
    }

    [Test]
    public void MultiHeadAttention_MaskWithoutGrad_LeavesMaskGradNull()
    {
        // The documented contract: a constant mask gets no gradient. Pinned so the
        // removal of the mask from the node's input list stays a recorded decision
        // rather than drifting back into an implied dependency.
        var q = Mat2D(Rand(16, 13), 4, 4, requiresGrad: true);
        var k = Mat2D(Rand(16, 14), 4, 4, requiresGrad: true);
        var v = Mat2D(Rand(16, 15), 4, 4, requiresGrad: true);
        var mask = Mat2D(CausalMask(4), 4, 4, requiresGrad: false);

        var outTensor = ReverseGradOperations.MultiHeadAttention(q, k, v, 2, 0.5f, mask);
        outTensor.Backward(Mat2D(Rand(16, 16), 4, 4, requiresGrad: false));

        Assert.That(mask.Grad, Is.Null);
        Assert.That(q.Grad, Is.Not.Null, "query gradient should still flow");
        Assert.That(k.Grad, Is.Not.Null, "key gradient should still flow");
        Assert.That(v.Grad, Is.Not.Null, "value gradient should still flow");
    }

    [Test]
    public void MultiHeadAttention_NodeDeclaresOnlyQueryKeyValue()
    {
        var q = Mat2D(Rand(16, 17), 4, 4, requiresGrad: true);
        var k = Mat2D(Rand(16, 18), 4, 4, requiresGrad: true);
        var v = Mat2D(Rand(16, 19), 4, 4, requiresGrad: true);
        var mask = Mat2D(CausalMask(4), 4, 4, requiresGrad: false);

        var outTensor = ReverseGradOperations.MultiHeadAttention(q, k, v, 2, 0.5f, mask);

        var inputs = outTensor.GradFn!.Inputs;
        Assert.That(inputs, Has.Count.EqualTo(3));
        Assert.That(inputs, Does.Contain(q));
        Assert.That(inputs, Does.Contain(k));
        Assert.That(inputs, Does.Contain(v));
        Assert.That(inputs, Does.Not.Contain(mask));
    }

    [Test]
    public void BatchedMultiHeadAttention_NodeDeclaresOnlyQueryKeyValue()
    {
        var q = Mat3D(Rand(32, 20), 2, 4, 4, requiresGrad: true);
        var k = Mat3D(Rand(32, 21), 2, 4, 4, requiresGrad: true);
        var v = Mat3D(Rand(32, 22), 2, 4, 4, requiresGrad: true);
        var mask = Mat3D(CausalMask(2, 4), 2, 4, 4, requiresGrad: false);

        var outTensor = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 2, 0.5f, mask);

        var inputs = outTensor.GradFn!.Inputs;
        Assert.That(inputs, Has.Count.EqualTo(3));
        Assert.That(inputs, Does.Contain(q));
        Assert.That(inputs, Does.Contain(k));
        Assert.That(inputs, Does.Contain(v));
        Assert.That(inputs, Does.Not.Contain(mask));
    }

    static float[] CausalMask(int len)
    {
        var mask = new float[len * len];
        for (int i = 0; i < len; i++)
            for (int j = 0; j < len; j++)
                if (j > i)
                    mask[i * len + j] = float.NegativeInfinity;
        return mask;
    }

    static float[] CausalMask(int batch, int len)
    {
        var perSeq = CausalMask(len);
        var mask = new float[batch * len * len];
        for (int b = 0; b < batch; b++)
            Array.Copy(perSeq, 0, mask, b * len * len, len * len);
        return mask;
    }

    static float[] Rand(int count, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1);
        return arr;
    }

    static ReverseGradTensor<float> Mat2D(float[] data, int rows, int cols, bool requiresGrad)
        => ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);

    static ReverseGradTensor<float> Mat3D(float[] data, int b, int l, int d, bool requiresGrad)
    {
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(b, l, d);
        return tensor;
    }
}
