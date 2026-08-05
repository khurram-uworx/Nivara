using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Tests.NivaraTorch;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class BatchedMultiHeadAttentionTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BatchedForward_SelfAttention_MatchesPerSequenceForwards()
    {
        const int B = 3, L = 5, D = 8, H = 2;
        var (qData, kData, vData, maskData) = GenerateSelfAttention(B, L, D);
        var q = Mat3D(qData, B, L, D, requiresGrad: false);
        var k = Mat3D(kData, B, L, D, requiresGrad: false);
        var v = Mat3D(vData, B, L, D, requiresGrad: false);
        var mask = Mat3D(maskData, B, L, L, requiresGrad: false);
        float scale = 1f / MathF.Sqrt(D / H);

        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);

        Assert.That(batched.Shape, Is.EqualTo(new[] { B, L, D }));
        for (int b = 0; b < B; b++)
        {
            var single = ReverseGradOperations.MultiHeadAttention(
                Mat2D(qData, b, L, D, false), Mat2D(kData, b, L, D, false),
                Mat2D(vData, b, L, D, false), H, scale, Mat2D(maskData, b, L, L, false));
            TestHelpers.AssertTensorEqual(
                TestHelpers.ExtractOutput(single), ExtractSlice(batched, b, L, D), label: $"batched self b{b}");
        }
    }

    [Test]
    public void BatchedBackward_SelfAttention_MatchesPerSequenceGradients()
    {
        const int B = 3, L = 5, D = 8, H = 2;
        var (qData, kData, vData, maskData) = GenerateSelfAttention(B, L, D);
        var dOut = RandFloats(B * L * D, 999);
        float scale = 1f / MathF.Sqrt(D / H);

        var q = Mat3D(qData, B, L, D, requiresGrad: true);
        var k = Mat3D(kData, B, L, D, requiresGrad: true);
        var v = Mat3D(vData, B, L, D, requiresGrad: true);
        var mask = Mat3D(maskData, B, L, L, requiresGrad: false);
        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);
        batched.Backward(Mat3D(dOut, B, L, D, requiresGrad: false));

        for (int b = 0; b < B; b++)
        {
            var qS = Mat2D(qData, b, L, D, true);
            var kS = Mat2D(kData, b, L, D, true);
            var vS = Mat2D(vData, b, L, D, true);
            var mS = Mat2D(maskData, b, L, L, false);
            var single = ReverseGradOperations.MultiHeadAttention(qS, kS, vS, H, scale, mS);
            single.Backward(Mat2D(dOut, b, L, D, false));

            TestHelpers.AssertTensorEqual(GradData(qS), GradSlice(q, b, L, D), label: $"batched dQ b{b}");
            TestHelpers.AssertTensorEqual(GradData(kS), GradSlice(k, b, L, D), label: $"batched dK b{b}");
            TestHelpers.AssertTensorEqual(GradData(vS), GradSlice(v, b, L, D), label: $"batched dV b{b}");
        }
    }

    [Test]
    public void BatchedForward_CrossAttention_MatchesPerSequenceForwards()
    {
        const int B = 2, QL = 4, KV = 6, D = 8, H = 2;
        var q = Mat3D(RandFloats(B * QL * D, 11), B, QL, D, false);
        var k = Mat3D(RandFloats(B * KV * D, 22), B, KV, D, false);
        var v = Mat3D(RandFloats(B * KV * D, 33), B, KV, D, false);
        var mask = Mat3D(RandFloats(B * QL * KV, 44), B, QL, KV, false);
        float scale = 1f / MathF.Sqrt(D / H);

        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);

        Assert.That(batched.Shape, Is.EqualTo(new[] { B, QL, D }));
        for (int b = 0; b < B; b++)
        {
            var single = ReverseGradOperations.MultiHeadAttention(
                Slice(q, b, QL, D), Slice(k, b, KV, D), Slice(v, b, KV, D), H, scale, Slice(mask, b, QL, KV));
            TestHelpers.AssertTensorEqual(
                TestHelpers.ExtractOutput(single), ExtractSlice(batched, b, QL, D), label: $"batched cross b{b}");
        }
    }

    [Test]
    public void Batched_NoMask_MatchesPerSequenceNoMask()
    {
        const int B = 2, L = 4, D = 8, H = 2;
        var q = Mat3D(RandFloats(B * L * D, 55), B, L, D, false);
        var k = Mat3D(RandFloats(B * L * D, 66), B, L, D, false);
        var v = Mat3D(RandFloats(B * L * D, 77), B, L, D, false);
        float scale = 1f / MathF.Sqrt(D / H);

        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale);
        Assert.That(batched.Shape, Is.EqualTo(new[] { B, L, D }));

        for (int b = 0; b < B; b++)
        {
            var single = ReverseGradOperations.MultiHeadAttention(
                Slice(q, b, L, D), Slice(k, b, L, D), Slice(v, b, L, D), H, scale);
            TestHelpers.AssertTensorEqual(
                TestHelpers.ExtractOutput(single), ExtractSlice(batched, b, L, D), label: $"batched no-mask b{b}");
        }
    }

    [Test]
    public void Batched_BatchSizeOne_MatchesSingleSequence()
    {
        const int L = 5, D = 8, H = 2;
        var (qData, kData, vData, maskData) = GenerateSelfAttention(1, L, D);
        float scale = 1f / MathF.Sqrt(D / H);

        var batched = ReverseGradOperations.BatchedMultiHeadAttention(
            Mat3D(qData, 1, L, D, false), Mat3D(kData, 1, L, D, false),
            Mat3D(vData, 1, L, D, false), H, scale, Mat3D(maskData, 1, L, L, false));
        var single = ReverseGradOperations.MultiHeadAttention(
            Mat2D(qData, 0, L, D, false), Mat2D(kData, 0, L, D, false),
            Mat2D(vData, 0, L, D, false), H, scale, Mat2D(maskData, 0, L, L, false));

        TestHelpers.AssertTensorEqual(TestHelpers.ExtractOutput(single), ExtractSlice(batched, 0, L, D), label: "B=1");
    }

    [Test]
    public void Batched_InferenceOutsideGradScope_ProducesOutputWithoutGraph()
    {
        gradScope?.Dispose();
        gradScope = null;

        const int B = 2, L = 4, D = 8, H = 2;
        var q = Mat3D(RandFloats(B * L * D, 5), B, L, D, requiresGrad: true);
        var k = Mat3D(RandFloats(B * L * D, 6), B, L, D, requiresGrad: true);
        var v = Mat3D(RandFloats(B * L * D, 7), B, L, D, requiresGrad: true);
        float scale = 1f / MathF.Sqrt(D / H);

        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale);

        Assert.That(batched.RequiresGrad, Is.False, "No graph nodes should be created outside Grad() scope");
        Assert.That(batched.Shape, Is.EqualTo(new[] { B, L, D }));
        for (int b = 0; b < B; b++)
        {
            var single = ReverseGradOperations.MultiHeadAttention(
                Slice(q, b, L, D), Slice(k, b, L, D), Slice(v, b, L, D), H, scale);
            TestHelpers.AssertTensorEqual(
                TestHelpers.ExtractOutput(single), ExtractSlice(batched, b, L, D), label: $"inference b{b}");
        }
    }

    [Test]
    public void BatchedMultiHeadAttention_InvalidShapes_Throw()
    {
        const int B = 2, L = 4, D = 8, H = 2;
        var q = Mat3D(RandFloats(B * L * D, 1), B, L, D, false);
        var k = Mat3D(RandFloats(B * L * D, 2), B, L, D, false);
        var v = Mat3D(RandFloats(B * L * D, 3), B, L, D, false);
        var mask = Mat3D(RandFloats(B * L * L, 4), B, L, L, false);
        float scale = 1f / MathF.Sqrt(D / H);

        var q2 = Mat2D(RandFloats(L * D, 9), 0, L, D, false);
        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q2, q2, q2, H, scale),
            Throws.ArgumentException, "rank-2 query must be rejected");

        var kB = Mat3D(RandFloats((B + 1) * L * D, 10), B + 1, L, D, false);
        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q, kB, v, H, scale),
            Throws.ArgumentException, "key batch mismatch must be rejected");

        var badD = Mat3D(RandFloats(B * L * (D + 1), 11), B, L, D + 1, false);
        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q, k, badD, H, scale),
            Throws.ArgumentException, "value width mismatch must be rejected");

        var kLen = Mat3D(RandFloats(B * (L + 1) * D, 12), B, L + 1, D, false);
        var vLen = Mat3D(RandFloats(B * L * D, 13), B, L, D, false);
        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q, kLen, vLen, H, scale),
            Throws.ArgumentException, "key/value sequence mismatch must be rejected");

        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, 3, scale),
            Throws.ArgumentException, "non-divisible head count must be rejected");

        var badMask = Mat3D(RandFloats(B * L * (L + 1), 14), B, L, L + 1, false);
        Assert.That(() => ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, badMask),
            Throws.ArgumentException, "mask shape mismatch must be rejected");
    }

    [Test]
    public void Batched_LargeParallelBatch_MatchesPerSequence()
    {
        const int B = 4, L = 128, D = 64, H = 4;
        var (qData, kData, vData, maskData) = GenerateSelfAttention(B, L, D);
        var dOut = RandFloats(B * L * D, 999);
        float scale = 1f / MathF.Sqrt(D / H);

        var q = Mat3D(qData, B, L, D, requiresGrad: true);
        var k = Mat3D(kData, B, L, D, requiresGrad: true);
        var v = Mat3D(vData, B, L, D, requiresGrad: true);
        var mask = Mat3D(maskData, B, L, L, requiresGrad: false);
        var batched = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);
        batched.Backward(Mat3D(dOut, B, L, D, requiresGrad: false));

        for (int b = 0; b < B; b++)
        {
            var qS = Mat2D(qData, b, L, D, true);
            var kS = Mat2D(kData, b, L, D, true);
            var vS = Mat2D(vData, b, L, D, true);
            var mS = Mat2D(maskData, b, L, L, false);
            var single = ReverseGradOperations.MultiHeadAttention(qS, kS, vS, H, scale, mS);
            single.Backward(Mat2D(dOut, b, L, D, false));

            TestHelpers.AssertTensorEqual(GradData(qS), GradSlice(q, b, L, D), label: $"parallel dQ b{b}");
            TestHelpers.AssertTensorEqual(GradData(kS), GradSlice(k, b, L, D), label: $"parallel dK b{b}");
            TestHelpers.AssertTensorEqual(GradData(vS), GradSlice(v, b, L, D), label: $"parallel dV b{b}");
        }
    }

    static (float[], float[], float[], float[]) GenerateSelfAttention(int B, int L, int D)
    {
        var q = RandFloats(B * L * D, 123);
        var k = RandFloats(B * L * D, 456);
        var v = RandFloats(B * L * D, 789);
        var mask = new float[B * L * L];
        var rng = new Random(2026);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < L; i++)
                for (int j = 0; j < L; j++)
                {
                    int idx = (b * L + i) * L + j;
                    if (j > i)
                        mask[idx] = float.NegativeInfinity;
                    else
                        mask[idx] = (float)(rng.NextDouble() * 0.5 - 0.25);
                }
        return (q, k, v, mask);
    }

    static float[] RandFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1);
        return arr;
    }

    static ReverseGradTensor<float> Mat3D(float[] data, int b, int l, int d, bool requiresGrad)
    {
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(b, l, d);
        return tensor;
    }

    static ReverseGradTensor<float> Mat2D(float[] data, int b, int rows, int cols, bool requiresGrad)
    {
        var arr = new float[rows * cols];
        Array.Copy(data, b * rows * cols, arr, 0, rows * cols);
        return ReverseGradTensor<float>.FromMatrix(arr, rows, cols, requiresGrad);
    }

    static ReverseGradTensor<float> Slice(ReverseGradTensor<float> tensor, int b, int rows, int cols)
    {
        var arr = ExtractSlice(tensor, b, rows, cols);
        return ReverseGradTensor<float>.FromMatrix(arr, rows, cols, requiresGrad: tensor.RequiresGrad);
    }

    static float[] ExtractSlice(ReverseGradTensor<float> tensor, int b, int rows, int cols)
    {
        var arr = new float[rows * cols];
        int off = b * rows * cols;
        for (int i = 0; i < arr.Length; i++)
            arr[i] = tensor[off + i];
        return arr;
    }

    static float[] GradSlice(ReverseGradTensor<float> tensor, int b, int rows, int cols)
    {
        Assert.That(tensor.Grad, Is.Not.Null);
        var grad = tensor.Grad!;
        var arr = new float[rows * cols];
        int off = b * rows * cols;
        for (int i = 0; i < arr.Length; i++)
            arr[i] = grad[off + i];
        return arr;
    }

    static float[] GradData(ReverseGradTensor<float> tensor)
    {
        Assert.That(tensor.Grad, Is.Not.Null);
        var grad = tensor.Grad!;
        var arr = new float[grad.Length];
        for (int i = 0; i < grad.Length; i++)
            arr[i] = grad[i];
        return arr;
    }
}
