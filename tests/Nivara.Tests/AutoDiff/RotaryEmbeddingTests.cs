using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class RotaryEmbeddingTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static (float[] cos, float[] sin) Precompute(int seqLen, int headDim, float theta = 10000f)
    {
        int half = headDim / 2;
        var cos = new float[seqLen * half];
        var sin = new float[seqLen * half];
        for (int p = 0; p < seqLen; p++)
        {
            for (int j = 0; j < half; j++)
            {
                float invFreq = MathF.Pow(theta, -2f * j / headDim);
                float angle = p * invFreq;
                cos[p * half + j] = MathF.Cos(angle);
                sin[p * half + j] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    static float RotateValue(float x0, float x1, float c, float s, bool isOdd)
        => isOdd ? x0 * s + x1 * c : x0 * c - x1 * s;

    [Test]
    public void Forward_MatchesManualHfFormula()
    {
        const int headDim = 4;
        const int seqLen = 3;
        using var rope = new RotaryEmbedding<float>(headDim, maxPositionEmbeddings: 64, ropeTheta: 10000f);

        var inputData = new float[seqLen * headDim];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = i + 1; // 1,2,...,12 → makes pairwise values distinct

        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, headDim, requiresGrad: false);
        var output = rope.Forward(input);

        var (cos, sin) = Precompute(seqLen, headDim);
        for (int p = 0; p < seqLen; p++)
        {
            for (int j = 0; j < headDim / 2; j++)
            {
                int i0 = p * headDim + j * 2;
                int i1 = i0 + 1;
                float c = cos[p * (headDim / 2) + j];
                float s = sin[p * (headDim / 2) + j];
                Assert.That(output[i0], Is.EqualTo(RotateValue(inputData[i0], inputData[i1], c, s, false)).Within(1e-5f));
                Assert.That(output[i1], Is.EqualTo(RotateValue(inputData[i0], inputData[i1], c, s, true)).Within(1e-5f));
            }
        }
    }

    [Test]
    public void Forward_ConservesNormPerRow()
    {
        const int headDim = 8;
        const int seqLen = 5;
        using var rope = new RotaryEmbedding<float>(headDim, maxPositionEmbeddings: 32, ropeTheta: 10000f);

        var inputData = new float[seqLen * headDim];
        var rand = new Random(1234);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rand.NextDouble() * 2 - 1);

        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, headDim, requiresGrad: false);
        var output = rope.Forward(input);

        for (int p = 0; p < seqLen; p++)
        {
            float inSq = 0, outSq = 0;
            for (int j = 0; j < headDim; j++)
            {
                inSq += inputData[p * headDim + j] * inputData[p * headDim + j];
                outSq += output[p * headDim + j] * output[p * headDim + j];
            }
            Assert.That(outSq, Is.EqualTo(inSq).Within(1e-4f),
                "RoPE is a per-position rotation and must preserve each row's norm.");
        }
    }

    [Test]
    public void Backward_MatchesAnalyticGradient()
    {
        const int headDim = 4;
        const int seqLen = 2;
        using var rope = new RotaryEmbedding<float>(headDim, maxPositionEmbeddings: 16, ropeTheta: 10000f);

        var inputData = new float[seqLen * headDim];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = i * 0.5f;

        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, headDim, requiresGrad: true);
        var output = rope.Forward(input);
        var sum = ReverseGradOperations.Sum(output);
        sum.Backward();

        var (cos, sin) = Precompute(seqLen, headDim);
        Assert.That(input.Grad, Is.Not.Null);
        for (int p = 0; p < seqLen; p++)
        {
            for (int j = 0; j < headDim / 2; j++)
            {
                int i0 = p * headDim + j * 2;
                int i1 = i0 + 1;
                float c = cos[p * (headDim / 2) + j];
                float s = sin[p * (headDim / 2) + j];
                // With dL/dout = 1, dL/dx0 = cos + sin, dL/dx1 = -sin + cos.
                Assert.That(input.Grad![i0], Is.EqualTo(c + s).Within(1e-5f));
                Assert.That(input.Grad![i1], Is.EqualTo(-s + c).Within(1e-5f));
            }
        }
    }

    [Test]
    public void InferencePath_BuildsNoGraphNodes()
    {
        using var rope = new RotaryEmbedding<float>(4, maxPositionEmbeddings: 8);
        var input = ReverseGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 4, requiresGrad: false);

        ReverseGradTensor<float> outside;
        using (GradientUtils.Grad())
            outside = rope.Forward(input);

        Assert.That(outside.RequiresGrad, Is.False);
        Assert.That(outside.IsLeaf, Is.True);
    }

    [Test]
    public void Forward_RejectsOddHeadDim()
    {
        Assert.Throws<ArgumentException>(() => new RotaryEmbedding<float>(5));
    }
}
