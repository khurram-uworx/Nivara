using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class GqaKvRepeatTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Forward_RepeatsKvHeadsToQueryHeadCount()
    {
        // numKvHeads=2, headDim=2, so kv width = 4. seqLen=2.
        // Source row: head0 = [a,b], head1 = [c,d].
        // Output (numHeads=4, repeat=2): heads [a,b],[a,b],[c,d],[c,d].
        int seqLen = 2, numKvHeads = 2, headDim = 2, numHeads = 4;
        var src = new float[]
        {
            // row 0: h0=[10,11], h1=[12,13]
            10f, 11f, 12f, 13f,
            // row 1: h0=[20,21], h1=[22,23]
            20f, 21f, 22f, 23f,
        };

        var input = ReverseGradTensor<float>.FromMatrix(src, seqLen, numKvHeads * headDim, requiresGrad: false);
        var output = ReverseGradOperations.GqaRepeatKV(input, numHeads, numKvHeads);

        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, numHeads * headDim }));

        // Row 0
        Assert.That(output[0], Is.EqualTo(10f)); // logical head 0 ← kv 0
        Assert.That(output[1], Is.EqualTo(11f));
        Assert.That(output[2], Is.EqualTo(10f)); // logical head 1 ← kv 0
        Assert.That(output[3], Is.EqualTo(11f));
        Assert.That(output[4], Is.EqualTo(12f)); // logical head 2 ← kv 1
        Assert.That(output[5], Is.EqualTo(13f));
        Assert.That(output[6], Is.EqualTo(12f)); // logical head 3 ← kv 1
        Assert.That(output[7], Is.EqualTo(13f));

        // Row 1
        Assert.That(output[8], Is.EqualTo(20f));
        Assert.That(output[9], Is.EqualTo(21f));
        Assert.That(output[10], Is.EqualTo(20f));
        Assert.That(output[11], Is.EqualTo(21f));
        Assert.That(output[12], Is.EqualTo(22f));
        Assert.That(output[13], Is.EqualTo(23f));
        Assert.That(output[14], Is.EqualTo(22f));
        Assert.That(output[15], Is.EqualTo(23f));
    }

    [Test]
    public void Backward_SumsGradientsAcrossRepeatedHeads()
    {
        int seqLen = 1, numKvHeads = 2, headDim = 1, numHeads = 4;
        // Single row, headDim=1: kv width = 2. Source [x0, x1].
        var src = new float[] { 5f, 9f };
        var input = ReverseGradTensor<float>.FromMatrix(src, seqLen, numKvHeads * headDim, requiresGrad: true);

        var output = ReverseGradOperations.GqaRepeatKV(input, numHeads, numKvHeads);
        var sum = ReverseGradOperations.Sum(output);
        sum.Backward();

        // Output logical heads: [x0, x0, x1, x1]. Sum of output = 2x0 + 2x1.
        // dL/dx0 = 2, dL/dx1 = 2.
        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(2));
        Assert.That(input.Grad![0], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(input.Grad![1], Is.EqualTo(2f).Within(1e-6f));
    }

    [Test]
    public void Construction_RejectsNonDivisibleHeadCounts()
    {
        var input = ReverseGradTensor<float>.FromMatrix(new float[] { 1, 2, 3, 4 }, 1, 4, requiresGrad: false);
        Assert.Throws<ArgumentException>(() => ReverseGradOperations.GqaRepeatKV(input, 4, 3));
    }

    [Test]
    public void InferencePath_BuildsNoGraphNodes()
    {
        int seqLen = 2, numKvHeads = 3, headDim = 2, numHeads = 9;
        var src = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < src.Length; i++) src[i] = i;
        var input = ReverseGradTensor<float>.FromMatrix(src, seqLen, numKvHeads * headDim, requiresGrad: false);

        ReverseGradTensor<float> outside;
        using (GradientUtils.Grad())
            outside = ReverseGradOperations.GqaRepeatKV(input, numHeads, numKvHeads);

        Assert.That(outside.RequiresGrad, Is.False);
        Assert.That(outside.IsLeaf, Is.True);
    }
}
