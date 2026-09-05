using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Utilities;
using Nivara.Tests.NivaraTorch;
using NUnit.Framework;
using System.Buffers.Binary;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Torch-parity fixture for the composed student MLP used by the <c>qwen distill</c> mode:
/// <c>Linear(4096 -&gt; 64) -&gt; ReLU -&gt; Linear(64 -&gt; 2) -&gt; CrossEntropyLoss(mean)</c>. The reference
/// values are produced by <c>samples/NivaraInference/Python/qwen_distill_reference.py</c>
/// (fixed seeded weights, one forward + one backward) and dumped as float32 little-endian
/// files into <c>samples/data/qwen-distill/</c>. The fixture set is model-independent
/// (no Qwen checkpoint), so this test can run in CI; it silently <c>Assert.Ignore</c>s when
/// the files are absent (CI/clean), mirroring the other fixture-gated Qwen tests.
///
/// Tolerances: forward logits ~1e-4 (float32, 4096-deep dot accumulation), mean-CE loss
/// ~1e-5, first-layer weight gradient ~5e-4 (matmul + Relu mask). Constants are the planned
/// values; adjust if the empirical run on this machine shows a wider honest spread.
/// </summary>
[TestFixture]
public class QwenDistillStudentParityTests
{
    const int Batch = 4;
    const int FeatDim = 4096;
    const int Hidden = 64;
    const int Classes = 2;

    static string FixtureDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen-distill");

    static readonly string[] RequiredFiles =
    {
        "l1_w.bin", "l1_b.bin", "l2_w.bin", "l2_b.bin", "x.bin", "t.bin",
        "logits.bin", "loss.bin", "grad_l1_w.bin",
    };

    [SetUp]
    public void RequireFixtures()
    {
        foreach (var name in RequiredFiles)
        {
            if (!File.Exists(Path.Combine(FixtureDir, name)))
                Assert.Ignore(
                    "Qwen student-distill fixtures absent; run " +
                    "samples/NivaraInference/Python/qwen_distill_reference.py to generate " +
                    "samples/data/qwen-distill/*.bin, then re-run.");
        }
    }

    [Test]
    public void DistillStudent_ForwardLogits_MatchesTorch()
    {
        var x = ReverseGradTensor<float>.FromMatrix(
            ReadFloat32("x.bin"), Batch, FeatDim, requiresGrad: false);
        var l1 = BuildLinear(FeatDim, Hidden, "l1_w.bin", "l1_b.bin");
        var l2 = BuildLinear(Hidden, Classes, "l2_w.bin", "l2_b.bin");

        ReverseGradTensor<float> logits;
        using (GradientUtils.Grad())
        {
            var hidden = Activation.Relu(l1.Forward(x));
            logits = l2.Forward(hidden);
        }

        var reference = ReadFloat32("logits.bin");
        ReportMaxDiff(reference, TestHelpers.ExtractOutput(logits), "forward logits");
        TestHelpers.AssertTensorEqual(
            reference, TestHelpers.ExtractOutput(logits), 1e-4f, 1e-4f, "forward logits");
    }

    [Test]
    public void DistillStudent_MeanCrossEntropy_MatchesTorch()
    {
        var x = ReverseGradTensor<float>.FromMatrix(
            ReadFloat32("x.bin"), Batch, FeatDim, requiresGrad: false);
        var l1 = BuildLinear(FeatDim, Hidden, "l1_w.bin", "l1_b.bin");
        var l2 = BuildLinear(Hidden, Classes, "l2_w.bin", "l2_b.bin");

        ReverseGradTensor<float> loss;
        using (GradientUtils.Grad())
        {
            var hidden = Activation.Relu(l1.Forward(x));
            var logits = l2.Forward(hidden);
            loss = new CrossEntropyLoss<float>().Forward(logits, targets: ReadInt32("t.bin"));
        }

        float actual = TestHelpers.ScalarOutput(loss);
        float reference = ReadFloat32("loss.bin")[0];
        TestContext.Out.WriteLine($"mean CE loss: torch={reference:G9} nivara={actual:G9} " +
            $"diff={MathF.Abs(reference - actual):G9}");
        TestHelpers.AssertScalarEqual(reference, actual, 1e-5f, 1e-5f, "mean CE loss");
    }

    [Test]
    public void DistillStudent_FirstLayerWeightGrad_MatchesTorch()
    {
        var x = ReverseGradTensor<float>.FromMatrix(
            ReadFloat32("x.bin"), Batch, FeatDim, requiresGrad: false);
        var l1 = BuildLinear(FeatDim, Hidden, "l1_w.bin", "l1_b.bin");
        var l2 = BuildLinear(Hidden, Classes, "l2_w.bin", "l2_b.bin");

        using (GradientUtils.Grad())
        {
            var hidden = Activation.Relu(l1.Forward(x));
            var logits = l2.Forward(hidden);
            var loss = new CrossEntropyLoss<float>().Forward(logits, targets: ReadInt32("t.bin"));
            loss.Backward();
        }

        var reference = ReadFloat32("grad_l1_w.bin");
        var grad = TestHelpers.ExtractGrad(l1.Weight!.Tensor);
        ReportMaxDiff(reference, grad, "dLoss/dW1");
        TestHelpers.AssertTensorEqual(reference, grad, 5e-4f, 5e-4f, "dLoss/dW1");
        Assert.That(l1.Bias!.Tensor.Grad, Is.Not.Null, "bias gradient must be populated too");
    }

    static Linear<float> BuildLinear(int inFeatures, int outFeatures, string weightFile, string biasFile)
    {
        var linear = new Linear<float>(inFeatures, outFeatures);
        linear.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(
            ReadFloat32(weightFile), outFeatures, inFeatures, requiresGrad: true);
        linear.Bias!.Tensor = ReverseGradTensor<float>.FromMatrix(
            ReadFloat32(biasFile), 1, outFeatures, requiresGrad: true);
        return linear;
    }

    static float[] ReadFloat32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(FixtureDir, name));
        var result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    static int[] ReadInt32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(FixtureDir, name));
        var result = new int[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    static void ReportMaxDiff(float[] expected, float[] actual, string label)
    {
        float maxDiff = 0f;
        float maxAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxDiff = MathF.Max(maxDiff, MathF.Abs(expected[i] - actual[i]));
            maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i]));
        }
        TestContext.Out.WriteLine($"{label}: n={expected.Length}, refMaxAbs={maxAbs:G6}, maxAbsDiff={maxDiff:G6}");
    }
}