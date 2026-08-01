using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class OperationTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Softmax_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("softmax_input.bin");
        var expected = TestHelpers.LoadBin("softmax_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 10);
        var output = ReverseGradOperations.Softmax(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Softmax");
    }

    [Test]
    public void LogSoftmax_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("log_softmax_input.bin");
        var expected = TestHelpers.LoadBin("log_softmax_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 10);
        var output = ReverseGradOperations.LogSoftmax(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "LogSoftmax");
    }

    [Test]
    public void MatMul_MatchesPyTorch()
    {
        var a = TestHelpers.LoadBin("matmul_a.bin");
        var b = TestHelpers.LoadBin("matmul_b.bin");
        var expected = TestHelpers.LoadBin("matmul_output.bin");

        var aTensor = ReverseGradTensor<float>.FromArray(a, requiresGrad: false);
        aTensor.Reshape(4, 8);
        var bTensor = ReverseGradTensor<float>.FromArray(b, requiresGrad: false);
        bTensor.Reshape(8, 16);

        var output = ReverseGradOperations.MatMul(aTensor, bTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "MatMul");
    }

    [Test]
    public void AddBias_MatchesPyTorch()
    {
        var a = TestHelpers.LoadBin("add_bias_a.bin");
        var bias = TestHelpers.LoadBin("add_bias_b.bin");
        var expected = TestHelpers.LoadBin("add_bias_output.bin");

        var aTensor = ReverseGradTensor<float>.FromArray(a, requiresGrad: false);
        aTensor.Reshape(4, 16);
        var biasTensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var output = ReverseGradOperations.AddBias(aTensor, biasTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "AddBias");
    }

    [Test]
    public void MatMulTransposedB_MatchesPyTorch()
    {
        gradScope?.Dispose();
        gradScope = null;

        var a = TestHelpers.LoadBin("matmul_transposed_b_a.bin");
        var b = TestHelpers.LoadBin("matmul_transposed_b_b.bin");
        var expected = TestHelpers.LoadBin("matmul_transposed_b_output.bin");

        var aTensor = ReverseGradTensor<float>.FromArray(a, requiresGrad: false);
        aTensor.Reshape(4, 8);
        var bTensor = ReverseGradTensor<float>.FromArray(b, requiresGrad: false);
        bTensor.Reshape(16, 8);

        var output = ReverseGradOperations.MatMulTransposedB(aTensor, bTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "MatMulTransposedB");
    }
}
