using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class DropoutTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Dropout_Eval_Passthrough_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("dropout_eval_input.bin");
        var expected = TestHelpers.LoadBin("dropout_eval_output.bin");

        using var dropout = new Dropout<float>(0.5);
        dropout.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 32);

        var output = dropout.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 0f, relTol: 0f, label: "Dropout_eval");
    }
}
