using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LossTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BCEWithLogitsLoss_Sum_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bce_with_logits_input.bin");
        var target = TestHelpers.LoadBin("bce_with_logits_target.bin");
        var expected = TestHelpers.LoadBin("bce_with_logits_sum_output.bin")[0];

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 10);
        var targetTensor = ReverseGradTensor<float>.FromArray(target, requiresGrad: false);
        targetTensor.Reshape(4, 10);

        var loss = new BCEWithLogitsLoss<float>();
        var output = loss.Forward(inputTensor, targetTensor, reduceToMean: false);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "BCEWithLogits_sum");
    }

    [Test]
    public void BCEWithLogitsLoss_Mean_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bce_with_logits_input.bin");
        var target = TestHelpers.LoadBin("bce_with_logits_target.bin");
        var expected = TestHelpers.LoadBin("bce_with_logits_mean_output.bin")[0];

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 10);
        var targetTensor = ReverseGradTensor<float>.FromArray(target, requiresGrad: false);
        targetTensor.Reshape(4, 10);

        var loss = new BCEWithLogitsLoss<float>();
        var output = loss.Forward(inputTensor, targetTensor, reduceToMean: true);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "BCEWithLogits_mean");
    }

    [Test]
    public void CrossEntropyLoss_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("cross_entropy_input.bin");
        var targets = TestHelpers.LoadInt64Bin("cross_entropy_target.bin");
        var expected = TestHelpers.LoadBin("cross_entropy_output.bin")[0];

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 10);

        var loss = new CrossEntropyLoss<float>();
        var output = loss.Forward(inputTensor, targets);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "CrossEntropy");
    }

    [Test]
    public void MSELoss_Sum_MatchesPyTorch()
    {
        var pred = TestHelpers.LoadBin("mse_loss_pred.bin");
        var target = TestHelpers.LoadBin("mse_loss_target.bin");
        var expected = TestHelpers.LoadBin("mse_loss_sum_output.bin")[0];

        var predTensor = ReverseGradTensor<float>.FromArray(pred, requiresGrad: false);
        predTensor.Reshape(4, 10);
        var targetTensor = ReverseGradTensor<float>.FromArray(target, requiresGrad: false);
        targetTensor.Reshape(4, 10);

        var loss = new MSELoss<float>();
        var output = loss.Forward(predTensor, targetTensor, reduceToMean: false);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "MSE_sum");
    }

    [Test]
    public void MSELoss_Mean_MatchesPyTorch()
    {
        var pred = TestHelpers.LoadBin("mse_loss_pred.bin");
        var target = TestHelpers.LoadBin("mse_loss_target.bin");
        var expected = TestHelpers.LoadBin("mse_loss_mean_output.bin")[0];

        var predTensor = ReverseGradTensor<float>.FromArray(pred, requiresGrad: false);
        predTensor.Reshape(4, 10);
        var targetTensor = ReverseGradTensor<float>.FromArray(target, requiresGrad: false);
        targetTensor.Reshape(4, 10);

        var loss = new MSELoss<float>();
        var output = loss.Forward(predTensor, targetTensor, reduceToMean: true);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "MSE_mean");
    }

    [Test]
    public void L1Loss_MatchesPyTorch()
    {
        var pred = TestHelpers.LoadBin("l1_loss_pred.bin");
        var target = TestHelpers.LoadBin("l1_loss_target.bin");
        var expected = TestHelpers.LoadBin("l1_loss_output.bin")[0];

        var predTensor = ReverseGradTensor<float>.FromArray(pred, requiresGrad: false);
        predTensor.Reshape(4, 10);
        var targetTensor = ReverseGradTensor<float>.FromArray(target, requiresGrad: false);
        targetTensor.Reshape(4, 10);

        var loss = new L1Loss<float>();
        var output = loss.Forward(predTensor, targetTensor);

        TestHelpers.AssertScalarEqual(expected, TestHelpers.ScalarOutput(output), label: "L1");
    }
}
