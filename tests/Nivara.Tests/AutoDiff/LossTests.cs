using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LossTests
{
    [Test]
    public void MSELoss_Forward_ComputesCorrectValue()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var loss = LossFunctions.MSE(predictions, targets);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(loss[0], Is.EqualTo(3f));
    }

    [Test]
    public void MSELoss_Backward_ProducesGradients()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var loss = LossFunctions.MSE(predictions, targets);
        loss.Backward();

        Assert.That(predictions.Grad, Is.Not.Null);
        Assert.That(predictions.Grad!.Length, Is.EqualTo(3));
        Assert.That(predictions.Grad[0], Is.EqualTo(2f));
        Assert.That(predictions.Grad[1], Is.EqualTo(2f));
        Assert.That(predictions.Grad[2], Is.EqualTo(2f));
    }

    [Test]
    public void L1Loss_Forward_ComputesCorrectValue()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var loss = LossFunctions.L1(predictions, targets);

        Assert.That(loss[0], Is.EqualTo(3f));
    }

    [Test]
    public void BCELoss_Forward_RequiresInputsInZeroOne()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.8f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var loss = LossFunctions.BCE(predictions, targets, eps: 1e-7);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(loss[0], Is.GreaterThan(0f));
        Assert.That(loss[0], Is.EqualTo(-(float)Math.Log(0.8)).Within(1e-5f));
    }

    [Test]
    public void BCEWithLogitsLoss_Forward_WiderInputRange()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var loss = LossFunctions.BCEWithLogits(logits, targets);

        // loss = maxX - x*z + log(1+exp(-|x|))
        // for x=0, z=1: 0 - 0 + log(2) = 0.693
        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(loss[0], Is.EqualTo(0.693147f).Within(1e-5f));
    }

    [Test]
    public void BCEWithLogitsLoss_Backward_ProducesGradients()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var loss = LossFunctions.BCEWithLogits(logits, targets);
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        Assert.That(logits.Grad!.Length, Is.EqualTo(1));
        Assert.That(float.IsNaN(logits.Grad[0]), Is.False);
    }

    [Test]
    public void CrossEntropyLoss_Forward_OneHotTargets()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 1f, 0.1f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 0f, 0f }), requiresGrad: false);

        var logSoftmax = GradOperations.LogSoftmax(logits);
        var nll = GradOperations.Negate(GradOperations.Sum(GradOperations.Multiply(logSoftmax, targets)));

        Assert.That(nll.Length, Is.EqualTo(1));
        Assert.That(nll[0], Is.GreaterThan(0f));
    }

    [Test]
    public void CrossEntropyLoss_Backward_ProducesGradients()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 1f, 0.1f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 0f, 0f }), requiresGrad: false);

        var logSoftmax = GradOperations.LogSoftmax(logits);
        var nll = GradOperations.Negate(GradOperations.Sum(GradOperations.Multiply(logSoftmax, targets)));
        nll.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        Assert.That(logits.Grad!.Length, Is.EqualTo(3));
        for (int i = 0; i < 3; i++)
            Assert.That(float.IsNaN(logits.Grad[i]), Is.False);
    }

    [Test]
    public void MSELoss_Backward_NonNullGradientsOnParameters()
    {
        var weight = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f, 1f }), requiresGrad: true);
        var targets = GradientUtils.Constant(new float[] { 1f, 1f });

        var loss = LossFunctions.MSE(weight, targets);
        loss.Backward();

        Assert.That(weight.Grad, Is.Not.Null);
        Assert.That(weight.Grad!.Length, Is.EqualTo(2));
    }

    [Test]
    public void L1Loss_Backward_ProducesGradients()
    {
        var pred = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: true);
        var target = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var loss = LossFunctions.L1(pred, target);
        loss.Backward();

        Assert.That(pred.Grad, Is.Not.Null);
        Assert.That(pred.Grad![0], Is.EqualTo(-1f));
    }

    [Test]
    public void Softmax_Forward_OutputSumToOne()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var output = GradOperations.Softmax(input);

        Assert.That(output.Length, Is.EqualTo(3));
        var sum = 0f;
        for (int i = 0; i < output.Length; i++)
        {
            Assert.That(float.IsNaN(output[i]), Is.False);
            sum += output[i];
        }
        Assert.That(sum, Is.EqualTo(1f).Within(1e-5f));
    }

    [Test]
    public void LogSoftmax_Forward_CorrectShape()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var output = GradOperations.LogSoftmax(input);

        Assert.That(output.Length, Is.EqualTo(3));
        for (int i = 0; i < output.Length; i++)
        {
            Assert.That(float.IsNaN(output[i]), Is.False);
            Assert.That(output[i], Is.LessThan(0f));
        }
    }

    [Test]
    public void MSELoss_BatchInput_ComputesCorrectShape()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f, 5f }), requiresGrad: true);
        predictions.Reshape(2, 2);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f }), requiresGrad: false);
        targets.Reshape(2, 2);

        var loss = LossFunctions.MSE(predictions, targets);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(loss[0], Is.EqualTo(4f));
    }

    [Test]
    public void CrossEntropyLoss_LogSoftmaxNll_BackwardFlows()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: true);
        var labels = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f, 1f, 0f }), requiresGrad: false);

        var lsm = GradOperations.LogSoftmax(logits);
        var loss = GradOperations.Negate(GradOperations.Sum(GradOperations.Multiply(lsm, labels)));
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        Assert.That(logits.Grad!.Length, Is.EqualTo(3));
        for (int i = 0; i < 3; i++)
            Assert.That(float.IsNaN(logits.Grad[i]), Is.False);
    }
}
