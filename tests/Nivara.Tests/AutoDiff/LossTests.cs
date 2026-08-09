using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LossTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void MSELoss_Forward_ComputesCorrectValue()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 3f, 4f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var mse = new MSELoss<float>();
        var loss = mse.Forward(predictions, targets);

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

        var mse = new MSELoss<float>();
        var loss = mse.Forward(predictions, targets);
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

        var l1 = new L1Loss<float>();
        var loss = l1.Forward(predictions, targets);

        Assert.That(loss[0], Is.EqualTo(3f));
    }

    [Test]
    public void BCELoss_Forward_RequiresInputsInZeroOne()
    {
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.8f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var bce = new BCELoss<float>(eps: 1e-7);
        var loss = bce.Forward(predictions, targets);

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

        var bceLogits = new BCEWithLogitsLoss<float>();
        var loss = bceLogits.Forward(logits, targets);

        // loss = maxX - x*z + log(1+exp(-|x|))
        // for x=0, z=1: 0 - 0 + log(2) = 0.693
        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(loss[0], Is.EqualTo(0.693147f).Within(1e-5f));
    }

    [Test]
    public void Activation_LeakyRelu_DefaultSlope_IsNotZero()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { -2f, -1f, 0f, 1f, 2f }), requiresGrad: true);

        var output = Activation.LeakyRelu(input);
        Assert.That(output[0], Is.EqualTo(-0.02f).Within(1e-6f));
        Assert.That(output[1], Is.EqualTo(-0.01f).Within(1e-6f));
        Assert.That(output[2], Is.EqualTo(0f).Within(1e-6f));
        Assert.That(output[3], Is.EqualTo(1f).Within(1e-6f));
        Assert.That(output[4], Is.EqualTo(2f).Within(1e-6f));

        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();
        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad![0], Is.EqualTo(0.01f).Within(1e-6f));
        Assert.That(input.Grad[1], Is.EqualTo(0.01f).Within(1e-6f));
        Assert.That(input.Grad[3], Is.EqualTo(1f).Within(1e-6f));
    }

    [Test]
    public void BCEWithLogitsLoss_Backward_AtZeroInput_ComputesCorrectGradient()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var bceLogits = new BCEWithLogitsLoss<float>();
        var loss = bceLogits.Forward(logits, targets);
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        float expected = MathF.Exp(0f) / (1f + MathF.Exp(0f)) - 1f;
        Assert.That(logits.Grad![0], Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void BCEWithLogitsLoss_Backward_AtNonZeroInput_ComputesCorrectGradient()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: false);

        var bceLogits = new BCEWithLogitsLoss<float>();
        var loss = bceLogits.Forward(logits, targets);
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        float sigmoid2 = 1f / (1f + MathF.Exp(-2f));
        Assert.That(logits.Grad![0], Is.EqualTo(sigmoid2).Within(1e-5f));
    }

    [Test]
    public void BCEWithLogitsLoss_Backward_ProducesGradients()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f }), requiresGrad: false);

        var bceLogits = new BCEWithLogitsLoss<float>();
        var loss = bceLogits.Forward(logits, targets);
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        Assert.That(logits.Grad!.Length, Is.EqualTo(1));
        Assert.That(float.IsNaN(logits.Grad[0]), Is.False);
    }

    [Test]
    public void BCEWithLogitsLoss_ReduceToMean_BackwardDividesGradientByN()
    {
        int n = 4;
        var logitsSum = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f, 0f, 0f, 0f }), requiresGrad: true);
        var targetsSum = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }), requiresGrad: false);

        var bceLogits = new BCEWithLogitsLoss<float>();
        var sumLoss = bceLogits.Forward(logitsSum, targetsSum, reduceToMean: false);
        sumLoss.Backward();
        float gradSum = logitsSum.Grad![0];

        var logitsMean = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f, 0f, 0f, 0f }), requiresGrad: true);
        var targetsMean = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }), requiresGrad: false);

        var meanLoss = bceLogits.Forward(logitsMean, targetsMean, reduceToMean: true);
        meanLoss.Backward();
        float gradMean = logitsMean.Grad![0];

        Assert.That(gradMean, Is.EqualTo(gradSum / n).Within(1e-5f));
    }

    [Test]
    public void BCEWithLogitsLoss_ReduceToMean_ReturnsMeanOfElementLosses()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f, 0f, 0f, 0f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }), requiresGrad: false);

        var bceLogits = new BCEWithLogitsLoss<float>();
        var sumLoss = bceLogits.Forward(logits, targets, reduceToMean: false);
        var meanLoss = bceLogits.Forward(logits, targets, reduceToMean: true);

        // sum = 4 * log(2) = 2.7726, mean = log(2) = 0.6931
        Assert.That(sumLoss[0], Is.EqualTo(4f * 0.693147f).Within(1e-4f));
        Assert.That(meanLoss[0], Is.EqualTo(0.693147f).Within(1e-4f));
        Assert.That(meanLoss[0], Is.EqualTo(sumLoss[0] / 4f).Within(1e-5f));
    }

    [Test]
    public void CrossEntropyLoss_Forward_OneHotTargets()
    {
        var logits = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 1f, 0.1f }), requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 0f, 0f }), requiresGrad: false);

        var logSoftmax = ReverseGradOperations.LogSoftmax(logits);
        var nll = ReverseGradOperations.Negate(ReverseGradOperations.Sum(ReverseGradOperations.Multiply(logSoftmax, targets)));

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

        var logSoftmax = ReverseGradOperations.LogSoftmax(logits);
        var nll = ReverseGradOperations.Negate(ReverseGradOperations.Sum(ReverseGradOperations.Multiply(logSoftmax, targets)));
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

        var mse = new MSELoss<float>();
        var loss = mse.Forward(weight, targets);
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

        var l1 = new L1Loss<float>();
        var loss = l1.Forward(pred, target);
        loss.Backward();

        Assert.That(pred.Grad, Is.Not.Null);
        Assert.That(pred.Grad![0], Is.EqualTo(-1f));
    }

    [Test]
    public void Softmax_Forward_OutputSumToOne()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var softmax = new Softmax<float>();
        var output = softmax.Forward(input);

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

        var logSoftmax = new LogSoftmax<float>();
        var output = logSoftmax.Forward(input);

        Assert.That(output.Length, Is.EqualTo(3));
        for (int i = 0; i < output.Length; i++)
        {
            Assert.That(float.IsNaN(output[i]), Is.False);
            Assert.That(output[i], Is.LessThan(0f));
        }
    }

    [Test]
    public void Softmax_DimZero_On2D_SoftmaxesOverRows()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 1.5f, 0.5f, 2.5f, 3.5f, -1f, 0f, 1f, 2f }),
            requiresGrad: false);
        input.Reshape(3, 4);

        var softmax = new Softmax<float>(dim: 0);
        var output = softmax.Forward(input);

        for (int col = 0; col < 4; col++)
        {
            float sum = output[col] + output[col + 4] + output[col + 8];
            Assert.That(sum, Is.EqualTo(1f).Within(1e-5f), $"Column {col}");
        }
    }

    [Test]
    public void LogSoftmax_DimZero_On2D_MatchesColumnWise()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 1.5f, 0.5f, 2.5f, 3.5f, -1f, 0f, 1f, 2f }),
            requiresGrad: false);
        input.Reshape(3, 4);

        var logSoftmax = new LogSoftmax<float>(dim: 0);
        var output = logSoftmax.Forward(input);

        for (int col = 0; col < 4; col++)
        {
            float max = Math.Max(input[col], Math.Max(input[col + 4], input[col + 8]));
            float sum = MathF.Exp(input[col] - max) + MathF.Exp(input[col + 4] - max) + MathF.Exp(input[col + 8] - max);
            Assert.That(output[col], Is.EqualTo(input[col] - max - MathF.Log(sum)).Within(1e-5f), $"Column {col}");
            Assert.That(output[col + 4], Is.EqualTo(input[col + 4] - max - MathF.Log(sum)).Within(1e-5f), $"Column {col}");
            Assert.That(output[col + 8], Is.EqualTo(input[col + 8] - max - MathF.Log(sum)).Within(1e-5f), $"Column {col}");
        }
    }

    [Test]
    public void Softmax_DimZero_Backward_FlowsGradients()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 1.5f, 0.5f, 2.5f, 3.5f, -1f, 0f, 1f, 2f }),
            requiresGrad: true);
        input.Reshape(3, 4);

        var softmax = new Softmax<float>(dim: 0);
        var output = softmax.Forward(input);

        var seed = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f, 1.1f, 1.2f, 1.3f }),
            requiresGrad: false);
        seed.Reshape(3, 4);
        output.Backward(seed);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(12));
        for (int i = 0; i < 12; i++)
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
    }

    [Test]
    public void Softmax_DimOutOfRange_Throws()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f }), requiresGrad: false);
        input.Reshape(2, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Softmax<float>(dim: 2).Forward(input));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Softmax<float>(dim: -3).Forward(input));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LogSoftmax<float>(dim: 2).Forward(input));
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

        var mse = new MSELoss<float>();
        var loss = mse.Forward(predictions, targets);

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

        var lsm = ReverseGradOperations.LogSoftmax(logits);
        var loss = ReverseGradOperations.Negate(ReverseGradOperations.Sum(ReverseGradOperations.Multiply(lsm, labels)));
        loss.Backward();

        Assert.That(logits.Grad, Is.Not.Null);
        Assert.That(logits.Grad!.Length, Is.EqualTo(3));
        for (int i = 0; i < 3; i++)
            Assert.That(float.IsNaN(logits.Grad[i]), Is.False);
    }
}
