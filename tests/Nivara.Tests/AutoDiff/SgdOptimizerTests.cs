using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class SgdOptimizerTests
{
    [Test]
    public void SgdUpdate_SimpleCase_UpdatesCorrectly()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        var loss = GradOperations.Sum(param);
        loss.Backward();

        var updated = SgdOptimizer.SgdUpdate(param, 0.1f);

        // param - lr * grad = [1, 2, 3] - 0.1 * [1, 1, 1] = [0.9, 1.9, 2.9]
        Assert.That(updated[0], Is.EqualTo(0.9f).Within(1e-6f));
        Assert.That(updated[1], Is.EqualTo(1.9f).Within(1e-6f));
        Assert.That(updated[2], Is.EqualTo(2.9f).Within(1e-6f));
        Assert.That(updated.RequiresGrad, Is.False);
    }

    [Test]
    public void SgdUpdate_MultipleSteps_AccumulatesCorrectly()
    {
        var data = NivaraColumn<float>.Create(new float[] { 10.0f, 20.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);

        float lr = 0.01f;

        // Step 1: loss = sum(param), gradient = [1, 1], param -= 0.01 * [1, 1] = [9.99, 19.99]
        var loss1 = GradOperations.Sum(param);
        loss1.Backward();
        var updated1 = SgdOptimizer.SgdUpdate(param, lr);
        Assert.That(updated1[0], Is.EqualTo(9.99f).Within(1e-6f));

        // Step 2: wrap updated1 as a new trainable parameter
        var param2 = new ReverseGradTensor<float>(updated1.ToColumn(), requiresGrad: true);
        var loss2 = GradOperations.Sum(param2);
        loss2.Backward();
        var updated2 = SgdOptimizer.SgdUpdate(param2, lr);
        Assert.That(updated2[0], Is.EqualTo(9.98f).Within(1e-6f));

        updated1.Dispose();
        param2.Dispose();
        updated2.Dispose();
    }

    [Test]
    public void SgdUpdate_PreservesShape()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        param.Reshape(2, 2);
        param.Grad = NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f, 1.0f, 1.0f });

        var updated = SgdOptimizer.SgdUpdate(param, 0.1f);

        Assert.That(updated.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void SgdUpdate_DoubleType_UpdatesCorrectly()
    {
        var data = NivaraColumn<double>.Create(new double[] { 5.0, 10.0 });
        var param = new ReverseGradTensor<double>(data, requiresGrad: true);
        var loss = GradOperations.Sum(param);
        loss.Backward();

        var updated = SgdOptimizer.SgdUpdate(param, 0.5);

        Assert.That(updated[0], Is.EqualTo(4.5).Within(1e-12));
        Assert.That(updated[1], Is.EqualTo(9.5).Within(1e-12));
    }

    [Test]
    public void SgdUpdate_NoGradient_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);

        Assert.That(() => SgdOptimizer.SgdUpdate(param, 0.1f),
            Throws.InvalidOperationException.With.Message.Contains("no gradient"));
    }

    [Test]
    public void SgdUpdate_NegativeLearningRate_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        var loss = GradOperations.Sum(param);
        loss.Backward();

        Assert.That(() => SgdOptimizer.SgdUpdate(param, -0.1f),
            Throws.ArgumentException.With.Message.Contains("positive"));
    }
}
