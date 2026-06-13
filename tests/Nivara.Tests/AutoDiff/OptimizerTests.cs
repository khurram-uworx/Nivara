using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class OptimizerTests
{
    [Test]
    public void Optimizer_DoesNotExposeTensorDictionaryParameterGroupOverload()
    {
        var hasUnsafeOverload = typeof(Optimizer<float>)
            .GetMethods()
            .Where(m => m.Name == nameof(Optimizer<float>.AddParameterGroup))
            .Any(m =>
            {
                var parameters = m.GetParameters();
                return parameters.Length > 0 &&
                    parameters[0].ParameterType == typeof(Dictionary<string, ReverseGradTensor<float>>);
            });

        Assert.That(hasUnsafeOverload, Is.False);
    }

    [Test]
    public void Sgd_Step_UpdatesParameterValues()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);

        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(param);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        sgd.Step();

        // param - lr * grad = [1, 2, 3] - 0.1 * [1, 1, 1] = [0.9, 1.9, 2.9]
        Assert.That(param.Tensor[0], Is.EqualTo(0.9f).Within(1e-6f));
        Assert.That(param.Tensor[1], Is.EqualTo(1.9f).Within(1e-6f));
        Assert.That(param.Tensor[2], Is.EqualTo(2.9f).Within(1e-6f));
    }

    [Test]
    public void Sgd_StepWithMomentum_UpdatesParameterValues()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);

        var sgd = new SGD<float>(0.1f, momentum: 0.9);
        sgd.AddParameterGroup(param);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        sgd.Step();

        // With momentum=0.9 and first step: v = 0 + 0.1*grad = 0.1*[1,1,1] = [0.1, 0.1, 0.1]
        // new param = [1, 2, 3] - [0.1, 0.1, 0.1] = [0.9, 1.9, 2.9]
        Assert.That(param.Tensor[0], Is.EqualTo(0.9f).Within(1e-6f));
        Assert.That(param.Tensor[1], Is.EqualTo(1.9f).Within(1e-6f));
        Assert.That(param.Tensor[2], Is.EqualTo(2.9f).Within(1e-6f));
    }

    [Test]
    public void Adam_Step_UpdatesParameterValues()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);

        var adam = new Adam<float>(beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adam.AddParameterGroup(param);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        var oldValues = new float[] { param.Tensor[0], param.Tensor[1], param.Tensor[2] };

        adam.Step();

        // Values should have changed
        Assert.That(param.Tensor[0], Is.Not.EqualTo(oldValues[0]));
        Assert.That(param.Tensor[1], Is.Not.EqualTo(oldValues[1]));
        Assert.That(param.Tensor[2], Is.Not.EqualTo(oldValues[2]));

        for (int i = 0; i < 3; i++)
            Assert.That(float.IsNaN(param.Tensor[i]) || float.IsInfinity(param.Tensor[i]), Is.False);
    }

    [Test]
    public void AdamW_Step_UpdatesParameterValues()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);

        var adamw = new AdamW<float>(beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamw.AddParameterGroup(param);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        var oldValues = new float[] { param.Tensor[0], param.Tensor[1], param.Tensor[2] };

        adamw.Step();

        Assert.That(param.Tensor[0], Is.Not.EqualTo(oldValues[0]));
        Assert.That(param.Tensor[1], Is.Not.EqualTo(oldValues[1]));
        Assert.That(param.Tensor[2], Is.Not.EqualTo(oldValues[2]));

        for (int i = 0; i < 3; i++)
            Assert.That(float.IsNaN(param.Tensor[i]) || float.IsInfinity(param.Tensor[i]), Is.False);
    }

    [Test]
    public void Optimizer_ParameterGroup_DifferentLearningRates()
    {
        var param1 = new Parameter<float>("p1", new float[] { 1f }, requiresGrad: true);
        var param2 = new Parameter<float>("p2", new float[] { 2f }, requiresGrad: true);

        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(param1, 0.01f);
        sgd.AddParameterGroup(param2, 0.1f);

        var loss1 = GradOperations.Sum(param1.Tensor);
        var loss2 = GradOperations.Sum(param2.Tensor);
        var loss = GradOperations.Add(loss1, loss2);
        loss.Backward();

        sgd.Step();

        // param1: 1.0 - 0.01 * 1.0 = 0.99
        Assert.That(param1.Tensor[0], Is.EqualTo(0.99f).Within(1e-6f));
        // param2: 2.0 - 0.1 * 1.0 = 1.9
        Assert.That(param2.Tensor[0], Is.EqualTo(1.9f).Within(1e-6f));
    }

    [Test]
    public void ZeroGrad_ClearsGradients()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);

        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(param);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        Assert.That(param.Tensor.Grad, Is.Not.Null);

        sgd.ZeroGrad();

        Assert.That(param.Tensor.Grad, Is.Null);
    }

    [Test]
    public void Sgd_MultipleSteps_AccumulatesCorrectly()
    {
        var param = new Parameter<float>("w", new float[] { 10f, 20f }, requiresGrad: true);

        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(param);

        // Step 1
        var loss1 = GradOperations.Sum(param.Tensor);
        loss1.Backward();
        sgd.Step();
        Assert.That(param.Tensor[0], Is.EqualTo(9.99f).Within(1e-6f));

        // Step 2 (gradients accumulate after ZeroGrad)
        sgd.ZeroGrad();
        var loss2 = GradOperations.Sum(param.Tensor);
        loss2.Backward();
        sgd.Step();
        Assert.That(param.Tensor[0], Is.EqualTo(9.98f).Within(1e-6f));
    }

    [Test]
    public void AdamW_WithWeightDecay_UpdatesValues()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);

        var adamw = new AdamW<float>(beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamw.AddParameterGroup(param, 0.01f, weightDecay: 0.01f);

        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        var old0 = param.Tensor[0];
        adamw.Step();

        Assert.That(param.Tensor[0], Is.Not.EqualTo(old0));
        Assert.That(float.IsNaN(param.Tensor[0]), Is.False);
        Assert.That(float.IsNaN(param.Tensor[1]), Is.False);
    }

    [Test]
    public void Sgd_MultipleParameterGroups_StepRespectsSeparateParams()
    {
        var paramA = new Parameter<float>("a", new float[] { 1f }, requiresGrad: true);
        var paramB = new Parameter<float>("b", new float[] { 1f }, requiresGrad: true);

        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(paramA, 0.1f);
        sgd.AddParameterGroup(paramB, 0.5f);

        var loss = GradOperations.Add(
            GradOperations.Sum(paramA.Tensor),
            GradOperations.Sum(paramB.Tensor));
        loss.Backward();

        sgd.Step();

        // paramA: 1.0 - 0.1 = 0.9
        Assert.That(paramA.Tensor[0], Is.EqualTo(0.9f).Within(1e-6f));
        // paramB: 1.0 - 0.5 = 0.5
        Assert.That(paramB.Tensor[0], Is.EqualTo(0.5f).Within(1e-6f));
    }

    [Test]
    public void AddParameterGroup_WithoutLearningRate_UsesOptimizerDefault()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.2f);

        sgd.AddParameterGroup(param);
        GradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        Assert.That(param.Tensor[0], Is.EqualTo(0.8f).Within(1e-6f));
    }

    [Test]
    public void AddParameterGroup_WithLearningRate_OverridesOptimizerDefault()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.2f);

        sgd.AddParameterGroup(param, 0.05f);
        GradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        Assert.That(param.Tensor[0], Is.EqualTo(0.95f).Within(1e-6f));
    }

    [Test]
    public void Adam_DefaultConstructor_UsesCommonDefaultLearningRate()
    {
        var adam = new Adam<float>();

        Assert.That(adam.LearningRate, Is.EqualTo(0.001f).Within(1e-8f));
    }

    [Test]
    public void AdamW_DefaultConstructor_UsesCommonDefaultLearningRate()
    {
        var adamw = new AdamW<float>();

        Assert.That(adamw.LearningRate, Is.EqualTo(0.001f).Within(1e-8f));
    }

    [Test]
    public void Adam_ConstructorLearningRate_IsUsedByDefaultParameterGroup()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var adam = new Adam<float>(0.01f);

        adam.AddParameterGroup(param);
        GradOperations.Sum(param.Tensor).Backward();
        adam.Step();

        Assert.That(param.Tensor[0], Is.LessThan(1f));
    }

    [Test]
    public void AdamW_ConstructorLearningRate_IsUsedByDefaultParameterGroup()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var adamw = new AdamW<float>(0.01f);

        adamw.AddParameterGroup(param);
        GradOperations.Sum(param.Tensor).Backward();
        adamw.Step();

        Assert.That(param.Tensor[0], Is.LessThan(1f));
    }

    [Test]
    public void Optimizer_Constructors_RejectNonPositiveLearningRates()
    {
        Assert.Throws<ArgumentException>(() => new SGD<float>(0f));
        Assert.Throws<ArgumentException>(() => new Adam<float>(0f));
        Assert.Throws<ArgumentException>(() => new AdamW<float>(0f));
        Assert.Throws<ArgumentException>(() => new SGD<float>(-0.01f));
        Assert.Throws<ArgumentException>(() => new Adam<float>(-0.01f));
        Assert.Throws<ArgumentException>(() => new AdamW<float>(-0.01f));
    }

    [Test]
    public void AddParameterGroup_RejectsNonPositiveLearningRates()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.1f);

        Assert.Throws<ArgumentException>(() => sgd.AddParameterGroup(param, 0f));
        Assert.Throws<ArgumentException>(() => sgd.AddParameterGroup(param, -0.1f));
    }
}
