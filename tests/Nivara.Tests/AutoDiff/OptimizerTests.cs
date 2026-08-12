using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class OptimizerTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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
    public void AdamW_MultipleElements_BackwardFlowUpdatesAll()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f, 4f }, requiresGrad: true);

        var adamw = new AdamW<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamw.AddParameterGroup(param, 0.01f, weightDecay: 0.01f);

        var loss = ReverseGradOperations.Sum(param.Tensor);
        loss.Backward();

        var oldValues = new float[4];
        for (int i = 0; i < 4; i++)
            oldValues[i] = param.Tensor[i];

        adamw.Step();

        for (int i = 0; i < 4; i++)
        {
            Assert.That(param.Tensor[i], Is.Not.EqualTo(oldValues[i]));
            Assert.That(float.IsNaN(param.Tensor[i]), Is.False);
            Assert.That(float.IsInfinity(param.Tensor[i]), Is.False);
        }
    }

    [Test]
    public void Optimizer_ParameterGroup_DifferentLearningRates()
    {
        var param1 = new Parameter<float>("p1", new float[] { 1f }, requiresGrad: true);
        var param2 = new Parameter<float>("p2", new float[] { 2f }, requiresGrad: true);

        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(param1, 0.01f);
        sgd.AddParameterGroup(param2, 0.1f);

        var loss1 = ReverseGradOperations.Sum(param1.Tensor);
        var loss2 = ReverseGradOperations.Sum(param2.Tensor);
        var loss = ReverseGradOperations.Add(loss1, loss2);
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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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
        var loss1 = ReverseGradOperations.Sum(param.Tensor);
        loss1.Backward();
        sgd.Step();
        Assert.That(param.Tensor[0], Is.EqualTo(9.99f).Within(1e-6f));

        // Step 2 (gradients accumulate after ZeroGrad)
        sgd.ZeroGrad();
        var loss2 = ReverseGradOperations.Sum(param.Tensor);
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

        var loss = ReverseGradOperations.Sum(param.Tensor);
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

        var loss = ReverseGradOperations.Add(
            ReverseGradOperations.Sum(paramA.Tensor),
            ReverseGradOperations.Sum(paramB.Tensor));
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
        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        Assert.That(param.Tensor[0], Is.EqualTo(0.8f).Within(1e-6f));
    }

    [Test]
    public void AddParameterGroup_WithLearningRate_OverridesOptimizerDefault()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.2f);

        sgd.AddParameterGroup(param, 0.05f);
        ReverseGradOperations.Sum(param.Tensor).Backward();
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
        ReverseGradOperations.Sum(param.Tensor).Backward();
        adam.Step();

        Assert.That(param.Tensor[0], Is.LessThan(1f));
    }

    [Test]
    public void AdamW_ConstructorLearningRate_IsUsedByDefaultParameterGroup()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var adamw = new AdamW<float>(0.01f);

        adamw.AddParameterGroup(param);
        ReverseGradOperations.Sum(param.Tensor).Backward();
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

    [Test]
    public void LearningRate_Set_UpdatesDefaultGroups()
    {
        var defaultParam = new Parameter<float>("default", new float[] { 1f }, requiresGrad: true);
        var overrideParam = new Parameter<float>("override", new float[] { 2f }, requiresGrad: true);

        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(defaultParam);         // uses optimizer.LearningRate
        sgd.AddParameterGroup(overrideParam, 0.5f);  // explicit override

        sgd.LearningRate = 0.1f;

        ReverseGradOperations.Add(
            ReverseGradOperations.Sum(defaultParam.Tensor),
            ReverseGradOperations.Sum(overrideParam.Tensor)).Backward();
        sgd.Step();

        // default group: 1.0 - 0.1 * 1.0 = 0.9
        Assert.That(defaultParam.Tensor[0], Is.EqualTo(0.9f).Within(1e-6f));
        // override group: 2.0 - 0.5 * 1.0 = 1.5
        Assert.That(overrideParam.Tensor[0], Is.EqualTo(1.5f).Within(1e-6f));
    }

    [Test]
    public void LearningRate_Set_RejectsNonPositive()
    {
        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(new Parameter<float>("w", new float[] { 1f }, requiresGrad: true));

        Assert.Throws<ArgumentException>(() => sgd.LearningRate = 0f);
        Assert.Throws<ArgumentException>(() => sgd.LearningRate = -0.1f);
        Assert.That(sgd.LearningRate, Is.EqualTo(0.1f));
    }

    [Test]
    public void LearningRate_Set_MatchesSetGroupLearningRate()
    {
        var baseParam = new Parameter<float>("a", new float[] { 1f }, requiresGrad: true);
        var sgdViaBase = new SGD<float>(0.01f);
        sgdViaBase.AddParameterGroup(baseParam);

        var groupParam = new Parameter<float>("b", new float[] { 1f }, requiresGrad: true);
        var sgdViaGroup = new SGD<float>(0.01f);
        sgdViaGroup.AddParameterGroup(groupParam);

        sgdViaBase.LearningRate = 0.05f;
        sgdViaGroup.SetGroupLearningRate(0, 0.05f);

        ReverseGradOperations.Sum(baseParam.Tensor).Backward();
        ReverseGradOperations.Sum(groupParam.Tensor).Backward();

        sgdViaBase.Step();
        sgdViaGroup.Step();

        Assert.That(baseParam.Tensor[0], Is.EqualTo(groupParam.Tensor[0]).Within(1e-6f));
        Assert.That(baseParam.Tensor[0], Is.EqualTo(0.95f).Within(1e-6f));
    }

    [Test]
    public void LearningRate_Set_DoesNotOverrideExplicitlyManagedGroup()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(param);

        sgd.SetGroupLearningRate(0, 0.05f);
        sgd.LearningRate = 0.2f;

        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        // Explicitly managed group keeps 0.05, not the new base 0.2
        Assert.That(param.Tensor[0], Is.EqualTo(0.95f).Within(1e-6f));
    }

    [Test]
    public void SetGroupWeightDecay_UpdatesGroup()
    {
        var param = new Parameter<float>("w", new float[] { 1f }, requiresGrad: true);
        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(param, 0.1f, weightDecay: 0f);

        sgd.SetGroupWeightDecay(0, 1f);

        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        // wd=1: param - lr * (wd * param + grad) = 1 - 0.1 * (1 * 1 + 1) = 0.8
        Assert.That(param.Tensor[0], Is.EqualTo(0.8f).Within(1e-6f));
    }

    [Test]
    public void ParameterGroup_MutableMembers_NotPublicSettable()
    {
        var groupType = typeof(Optimizer<float>.ParameterGroup);

        foreach (var name in new[]
        {
            nameof(Optimizer<float>.ParameterGroup.LearningRate),
            nameof(Optimizer<float>.ParameterGroup.WeightDecay),
        })
        {
            var setter = groupType.GetProperty(name)!.SetMethod;
            Assert.That(setter, Is.Not.Null);
            Assert.That(setter.IsPublic, Is.False, $"{name} must not be publicly settable");
            Assert.That(setter.IsAssembly, Is.True, $"{name} should be internal-settable");
        }
    }

    [Test]
    public void SgdUpdate_SimpleCase_UpdatesCorrectly()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        var loss = ReverseGradOperations.Sum(param);
        loss.Backward();

        var updated = SGD<float>.SgdUpdate(param, 0.1f);

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
        var loss1 = ReverseGradOperations.Sum(param);
        loss1.Backward();
        var updated1 = SGD<float>.SgdUpdate(param, lr);
        Assert.That(updated1[0], Is.EqualTo(9.99f).Within(1e-6f));

        // Step 2: wrap updated1 as a new trainable parameter
        var param2 = new ReverseGradTensor<float>(updated1.ToColumn(), requiresGrad: true);
        var loss2 = ReverseGradOperations.Sum(param2);
        loss2.Backward();
        var updated2 = SGD<float>.SgdUpdate(param2, lr);
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

        var updated = SGD<float>.SgdUpdate(param, 0.1f);

        Assert.That(updated.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void SgdUpdate_DoubleType_UpdatesCorrectly()
    {
        var data = NivaraColumn<double>.Create(new double[] { 5.0, 10.0 });
        var param = new ReverseGradTensor<double>(data, requiresGrad: true);
        var loss = ReverseGradOperations.Sum(param);
        loss.Backward();

        var updated = SGD<double>.SgdUpdate(param, 0.5);

        Assert.That(updated[0], Is.EqualTo(4.5).Within(1e-12));
        Assert.That(updated[1], Is.EqualTo(9.5).Within(1e-12));
    }

    [Test]
    public void SgdUpdate_NoGradient_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);

        Assert.That(() => SGD<float>.SgdUpdate(param, 0.1f),
            Throws.InvalidOperationException.With.Message.Contains("no gradient"));
    }

    [Test]
    public void SgdUpdate_NegativeLearningRate_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        var loss = ReverseGradOperations.Sum(param);
        loss.Backward();

        Assert.That(() => SGD<float>.SgdUpdate(param, -0.1f),
            Throws.ArgumentException.With.Message.Contains("positive"));
    }

    [Test]
    public void AdamUpdate_SimpleCase_MatchesHandComputedReference()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        ReverseGradOperations.Sum(param).Backward();

        var updated = Adam<float>.AdamUpdate(param, 0.01f, new float[3], new float[3], 1);

        // step=1: same math as Adam_Step_ValuesMatchHandComputedReference -> [0.99, 1.99, 2.99]
        Assert.That(updated[0], Is.EqualTo(0.99f).Within(1e-5f));
        Assert.That(updated[1], Is.EqualTo(1.99f).Within(1e-5f));
        Assert.That(updated[2], Is.EqualTo(2.99f).Within(1e-5f));
        Assert.That(updated.RequiresGrad, Is.False);
    }

    [Test]
    public void AdamUpdate_MultipleSteps_MatchesInstanceStep()
    {
        var paramInst = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamInst = new Adam<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamInst.AddParameterGroup(paramInst);
        for (int i = 0; i < 2; i++)
        {
            ReverseGradOperations.Sum(paramInst.Tensor).Backward();
            adamInst.Step();
            adamInst.ZeroGrad();
        }

        var current = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: true);
        var expAvg = new float[3];
        var expAvgSq = new float[3];
        for (int step = 1; step <= 2; step++)
        {
            ReverseGradOperations.Sum(current).Backward();
            using var updated = Adam<float>.AdamUpdate(current, 0.01f, expAvg, expAvgSq, step);
            current.Dispose();
            current = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(updated.ToColumn().ToArray()), requiresGrad: true);
        }

        for (int i = 0; i < 3; i++)
            Assert.That(current[i], Is.EqualTo(paramInst.Tensor[i]).Within(1e-5f));
    }

    [Test]
    public void AdamUpdate_NoGradient_Throws()
    {
        var param = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }), requiresGrad: true);

        Assert.That(() => Adam<float>.AdamUpdate(param, 0.1f, new float[2], new float[2], 1),
            Throws.InvalidOperationException.With.Message.Contains("no gradient"));
    }

    [Test]
    public void AdamUpdate_NegativeLearningRate_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1f, 2f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        ReverseGradOperations.Sum(param).Backward();

        Assert.That(() => Adam<float>.AdamUpdate(param, -0.1f, new float[2], new float[2], 1),
            Throws.ArgumentException.With.Message.Contains("positive"));
    }

    [Test]
    public void AdamWUpdate_SimpleCase_MatchesHandComputedReference()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1f, 2f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        ReverseGradOperations.Sum(param).Backward();

        var updated = AdamW<float>.AdamWUpdate(param, 0.01f, new float[2], new float[2], 1, weightDecay: 0.01f);

        // step=1: matches AdamW_StepWithWeightDecay_ValuesMatchHandComputedReference
        Assert.That(updated[0], Is.EqualTo(0.9899f).Within(1e-5f));
        Assert.That(updated[1], Is.EqualTo(1.9898f).Within(1e-5f));
        Assert.That(updated.RequiresGrad, Is.False);
    }

    [Test]
    public void AdamWUpdate_MultipleSteps_MatchesInstanceStep()
    {
        var paramInst = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);
        var adamwInst = new AdamW<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamwInst.AddParameterGroup(paramInst, 0.01f, weightDecay: 0.01f);
        for (int i = 0; i < 2; i++)
        {
            ReverseGradOperations.Sum(paramInst.Tensor).Backward();
            adamwInst.Step();
            adamwInst.ZeroGrad();
        }

        var current = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }), requiresGrad: true);
        var expAvg = new float[2];
        var expAvgSq = new float[2];
        for (int step = 1; step <= 2; step++)
        {
            ReverseGradOperations.Sum(current).Backward();
            using var updated = AdamW<float>.AdamWUpdate(current, 0.01f, expAvg, expAvgSq, step, weightDecay: 0.01f);
            current.Dispose();
            current = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(updated.ToColumn().ToArray()), requiresGrad: true);
        }

        for (int i = 0; i < 2; i++)
            Assert.That(current[i], Is.EqualTo(paramInst.Tensor[i]).Within(1e-5f));
    }

    [Test]
    public void AdamWUpdate_NoGradient_Throws()
    {
        var param = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }), requiresGrad: true);

        Assert.That(() => AdamW<float>.AdamWUpdate(param, 0.1f, new float[2], new float[2], 1),
            Throws.InvalidOperationException.With.Message.Contains("no gradient"));
    }

    [Test]
    public void AdamWUpdate_NegativeLearningRate_Throws()
    {
        var data = NivaraColumn<float>.Create(new float[] { 1f, 2f });
        var param = new ReverseGradTensor<float>(data, requiresGrad: true);
        ReverseGradOperations.Sum(param).Backward();

        Assert.That(() => AdamW<float>.AdamWUpdate(param, -0.1f, new float[2], new float[2], 1),
            Throws.ArgumentException.With.Message.Contains("positive"));
    }

    [Test]
    public void Step_WithoutZeroGrad_AccumulatesGradients()
    {
        var param = new Parameter<float>("w", new float[] { 10f, 20f }, requiresGrad: true);
        var sgd = new SGD<float>(0.01f);
        sgd.AddParameterGroup(param);

        // Step 1: grad = [1, 1]; param -> [9.99, 19.99]
        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        // Step 2 WITHOUT ZeroGrad: backward accumulates onto the existing Grad slot.
        // Grad was [1, 1]; the new gradient [1, 1] accumulates to [2, 2].
        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        // param = 10 - 0.01*1 (step 1) - 0.01*(1+1) (step 2) = 9.97
        Assert.That(param.Tensor[0], Is.EqualTo(9.97f).Within(1e-6f));
        Assert.That(param.Tensor[1], Is.EqualTo(19.97f).Within(1e-6f));
    }

    [Test]
    public void Sgd_Step_ReusesParameterTensorInPlace()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);
        var sgd = new SGD<float>(0.1f);
        sgd.AddParameterGroup(param);

        var original = param.Tensor;
        var version = param.Version;

        ReverseGradOperations.Sum(param.Tensor).Backward();
        sgd.Step();

        Assert.That(object.ReferenceEquals(param.Tensor, original), Is.True);
        Assert.That(param.Version, Is.EqualTo(version + 1));
        Assert.That(param.Tensor[0], Is.EqualTo(0.9f).Within(1e-6f));
    }

    [Test]
    public void Sgd_StepWithMomentum_MultipleSteps_MatchesHandComputedReference()
    {
        var param = new Parameter<float>("w", new float[] { 10f, 20f }, requiresGrad: true);
        var sgd = new SGD<float>(0.01f, momentum: 0.9);
        sgd.AddParameterGroup(param);

        var lr = 0.01f;
        var velocity = new[] { 0f, 0f };
        var expected = new[] { 10f, 20f };

        for (int step = 0; step < 3; step++)
        {
            sgd.ZeroGrad();
            ReverseGradOperations.Sum(param.Tensor).Backward();
            sgd.Step();

            for (int i = 0; i < 2; i++)
            {
                velocity[i] = 0.9f * velocity[i] + lr;
                expected[i] -= velocity[i];
                Assert.That(param.Tensor[i], Is.EqualTo(expected[i]).Within(1e-5f));
            }
        }
    }

    [Test]
    public void Adam_Step_ReusesParameterTensorInPlace()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adam = new Adam<float>(beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adam.AddParameterGroup(param);

        var original = param.Tensor;
        var version = param.Version;

        ReverseGradOperations.Sum(param.Tensor).Backward();
        adam.Step();

        Assert.That(object.ReferenceEquals(param.Tensor, original), Is.True);
        Assert.That(param.Version, Is.EqualTo(version + 1));
    }

    [Test]
    public void Adam_Step_ValuesMatchHandComputedReference()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adam = new Adam<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adam.AddParameterGroup(param);

        ReverseGradOperations.Sum(param.Tensor).Backward();
        adam.Step();

        // step=1: biasCorr1 = 1 - 0.9 = 0.1, biasCorr2 = 1 - 0.999 = 0.001
        // expAvg = 0.1*grad = 0.1, expAvgSq = 0.001*grad^2 = 0.001
        // mHat = 1, vHat = 1, update = lr * 1 / (sqrt(1) + eps) = 0.01
        // new = data - update = [0.99, 1.99, 2.99]
        Assert.That(param.Tensor[0], Is.EqualTo(0.99f).Within(1e-5f));
        Assert.That(param.Tensor[1], Is.EqualTo(1.99f).Within(1e-5f));
        Assert.That(param.Tensor[2], Is.EqualTo(2.99f).Within(1e-5f));
    }

    [Test]
    public void AdamW_Step_ReusesParameterTensorInPlace()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamw = new AdamW<float>(beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamw.AddParameterGroup(param);

        var original = param.Tensor;
        var version = param.Version;

        ReverseGradOperations.Sum(param.Tensor).Backward();
        adamw.Step();

        Assert.That(object.ReferenceEquals(param.Tensor, original), Is.True);
        Assert.That(param.Version, Is.EqualTo(version + 1));
    }

    [Test]
    public void AdamW_StepWithWeightDecay_ValuesMatchHandComputedReference()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);
        var adamw = new AdamW<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamw.AddParameterGroup(param, 0.01f, weightDecay: 0.01f);

        ReverseGradOperations.Sum(param.Tensor).Backward();
        adamw.Step();

        // step=1: update = lr * mHat/denom + lr * wd * data = 0.01 + 0.01*0.01*data
        // data=1 -> 0.01 + 0.0001 = 0.0101 -> new = 0.9899
        // data=2 -> 0.01 + 0.0002 = 0.0102 -> new = 1.9898
        Assert.That(param.Tensor[0], Is.EqualTo(0.9899f).Within(1e-5f));
        Assert.That(param.Tensor[1], Is.EqualTo(1.9898f).Within(1e-5f));
    }

    [Test]
    public void Adam_Step_FloatAndDouble_ProduceEquivalentValues()
    {
        var data = new double[] { 1.0, 2.0, 3.0 };

        var paramF = new Parameter<float>("wf", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamF = new Adam<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamF.AddParameterGroup(paramF);
        ReverseGradOperations.Sum(paramF.Tensor).Backward();
        adamF.Step();

        var paramD = new Parameter<double>("wd", data, requiresGrad: true);
        var adamD = new Adam<double>(0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamD.AddParameterGroup(paramD);
        ReverseGradOperations.Sum(paramD.Tensor).Backward();
        adamD.Step();

        for (int i = 0; i < 3; i++)
            Assert.That((double)paramF.Tensor[i], Is.EqualTo(paramD.Tensor[i]).Within(1e-6));
    }

    [Test]
    public void Adam_Step_Half_ProducesValuesCloseToFloat()
    {
        var paramF = new Parameter<float>("wf", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamF = new Adam<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamF.AddParameterGroup(paramF);
        ReverseGradOperations.Sum(paramF.Tensor).Backward();
        adamF.Step();

        var paramH = new Parameter<Half>("wh", new Half[] { (Half)1f, (Half)2f, (Half)3f }, requiresGrad: true);
        var adamH = new Adam<Half>((Half)0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamH.AddParameterGroup(paramH);
        ReverseGradOperations.Sum(paramH.Tensor).Backward();
        adamH.Step();

        for (int i = 0; i < 3; i++)
            Assert.That((double)paramH.Tensor[i], Is.EqualTo((double)paramF.Tensor[i]).Within(1e-3));
    }

    [Test]
    public void AdamW_Step_Half_ProducesValuesCloseToFloat()
    {
        var paramF = new Parameter<float>("wf", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamwF = new AdamW<float>(0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamwF.AddParameterGroup(paramF);
        ReverseGradOperations.Sum(paramF.Tensor).Backward();
        adamwF.Step();

        var paramH = new Parameter<Half>("wh", new Half[] { (Half)1f, (Half)2f, (Half)3f }, requiresGrad: true);
        var adamwH = new AdamW<Half>((Half)0.01f, beta1: 0.9, beta2: 0.999, eps: 1e-8);
        adamwH.AddParameterGroup(paramH);
        ReverseGradOperations.Sum(paramH.Tensor).Backward();
        adamwH.Step();

        for (int i = 0; i < 3; i++)
            Assert.That((double)paramH.Tensor[i], Is.EqualTo((double)paramF.Tensor[i]).Within(1e-3));
    }

    [Test]
    public void Adam_StateDict_LoadStateDict_FreshOptimizer_RestoresState()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adam = new Adam<float>(0.01f);
        adam.AddParameterGroup(param);

        for (int i = 0; i < 3; i++)
        {
            var loss = ReverseGradOperations.Sum(param.Tensor);
            loss.Backward();
            adam.Step();
            adam.ZeroGrad();
        }

        var state = adam.StateDict();

        var freshParam = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var restored = new Adam<float>(0.01f);
        restored.AddParameterGroup(freshParam);
        restored.LoadStateDict(state);

        var reloadedState = restored.StateDict();

        Assert.That(reloadedState.Keys, Is.EquivalentTo(state.Keys));
        Assert.That(reloadedState["step"][0], Is.EqualTo(state["step"][0]));
        int bufIdx = 0;
        while (state.ContainsKey($"expAvg_{bufIdx}"))
        {
            Assert.That(reloadedState[$"expAvg_{bufIdx}"], Is.EqualTo(state[$"expAvg_{bufIdx}"]).Within(1e-6f));
            Assert.That(reloadedState[$"expAvgSq_{bufIdx}"], Is.EqualTo(state[$"expAvgSq_{bufIdx}"]).Within(1e-6f));
            bufIdx++;
        }
    }

    [Test]
    public void AdamW_StateDict_LoadStateDict_FreshOptimizer_RestoresState()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var adamw = new AdamW<float>(0.01f);
        adamw.AddParameterGroup(param);

        for (int i = 0; i < 3; i++)
        {
            var loss = ReverseGradOperations.Sum(param.Tensor);
            loss.Backward();
            adamw.Step();
            adamw.ZeroGrad();
        }

        var state = adamw.StateDict();

        var freshParam = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var restored = new AdamW<float>(0.01f);
        restored.AddParameterGroup(freshParam);
        restored.LoadStateDict(state);

        var reloadedState = restored.StateDict();

        Assert.That(reloadedState.Keys, Is.EquivalentTo(state.Keys));
        Assert.That(reloadedState["step"][0], Is.EqualTo(state["step"][0]));
        int bufIdx = 0;
        while (state.ContainsKey($"expAvg_{bufIdx}"))
        {
            Assert.That(reloadedState[$"expAvg_{bufIdx}"], Is.EqualTo(state[$"expAvg_{bufIdx}"]).Within(1e-6f));
            Assert.That(reloadedState[$"expAvgSq_{bufIdx}"], Is.EqualTo(state[$"expAvgSq_{bufIdx}"]).Within(1e-6f));
            bufIdx++;
        }
    }

    [Test]
    public void Sgd_StateDict_LoadStateDict_FreshOptimizer_RestoresMomentum()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var sgd = new SGD<float>(0.01f, momentum: 0.9);
        sgd.AddParameterGroup(param);

        for (int i = 0; i < 3; i++)
        {
            var loss = ReverseGradOperations.Sum(param.Tensor);
            loss.Backward();
            sgd.Step();
            sgd.ZeroGrad();
        }

        var state = sgd.StateDict();
        Assert.That(state, Does.ContainKey("velocity_0"));

        var freshParam = new Parameter<float>("w", new float[] { 1f, 2f, 3f }, requiresGrad: true);
        var restored = new SGD<float>(0.01f, momentum: 0.9);
        restored.AddParameterGroup(freshParam);
        restored.LoadStateDict(state);

        var reloadedState = restored.StateDict();
        Assert.That(reloadedState.Keys, Is.EquivalentTo(state.Keys));
        Assert.That(reloadedState["velocity_0"], Is.EqualTo(state["velocity_0"]).Within(1e-6f));
    }
}
