using Nivara.AutoDiff;
using Nivara.AutoDiff.Extensions;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class MixedPrecisionTrainingTests
{
    [Test]
    public void LinearMixed_Bf16Weights_TrainingLossDecreases()
    {
        using var model = new Linear<float, BFloat16>(1, 1, weightInitializer: new DeterministicInit<BFloat16>());
        var optimizer = new SGD<BFloat16>(BFloat16.CreateChecked(0.05f));
        optimizer.AddParameterGroup(model.Weight!);

        var input = ReverseGradTensor<float>.FromMatrix(new[] { 0f, 1f, 2f, 3f }, 4, 1, requiresGrad: false);
        var target = ReverseGradTensor<float>.FromMatrix(new[] { 1f, 3f, 5f, 7f }, 4, 1, requiresGrad: false);

        var losses = new List<float>();
        for (int epoch = 0; epoch < 10; epoch++)
        {
            using (GradientUtils.Grad())
            {
                var pred = model.Forward(input);
                var diff = pred - target;
                var loss = ReverseGradOperations.Sum(diff * diff);
                losses.Add(loss[0]);
                GradientUtils.ZeroGrad(model.Parameters().Values);
                // zero mixed param grads manually
                model.Weight!.Tensor.ZeroGrad();
                loss.Backward();
            }
            optimizer.Step();
        }

        Assert.That(losses.All(float.IsFinite), Is.True);
        Assert.That(losses[^1], Is.LessThan(losses[0]));
    }
}
