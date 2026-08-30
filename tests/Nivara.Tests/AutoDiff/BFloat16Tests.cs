using Nivara.AutoDiff;
using Nivara.AutoDiff.Extensions;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Kernel tests for BFloat16 AutoDiff support (issue #137). BFloat16 satisfies
/// IFloatingPointIeee754&lt;T&gt; on .NET 11, so every generic op compiles against it;
/// these tests verify runtime admission and numeric parity against float references.
/// BFloat16 has ~8 mantissa bits, so comparisons use generous tolerances.
/// </summary>
[TestFixture]
public class BFloat16Tests
{
    const float RelTol = 0.02f;
    const float AbsTol = 0.01f;

    static BFloat16[] ToBFloat16(float[] values)
    {
        var result = new BFloat16[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = BFloat16.CreateChecked(values[i]);
        return result;
    }

    static float[] ToFloats<T>(ReverseGradTensor<T> tensor) where T : struct, IFloatingPointIeee754<T>
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
            result[i] = float.CreateChecked(tensor[i]);
        return result;
    }

    static ReverseGradTensor<BFloat16> Tensor(float[] values, bool requiresGrad = false) =>
        new(NivaraColumn<BFloat16>.Create(ToBFloat16(values)), requiresGrad);

    static ReverseGradTensor<BFloat16> Matrix(float[] values, int rows, int cols, bool requiresGrad = false) =>
        ReverseGradTensor<BFloat16>.FromMatrix(ToBFloat16(values), rows, cols, requiresGrad);

    static ReverseGradTensor<float> FloatTensor(float[] values, bool requiresGrad = false) =>
        new(NivaraColumn<float>.Create(values), requiresGrad);

    static void AssertClose(float expected, float actual, string label)
    {
        var tolerance = Math.Max(AbsTol, RelTol * Math.Abs(expected));
        Assert.That(Math.Abs(actual - expected), Is.LessThanOrEqualTo(tolerance), label);
    }

    static void AssertParity(float[] expected, float[] actual, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label}: length");
        for (int i = 0; i < expected.Length; i++)
            AssertClose(expected[i], actual[i], $"{label}[{i}]");
    }

    [Test]
    public void TypeValidator_BFloat16_IsSupportedType()
    {
        Assert.That(TypeValidator.IsSupportedType(typeof(BFloat16)), Is.True);
        Assert.That(TypeValidator.IsSupported<BFloat16>(), Is.True);
        Assert.That(TypeValidator.GetSupportedTypes(), Does.Contain(typeof(BFloat16)));
    }

    [Test]
    public void ElementwiseOps_BFloat16Forward_ParityWithFloatReference()
    {
        float[] a = [-2f, -1f, 0f, 1f, 2f];
        float[] b = [0.5f, 1f, -1f, 2f, 0f];

        var bfa = FloatTensor(a);
        var bfb = FloatTensor(b);
        var bfA = Tensor(a);
        var bfB = Tensor(b);

        AssertParity(ToFloats(ReverseGradOperations.Add(bfa, bfb)), ToFloats(ReverseGradOperations.Add(bfA, bfB)), "Add");
        AssertParity(ToFloats(ReverseGradOperations.Subtract(bfa, bfb)), ToFloats(ReverseGradOperations.Subtract(bfA, bfB)), "Subtract");
        AssertParity(ToFloats(ReverseGradOperations.Multiply(bfa, bfb)), ToFloats(ReverseGradOperations.Multiply(bfA, bfB)), "Multiply");
        AssertParity(ToFloats(ReverseGradOperations.Sigmoid(bfa)), ToFloats(ReverseGradOperations.Sigmoid(bfA)), "Sigmoid");
        AssertParity(ToFloats(ReverseGradOperations.Tanh(bfa)), ToFloats(ReverseGradOperations.Tanh(bfA)), "Tanh");
        AssertParity(ToFloats(ReverseGradOperations.Relu(bfa)), ToFloats(ReverseGradOperations.Relu(bfA)), "Relu");
        AssertParity(ToFloats(ReverseGradOperations.Gelu(bfa)), ToFloats(ReverseGradOperations.Gelu(bfA)), "Gelu");
    }

    [Test]
    public void ElementwiseOps_BFloat16Backward_GradientsMatchFloatReference()
    {
        float[] aData = [-2f, -1f, 0f, 1f, 2f];
        float[] bData = [0.5f, 1f, -1f, 2f, 0f];

        var bfA = Tensor(aData, requiresGrad: true);
        var bfB = Tensor(bData, requiresGrad: true);
        using (GradientUtils.Grad())
        {
            var z = ReverseGradOperations.Sum(ReverseGradOperations.Add(
                ReverseGradOperations.Multiply(bfA, bfB), bfB));
            z.Backward();
        }

        Assert.That(bfA.Grad, Is.Not.Null);
        Assert.That(bfB.Grad, Is.Not.Null);
        for (int i = 0; i < aData.Length; i++)
        {
            AssertClose(bData[i], float.CreateChecked(bfA.Grad![i]), $"dL/da[{i}] should equal b[{i}]");
            AssertClose(aData[i] + 1f, float.CreateChecked(bfB.Grad![i]), $"dL/db[{i}] should equal a[{i}]+1");
        }
    }

    [Test]
    public void Softmax_BFloat16Forward_ParityWithFloatReference()
    {
        float[] data = [-2f, -1f, 0f, 1f, 2f];

        var floatResult = ToFloats(ReverseGradOperations.Softmax(FloatTensor(data), 0));
        var bfResult = ToFloats(ReverseGradOperations.Softmax(Tensor(data), 0));

        AssertParity(floatResult, bfResult, "Softmax");

        float sum = 0;
        foreach (var v in bfResult) sum += v;
        Assert.That(sum, Is.EqualTo(1f).Within(RelTol));
    }

    [Test]
    public void MatMul_BFloat16Forward_ParityWithFloatReference()
    {
        float[] aData = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] bData = [7f, 8f, 9f, 10f, 11f, 12f];

        var floatResult = ToFloats(ReverseGradOperations.MatMul(
            ReverseGradTensor<float>.FromMatrix(aData, 2, 3, requiresGrad: false),
            ReverseGradTensor<float>.FromMatrix(bData, 3, 2, requiresGrad: false)));
        var bfResult = ToFloats(ReverseGradOperations.MatMul(Matrix(aData, 2, 3), Matrix(bData, 3, 2)));

        AssertParity(floatResult, bfResult, "MatMul");
    }

    [Test]
    public void LinearModule_BFloat16SgdTraining_LossDecreases()
    {
        using var model = new Linear<BFloat16>(1, 1);
        var optimizer = new SGD<BFloat16>(BFloat16.CreateChecked(0.05f));
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var input = Matrix([0f, 1f, 2f, 3f], 4, 1);
        var target = Matrix([1f, 3f, 5f, 7f], 4, 1);

        var losses = new List<float>();
        for (int epoch = 0; epoch < 20; epoch++)
        {
            using (GradientUtils.Grad())
            {
                var loss = new MSELoss<BFloat16>(Reduction.Sum).Forward(model.Forward(input), target);
                losses.Add(float.CreateChecked(loss[0]));
                GradientUtils.ZeroGrad(model.Parameters().Values);
                loss.Backward();
            }
            optimizer.Step();
        }

        Assert.That(losses.All(float.IsFinite), Is.True, "losses must stay finite");
        Assert.That(losses[^1], Is.LessThan(losses[0]));
    }

    [Test]
    public void LinearModule_BFloat16AdamStep_UpdatesParametersWithoutNaN()
    {
        using var model = new Linear<BFloat16>(1, 1);
        var optimizer = new Adam<BFloat16>(BFloat16.CreateChecked(0.01f));
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var input = Matrix([0f, 1f, 2f], 3, 1);
        var target = Matrix([1f, 2f, 3f], 3, 1);

        for (int epoch = 0; epoch < 5; epoch++)
        {
            using (GradientUtils.Grad())
            {
                var loss = new MSELoss<BFloat16>(Reduction.Sum).Forward(model.Forward(input), target);
                GradientUtils.ZeroGrad(model.Parameters().Values);
                loss.Backward();
            }
            optimizer.Step();
        }

        foreach (var parameter in model.Parameters().Values)
        {
            Assert.That(parameter.Length, Is.GreaterThan(0));
            for (int i = 0; i < parameter.Length; i++)
                Assert.That(float.IsFinite(float.CreateChecked(parameter[i])), Is.True,
                    "updated parameters must be finite after Adam steps");
        }
    }

    [Test]
    public void InferenceDefault_BFloat16OpsOutsideGradScope_BuildNoGraph()
    {
        var a = Tensor([-1f, 2f], requiresGrad: true);
        var b = Tensor([3f, -4f], requiresGrad: true);

        var result = ReverseGradOperations.Sum(a + b);
        var info = GradientUtils.GetGraphInfo(result);

        Assert.That(result.RequiresGrad, Is.False);
        Assert.That(info.TotalNodes, Is.EqualTo(0));
    }

    [Test]
    public void ToReverseGradTensorsAuto_BFloat16Column_ConvertsToGradTensor()
    {
        var frame = NivaraFrame.Create(
            ("x", NivaraColumn<BFloat16>.Create(ToBFloat16([1f, 2f, 3f]))),
            ("y", NivaraColumn<float>.Create([1f, 2f, 3f])));

        var tensors = frame.ToReverseGradTensorsAuto();

        Assert.That(tensors.ContainsKey("x"), Is.True, "BFloat16 column must convert");
        Assert.That(tensors.ContainsKey("y"), Is.True);
        Assert.That(tensors["x"], Is.TypeOf<ReverseGradTensor<BFloat16>>());
        Assert.That(tensors["y"], Is.TypeOf<ReverseGradTensor<float>>());
    }
}
