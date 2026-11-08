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
/// Kernel tests for Half (f16) AutoDiff support (issue #1021). Half satisfies
/// IFloatingPointIeee754&lt;T&gt; on .NET 11, so every generic op compiles against it;
/// these tests verify runtime admission and numeric parity against float references.
/// </summary>
[TestFixture]
public class HalfTests
{
    const float RelTol = 0.02f;
    const float AbsTol = 0.02f;

    static Half[] ToHalf(float[] values)
    {
        var result = new Half[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = Half.CreateChecked(values[i]);
        return result;
    }

    static float[] ToFloats<T>(ReverseGradTensor<T> tensor) where T : struct, IFloatingPointIeee754<T>
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
            result[i] = float.CreateChecked(tensor[i]);
        return result;
    }

    static ReverseGradTensor<Half> Tensor(float[] values, bool requiresGrad = false) =>
        new(NivaraColumn<Half>.Create(ToHalf(values)), requiresGrad);

    static ReverseGradTensor<Half> Matrix(float[] values, int rows, int cols, bool requiresGrad = false) =>
        ReverseGradTensor<Half>.FromMatrix(ToHalf(values), rows, cols, requiresGrad);

    static ReverseGradTensor<float> FloatTensor(float[] values, bool requiresGrad = false) =>
        new(NivaraColumn<float>.Create(values), requiresGrad);

    static ReverseGradTensor<float> FloatMatrix(float[] values, int rows, int cols, bool requiresGrad = false) =>
        ReverseGradTensor<float>.FromMatrix(values, rows, cols, requiresGrad);

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
    public void TypeValidator_Half_IsSupportedType()
    {
        Assert.That(TypeValidator.IsSupportedType(typeof(Half)), Is.True);
        Assert.That(TypeValidator.IsSupported<Half>(), Is.True);
        Assert.That(TypeValidator.GetSupportedTypes(), Does.Contain(typeof(Half)));
    }

    [Test]
    public void ElementwiseOps_HalfForward_ParityWithFloatReference()
    {
        float[] a = [-2f, -1f, 0f, 1f, 2f];
        float[] b = [0.5f, 1f, -1f, 2f, 0f];

        var fa = FloatTensor(a);
        var fb = FloatTensor(b);
        var ha = Tensor(a);
        var hb = Tensor(b);

        AssertParity(ToFloats(ReverseGradOperations.Add(fa, fb)), ToFloats(ReverseGradOperations.Add(ha, hb)), "Add");
        AssertParity(ToFloats(ReverseGradOperations.Subtract(fa, fb)), ToFloats(ReverseGradOperations.Subtract(ha, hb)), "Subtract");
        AssertParity(ToFloats(ReverseGradOperations.Multiply(fa, fb)), ToFloats(ReverseGradOperations.Multiply(ha, hb)), "Multiply");
        AssertParity(ToFloats(ReverseGradOperations.Sigmoid(fa)), ToFloats(ReverseGradOperations.Sigmoid(ha)), "Sigmoid");
        AssertParity(ToFloats(ReverseGradOperations.Tanh(fa)), ToFloats(ReverseGradOperations.Tanh(ha)), "Tanh");
        AssertParity(ToFloats(ReverseGradOperations.Relu(fa)), ToFloats(ReverseGradOperations.Relu(ha)), "Relu");
        AssertParity(ToFloats(ReverseGradOperations.Gelu(fa)), ToFloats(ReverseGradOperations.Gelu(ha)), "Gelu");
    }

    [Test]
    public void ElementwiseOps_HalfBackward_GradientsMatchFloatReference()
    {
        float[] aData = [-2f, -1f, 0f, 1f, 2f];
        float[] bData = [0.5f, 1f, -1f, 2f, 0f];

        var hA = Tensor(aData, requiresGrad: true);
        var hB = Tensor(bData, requiresGrad: true);
        using (GradientUtils.Grad())
        {
            var z = ReverseGradOperations.Sum(ReverseGradOperations.Add(
                ReverseGradOperations.Multiply(hA, hB), hB));
            z.Backward();
        }

        Assert.That(hA.Grad, Is.Not.Null);
        Assert.That(hB.Grad, Is.Not.Null);
        for (int i = 0; i < aData.Length; i++)
        {
            AssertClose(bData[i], float.CreateChecked(hA.Grad![i]), $"dL/da[{i}] should equal b[{i}]");
            AssertClose(aData[i] + 1f, float.CreateChecked(hB.Grad![i]), $"dL/db[{i}] should equal a[{i}]+1");
        }
    }

    [Test]
    public void Softmax_HalfForward_ParityWithFloatReference()
    {
        float[] data = [-2f, -1f, 0f, 1f, 2f];

        var floatResult = ToFloats(ReverseGradOperations.Softmax(FloatTensor(data), 0));
        var halfResult = ToFloats(ReverseGradOperations.Softmax(Tensor(data), 0));

        AssertParity(floatResult, halfResult, "Softmax");

        float sum = 0;
        foreach (var v in halfResult) sum += v;
        Assert.That(sum, Is.EqualTo(1f).Within(RelTol));
    }

    [Test]
    public void MatMul_HalfForward_ParityWithFloatReference()
    {
        float[] aData = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] bData = [7f, 8f, 9f, 10f, 11f, 12f];

        var floatResult = ToFloats(ReverseGradOperations.MatMul(
            ReverseGradTensor<float>.FromMatrix(aData, 2, 3, requiresGrad: false),
            ReverseGradTensor<float>.FromMatrix(bData, 3, 2, requiresGrad: false)));
        var halfResult = ToFloats(ReverseGradOperations.MatMul(Matrix(aData, 2, 3), Matrix(bData, 3, 2)));

        AssertParity(floatResult, halfResult, "MatMul");
    }

    [Test]
    public void Attention_MultiHeadForward_ParityWithFloatReference()
    {
        float[] aData = [1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f];

        var fQ = FloatMatrix(aData, 2, 4);
        var fK = FloatMatrix(aData, 2, 4);
        var fV = FloatMatrix(aData, 2, 4);
        var hQ = Matrix(aData, 2, 4);
        var hK = Matrix(aData, 2, 4);
        var hV = Matrix(aData, 2, 4);

        var fResult = ToFloats(ReverseGradOperations.MultiHeadAttention(fQ, fK, fV, 2, (float)(1.0 / Math.Sqrt(2.0))));
        var hResult = ToFloats(ReverseGradOperations.MultiHeadAttention(hQ, hK, hV, 2, Half.CreateChecked((float)(1.0 / Math.Sqrt(2.0)))));

        AssertParity(fResult, hResult, "MultiHeadAttention");
    }

    [Test]
    public void LinearModule_HalfSgdTraining_LossDecreases()
    {
        using var model = new Linear<Half>(1, 1, weightInitializer: new DeterministicInit<Half>());
        var optimizer = new SGD<Half>(Half.CreateChecked(0.05f));
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var input = Matrix([0f, 1f, 2f, 3f], 4, 1);
        var target = Matrix([1f, 3f, 5f, 7f], 4, 1);

        var losses = new List<float>();
        for (int epoch = 0; epoch < 20; epoch++)
        {
            using (GradientUtils.Grad())
            {
                var loss = new MSELoss<Half>(Reduction.Sum).Forward(model.Forward(input), target);
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
    public void LinearModule_HalfAdamStep_UpdatesParametersWithoutNaN()
    {
        using var model = new Linear<Half>(1, 1);
        var optimizer = new Adam<Half>(Half.CreateChecked(0.01f));
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var input = Matrix([0f, 1f, 2f], 3, 1);
        var target = Matrix([1f, 2f, 3f], 3, 1);

        for (int epoch = 0; epoch < 5; epoch++)
        {
            using (GradientUtils.Grad())
            {
                var loss = new MSELoss<Half>(Reduction.Sum).Forward(model.Forward(input), target);
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
    public void InferenceDefault_HalfOpsOutsideGradScope_BuildNoGraph()
    {
        var a = Tensor([-1f, 2f], requiresGrad: true);
        var b = Tensor([3f, -4f], requiresGrad: true);

        var result = ReverseGradOperations.Sum(a + b);
        var info = GradientUtils.GetGraphInfo(result);

        Assert.That(result.RequiresGrad, Is.False);
        Assert.That(info.TotalNodes, Is.EqualTo(0));
    }

    [Test]
    public void ToReverseGradTensorsAuto_HalfColumn_ConvertsToGradTensor()
    {
        var frame = NivaraFrame.Create(
            ("x", NivaraColumn<Half>.Create(ToHalf([1f, 2f, 3f]))),
            ("y", NivaraColumn<float>.Create([1f, 2f, 3f])));

        var tensors = frame.ToReverseGradTensorsAuto();

        Assert.That(tensors.ContainsKey("x"), Is.True, "Half column must convert");
        Assert.That(tensors.ContainsKey("y"), Is.True);
        Assert.That(tensors["x"], Is.TypeOf<ReverseGradTensor<Half>>());
        Assert.That(tensors["y"], Is.TypeOf<ReverseGradTensor<float>>());
    }
}
