using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class NnTests
{
    private sealed class ModuleWithParams : Module<float>
    {
        public Parameter<float> Weight { get; }
        public Parameter<float> Bias { get; }

        public ModuleWithParams()
        {
            Weight = new Parameter<float>("Weight", new float[] { 1f, 2f, 3f }, requiresGrad: true);
            Bias = new Parameter<float>("Bias", new float[] { 0.5f }, requiresGrad: true);
            RegisterParameters(Weight, Bias);
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => input;
    }

    [Test]
    public void Parameter_Creation_ProvidesTensorAccess()
    {
        var param = new Parameter<float>("test", new float[] { 1f, 2f, 3f }, requiresGrad: true);

        Assert.That(param.Name, Is.EqualTo("test"));
        Assert.That(param.Length, Is.EqualTo(3));
        Assert.That(param.Tensor.RequiresGrad, Is.True);
        Assert.That(param.Tensor[0], Is.EqualTo(1f));
        Assert.That(param.Tensor[1], Is.EqualTo(2f));
        Assert.That(param.Tensor[2], Is.EqualTo(3f));
    }

    [Test]
    public void Parameter_TensorWithRequiresGrad_HasGradAccess()
    {
        var param = new Parameter<float>("w", new float[] { 1f, 2f }, requiresGrad: true);
        var loss = GradOperations.Sum(param.Tensor);
        loss.Backward();

        Assert.That(param.Tensor.Grad, Is.Not.Null);
        Assert.That(param.Tensor.Grad!.Length, Is.EqualTo(2));
    }

    [Test]
    public void Linear_Forward_BasicShape()
    {
        using var linear = new Linear<float>(2, 3, bias: false);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }), requiresGrad: false);
        input.Reshape(1, 2);

        var output = linear.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3 }));
        Assert.That(output.Length, Is.EqualTo(3));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Linear_WeightAndBias_HaveCorrectShapes()
    {
        using var linear = new Linear<float>(5, 2, bias: true);

        Assert.That(linear.Weight.Shape, Is.EqualTo(new[] { 2, 5 }));
        Assert.That(linear.InFeatures, Is.EqualTo(5));
        Assert.That(linear.OutFeatures, Is.EqualTo(2));

        Assert.That(linear.Bias, Is.Not.Null);
        Assert.That(linear.Bias!.Shape, Is.EqualTo(new[] { 1, 2 }));
    }

    [Test]
    public void Sequential_Forward_ChainLayers()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(3, 4, bias: false),
            new Linear<float>(4, 2, bias: false));

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);
        input.Reshape(1, 3);

        var output = seq.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 2 }));
        Assert.That(output.Length, Is.EqualTo(2));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Dropout_EvalMode_IsNoOp()
    {
        using var dropout = new Dropout<float>(0.5);
        dropout.Eval();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }), requiresGrad: false);

        var output = dropout.Forward(input);

        Assert.That(output[0], Is.EqualTo(1f));
        Assert.That(output[1], Is.EqualTo(2f));
        Assert.That(output[2], Is.EqualTo(3f));
    }

    [Test]
    public void Activation_Relu_Forward()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { -2f, -1f, 0f, 1f, 2f }), requiresGrad: false);

        var result = Activation.Relu(input);

        Assert.That(result[0], Is.EqualTo(0f));
        Assert.That(result[1], Is.EqualTo(0f));
        Assert.That(result[2], Is.EqualTo(0f));
        Assert.That(result[3], Is.EqualTo(1f));
        Assert.That(result[4], Is.EqualTo(2f));
    }

    [Test]
    public void Activation_Sigmoid_Forward()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: false);

        var result = Activation.Sigmoid(input);

        Assert.That(result[0], Is.EqualTo(0.5f).Within(1e-6f));
    }

    [Test]
    public void Activation_Tanh_Forward()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0f }), requiresGrad: false);

        var result = Activation.Tanh(input);

        Assert.That(result[0], Is.EqualTo(0f).Within(1e-6f));
    }

    [Test]
    public void KaimingUniform_InitializesWithCorrectShapes()
    {
        using var linear = new Linear<float>(4, 3);
        var parameters = linear.Parameters();
        KaimingUniform.Init(parameters);

        Assert.That(parameters["Weight"].Shape, Is.EqualTo(new[] { 3, 4 }));
        Assert.That(parameters["Bias"].Shape, Is.EqualTo(new[] { 1, 3 }));

        for (int i = 0; i < parameters["Weight"].Length; i++)
            Assert.That(float.IsNaN(parameters["Weight"][i]) || float.IsInfinity(parameters["Weight"][i]), Is.False);

        for (int i = 0; i < parameters["Bias"].Length; i++)
            Assert.That(float.IsNaN(parameters["Bias"][i]) || float.IsInfinity(parameters["Bias"][i]), Is.False);
    }

    [Test]
    public void XavierUniform_InitializesWithCorrectShapes()
    {
        using var linear = new Linear<float>(4, 3);
        var parameters = linear.Parameters();
        XavierUniform.Init(parameters);

        Assert.That(parameters["Weight"].Shape, Is.EqualTo(new[] { 3, 4 }));
        for (int i = 0; i < parameters["Weight"].Length; i++)
            Assert.That(float.IsNaN(parameters["Weight"][i]), Is.False);
    }

    [Test]
    public void Normal_Initializer_ProducesNonNanValues()
    {
        using var linear = new Linear<float>(4, 3);
        var parameters = linear.Parameters();
        Normal.Init(parameters);

        for (int i = 0; i < parameters["Weight"].Length; i++)
            Assert.That(float.IsNaN(parameters["Weight"][i]), Is.False);
    }

    [Test]
    public void Uniform_Initializer_ProducesNonNanValues()
    {
        using var linear = new Linear<float>(4, 3);
        var parameters = linear.Parameters();
        Uniform.Init(parameters);

        for (int i = 0; i < parameters["Weight"].Length; i++)
            Assert.That(float.IsNaN(parameters["Weight"][i]), Is.False);
    }

    [Test]
    public void Module_TrainEval_TogglesIsTraining()
    {
        var module = new ModuleWithParams();

        Assert.That(module.IsTraining, Is.True);

        module.Eval();
        Assert.That(module.IsTraining, Is.False);

        module.Train();
        Assert.That(module.IsTraining, Is.True);
    }

    [Test]
    public void ModuleParameters_ReturnsRegisteredParameters()
    {
        var module = new ModuleWithParams();
        var parameters = module.Parameters();

        Assert.That(parameters.Count, Is.EqualTo(2));
        Assert.That(parameters.ContainsKey("Weight"), Is.True);
        Assert.That(parameters.ContainsKey("Bias"), Is.True);
    }

    [Test]
    public void ModuleParameters_ValuesAreWritable()
    {
        var module = new ModuleWithParams();
        var parameters = module.Parameters();

        var weightTensor = parameters["Weight"];
        Assert.That(weightTensor[0], Is.EqualTo(1f));
        Assert.That(weightTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void Linear_WeightProperty_HasRequiresGradTrue()
    {
        using var linear = new Linear<float>(3, 2);
        Assert.That(linear.Weight.RequiresGrad, Is.True);
    }

    protected static void DisposeModules(params Module<float>[] modules)
    {
        foreach (var m in modules)
            m.Dispose();
    }
}
