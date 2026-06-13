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

        Assert.That(output, Is.SameAs(input));
        Assert.That(output[0], Is.EqualTo(1f));
        Assert.That(output[1], Is.EqualTo(2f));
        Assert.That(output[2], Is.EqualTo(3f));
    }

    [Test]
    public void Dropout_TrainingMode_ParticipatesInBackwardPass()
    {
        using var dropout = new Dropout<float>(0.25);
        dropout.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f }), requiresGrad: true);
        var output = dropout.Forward(input);
        var gradient = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }), requiresGrad: false);

        output.Backward(gradient);

        Assert.That(output, Is.Not.SameAs(input));
        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(input.Length));
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);
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

    [Test]
    public void Sequential_Parameters_ReturnsAllLayers()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(3, 4),
            new Linear<float>(4, 1));
        var p = seq.Parameters();

        Assert.That(p.Count, Is.EqualTo(4), "Should have 4 params (Weight+Bias for 2 layers)");
        Assert.That(p.ContainsKey("Module_0.Weight"), Is.True, "First layer Weight");
        Assert.That(p.ContainsKey("Module_0.Bias"), Is.True, "First layer Bias");
        Assert.That(p.ContainsKey("Module_1.Weight"), Is.True, "Second layer Weight");
        Assert.That(p.ContainsKey("Module_1.Bias"), Is.True, "Second layer Bias");
    }

    [Test]
    public void Sequential_GetParameters_ReturnsAllLayers()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(3, 4),
            new Linear<float>(4, 1));
        var p = seq.GetParameters();

        Assert.That(p.Count, Is.EqualTo(4), "Should have 4 params (Weight+Bias for 2 layers)");
        Assert.That(p.ContainsKey("Module_0.Weight"), Is.True);
        Assert.That(p.ContainsKey("Module_0.Bias"), Is.True);
        Assert.That(p.ContainsKey("Module_1.Weight"), Is.True);
        Assert.That(p.ContainsKey("Module_1.Bias"), Is.True);
    }

    [Test]
    public void Sequential_Parameters_SingleLayer_BackwardCompatible()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(3, 2));
        var p = seq.Parameters();

        Assert.That(p.Count, Is.EqualTo(2), "Single layer should still have 2 params");
        Assert.That(p.ContainsKey("Module_0.Weight"), Is.True);
        Assert.That(p.ContainsKey("Module_0.Bias"), Is.True);
    }

    [Test]
    public void Sequential_Parameters_ThreeLayerModel_CorrectCount()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(4, 8),
            new Linear<float>(8, 6),
            new Linear<float>(6, 3));
        var p = seq.Parameters();

        Assert.That(p.Count, Is.EqualTo(6), "3 layers × 2 params each = 6");
        Assert.That(p.ContainsKey("Module_0.Weight"), Is.True);
        Assert.That(p.ContainsKey("Module_2.Bias"), Is.True, "Last layer bias exists");
    }

    [Test]
    public void Linear_DefaultInit_MatchesCurrentBehavior()
    {
        using var linear = new Linear<float>(4, 3);
        var w = linear.Weight;

        Assert.That(w.Shape, Is.EqualTo(new[] { 3, 4 }));
        Assert.That(w.RequiresGrad, Is.True);
        for (int i = 0; i < w.Length; i++)
            Assert.That(float.IsNaN(w[i]) || float.IsInfinity(w[i]), Is.False);
    }

    [Test]
    public void Linear_CustomWeightInit_ChangesValues()
    {
        using var linear = new Linear<float>(4, 3, bias: false,
            weightInitializer: XavierUniformInitializer<float>.Instance);

        var w = linear.Weight;
        Assert.That(w.Shape, Is.EqualTo(new[] { 3, 4 }));
        for (int i = 0; i < w.Length; i++)
            Assert.That(float.IsNaN(w[i]) || float.IsInfinity(w[i]), Is.False);
    }

    [Test]
    public void Linear_CustomBiasInit_InitializesBias()
    {
        using var linear = new Linear<float>(4, 3, bias: true,
            weightInitializer: KaimingUniformInitializer<float>.Instance,
            biasInitializer: new UniformInitializer<float>(-0.1f, 0.1f));

        Assert.That(linear.Bias, Is.Not.Null);
        for (int i = 0; i < linear.Bias!.Length; i++)
        {
            Assert.That(float.IsNaN(linear.Bias[i]) || float.IsInfinity(linear.Bias[i]), Is.False);
            Assert.That(linear.Bias[i], Is.InRange(-0.1f, 0.1f));
        }
    }

    [Test]
    public void Linear_NullBiasInit_BiasStaysZeros()
    {
        using var linear = new Linear<float>(4, 3, bias: true,
            weightInitializer: null,
            biasInitializer: null);

        for (int i = 0; i < linear.Bias!.Length; i++)
            Assert.That(linear.Bias[i], Is.EqualTo(0f));
    }

    [Test]
    public void KaimingUniformInitializer_Interface_ProducesCorrectShape()
    {
        using var linear = new Linear<float>(4, 3,
            weightInitializer: KaimingUniformInitializer<float>.Instance);
        var w = linear.Weight;

        Assert.That(w.Shape, Is.EqualTo(new[] { 3, 4 }));
        for (int i = 0; i < w.Length; i++)
            Assert.That(float.IsNaN(w[i]) || float.IsInfinity(w[i]), Is.False);
    }

    [Test]
    public void XavierUniformInitializer_Interface_ProducesCorrectShape()
    {
        using var linear = new Linear<float>(4, 3,
            weightInitializer: XavierUniformInitializer<float>.Instance);
        var w = linear.Weight;

        Assert.That(w.Shape, Is.EqualTo(new[] { 3, 4 }));
        for (int i = 0; i < w.Length; i++)
            Assert.That(float.IsNaN(w[i]), Is.False);
    }

    [Test]
    public void NormalInitializer_WithCustomParams_AppliesMeanStd()
    {
        var init = new NormalInitializer<float>(2.0f, 0.5f);
        var param = new Parameter<float>("test", new float[1000], requiresGrad: true);
        init.Initialize(param);

        double sum = 0;
        for (int i = 0; i < param.Length; i++)
            sum += param.Tensor[i];
        var mean = sum / param.Length;

        Assert.That(mean, Is.EqualTo(2.0).Within(0.2));
    }

    [Test]
    public void UniformInitializer_WithCustomBounds_ProducesCorrectRange()
    {
        var init = new UniformInitializer<float>(5.0f, 10.0f);
        var param = new Parameter<float>("test", new float[1000], requiresGrad: true);
        init.Initialize(param);

        for (int i = 0; i < param.Length; i++)
        {
            Assert.That(param.Tensor[i], Is.GreaterThanOrEqualTo(5.0f));
            Assert.That(param.Tensor[i], Is.LessThanOrEqualTo(10.0f));
        }
    }

    [Test]
    public void PyTorchDefaultInitializer_ProducesExpectedBound()
    {
        var init = PyTorchDefaultInitializer<float>.Instance;
        var param = new Parameter<float>("test", new float[1000], requiresGrad: true);
        param.Tensor.Reshape(10, 100); // fanIn = 100
        init.Initialize(param);

        // PyTorch bound = 1/sqrt(fanIn) = 1/sqrt(100) = 0.1
        var bound = 1.0f / MathF.Sqrt(100);
        for (int i = 0; i < param.Length; i++)
        {
            Assert.That(float.IsNaN(param.Tensor[i]) || float.IsInfinity(param.Tensor[i]), Is.False);
            Assert.That(param.Tensor[i], Is.InRange(-bound, bound));
        }
    }

    [Test]
    public void Sequential_CustomInitForAllLayers()
    {
        using var seq = new Sequential<float>(
            new Linear<float>(3, 4, bias: true,
                weightInitializer: XavierUniformInitializer<float>.Instance,
                biasInitializer: new UniformInitializer<float>(-0.05f, 0.05f)),
            new Linear<float>(4, 1, bias: true,
                weightInitializer: XavierUniformInitializer<float>.Instance,
                biasInitializer: new UniformInitializer<float>(-0.05f, 0.05f)));

        var p = seq.Parameters();
        Assert.That(p.Count, Is.EqualTo(4));
        foreach (var (_, tensor) in p)
        {
            for (int i = 0; i < tensor.Length; i++)
                Assert.That(float.IsNaN(tensor[i]) || float.IsInfinity(tensor[i]), Is.False);
        }
    }

    [Test]
    public void Parameter_Dispose_DisposesTensorAndIsIdempotent()
    {
        var param = new Parameter<float>("test", new float[] { 1f, 2f }, requiresGrad: true);
        var tensor = param.Tensor;

        Assert.DoesNotThrow(() => param.Dispose());
        Assert.DoesNotThrow(() => param.Dispose());

        Assert.Throws<ObjectDisposedException>(() => _ = tensor.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = param.Tensor);
    }

    [Test]
    public void Linear_Dispose_DisposesOwnedParameters()
    {
        var linear = new Linear<float>(2, 3, bias: true);
        var weight = linear.Weight;
        var bias = linear.Bias;

        linear.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = weight.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = bias!.Length);
        Assert.DoesNotThrow(() => linear.Dispose());
    }

    [Test]
    public void Sequential_Dispose_DisposesChildModuleParameters()
    {
        var first = new Linear<float>(3, 4);
        var second = new Linear<float>(4, 2);
        var firstWeight = first.Weight;
        var secondBias = second.Bias;
        var seq = new Sequential<float>(first, second);

        seq.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = firstWeight.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = secondBias!.Length);
        Assert.DoesNotThrow(() => seq.Dispose());
    }

    protected static void DisposeModules(params Module<float>[] modules)
    {
        foreach (var m in modules)
            m.Dispose();
    }
}
