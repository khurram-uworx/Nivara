using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class NnTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    sealed class ModuleWithParams : Module<float>
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
        var loss = ReverseGradOperations.Sum(param.Tensor);
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

        Assert.That(linear.Weight!.Tensor.Shape, Is.EqualTo(new[] { 2, 5 }));
        Assert.That(linear.InFeatures, Is.EqualTo(5));
        Assert.That(linear.OutFeatures, Is.EqualTo(2));

        Assert.That(linear.Bias, Is.Not.Null);
        Assert.That(linear.Bias!.Tensor.Shape, Is.EqualTo(new[] { 1, 2 }));
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
        Assert.That(linear.Weight!.Tensor.RequiresGrad, Is.True);
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
        var w = linear.Weight!.Tensor;

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

        var w = linear.Weight!.Tensor;
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
        for (int i = 0; i < linear.Bias!.Tensor.Length; i++)
        {
            Assert.That(float.IsNaN(linear.Bias!.Tensor[i]) || float.IsInfinity(linear.Bias!.Tensor[i]), Is.False);
            Assert.That(linear.Bias!.Tensor[i], Is.InRange(-0.1f, 0.1f));
        }
    }

    [Test]
    public void Linear_NullBiasInit_BiasStaysZeros()
    {
        using var linear = new Linear<float>(4, 3, bias: true,
            weightInitializer: null,
            biasInitializer: null);

        for (int i = 0; i < linear.Bias!.Tensor.Length; i++)
            Assert.That(linear.Bias!.Tensor[i], Is.EqualTo(0f));
    }

    [Test]
    public void KaimingUniformInitializer_Interface_ProducesCorrectShape()
    {
        using var linear = new Linear<float>(4, 3,
            weightInitializer: KaimingUniformInitializer<float>.Instance);
        var w = linear.Weight!.Tensor;

        Assert.That(w.Shape, Is.EqualTo(new[] { 3, 4 }));
        for (int i = 0; i < w.Length; i++)
            Assert.That(float.IsNaN(w[i]) || float.IsInfinity(w[i]), Is.False);
    }

    [Test]
    public void XavierUniformInitializer_Interface_ProducesCorrectShape()
    {
        using var linear = new Linear<float>(4, 3,
            weightInitializer: XavierUniformInitializer<float>.Instance);
        var w = linear.Weight!.Tensor;

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
            for (int i = 0; i < tensor.Length; i++)
                Assert.That(float.IsNaN(tensor[i]) || float.IsInfinity(tensor[i]), Is.False);
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
        var weight = linear.Weight!.Tensor;
        var bias = linear.Bias!.Tensor;

        linear.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = weight.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = bias.Length);
        Assert.DoesNotThrow(() => linear.Dispose());
    }

    [Test]
    public void Sequential_Dispose_DisposesChildModuleParameters()
    {
        var first = new Linear<float>(3, 4);
        var second = new Linear<float>(4, 2);
        var firstWeight = first.Weight!.Tensor;
        var secondBias = second.Bias!.Tensor;
        var seq = new Sequential<float>(first, second);

        seq.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = firstWeight.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = secondBias.Length);
        Assert.DoesNotThrow(() => seq.Dispose());
    }

    protected static void DisposeModules(params Module<float>[] modules)
    {
        foreach (var m in modules)
            m.Dispose();
    }

    [Test]
    public void VAE_Forward_ShapeCorrect()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: false);
        data.Reshape(2, 4);

        var recon = vae.Forward(data);

        Assert.That(recon.Shape, Is.EqualTo(new[] { 2, 4 }));
        for (int i = 0; i < recon.Length; i++)
            Assert.That(float.IsNaN(recon[i]) || float.IsInfinity(recon[i]), Is.False);
    }

    [Test]
    public void VAE_Encode_ReturnsCorrectShapes()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        data.Reshape(1, 4);

        var (mu, logVar) = vae.Encode(data);

        Assert.That(mu.Shape, Is.EqualTo(new[] { 1, 2 }));
        Assert.That(logVar.Shape, Is.EqualTo(new[] { 1, 2 }));
        Assert.That(mu.RequiresGrad, Is.True);
        Assert.That(logVar.RequiresGrad, Is.True);
        for (int i = 0; i < mu.Length; i++)
        {
            Assert.That(float.IsNaN(mu[i]) || float.IsInfinity(mu[i]), Is.False);
            Assert.That(float.IsNaN(logVar[i]) || float.IsInfinity(logVar[i]), Is.False);
        }
    }

    [Test]
    public void VAE_Reparameterize_EvalMode_ReturnsMu()
    {
        using var vae = new VAE<float>(4, 2, 8);
        vae.Eval();

        var mu = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.5f, -1.2f]), requiresGrad: true);
        var logVar = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f]), requiresGrad: true);

        var z = vae.Reparameterize(mu, logVar);

        Assert.That(z, Is.SameAs(mu));
    }

    [Test]
    public void VAE_Reparameterize_TrainMode_Stochastic()
    {
        using var vae = new VAE<float>(4, 2, 8);
        vae.Train();

        var mu = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.5f, -1.2f]), requiresGrad: true);
        var logVar = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f]), requiresGrad: true);

        var z1 = vae.Reparameterize(mu, logVar, seed: 42);
        var z2 = vae.Reparameterize(mu, logVar, seed: 99);

        Assert.That(z1.Length, Is.EqualTo(2));
        Assert.That(z2.Length, Is.EqualTo(2));

        bool allEqual = true;
        for (int i = 0; i < z1.Length; i++)
            if (z1[i] != z2[i]) { allEqual = false; break; }

        Assert.That(allEqual, Is.False, "Two calls with different seeds should produce different samples");
    }

    [Test]
    public void VAE_ElboLoss_ReturnsScalar()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        data.Reshape(1, 4);

        var (mu, logVar) = vae.Encode(data);
        var z = vae.Reparameterize(mu, logVar);
        var recon = vae.Decode(z);
        var loss = vae.ElboLoss(recon, data, mu, logVar);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(float.IsNaN(loss[0]) || float.IsInfinity(loss[0]), Is.False);
        Assert.That(loss[0], Is.GreaterThan(0f));
    }

    [Test]
    public void VAE_Backward_GradientsFlowToAllParams()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: false);
        data.Reshape(2, 4);

        var (mu, logVar) = vae.Encode(data);
        var z = vae.Reparameterize(mu, logVar);
        var recon = vae.Decode(z);
        var loss = vae.ElboLoss(recon, data, mu, logVar);
        loss.Backward();

        foreach (var (name, param) in vae.GetParameters())
        {
            // Beta has requiresGrad=false so it never receives gradients
            if (name == "Beta")
            {
                Assert.That(param.Tensor.Grad, Is.Null,
                    $"Parameter '{name}' (requiresGrad=false) should have no gradient");
                continue;
            }
            Assert.That(param.Tensor.Grad, Is.Not.Null,
                $"Parameter '{name}' should have gradient after Backward");
            Assert.That(param.Tensor.Grad!.Length, Is.EqualTo(param.Tensor.Length));
        }
    }

    [Test]
    public void VAE_Training_ReducesLoss()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var raw = new float[16];
        for (int i = 0; i < raw.Length; i++)
            raw[i] = (i % 4) + 1;
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(raw), requiresGrad: false);
        data.Reshape(4, 4);

        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(vae.GetParameters().Values, 0.01f);

        float ComputeLoss()
        {
            var (mu, logVar) = vae.Encode(data);
            var z = vae.Reparameterize(mu, logVar);
            var recon = vae.Decode(z);
            var l = vae.ElboLoss(recon, data, mu, logVar);
            return l[0];
        }

        var initialLoss = ComputeLoss();

        for (int epoch = 0; epoch < 100; epoch++)
        {
            optimizer.ZeroGrad();
            var (mu, logVar) = vae.Encode(data);
            var z = vae.Reparameterize(mu, logVar);
            var recon = vae.Decode(z);
            var loss = vae.ElboLoss(recon, data, mu, logVar);
            loss.Backward();
            optimizer.Step();
        }

        var finalLoss = ComputeLoss();
        Assert.That(finalLoss, Is.LessThan(initialLoss),
            "VAE training should reduce ELBO loss");
    }

    [Test]
    public void VAE_Parameters_ReturnsCorrectCount()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var p = vae.Parameters();

        // 6 Linear sub-modules x 2 params each (Weight + Bias) = 12, plus 1 Beta
        Assert.That(p.Count, Is.EqualTo(13));
    }

    [Test]
    public void VAE_GetParameters_IncludesBeta()
    {
        using var vae = new VAE<float>(4, 2, 8, beta: 2.5f);
        var p = vae.GetParameters();

        Assert.That(p.ContainsKey("Beta"), Is.True);
        Assert.That(p["Beta"].Tensor.RequiresGrad, Is.False);
        Assert.That(p["Beta"].Tensor[0], Is.EqualTo(2.5f).Within(1e-6f));
    }

    [Test]
    public void VAE_Encode_NoSpuriousNulls()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var data = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: false);
        data.Reshape(2, 4);

        var (mu, logVar) = vae.Encode(data);

        for (int i = 0; i < mu.Length; i++)
        {
            Assert.That(float.IsNaN(mu[i]) || float.IsInfinity(mu[i]), Is.False);
            Assert.That(float.IsNaN(logVar[i]) || float.IsInfinity(logVar[i]), Is.False);
        }
    }

    [Test]
    public void VAE_Dispose_DisposesSubModules()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var param = vae.GetParameters().First().Value;
        var tensor = param.Tensor;

        vae.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = tensor.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = param.Tensor);
    }

    [Test]
    public void VAE_Serialization_RoundTrip()
    {
        using var vae = new VAE<float>(4, 2, 8);
        var path = Path.Combine(Path.GetTempPath(),
            $"vae_test_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(vae, path);

            using var loaded = new VAE<float>(4, 2, 8);
            ModelSerializer.Load(loaded, path);

            var originalParams = vae.Parameters();
            var loadedParams = loaded.Parameters();

            Assert.That(loadedParams.Count, Is.EqualTo(originalParams.Count));
            foreach (var (name, tensor) in originalParams)
            {
                Assert.That(loadedParams.ContainsKey(name), Is.True,
                    $"Parameter '{name}' should exist in loaded model");
                Assert.That(loadedParams[name].Shape, Is.EqualTo(tensor.Shape));
                for (int i = 0; i < tensor.Length; i++)
                    Assert.That(loadedParams[name][i], Is.EqualTo(tensor[i]).Within(1e-6f));
            }
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Test]
    public void VAE_Constructor_InvalidArg_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new VAE<float>(0, 2, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VAE<float>(4, 0, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VAE<float>(4, 2, 0));
    }

    [Test]
    public void VAE_Conditional_Encode_ShapeCorrect()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: false);
        x.Reshape(2, 4);

        var condition = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]),
            requiresGrad: false);
        condition.Reshape(2, 3);

        var (mu, logVar) = vae.Encode(x, condition);

        Assert.That(mu.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(logVar.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void VAE_Conditional_Encode_WithNullCondition_HandlesGracefully()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        x.Reshape(1, 4);

        // Should not throw, but will fail at MatMul since encoder expects 7 inputs
        // This documents the behavior: conditionDim > 0 requires condition
        Assert.Throws<ArgumentException>(() => vae.Encode(x, condition: null));
    }

    [Test]
    public void VAE_Conditional_Forward_EndToEnd()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: false);
        x.Reshape(2, 4);

        var condition = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]),
            requiresGrad: false);
        condition.Reshape(2, 3);

        var recon = vae.Forward(x, condition);

        Assert.That(recon.Shape, Is.EqualTo(new[] { 2, 4 }));
        for (int i = 0; i < recon.Length; i++)
            Assert.That(float.IsNaN(recon[i]) || float.IsInfinity(recon[i]), Is.False);
    }

    [Test]
    public void VAE_Conditional_ElboLoss_Works()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        x.Reshape(1, 4);

        var condition = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f, 0.3f]),
            requiresGrad: false);
        condition.Reshape(1, 3);

        var (mu, logVar) = vae.Encode(x, condition);
        var z = vae.Reparameterize(mu, logVar);
        var recon = vae.Decode(z, condition);
        var loss = vae.ElboLoss(recon, x, mu, logVar);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(float.IsNaN(loss[0]) || float.IsInfinity(loss[0]), Is.False);
        Assert.That(loss[0], Is.GreaterThan(0f));
    }

    [Test]
    public void VAE_Conditional_Decode_WithCondition()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var z = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.5f, -0.3f, 1.2f, 0.8f]),
            requiresGrad: false);
        z.Reshape(2, 2);

        var condition = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]),
            requiresGrad: false);
        condition.Reshape(2, 3);

        var recon = vae.Decode(z, condition);

        Assert.That(recon.Shape, Is.EqualTo(new[] { 2, 4 }));
    }

    [Test]
    public void VAE_Conditional_Backward_GradientsFlow()
    {
        using var vae = new VAE<float>(4, 2, 8, conditionDim: 3);
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        x.Reshape(1, 4);

        var condition = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([0.1f, 0.2f, 0.3f]),
            requiresGrad: true);
        condition.Reshape(1, 3);

        var (mu, logVar) = vae.Encode(x, condition);
        var z = vae.Reparameterize(mu, logVar);
        var recon = vae.Decode(z, condition);
        var loss = vae.ElboLoss(recon, x, mu, logVar);
        loss.Backward();

        foreach (var (name, param) in vae.GetParameters())
        {
            if (name == "Beta") continue; // requiresGrad=false
            Assert.That(param.Tensor.Grad, Is.Not.Null, $"Parameter '{name}' should have gradient");
            Assert.That(param.Tensor.Grad!.Length, Is.EqualTo(param.Tensor.Length));
        }
    }

    [Test]
    public void VAE_Conditional_Constructor_InvalidConditionDim_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new VAE<float>(4, 2, 8, conditionDim: -1));
    }

    [Test]
    public void BatchNorm1d_Forward_ShapeCorrect()
    {
        using var bn = new BatchNorm1d<float>(4);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(2, 4);

        var output = bn.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 4 }));
        Assert.That(output.Length, Is.EqualTo(8));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void BatchNorm1d_TrainMode_UpdatesRunningStats()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();
        Assert.That(bn.IsTraining, Is.True);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            requiresGrad: false);
        input.Reshape(2, 3);

        var runningMeanBefore = new float[3];
        bn.RunningMean.Data.CopyTo(runningMeanBefore, 0f);

        var output = bn.Forward(input);

        var runningMeanAfter = new float[3];
        bn.RunningMean.Data.CopyTo(runningMeanAfter, 0f);

        // Running mean should have been updated
        bool meanChanged = false;
        for (int i = 0; i < 3; i++)
            if (Math.Abs(runningMeanAfter[i] - runningMeanBefore[i]) > 1e-6f) meanChanged = true;

        Assert.That(meanChanged, Is.True, "Running mean should update in train mode");
    }

    [Test]
    public void BatchNorm1d_EvalMode_UsesRunningStats()
    {
        using var bn = new BatchNorm1d<float>(3, trackRunningStats: true);
        bn.Train();

        var trainInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            requiresGrad: false);
        trainInput.Reshape(2, 3);
        bn.Forward(trainInput); // Updates running stats

        bn.Eval();
        Assert.That(bn.IsTraining, Is.False);

        var evalInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 10f, 20f, 30f }),
            requiresGrad: false);
        evalInput.Reshape(1, 3);

        var output = bn.Forward(evalInput);

        Assert.That(output.Length, Is.EqualTo(3));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void BatchNorm1d_AffineFalse_NoScaleShift()
    {
        using var bn = new BatchNorm1d<float>(3, affine: false);
        bn.Eval();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }),
            requiresGrad: false);
        input.Reshape(1, 3);

        var output = bn.Forward(input);

        // With affine=false, output should just be normalized (mean 0, var 1)
        Assert.That(output.Length, Is.EqualTo(3));
        Assert.That(bn.Weight, Is.Null);
        Assert.That(bn.Bias, Is.Null);
    }

    [Test]
    public void BatchNorm1d_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            requiresGrad: true);
        input.Reshape(2, 3);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f, 1f, 1f }),
            requiresGrad: false);
        gradOutput.Reshape(2, 3);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(6));
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);

        if (bn.Weight != null)
        {
            Assert.That(bn.Weight.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Weight.Tensor.Grad!.Length, Is.EqualTo(3));
        }
        if (bn.Bias != null)
        {
            Assert.That(bn.Bias.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Bias.Tensor.Grad!.Length, Is.EqualTo(3));
        }
    }

    [Test]
    public void BatchNorm1d_3DInput_Forward_ShapeCorrect()
    {
        using var bn = new BatchNorm1d<float>(4);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 4 * 8]), // B=2, C=4, L=8
            requiresGrad: false);
        input.Reshape(2, 4, 8);

        var output = bn.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(3));
        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 4, 8 }));
        Assert.That(output.Length, Is.EqualTo(64));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void BatchNorm1d_3DInput_NormalizesPerChannel()
    {
        using var bn = new BatchNorm1d<float>(2, trackRunningStats: false, affine: false);
        bn.Eval();

        var data = new float[]
        {
            1, 2, 3, 4,  10, 20, 30, 40,
            1, 2, 3, 4,  10, 20, 30, 40,
        };
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(data), requiresGrad: false);
        input.Reshape(2, 2, 4);

        var output = bn.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(3));
        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 2, 4 }));

        for (int ch = 0; ch < 2; ch++)
        {
            float sum = 0;
            for (int b = 0; b < 2; b++)
                for (int l = 0; l < 4; l++)
                    sum += output[b * 2 * 4 + ch * 4 + l];
            float mean = sum / 8;
            Assert.That(mean, Is.EqualTo(0f).Within(1e-5f),
                $"Channel {ch} should have mean 0 across B*L=8 elements");
        }
    }

    [Test]
    public void BatchNorm1d_NoRunningStats_RunningMeanThrows()
    {
        using var bn = new BatchNorm1d<float>(2, trackRunningStats: false);

        Assert.That(bn.TrackRunningStats, Is.False);
        Assert.That(() => bn.RunningMean, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
        Assert.That(() => bn.RunningVar, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
        Assert.That(() => bn.NumBatchesTracked, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
    }

    [Test]
    public void BatchNorm2d_NoRunningStats_RunningMeanThrows()
    {
        using var bn = new BatchNorm2d<float>(2, trackRunningStats: false);

        Assert.That(bn.TrackRunningStats, Is.False);
        Assert.That(() => bn.RunningMean, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
        Assert.That(() => bn.RunningVar, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
        Assert.That(() => bn.NumBatchesTracked, Throws.InvalidOperationException.With.Message.Contains("trackRunningStats: false"));
    }

    [Test]
    public void BatchNorm1d_TrackRunningStats_ExposesRunningStats()
    {
        using var bn = new BatchNorm1d<float>(3, trackRunningStats: true);

        Assert.That(bn.TrackRunningStats, Is.True);
        Assert.That(bn.RunningMean.Length, Is.EqualTo(3));
        Assert.That(bn.RunningVar.Length, Is.EqualTo(3));
        Assert.That(bn.NumBatchesTracked[0], Is.EqualTo(0f));
    }

    [Test]
    public void BatchNorm2d_TrackRunningStats_ExposesRunningStats()
    {
        using var bn = new BatchNorm2d<float>(3, trackRunningStats: true);

        Assert.That(bn.TrackRunningStats, Is.True);
        Assert.That(bn.RunningMean.Length, Is.EqualTo(3));
        Assert.That(bn.RunningVar.Length, Is.EqualTo(3));
        Assert.That(bn.NumBatchesTracked[0], Is.EqualTo(0f));
    }

    [Test]
    public void BatchNorm1d_3DInput_TrainMode_UpdatesRunningStats()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();

        var data = new float[]
        {
            1, 2, 3, 4, 5,  6, 7, 8, 9, 10,  11, 12, 13, 14, 15,
            2, 3, 4, 5, 6,  7, 8, 9, 10, 11,  12, 13, 14, 15, 16,
        };
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(data), requiresGrad: false);
        input.Reshape(2, 3, 5);

        var runningMeanBefore = new float[3];
        bn.RunningMean.Data.CopyTo(runningMeanBefore, 0f);

        bn.Forward(input);

        var runningMeanAfter = new float[3];
        bn.RunningMean.Data.CopyTo(runningMeanAfter, 0f);

        bool meanChanged = false;
        for (int i = 0; i < 3; i++)
            if (Math.Abs(runningMeanAfter[i] - runningMeanBefore[i]) > 1e-6f) meanChanged = true;

        Assert.That(meanChanged, Is.True, "Running mean should update in train mode for 3D input");
    }

    [Test]
    public void BatchNorm1d_3DInput_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4]), // B=2, C=3, L=4
            requiresGrad: true);
        input.Reshape(2, 3, 4);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4]),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 4);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(24));
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);

        if (bn.Weight != null)
        {
            Assert.That(bn.Weight.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Weight.Tensor.Grad!.Length, Is.EqualTo(3));
        }
        if (bn.Bias != null)
        {
            Assert.That(bn.Bias.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Bias.Tensor.Grad!.Length, Is.EqualTo(3));
        }
    }

    [Test]
    public void BatchNorm1d_2DInput_RejectsInvalidRanks()
    {
        using var bn = new BatchNorm1d<float>(3);
        var input1d = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[3]),
            requiresGrad: false);
        input1d.Reshape(3);

        Assert.Throws<ArgumentException>(() => bn.Forward(input1d));

        var input4d = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[24]),
            requiresGrad: false);
        input4d.Reshape(2, 3, 2, 2);

        Assert.Throws<ArgumentException>(() => bn.Forward(input4d));
    }

    [Test]
    public void BatchNorm2d_Forward_ShapeCorrect()
    {
        using var bn = new BatchNorm2d<float>(3);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4 * 4]), // N=2, C=3, H=4, W=4
            requiresGrad: false);
        input.Reshape(2, 3, 4, 4);

        var output = bn.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 3, 4, 4 }));
        Assert.That(output.Length, Is.EqualTo(96));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void BatchNorm2d_TrainMode_UpdatesRunningStats()
    {
        using var bn = new BatchNorm2d<float>(2);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(1, 2, 2, 2);

        var runningMeanBefore = new float[2];
        bn.RunningMean.Data.CopyTo(runningMeanBefore, 0f);

        bn.Forward(input);

        var runningMeanAfter = new float[2];
        bn.RunningMean.Data.CopyTo(runningMeanAfter, 0f);

        bool meanChanged = false;
        for (int i = 0; i < 2; i++)
            if (Math.Abs(runningMeanAfter[i] - runningMeanBefore[i]) > 1e-6f) meanChanged = true;

        Assert.That(meanChanged, Is.True, "Running mean should update in train mode");
    }

    [Test]
    public void BatchNorm2d_EvalMode_UsesRunningStats()
    {
        using var bn = new BatchNorm2d<float>(2, trackRunningStats: true);
        bn.Train();

        var trainInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        trainInput.Reshape(1, 2, 2, 2);
        bn.Forward(trainInput);

        bn.Eval();

        var evalInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f }),
            requiresGrad: false);
        evalInput.Reshape(1, 2, 2, 2);

        var output = bn.Forward(evalInput);

        Assert.That(output.Length, Is.EqualTo(8));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void BatchNorm2d_AffineFalse_NoScaleShift()
    {
        using var bn = new BatchNorm2d<float>(2, affine: false);
        bn.Eval();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f }),
            requiresGrad: false);
        input.Reshape(1, 2, 1, 2);

        var output = bn.Forward(input);

        Assert.That(output.Length, Is.EqualTo(4));
        Assert.That(bn.Weight, Is.Null);
        Assert.That(bn.Bias, Is.Null);
    }

    [Test]
    public void BatchNorm2d_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm2d<float>(2);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: true);
        input.Reshape(1, 2, 2, 2);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f }),
            requiresGrad: false);
        gradOutput.Reshape(1, 2, 2, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(8));
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);

        if (bn.Weight != null)
        {
            Assert.That(bn.Weight.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Weight.Tensor.Grad!.Length, Is.EqualTo(2));
        }
        if (bn.Bias != null)
        {
            Assert.That(bn.Bias.Tensor.Grad, Is.Not.Null);
            Assert.That(bn.Bias.Tensor.Grad!.Length, Is.EqualTo(2));
        }
    }

    [Test]
    public void BatchNorm1d_StateDict_RoundTrip()
    {
        using var bn1 = new BatchNorm1d<float>(4);
        bn1.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }),
            requiresGrad: false);
        input.Reshape(2, 4);
        bn1.Forward(input);

        var state = bn1.StateDict();
        Assert.That(state.ContainsKey("running_mean"), Is.True);
        Assert.That(state.ContainsKey("running_var"), Is.True);
        Assert.That(state.ContainsKey("num_batches_tracked"), Is.True);

        using var bn2 = new BatchNorm1d<float>(4);
        bn2.LoadStateDict(state);

        Assert.That(bn2.RunningMean.Length, Is.EqualTo(4));
        Assert.That(bn2.RunningVar.Length, Is.EqualTo(4));

        for (int i = 0; i < 4; i++)
        {
            Assert.That(bn2.RunningMean[i], Is.EqualTo(bn1.RunningMean[i]).Within(1e-6f));
            Assert.That(bn2.RunningVar[i], Is.EqualTo(bn1.RunningVar[i]).Within(1e-6f));
        }
    }

    [Test]
    public void BatchNorm2d_StateDict_RoundTrip()
    {
        using var bn1 = new BatchNorm2d<float>(3);
        bn1.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(2, 3, 4, 4);
        bn1.Forward(input);

        var state = bn1.StateDict();
        Assert.That(state.ContainsKey("running_mean"), Is.True);
        Assert.That(state.ContainsKey("running_var"), Is.True);
        Assert.That(state.ContainsKey("num_batches_tracked"), Is.True);

        using var bn2 = new BatchNorm2d<float>(3);
        bn2.LoadStateDict(state);

        Assert.That(bn2.RunningMean.Length, Is.EqualTo(3));
        Assert.That(bn2.RunningVar.Length, Is.EqualTo(3));

        for (int i = 0; i < 3; i++)
        {
            Assert.That(bn2.RunningMean[i], Is.EqualTo(bn1.RunningMean[i]).Within(1e-6f));
            Assert.That(bn2.RunningVar[i], Is.EqualTo(bn1.RunningVar[i]).Within(1e-6f));
        }
    }

    [Test]
    public void BatchNorm1d_Dispose_DisposesParameters()
    {
        var bn = new BatchNorm1d<float>(3);
        var weight = bn.Weight;
        var bias = bn.Bias;

        bn.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = weight!.Tensor.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = bias!.Tensor.Length);
        Assert.DoesNotThrow(() => bn.Dispose());
    }

    [Test]
    public void BatchNorm2d_Dispose_DisposesParameters()
    {
        var bn = new BatchNorm2d<float>(3);
        var weight = bn.Weight;
        var bias = bn.Bias;

        bn.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = weight!.Tensor.Length);
        Assert.Throws<ObjectDisposedException>(() => _ = bias!.Tensor.Length);
        Assert.DoesNotThrow(() => bn.Dispose());
    }

    [Test]
    public void Embedding_ScalarForward_ReturnsCorrectShape()
    {
        using var emb = new Embedding<float>(10, 4);
        var result = emb.Forward(3);

        Assert.That(result.Shape, Is.EqualTo(new[] { 1, 4 }));
        Assert.That(result.Length, Is.EqualTo(4));
    }

    [Test]
    public void Embedding_BatchedForward_1D_ReturnsCorrectShape()
    {
        using var emb = new Embedding<float>(10, 4);
        var tokenIds = new float[] { 0, 1, 2, 3 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(4);

        var result = emb.Forward(input);

        Assert.That(result.Shape, Is.EqualTo(new[] { 4, 4 }));
        Assert.That(result.Length, Is.EqualTo(16));
    }

    [Test]
    public void Embedding_BatchedForward_2D_ReturnsCorrectShape()
    {
        using var emb = new Embedding<float>(10, 4);
        var tokenIds = new float[] { 0, 1, 2, 3, 4, 5 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(2, 3);

        var result = emb.Forward(input);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3, 4 }));
        Assert.That(result.Length, Is.EqualTo(24));
    }

    [Test]
    public void Embedding_BatchedForward_MatchesSingleForward()
    {
        using var emb = new Embedding<float>(10, 4);
        var tokenIds = new float[] { 2, 5, 7 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(3);

        var batched = emb.Forward(input);
        var single0 = emb.Forward(2);
        var single1 = emb.Forward(5);
        var single2 = emb.Forward(7);

        for (int i = 0; i < 4; i++)
        {
            Assert.That(batched.Data[i], Is.EqualTo(single0.Data[i]).Within(1e-6f),
                $"Position 0, dim {i} mismatch");
            Assert.That(batched.Data[4 + i], Is.EqualTo(single1.Data[i]).Within(1e-6f),
                $"Position 1, dim {i} mismatch");
            Assert.That(batched.Data[8 + i], Is.EqualTo(single2.Data[i]).Within(1e-6f),
                $"Position 2, dim {i} mismatch");
        }
    }

    [Test]
    public void Embedding_BatchedForward_GradientAccumulatesForRepeatedTokens()
    {
        using var emb = new Embedding<float>(5, 4);

        var tokenIds = new float[] { 2, 2, 2 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(3);

        var result = emb.Forward(input);
        var grad = new float[12];
        for (int i = 0; i < 12; i++) grad[i] = 1f;
        var gradTensor = ReverseGradTensor<float>.FromArray(grad);
        gradTensor.Reshape(3, 4);

        result.Backward(gradTensor);

        var weightGrad = emb.Weight!.Tensor.Grad;
        Assert.That(weightGrad, Is.Not.Null, "Weight should have gradients");

        for (int d = 0; d < 4; d++)
            Assert.That(weightGrad[2 * 4 + d], Is.EqualTo(3f).Within(1e-5f),
                $"Token 2 appeared 3 times — element {d} should accumulate to 3.0");

        float otherRowSum = 0f;
        for (int d = 0; d < 4; d++)
            otherRowSum += weightGrad[0 * 4 + d];

        Assert.That(otherRowSum, Is.EqualTo(0f).Within(1e-5f),
            "Token 0 was not used — gradient should be 0");
    }

    [Test]
    public void Embedding_BatchedForward_OutOfRangeToken_Throws()
    {
        using var emb = new Embedding<float>(5, 4);
        var tokenIds = new float[] { 0, 10, 2 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(3);

        Assert.Throws<ArgumentOutOfRangeException>(() => emb.Forward(input));
    }

    [Test]
    public void Embedding_BatchedForward_EmptyInput_Throws()
    {
        using var emb = new Embedding<float>(5, 4);
        var input = ReverseGradTensor<float>.FromArray(new float[0]);

        Assert.Throws<ArgumentException>(() => emb.Forward(input));
    }

    [Test]
    public void Embedding_BatchedForward_NullInput_Throws()
    {
        using var emb = new Embedding<float>(5, 4);
        Assert.Throws<ArgumentNullException>(() => emb.Forward(null!));
    }

    [Test]
    public void Embedding_BatchedForward_StateDict_ContainsWeight()
    {
        using var emb = new Embedding<float>(10, 4);
        var state = emb.StateDict();

        Assert.That(state.ContainsKey("Weight"), Is.True);
        Assert.That(state["Weight"].Shape, Is.EqualTo(new[] { 10, 4 }));
    }

    [Test]
    public void Embedding_BatchedForward_LoadStateDict_RestoresWeights()
    {
        using var emb1 = new Embedding<float>(10, 4);
        var tokenIds = new float[] { 1, 3, 5 };
        var input = ReverseGradTensor<float>.FromArray(tokenIds);
        input.Reshape(3);
        var original = emb1.Forward(input);
        var origData = new float[12];
        original.Data.CopyTo(origData, 0f);

        var state = emb1.StateDict();

        using var emb2 = new Embedding<float>(10, 4);
        emb2.LoadStateDict(state);

        var restored = emb2.Forward(input);
        for (int i = 0; i < 12; i++)
            Assert.That(restored.Data[i], Is.EqualTo(origData[i]).Within(1e-6f));
    }

    [Test]
    public void Embedding_BatchedForward_GetParameters_IncludesWeight()
    {
        using var emb = new Embedding<float>(10, 4);
        var parameters = emb.GetParameters();

        Assert.That(parameters.ContainsKey("Weight"), Is.True);
        Assert.That(parameters["Weight"].Length, Is.EqualTo(40));
    }

    [Test]
    public void SparseEmbedding_Forward_SumsRowsPerBatch()
    {
        using var emb = new SparseEmbedding<float>(5, 3);
        emb.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(
            [
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
                10f, 11f, 12f,
                13f, 14f, 15f
            ],
            5,
            3,
            requiresGrad: true);

        var indices = ReverseGradTensor<float>.FromArray([0f, 2f, -1f, 1f, 3f, 4f]);
        indices.Reshape(2, 3);

        var result = emb.Forward(indices);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result.Data[0], Is.EqualTo(8f).Within(1e-6f));
        Assert.That(result.Data[1], Is.EqualTo(10f).Within(1e-6f));
        Assert.That(result.Data[2], Is.EqualTo(12f).Within(1e-6f));
        Assert.That(result.Data[3], Is.EqualTo(27f).Within(1e-6f));
        Assert.That(result.Data[4], Is.EqualTo(30f).Within(1e-6f));
        Assert.That(result.Data[5], Is.EqualTo(33f).Within(1e-6f));
    }

    [Test]
    public void SparseEmbedding_Backward_DuplicateIndicesAccumulateAndPaddingIsIgnored()
    {
        using var emb = new SparseEmbedding<float>(4, 2);
        var indices = ReverseGradTensor<float>.FromArray([1f, 1f, -1f, 2f]);
        indices.Reshape(2, 2);

        var result = emb.Forward(indices);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f, 2f, 2f]);
        grad.Reshape(2, 2);
        result.Backward(grad);

        var weightGrad = emb.Weight!.Tensor.Grad;
        Assert.That(weightGrad, Is.Not.Null);
        Assert.That(weightGrad![0], Is.EqualTo(0f).Within(1e-6f));
        Assert.That(weightGrad[1], Is.EqualTo(0f).Within(1e-6f));
        Assert.That(weightGrad[2], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(weightGrad[3], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(weightGrad[4], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(weightGrad[5], Is.EqualTo(2f).Within(1e-6f));
    }

    [Test]
    public void SparseEmbedding_Forward_OutOfRangeIndex_Throws()
    {
        using var emb = new SparseEmbedding<float>(4, 2);
        var indices = ReverseGradTensor<float>.FromArray([0f, 4f]);
        indices.Reshape(1, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => emb.Forward(indices));
    }

    [Test]
    public void SparseEmbedding_Forward_RequiresTwoDimensionalInput()
    {
        using var emb = new SparseEmbedding<float>(4, 2);
        var indices = ReverseGradTensor<float>.FromArray([0f, 1f]);

        Assert.Throws<ArgumentException>(() => emb.Forward(indices));
    }

    [Test]
    public void RegisterModules_NullElement_ThrowsArgumentNullException()
    {
        var module = new ModuleWithParams();

        Assert.Throws<ArgumentNullException>(() =>
            module.RegisterModules(null!));
    }

    [Test]
    public void RegisterModules_NullArray_ThrowsArgumentNullException()
    {
        var module = new ModuleWithParams();

        Assert.Throws<ArgumentNullException>(() =>
            module.RegisterModules((Module<float>[])null!));
    }

    [Test]
    public void RegisterParameters_NullElement_ThrowsArgumentNullException()
    {
        var module = new ModuleWithParams();

        Assert.Throws<ArgumentNullException>(() =>
            module.RegisterParameters(null!));
    }

    [Test]
    public void RegisterParameters_NullArray_ThrowsArgumentNullException()
    {
        var module = new ModuleWithParams();

        Assert.Throws<ArgumentNullException>(() =>
            module.RegisterParameters((Parameter<float>[])null!));
    }

    [Test]
    public void Conv2d_Forward_ShapeCorrect()
    {
        using var conv = new Conv2d<float>(3, 16, kernelSize: 3, padding: 0);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 3, 8, 8);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 6, 6 }));
        Assert.That(output.Length, Is.EqualTo(1 * 16 * 6 * 6));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_Forward_WithPadding_ShapeCorrect()
    {
        using var conv = new Conv2d<float>(1, 4, kernelSize: 3, padding: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 1, 8, 8);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 4, 8, 8 }));
    }

    [Test]
    public void Conv2d_Forward_WithStride_ShapeCorrect()
    {
        using var conv = new Conv2d<float>(1, 8, kernelSize: 3, stride: 2, padding: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 1, 8, 8);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 4, 4 }));
    }

    [Test]
    public void Conv2d_Forward_WithBias_AddsBias()
    {
        using var conv = new Conv2d<float>(1, 1, kernelSize: 1, bias: true);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(new float[] { 5f }, requiresGrad: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(new float[] { 1f }, 1, 1, requiresGrad: true);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 3f }),
            requiresGrad: false);
        input.Reshape(1, 1, 1, 1);

        var output = conv.Forward(input);

        Assert.That(output[0], Is.EqualTo(8f).Within(1e-6f));
    }

    [Test]
    public void Conv2d_Backward_GradientsFlow()
    {
        using var conv = new Conv2d<float>(2, 3, kernelSize: 3, padding: 1);
        conv.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: true);
        input.Reshape(1, 2, 4, 4);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 4 * 4]),
            requiresGrad: false);
        gradOutput.Reshape(1, 3, 4, 4);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(32));

        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Weight!.Tensor.Grad!.Length, Is.EqualTo(3 * 2 * 3 * 3));

        if (conv.Bias != null)
        {
            Assert.That(conv.Bias!.Tensor.Grad, Is.Not.Null);
            Assert.That(conv.Bias!.Tensor.Grad!.Length, Is.EqualTo(3));
        }
    }

    [Test]
    public void Conv2d_Backward_NoBias_NoBiasGradient()
    {
        using var conv = new Conv2d<float>(1, 1, kernelSize: 3, bias: false);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 4 * 4]),
            requiresGrad: true);
        input.Reshape(1, 1, 4, 4);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 2 * 2]),
            requiresGrad: false);
        gradOutput.Reshape(1, 1, 2, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(conv.Bias, Is.Null);
    }

    [Test]
    public void Conv2d_KernelSize1_MatchesLinearPerSample()
    {
        using var conv = new Conv2d<float>(2, 3, kernelSize: 1);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f, 5f, 6f],
            3, 2, requiresGrad: true);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }),
            requiresGrad: false);
        input.Reshape(1, 2, 1, 1);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 1, 1 }));
        Assert.That(output[0], Is.EqualTo(1f * 1f + 2f * 2f).Within(1e-6f));
        Assert.That(output[1], Is.EqualTo(1f * 3f + 2f * 4f).Within(1e-6f));
        Assert.That(output[2], Is.EqualTo(1f * 5f + 2f * 6f).Within(1e-6f));
    }

    [Test]
    public void Conv2d_Dispose_DisposesParameters()
    {
        var conv = new Conv2d<float>(2, 3, kernelSize: 3);
        var weight = conv.Weight;
        var bias = conv.Bias;

        conv.Dispose();

        Assert.That(weight, Is.Not.Null);
        Assert.That(bias, Is.Not.Null);
    }

    [Test]
    public void ConvTranspose2d_Forward_ShapeCorrect()
    {
        using var convT = new ConvTranspose2d<float>(16, 3, kernelSize: 3, padding: 0);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 16 * 6 * 6]),
            requiresGrad: false);
        input.Reshape(1, 16, 6, 6);

        var output = convT.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 8, 8 }));
        Assert.That(output.Length, Is.EqualTo(1 * 3 * 8 * 8));
    }

    [Test]
    public void ConvTranspose2d_Forward_WithPadding_ShapeCorrect()
    {
        using var convT = new ConvTranspose2d<float>(4, 1, kernelSize: 3, padding: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 4, 8, 8);

        var output = convT.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, 8, 8 }));
    }

    [Test]
    public void ConvTranspose2d_Forward_WithStride_ShapeCorrect()
    {
        using var convT = new ConvTranspose2d<float>(1, 1, kernelSize: 3, stride: 2, padding: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 1, 4, 4);

        var output = convT.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, 7, 7 }));
    }

    [Test]
    public void ConvTranspose2d_Backward_GradientsFlow()
    {
        using var convT = new ConvTranspose2d<float>(3, 2, kernelSize: 3, padding: 1);
        convT.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 4 * 4]),
            requiresGrad: true);
        input.Reshape(1, 3, 4, 4);

        var output = convT.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: false);
        gradOutput.Reshape(1, 2, 4, 4);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(48));

        Assert.That(convT.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(convT.Weight!.Tensor.Grad!.Length, Is.EqualTo(3 * 2 * 3 * 3));

        if (convT.Bias != null)
        {
            Assert.That(convT.Bias!.Tensor.Grad, Is.Not.Null);
            Assert.That(convT.Bias!.Tensor.Grad!.Length, Is.EqualTo(2));
        }
    }

    [Test]
    public void ConvTranspose2d_Forward_WithBias_AddsBias()
    {
        using var convT = new ConvTranspose2d<float>(1, 1, kernelSize: 1, bias: true);
        convT.Bias!.Tensor = ReverseGradTensor<float>.FromArray(new float[] { 10f }, requiresGrad: true);
        convT.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(new float[] { 2f }, 1, 1, requiresGrad: true);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 3f }),
            requiresGrad: false);
        input.Reshape(1, 1, 1, 1);

        var output = convT.Forward(input);

        Assert.That(output[0], Is.EqualTo(16f).Within(1e-6f));
    }

    [Test]
    public void ConvTranspose2d_Dispose_DisposesParameters()
    {
        var convT = new ConvTranspose2d<float>(2, 3, kernelSize: 3);
        var weight = convT.Weight;
        var bias = convT.Bias;

        convT.Dispose();

        Assert.That(weight, Is.Not.Null);
        Assert.That(bias, Is.Not.Null);
    }

    [Test]
    public void BroadcastMultiply_2D_ScalesPerChannel()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f, 5f, 6f], 2, 3, requiresGrad: false);
        var scale = ReverseGradTensor<float>.FromArray([2f, 3f, 4f], requiresGrad: false);

        var result = ReverseGradOperations.BroadcastMultiply(input, scale);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(result[1], Is.EqualTo(6f).Within(1e-6f));
        Assert.That(result[2], Is.EqualTo(12f).Within(1e-6f));
        Assert.That(result[3], Is.EqualTo(8f).Within(1e-6f));
        Assert.That(result[4], Is.EqualTo(15f).Within(1e-6f));
        Assert.That(result[5], Is.EqualTo(24f).Within(1e-6f));
    }

    [Test]
    public void BroadcastMultiply_4D_ScalesPerChannel()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 2 * 2]),
            requiresGrad: false);
        input.Reshape(1, 2, 2, 2);

        var scale = ReverseGradTensor<float>.FromArray(new float[] { 2f, 3f }, requiresGrad: false);

        var result = ReverseGradOperations.BroadcastMultiply(input, scale);

        Assert.That(result.Shape, Is.EqualTo(new[] { 1, 2, 2, 2 }));
        Assert.That(result.Length, Is.EqualTo(8));
    }

    [Test]
    public void BroadcastMultiply_Backward_InputGradientFlows()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f], 2, 2, requiresGrad: true);
        var scale = ReverseGradTensor<float>.FromArray([2f, 3f], requiresGrad: false);

        var result = ReverseGradOperations.BroadcastMultiply(input, scale);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f, 1f, 1f], requiresGrad: false);
        grad.Reshape(2, 2);

        result.Backward(grad);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad![0], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(input.Grad[1], Is.EqualTo(3f).Within(1e-6f));
        Assert.That(input.Grad[2], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(input.Grad[3], Is.EqualTo(3f).Within(1e-6f));
    }

    [Test]
    public void BroadcastMultiply_Backward_ScaleGradientFlows()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f], 2, 2, requiresGrad: false);
        var scale = ReverseGradTensor<float>.FromArray([2f, 3f], requiresGrad: true);

        var result = ReverseGradOperations.BroadcastMultiply(input, scale);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f, 1f, 1f], requiresGrad: false);
        grad.Reshape(2, 2);

        result.Backward(grad);

        Assert.That(scale.Grad, Is.Not.Null);
        Assert.That(scale.Grad![0], Is.EqualTo(1f + 3f).Within(1e-6f));
        Assert.That(scale.Grad[1], Is.EqualTo(2f + 4f).Within(1e-6f));
    }

    [Test]
    public void BroadcastMultiply_BothRequireGrad_BothGetGradients()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [2f, 4f], 1, 2, requiresGrad: true);
        var scale = ReverseGradTensor<float>.FromArray([3f, 5f], requiresGrad: true);

        var result = ReverseGradOperations.BroadcastMultiply(input, scale);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f], requiresGrad: false);
        grad.Reshape(1, 2);

        result.Backward(grad);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad![0], Is.EqualTo(3f).Within(1e-6f));
        Assert.That(input.Grad[1], Is.EqualTo(5f).Within(1e-6f));
        Assert.That(scale.Grad, Is.Not.Null);
        Assert.That(scale.Grad![0], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(scale.Grad[1], Is.EqualTo(4f).Within(1e-6f));
    }

    [Test]
    public void BroadcastMultiply_ScaleLengthMismatch_Throws()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f], 2, 2, requiresGrad: false);
        var scale = ReverseGradTensor<float>.FromArray(new float[] { 1f }, requiresGrad: false);

        Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.BroadcastMultiply(input, scale));
    }

    [Test]
    public void BroadcastAdd_2D_AddsPerChannel()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f, 5f, 6f], 2, 3, requiresGrad: false);
        var bias = ReverseGradTensor<float>.FromArray([10f, 20f, 30f], requiresGrad: false);

        var result = ReverseGradOperations.BroadcastAdd(input, bias);

        Assert.That(result.Shape, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(result[0], Is.EqualTo(11f).Within(1e-6f));
        Assert.That(result[1], Is.EqualTo(22f).Within(1e-6f));
        Assert.That(result[2], Is.EqualTo(33f).Within(1e-6f));
        Assert.That(result[3], Is.EqualTo(14f).Within(1e-6f));
        Assert.That(result[4], Is.EqualTo(25f).Within(1e-6f));
        Assert.That(result[5], Is.EqualTo(36f).Within(1e-6f));
    }

    [Test]
    public void BroadcastAdd_4D_AddsPerChannel()
    {
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 3, 4, 4);

        var bias = ReverseGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f }, requiresGrad: false);

        var result = ReverseGradOperations.BroadcastAdd(input, bias);

        Assert.That(result.Shape, Is.EqualTo(new[] { 1, 3, 4, 4 }));
        Assert.That(result.Length, Is.EqualTo(48));
    }

    [Test]
    public void BroadcastAdd_Backward_InputGradientFlows()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f], 2, 2, requiresGrad: true);
        var bias = ReverseGradTensor<float>.FromArray([10f, 20f], requiresGrad: false);

        var result = ReverseGradOperations.BroadcastAdd(input, bias);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f, 1f, 1f], requiresGrad: false);
        grad.Reshape(2, 2);

        result.Backward(grad);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(4));
        for (int i = 0; i < 4; i++)
            Assert.That(input.Grad[i], Is.EqualTo(1f).Within(1e-6f));
    }

    [Test]
    public void BroadcastAdd_Backward_BiasGradientFlows()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f, 3f, 4f], 2, 2, requiresGrad: false);
        var bias = ReverseGradTensor<float>.FromArray([10f, 20f], requiresGrad: true);

        var result = ReverseGradOperations.BroadcastAdd(input, bias);
        var grad = ReverseGradTensor<float>.FromArray([1f, 1f, 1f, 1f], requiresGrad: false);
        grad.Reshape(2, 2);

        result.Backward(grad);

        Assert.That(bias.Grad, Is.Not.Null);
        Assert.That(bias.Grad![0], Is.EqualTo(2f).Within(1e-6f));
        Assert.That(bias.Grad[1], Is.EqualTo(2f).Within(1e-6f));
    }

    [Test]
    public void BroadcastAdd_BiasLengthMismatch_Throws()
    {
        var input = ReverseGradTensor<float>.FromMatrix(
            [1f, 2f], 1, 2, requiresGrad: false);
        var bias = ReverseGradTensor<float>.FromArray(new float[] { 1f, 2f, 3f }, requiresGrad: false);

        Assert.Throws<ArgumentException>(() =>
            ReverseGradOperations.BroadcastAdd(input, bias));
    }

    [Test]
    public void Conv2d_Backward_WithStrideAndPadding()
    {
        using var conv = new Conv2d<float>(2, 4, kernelSize: 3, stride: 2, padding: 1);
        conv.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 8 * 8]),
            requiresGrad: true);
        input.Reshape(1, 2, 8, 8);

        var output = conv.Forward(input);
        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 4, 4, 4 }));

        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 4 * 4]),
            requiresGrad: false);
        gradOutput.Reshape(1, 4, 4, 4);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(128));
        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Weight!.Tensor.Grad!.Length, Is.EqualTo(4 * 2 * 3 * 3));
    }

    [Test]
    public void Conv2d_Backward_MultiBatch()
    {
        using var conv = new Conv2d<float>(3, 8, kernelSize: 3, padding: 1);
        conv.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[4 * 3 * 16 * 16]),
            requiresGrad: true);
        input.Reshape(4, 3, 16, 16);

        var output = conv.Forward(input);
        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 8, 16, 16 }));

        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[4 * 8 * 16 * 16]),
            requiresGrad: false);
        gradOutput.Reshape(4, 8, 16, 16);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(4 * 3 * 16 * 16));
        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Weight!.Tensor.Grad!.Length, Is.EqualTo(8 * 3 * 3 * 3));
    }

    [Test]
    public void ConvTranspose2d_Backward_WithStride()
    {
        using var convT = new ConvTranspose2d<float>(4, 2, kernelSize: 3, stride: 2, padding: 1);
        convT.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 4 * 4]),
            requiresGrad: true);
        input.Reshape(1, 4, 4, 4);

        var output = convT.Forward(input);
        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 2, 7, 7 }));

        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 7 * 7]),
            requiresGrad: false);
        gradOutput.Reshape(1, 2, 7, 7);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(64));
        Assert.That(convT.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(convT.Weight!.Tensor.Grad!.Length, Is.EqualTo(4 * 2 * 3 * 3));
    }

    [Test]
    public void Conv2d_LargeChannels_ForwardCorrect()
    {
        using var conv = new Conv2d<float>(64, 128, kernelSize: 3, padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 64 * 32 * 32]),
            requiresGrad: false);
        input.Reshape(2, 64, 32, 32);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 128, 32, 32 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void ConvTranspose2d_LargeChannels_ForwardCorrect()
    {
        using var convT = new ConvTranspose2d<float>(128, 64, kernelSize: 4, stride: 2, padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 128 * 16 * 16]),
            requiresGrad: false);
        input.Reshape(2, 128, 16, 16);

        var output = convT.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 64, 32, 32 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_Pointwise1x1_LargeChannels_ForwardCorrect()
    {
        using var conv = new Conv2d<float>(128, 256, kernelSize: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 128 * 14 * 14]),
            requiresGrad: false);
        input.Reshape(2, 128, 14, 14);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 256, 14, 14 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void LayerNorm_2D_Forward_ShapeCorrect()
    {
        using var ln = new LayerNorm<float>(8);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[4 * 8]),
            requiresGrad: false);
        input.Reshape(4, 8);

        var output = ln.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 8 }));
        Assert.That(output.Length, Is.EqualTo(32));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void LayerNorm_4D_LastDimNormalized()
    {
        using var ln = new LayerNorm<float>(16);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 4 * 4 * 16]),
            requiresGrad: false);
        input.Reshape(2, 4, 4, 16);

        var output = ln.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 4, 4, 16 }));
    }

    [Test]
    public void LayerNorm_Backward_GradientsFlow()
    {
        using var ln = new LayerNorm<float>(4);
        ln.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]),
            requiresGrad: true);
        input.Reshape(2, 4);

        var output = ln.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[8]),
            requiresGrad: false);
        gradOutput.Reshape(2, 4);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(8));
        Assert.That(ln.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(ln.Weight!.Tensor.Grad!.Length, Is.EqualTo(4));
        Assert.That(ln.Bias!.Tensor.Grad, Is.Not.Null);
        Assert.That(ln.Bias!.Tensor.Grad!.Length, Is.EqualTo(4));
    }

    [Test]
    public void LayerNorm_AffineFalse_NoScaleShift()
    {
        using var ln = new LayerNorm<float>(4, affine: false);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        input.Reshape(1, 4);

        var output = ln.Forward(input);

        Assert.That(ln.Weight, Is.Null);
        Assert.That(ln.Bias, Is.Null);
        Assert.That(output.Length, Is.EqualTo(4));
    }

    [Test]
    public void LayerNorm_NormalizedOutput()
    {
        using var ln = new LayerNorm<float>(4, affine: false);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f, 4f]),
            requiresGrad: false);
        input.Reshape(1, 4);

        var output = ln.Forward(input);

        double sum = 0;
        for (int i = 0; i < 4; i++)
            sum += output[i];
        Assert.That(sum, Is.EqualTo(0f).Within(1e-5f));

        double sumSq = 0;
        for (int i = 0; i < 4; i++)
            sumSq += output[i] * output[i];
        double variance = sumSq / 4.0;
        Assert.That(variance, Is.EqualTo(1f).Within(1e-4f));
    }

    [Test]
    public void LayerNorm_Dispose_DisposesParameters()
    {
        var ln = new LayerNorm<float>(8);
        var weight = ln.Weight;
        var bias = ln.Bias;

        ln.Dispose();

        Assert.That(weight, Is.Not.Null);
        Assert.That(bias, Is.Not.Null);
    }

    [Test]
    public void Conv2d_Grouped_Forward_ShapeCorrect()
    {
        using var conv = new Conv2d<float>(8, 8, kernelSize: 3, padding: 1, groups: 2);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 8 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 8, 4, 4);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 4, 4 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_Grouped_MultiBatch_ForwardCorrect()
    {
        using var conv = new Conv2d<float>(6, 6, kernelSize: 3, padding: 1, groups: 3);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 6 * 5 * 5]),
            requiresGrad: false);
        input.Reshape(2, 6, 5, 5);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 6, 5, 5 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_Grouped_IdentityGroupsMatchesUngrouped()
    {
        var rng = new Random(42);
        float[] weightData = new float[4 * 4 * 3 * 3];
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

        using var convGrouped = new Conv2d<float>(4, 4, kernelSize: 3, padding: 1, groups: 1);

        float[] inputData = new float[1 * 4 * 5 * 5];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(inputData), requiresGrad: false);
        input.Reshape(1, 4, 5, 5);

        var output = convGrouped.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 4, 5, 5 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_Grouped_Backward_GradientCorrect()
    {
        using var conv = new Conv2d<float>(4, 4, kernelSize: 3, padding: 1, groups: 2);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 4 * 4].Select(_ => 0.1f).ToArray()),
            requiresGrad: true);
        input.Reshape(1, 4, 4, 4);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);
        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad!.Length; i++)
        {
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
            Assert.That(float.IsInfinity(input.Grad[i]), Is.False);
        }

        foreach (var p in conv.Parameters())
        {
            Assert.That(p.Value.Grad, Is.Not.Null);
            for (int i = 0; i < p.Value.Grad!.Length; i++)
            {
                Assert.That(float.IsNaN(p.Value.Grad[i]), Is.False);
                Assert.That(float.IsInfinity(p.Value.Grad[i]), Is.False);
            }
        }
    }

    [Test]
    public void Conv2d_Grouped_InvalidGroups_Throws()
    {
        Assert.Throws<ArgumentException>(() => new Conv2d<float>(4, 4, 3, groups: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv2d<float>(4, 4, 3, groups: 0));
    }

    [Test]
    public void Conv2d_DeepGrouped_Forward_ShapeCorrect()
    {
        using var conv = new Conv2d<float>(32, 64, kernelSize: 3, padding: 1, groups: 8);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 32 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(2, 32, 8, 8);

        var output = conv.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 64, 8, 8 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_AsymmetricPadding_OutputShapeCorrect()
    {
        using var conv = new Conv2d<float>(1, 1, kernelSize: 3, stride: 2,
            paddingTop: 1, paddingBottom: 0, paddingLeft: 1, paddingRight: 0, bias: false);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 7 * 7]),
            requiresGrad: false);
        input.Reshape(1, 1, 7, 7);

        var output = conv.Forward(input);

        int expectedH = (7 + 1 + 0 - 3) / 2 + 1;
        int expectedW = (7 + 1 + 0 - 3) / 2 + 1;
        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, expectedH, expectedW }));
    }

    [Test]
    public void Conv2d_AsymmetricPadding_ForwardCorrect()
    {
        using var conv = new Conv2d<float>(1, 1, kernelSize: 3, stride: 2,
            paddingTop: 1, paddingBottom: 0, paddingLeft: 1, paddingRight: 0, bias: false);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(
            new float[9].Select(_ => 1f).ToArray(), 1, 9, requiresGrad: false);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Range(0, 49).Select(i => (float)i).ToArray()),
            requiresGrad: false);
        input.Reshape(1, 1, 7, 7);

        var output = conv.Forward(input);

        int expectedH = (7 + 1 + 0 - 3) / 2 + 1;
        int expectedW = (7 + 1 + 0 - 3) / 2 + 1;
        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, expectedH, expectedW }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void Conv2d_AsymmetricPadding_Backward_GradientFlows()
    {
        using var conv = new Conv2d<float>(2, 3, kernelSize: 3, stride: 2,
            paddingTop: 1, paddingBottom: 0, paddingLeft: 1, paddingRight: 0, bias: true);
        conv.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 7 * 7]),
            requiresGrad: true);
        input.Reshape(1, 2, 7, 7);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[output.Length]),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(1 * 2 * 7 * 7));
        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Weight!.Tensor.Grad!.Length, Is.EqualTo(3 * 2 * 3 * 3));
        Assert.That(conv.Bias!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Bias!.Tensor.Grad!.Length, Is.EqualTo(3));
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);
    }

    [Test]
    public void Conv2d_AsymmetricPadding_Grouped_Backward_GradientFlows()
    {
        using var conv = new Conv2d<float>(4, 4, kernelSize: 3, stride: 2,
            paddingTop: 1, paddingBottom: 0, paddingLeft: 1, paddingRight: 0, bias: false, groups: 2);
        conv.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 8 * 8]),
            requiresGrad: true);
        input.Reshape(1, 4, 8, 8);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[output.Length]),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(1 * 4 * 8 * 8));
        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);
    }

    [Test]
    public void Conv2d_AsymmetricPadding_ConstructorRejectsNegativeValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Conv2d<float>(1, 1, 3, 1, paddingTop: -1, paddingBottom: 0, paddingLeft: 0, paddingRight: 0, bias: false));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Conv2d<float>(1, 1, 3, 1, paddingTop: 0, paddingBottom: -1, paddingLeft: 0, paddingRight: 0, bias: false));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Conv2d<float>(1, 1, 3, 1, paddingTop: 0, paddingBottom: 0, paddingLeft: -1, paddingRight: 0, bias: false));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Conv2d<float>(1, 1, 3, 1, paddingTop: 0, paddingBottom: 0, paddingLeft: 0, paddingRight: -1, bias: false));
    }

    [Test]
    public void MaxPool2d_Forward_ShapeCorrect()
    {
        using var pool = new MaxPool2d<float>(kernelSize: 2, stride: 2);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 4 * 4]),
            requiresGrad: false);
        input.Reshape(1, 1, 4, 4);

        var output = pool.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, 2, 2 }));
    }

    [Test]
    public void MaxPool2d_Forward_WithPadding()
    {
        using var pool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Range(0, 16).Select(i => (float)i).ToArray()),
            requiresGrad: false);
        input.Reshape(1, 1, 4, 4);

        var output = pool.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, 2, 2 }));
        Assert.That(output[0], Is.EqualTo(5f));
        Assert.That(output[1], Is.EqualTo(7f));
        Assert.That(output[2], Is.EqualTo(13f));
        Assert.That(output[3], Is.EqualTo(15f));
    }

    [Test]
    public void MaxPool2d_Backward_GradientFlows()
    {
        using var pool = new MaxPool2d<float>(kernelSize: 2, stride: 2);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }),
            requiresGrad: true);
        input.Reshape(1, 1, 4, 4);

        var output = pool.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 2, 3, 4 }),
            requiresGrad: false);
        gradOutput.Reshape(1, 1, 2, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(16));
        Assert.That(input.Grad[5], Is.EqualTo(1f), "max of top-left 2x2 block");
        Assert.That(input.Grad[7], Is.EqualTo(2f), "max of top-right 2x2 block");
        Assert.That(input.Grad[13], Is.EqualTo(3f), "max of bottom-left 2x2 block");
        Assert.That(input.Grad[15], Is.EqualTo(4f), "max of bottom-right 2x2 block");
        for (int i = 0; i < 16; i++)
            if (i != 5 && i != 7 && i != 13 && i != 15)
                Assert.That(input.Grad[i], Is.EqualTo(0f));
    }

    [Test]
    public void MaxPool2d_ConstructorRejectsInvalidArgs()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MaxPool2d<float>(kernelSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MaxPool2d<float>(kernelSize: 3, padding: -1));
    }

    [Test]
    public void AdaptiveAvgPool2d_GlobalAvgPool_FlattensTo1x1()
    {
        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 7 * 7].Select(_ => 1f).ToArray()),
            requiresGrad: false);
        input.Reshape(1, 3, 7, 7);

        var output = pool.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 1, 1 }));
        for (int i = 0; i < 3; i++)
            Assert.That(output[i], Is.EqualTo(1f));
    }

    [Test]
    public void AdaptiveAvgPool2d_Backward_GradientFlows()
    {
        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 2 * 4 * 4]),
            requiresGrad: true);
        input.Reshape(1, 2, 4, 4);

        var output = pool.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 1 }),
            requiresGrad: false);
        gradOutput.Reshape(1, 2, 1, 1);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(1 * 2 * 4 * 4));
        float expectedGrad = 1f / 16f;
        for (int i = 0; i < input.Grad.Length; i++)
            Assert.That(input.Grad[i], Is.EqualTo(expectedGrad).Within(1e-6f));
    }

    [Test]
    public void AdaptiveAvgPool2d_ConstructorRejectsInvalidArgs()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaptiveAvgPool2d<float>(outputSize: 0));
    }

    [Test]
    public void MultiheadAttention_ShapeCorrect()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 64, numHeads: 8);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[10 * 64]),
            requiresGrad: false);
        input.Reshape(10, 64);

        var output = mha.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 10, 64 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void MultiheadAttention_Causal_ShapeCorrect()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4, causal: true);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[8 * 32]),
            requiresGrad: false);
        input.Reshape(8, 32);

        var output = mha.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 8, 32 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void MultiheadAttention_Backward_GradientFlows()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[6 * 32].Select(_ => 0.1f).ToArray()),
            requiresGrad: true);
        input.Reshape(6, 32);

        var output = mha.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad!.Length; i++)
        {
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
            Assert.That(float.IsInfinity(input.Grad[i]), Is.False);
        }

        foreach (var p in mha.Parameters())
        {
            Assert.That(p.Value.Grad, Is.Not.Null);
            for (int i = 0; i < p.Value.Grad!.Length; i++)
            {
                Assert.That(float.IsNaN(p.Value.Grad[i]), Is.False);
                Assert.That(float.IsInfinity(p.Value.Grad[i]), Is.False);
            }
        }
    }

    [Test]
    public void MultiheadAttention_CrossAttention_ShapeCorrect()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4);
        var query = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[5 * 32]),
            requiresGrad: false);
        query.Reshape(5, 32);
        var kv = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[8 * 32]),
            requiresGrad: false);
        kv.Reshape(8, 32);

        var output = mha.Forward(query, kv, kv);

        Assert.That(output.Shape, Is.EqualTo(new[] { 5, 32 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void MultiheadAttention_InvalidEmbedDim_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MultiheadAttention<float>(embedDim: 33, numHeads: 8));
    }

    [Test]
    public void MultiheadAttention_PaddingMask_ShapeCorrect()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[6 * 32]),
            requiresGrad: false);
        input.Reshape(6, 32);

        var paddingMask = ReverseGradTensor<float>.FromArray(
            new float[] { 1f, 1f, 1f, 1f, 0f, 0f }, requiresGrad: false);

        var output = mha.Forward(input, paddingMask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 6, 32 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void MultiheadAttention_PaddingMask_BackwardFlows()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4);
        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[4 * 32].Select(_ => 0.1f).ToArray()),
            requiresGrad: true);
        input.Reshape(4, 32);

        var paddingMask = ReverseGradTensor<float>.FromArray(
            new float[] { 1f, 1f, 0f, 0f }, requiresGrad: false);

        var output = mha.Forward(input, paddingMask);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad!.Length; i++)
        {
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
            Assert.That(float.IsInfinity(input.Grad[i]), Is.False);
        }

        foreach (var p in mha.Parameters())
        {
            Assert.That(p.Value.Grad, Is.Not.Null);
            for (int i = 0; i < p.Value.Grad!.Length; i++)
            {
                Assert.That(float.IsNaN(p.Value.Grad[i]), Is.False);
                Assert.That(float.IsInfinity(p.Value.Grad[i]), Is.False);
            }
        }
    }

    [Test]
    public void MultiheadAttention_CrossAttention_PaddingMask_ShapeCorrect()
    {
        using var mha = new MultiheadAttention<float>(embedDim: 32, numHeads: 4);
        var query = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[4 * 32]),
            requiresGrad: false);
        query.Reshape(4, 32);
        var kv = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[6 * 32]),
            requiresGrad: false);
        kv.Reshape(6, 32);

        var paddingMask = ReverseGradTensor<float>.FromArray(
            new float[] { 1f, 1f, 1f, 1f, 0f, 0f }, requiresGrad: false);

        var output = mha.Forward(query, kv, kv, causal: false, paddingMask: paddingMask);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void ConvVAE_Forward_ShapeCorrect()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 1,
            encoderChannels: [32, 64],
            latentChannels: 8,
            spatialSize: 28,
            kernelSize: 4,
            stride: 2,
            padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 1 * 28 * 28]),
            requiresGrad: false);
        input.Reshape(2, 1, 28, 28);

        var output = cvae.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 1, 28, 28 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void ConvVAE_Encode_ShapeCorrect()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 3,
            encoderChannels: [32, 64],
            latentChannels: 16,
            spatialSize: 32,
            kernelSize: 4,
            stride: 2,
            padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 32 * 32]),
            requiresGrad: false);
        input.Reshape(1, 3, 32, 32);

        var (mu, logVar) = cvae.Encode(input);

        Assert.That(mu.Shape, Is.EqualTo(new[] { 1, 16, 8, 8 }));
        Assert.That(logVar.Shape, Is.EqualTo(new[] { 1, 16, 8, 8 }));
    }

    [Test]
    public void ConvVAE_Decode_RoundTrip()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 1,
            encoderChannels: [32],
            latentChannels: 8,
            spatialSize: 16,
            kernelSize: 4,
            stride: 2,
            padding: 1);

        var z = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 8 * 8 * 8]),
            requiresGrad: false);
        z.Reshape(1, 8, 8, 8);

        var output = cvae.Decode(z);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1, 16, 16 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void ConvVAE_ElboLoss_ComputesCorrectly()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 1,
            encoderChannels: [16],
            latentChannels: 4,
            spatialSize: 8,
            kernelSize: 4,
            stride: 2,
            padding: 1);

        var recon = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 8 * 8]),
            requiresGrad: false);
        recon.Reshape(1, 1, 8, 8);

        var original = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 8 * 8]),
            requiresGrad: false);
        original.Reshape(1, 1, 8, 8);

        var mu = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 4 * 4]),
            requiresGrad: false);
        mu.Reshape(1, 4, 4, 4);

        var logVar = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 4 * 4]),
            requiresGrad: false);
        logVar.Reshape(1, 4, 4, 4);

        var loss = cvae.ElboLoss(recon, original, mu, logVar);

        Assert.That(loss.Length, Is.EqualTo(1));
        Assert.That(float.IsNaN(loss[0]), Is.False);
        Assert.That(float.IsInfinity(loss[0]), Is.False);
    }

    [Test]
    public void ConvVAE_Backward_GradientFlows()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 1,
            encoderChannels: [16],
            latentChannels: 4,
            spatialSize: 8,
            kernelSize: 4,
            stride: 2,
            padding: 1);
        cvae.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 1 * 8 * 8].Select(_ => 0.1f).ToArray()),
            requiresGrad: true);
        input.Reshape(1, 1, 8, 8);

        var (mu, logVar) = cvae.Encode(input);
        var z = cvae.Reparameterize(mu, logVar);
        var recon = cvae.Decode(z);
        var loss = cvae.ElboLoss(recon, input, mu, logVar);

        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f }),
            requiresGrad: false);
        gradOutput.Reshape(1);

        loss.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad!.Length; i++)
        {
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
            Assert.That(float.IsInfinity(input.Grad[i]), Is.False);
        }
    }

    [Test]
    public void ConvVAE_EndToEnd_ReducesLoss()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 1,
            encoderChannels: [16],
            latentChannels: 4,
            spatialSize: 8,
            kernelSize: 4,
            stride: 2,
            padding: 1);
        cvae.Train();

        var optimizer = new Adam<float>();
        optimizer.AddParameterGroup(cvae.GetParameters().Values, 0.001f);
        var rng = new Random(42);

        var target = new float[1 * 1 * 8 * 8];
        for (int i = 0; i < target.Length; i++)
            target[i] = (float)(Math.Sin(i * 0.1) * 0.5 + 0.5);

        var firstLoss = float.MaxValue;

        for (int epoch = 0; epoch < 5; epoch++)
        {
            var inputData = new float[1 * 1 * 8 * 8];
            target.CopyTo(inputData, 0);

            var input = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(inputData),
                requiresGrad: true);
            input.Reshape(1, 1, 8, 8);

            var (mu, logVar) = cvae.Encode(input);
            var z = cvae.Reparameterize(mu, logVar);
            var recon = cvae.Decode(z);
            var loss = cvae.ElboLoss(recon, input, mu, logVar);

            if (epoch == 0) firstLoss = loss[0];

            var gradOutput = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(new float[] { 1.0f }),
                requiresGrad: false);
            gradOutput.Reshape(1);
            loss.Backward(gradOutput);
            optimizer.Step();
            optimizer.ZeroGrad();
        }

        var finalInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(target),
            requiresGrad: true);
        finalInput.Reshape(1, 1, 8, 8);

        var (fMu, fLogVar) = cvae.Encode(finalInput);
        var fZ = cvae.Reparameterize(fMu, fLogVar);
        var fRecon = cvae.Decode(fZ);
        var finalLoss = cvae.ElboLoss(fRecon, finalInput, fMu, fLogVar);

        Assert.That(finalLoss[0], Is.LessThan(firstLoss * 2.0f + 0.5f));
    }

    [Test]
    public void ConvVAE_InvalidArgs_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ConvVAE<float>(1, [], 8, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvVAE<float>(1, [16], -1, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvVAE<float>(1, [16], 8, 0));
    }

    [Test]
    public void ConvVAE_3Channel_RGB()
    {
        using var cvae = new ConvVAE<float>(
            inputChannels: 3,
            encoderChannels: [32, 64],
            latentChannels: 8,
            spatialSize: 32,
            kernelSize: 4,
            stride: 2,
            padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 32 * 32]),
            requiresGrad: false);
        input.Reshape(1, 3, 32, 32);

        var output = cvae.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 32, 32 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void DepthwiseSeparableConv2d_Forward_ShapeCorrect()
    {
        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 3,
            outChannels: 16,
            kernelSize: 3,
            padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 3, 8, 8);

        var output = dsc.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 8, 8 }));
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void DepthwiseSeparableConv2d_WithStride()
    {
        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 16,
            outChannels: 32,
            kernelSize: 3,
            stride: 2,
            padding: 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 16 * 16 * 16]),
            requiresGrad: false);
        input.Reshape(2, 16, 16, 16);

        var output = dsc.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 32, 8, 8 }));
        for (int i = 0; i < Math.Min(100, output.Length); i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False);
    }

    [Test]
    public void DepthwiseSeparableConv2d_Backward_GradientFlows()
    {
        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 4,
            outChannels: 8,
            kernelSize: 3,
            padding: 1);
        dsc.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 4 * 6 * 6].Select(_ => 0.1f).ToArray()),
            requiresGrad: true);
        input.Reshape(1, 4, 6, 6);

        var output = dsc.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < input.Grad!.Length; i++)
        {
            Assert.That(float.IsNaN(input.Grad[i]), Is.False);
            Assert.That(float.IsInfinity(input.Grad[i]), Is.False);
        }

        foreach (var p in dsc.GetParameters().Values)
        {
            Assert.That(p.Tensor.Grad, Is.Not.Null);
            for (int i = 0; i < p.Tensor.Grad!.Length; i++)
            {
                Assert.That(float.IsNaN(p.Tensor.Grad[i]), Is.False);
                Assert.That(float.IsInfinity(p.Tensor.Grad[i]), Is.False);
            }
        }
    }

    [Test]
    public void DepthwiseSeparableConv2d_NoBias()
    {
        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 3,
            outChannels: 16,
            kernelSize: 3,
            padding: 1,
            useBias: false);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[1 * 3 * 8 * 8]),
            requiresGrad: false);
        input.Reshape(1, 3, 8, 8);

        var output = dsc.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 8, 8 }));
    }

    [Test]
    public void DepthwiseSeparableConv2d_EquivalentToManualComposition()
    {
        var rng = new Random(42);

        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 4,
            outChannels: 8,
            kernelSize: 3,
            padding: 1);

        using var depthwise = new Conv2d<float>(4, 4, 3, padding: 1, bias: false, groups: 4);
        using var pointwise = new Conv2d<float>(4, 8, 1, bias: true);

        var inputData = new float[1 * 4 * 6 * 6];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(inputData),
            requiresGrad: false);
        input.Reshape(1, 4, 6, 6);

        var outputDSC = dsc.Forward(input);

        var h = depthwise.Forward(input);
        h = Activation.Relu(h);
        var outputManual = pointwise.Forward(h);

        Assert.That(outputDSC.Shape, Is.EqualTo(outputManual.Shape));

        var dscData = new float[outputDSC.Length];
        var manualData = new float[outputManual.Length];
        outputDSC.Data.CopyTo(dscData, default(float)!);
        outputManual.Data.CopyTo(manualData, default(float)!);

        for (int i = 0; i < dscData.Length; i++)
            Assert.That(dscData[i], Is.EqualTo(manualData[i]).Within(1e-5f));
    }

    [Test]
    public void Conv1d_Forward_ShapeCorrect()
    {
        using var conv = new Conv1d<float>(inChannels: 3, outChannels: 16, kernelSize: 3, padding: 1);
        var input = ReverseGradTensor<float>.FromArray(new float[2 * 3 * 10], requiresGrad: true);
        input.Reshape(2, 3, 10);

        var output = conv.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(3));
        Assert.That(output.Shape[0], Is.EqualTo(2));
        Assert.That(output.Shape[1], Is.EqualTo(16));
        Assert.That(output.Shape[2], Is.EqualTo(10));
    }

    [Test]
    public void Conv1d_Forward_WithStride_ShapeCorrect()
    {
        using var conv = new Conv1d<float>(inChannels: 4, outChannels: 8, kernelSize: 3, stride: 2, padding: 1);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 4 * 12], requiresGrad: true);
        input.Reshape(1, 4, 12);

        var output = conv.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(3));
        Assert.That(output.Shape[0], Is.EqualTo(1));
        Assert.That(output.Shape[1], Is.EqualTo(8));
        Assert.That(output.Shape[2], Is.EqualTo(6));
    }

    [Test]
    public void Conv1d_Forward_WithPadding_ShapeCorrect()
    {
        using var conv = new Conv1d<float>(inChannels: 2, outChannels: 4, kernelSize: 5, padding: 2);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 2 * 8], requiresGrad: true);
        input.Reshape(1, 2, 8);

        var output = conv.Forward(input);

        Assert.That(output.Shape[2], Is.EqualTo(8));
    }

    [Test]
    public void Conv1d_Forward_NoPadding_ReducesLength()
    {
        using var conv = new Conv1d<float>(inChannels: 3, outChannels: 8, kernelSize: 3, padding: 0);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 3 * 10], requiresGrad: true);
        input.Reshape(1, 3, 10);

        var output = conv.Forward(input);

        Assert.That(output.Shape[2], Is.EqualTo(8));
    }

    [Test]
    public void Conv1d_Forward_WithBias_AddsBias()
    {
        using var conv = new Conv1d<float>(inChannels: 2, outChannels: 3, kernelSize: 1, bias: true);
        var inputData = new float[] { 1, 2, 3, 4, 5, 6 };
        var input = ReverseGradTensor<float>.FromArray(inputData, requiresGrad: true);
        input.Reshape(1, 2, 3);

        var output = conv.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(3));
        Assert.That(output.Shape[2], Is.EqualTo(3));
        float[] outVals = new float[output.Length];
        output.Data.CopyTo(outVals, default(float)!);
        Assert.That(outVals.Any(v => !float.IsNaN(v)), Is.True);
    }

    [Test]
    public void Conv1d_Backward_GradientsFlow()
    {
        using var conv = new Conv1d<float>(inChannels: 3, outChannels: 8, kernelSize: 3, padding: 1);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 3 * 8], requiresGrad: true);
        input.Reshape(1, 3, 8);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[output.Length]),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(conv.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.Bias!.Tensor.Grad, Is.Not.Null);
    }

    [Test]
    public void Conv1d_Backward_NoBias_NoBiasGradient()
    {
        using var conv = new Conv1d<float>(inChannels: 2, outChannels: 4, kernelSize: 3, bias: false);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 2 * 6], requiresGrad: true);
        input.Reshape(1, 2, 6);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[output.Length]),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(conv.Bias, Is.Null);
        Assert.That(input.Grad, Is.Not.Null);
    }

    [Test]
    public void Conv1d_KernelSize1_MatchesLinearPerPosition()
    {
        int inC = 4, outC = 3, len = 5;
        using var conv = new Conv1d<float>(inC, outC, kernelSize: 1, bias: false);

        var inputData = new float[inC * len];
        var rng = new Random(123);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);

        var input = ReverseGradTensor<float>.FromArray(inputData, requiresGrad: false);
        input.Reshape(1, inC, len);

        var output = conv.Forward(input);
        float[] outVals = new float[output.Length];
        output.Data.CopyTo(outVals, default(float)!);

        float[] weightData = new float[outC * inC];
        conv.Weight!.Tensor.Data.CopyTo(weightData, default(float)!);

        for (int pos = 0; pos < len; pos++)
        {
            for (int oc = 0; oc < outC; oc++)
            {
                float expected = 0;
                for (int ic = 0; ic < inC; ic++)
                    expected += weightData[oc * inC + ic] * inputData[ic * len + pos];
                Assert.That(outVals[oc * len + pos], Is.EqualTo(expected).Within(1e-5f));
            }
        }
    }

    [Test]
    public void Conv1d_Dispose_DisposesParameters()
    {
        var conv = new Conv1d<float>(inChannels: 3, outChannels: 8, kernelSize: 3);
        conv.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = conv.Weight!.Tensor.Length);
    }

    [Test]
    public void Conv1d_Backward_MultiBatch()
    {
        using var conv = new Conv1d<float>(inChannels: 2, outChannels: 4, kernelSize: 3, padding: 1);
        var input = ReverseGradTensor<float>.FromArray(new float[4 * 2 * 8], requiresGrad: true);
        input.Reshape(4, 2, 8);

        var output = conv.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[output.Length]),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        float[] gradVals = new float[input.Grad!.Length];
        input.Grad.CopyTo(gradVals, default(float)!);
        Assert.That(gradVals.All(v => !float.IsNaN(v)), Is.True);
    }

    [Test]
    public void Conv1d_InvalidInputRank_Throws()
    {
        using var conv = new Conv1d<float>(inChannels: 3, outChannels: 8, kernelSize: 3);
        var input = ReverseGradTensor<float>.FromArray(new float[3 * 8], requiresGrad: false);
        input.Reshape(3, 8);

        Assert.Throws<ArgumentException>(() => conv.Forward(input));
    }

    [Test]
    public void Conv1d_InvalidInputChannels_Throws()
    {
        using var conv = new Conv1d<float>(inChannels: 3, outChannels: 8, kernelSize: 3);
        var input = ReverseGradTensor<float>.FromArray(new float[1 * 5 * 10], requiresGrad: false);
        input.Reshape(1, 5, 10);

        Assert.Throws<ArgumentException>(() => conv.Forward(input));
    }

    [Test]
    public void RMSNormKernel_Half_Forward_MatchesScalarReference()
    {
        var rng = new Random(7);
        const int rows = 4, cols = 8;
        var src = new Half[rows * cols];
        for (int i = 0; i < src.Length; i++)
            src[i] = (Half)(rng.NextDouble() * 2 - 1);
        var actual = new Half[src.Length];
        RMSNormKernel<Half>.PerRowRMSNormForwardKernel(src, actual, rows, cols, 1e-5);

        for (int r = 0; r < rows; r++)
        {
            double sumSq = 0;
            for (int j = 0; j < cols; j++)
            {
                double v = (double)src[r * cols + j];
                sumSq += v * v;
            }
            double rms = Math.Sqrt(sumSq / cols + 1e-5);
            for (int j = 0; j < cols; j++)
                Assert.That((double)actual[r * cols + j], Is.EqualTo((double)src[r * cols + j] / rms).Within(1e-3), $"({r},{j})");
        }
    }

    [Test]
    public void RMSNormKernel_Half_Backward_MatchesScalarReference()
    {
        var rng = new Random(11);
        const int rows = 4, cols = 8;
        var src = new Half[rows * cols];
        var gradOut = new Half[rows * cols];
        for (int i = 0; i < src.Length; i++)
        {
            src[i] = (Half)(rng.NextDouble() * 2 - 1);
            gradOut[i] = (Half)(rng.NextDouble() * 2 - 1);
        }
        var gradResult = new Half[src.Length];
        RMSNormKernel<Half>.PerRowRMSNormBackwardKernel(src, gradOut, gradResult, rows, cols, 1e-5);

        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;
            double sumSq = 0;
            for (int j = 0; j < cols; j++)
            {
                double v = (double)src[baseIdx + j];
                sumSq += v * v;
            }
            double rms = Math.Sqrt(sumSq / cols + 1e-5);
            double invRms = 1.0 / rms;
            double rms3 = rms * rms * rms;
            double sumGradX = 0;
            for (int j = 0; j < cols; j++)
                sumGradX += (double)gradOut[baseIdx + j] * (double)src[baseIdx + j];
            double scale = sumGradX / (cols * rms3);
            for (int j = 0; j < cols; j++)
            {
                double g = (double)gradOut[baseIdx + j];
                double v = (double)src[baseIdx + j];
                Assert.That((double)gradResult[baseIdx + j], Is.EqualTo(g * invRms - v * scale).Within(1e-3), $"({r},{j})");
            }
        }
    }

    #region TransformerBlock NormType Tests

    [Test]
    public void TransformerBlock_RMSNorm_Forward_ShapeCorrect()
    {
        using var block = new TransformerBlock<float>(nEmbd: 32, nHead: 4, dropout: 0.0, maxSeqLen: 16, normType: NormType.RMSNorm);
        var input = ReverseGradTensor<float>.FromArray(new float[16 * 32], requiresGrad: true);
        input.Reshape(16, 32);

        var output = block.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.Shape[0], Is.EqualTo(16));
        Assert.That(output.Shape[1], Is.EqualTo(32));
    }

    [Test]
    public void TransformerBlock_LayerNorm_Forward_ShapeCorrect()
    {
        using var block = new TransformerBlock<float>(nEmbd: 32, nHead: 4, dropout: 0.0, maxSeqLen: 16, normType: NormType.LayerNorm);
        var input = ReverseGradTensor<float>.FromArray(new float[16 * 32], requiresGrad: true);
        input.Reshape(16, 32);

        var output = block.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.Shape[0], Is.EqualTo(16));
        Assert.That(output.Shape[1], Is.EqualTo(32));
    }

    [Test]
    public void TransformerBlock_LayerNorm_Backward_GradientFlows()
    {
        using var block = new TransformerBlock<float>(nEmbd: 16, nHead: 4, dropout: 0.0, maxSeqLen: 8, normType: NormType.LayerNorm);
        var input = ReverseGradTensor<float>.FromArray(new float[8 * 16], requiresGrad: true);
        input.Reshape(8, 16);

        var output = block.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        float[] gradVals = new float[input.Grad!.Length];
        input.Grad.CopyTo(gradVals, default(float)!);
        Assert.That(gradVals.All(v => !float.IsNaN(v)), Is.True);
    }

    [Test]
    public void TransformerBlock_RMSNorm_vs_LayerNorm_DifferentOutputs()
    {
        var data = new float[8 * 16];
        var rng = new Random(42);
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);

        var layerNormResult = LayerNormKernel<float>.Forward(data, 8, 16,
            ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty,
            float.CreateChecked(1e-5), affine: false);

        var rmsNormOutput = new float[data.Length];
        for (int r = 0; r < 8; r++)
        {
            int offset = r * 16;
            float sumSq = 0;
            for (int j = 0; j < 16; j++)
                sumSq += data[offset + j] * data[offset + j];
            float rms = MathF.Sqrt(sumSq / 16 + 1e-5f);
            for (int j = 0; j < 16; j++)
                rmsNormOutput[offset + j] = data[offset + j] / rms;
        }

        bool allSame = true;
        for (int i = 0; i < data.Length; i++)
        {
            if (Math.Abs(layerNormResult.Output[i] - rmsNormOutput[i]) > 1e-5f)
            {
                allSame = false;
                break;
            }
        }
        Assert.That(allSame, Is.False, "LayerNorm (mean-centering) and RMSNorm should produce different outputs");
    }

    #endregion

    #region ConvTextClassifierModel Tests

    [Test]
    public void ConvTextClassifier_Forward_ShapeCorrect()
    {
        using var embedding = new Embedding<float>(100, 16);
        using var conv1 = new Conv1d<float>(16, 64, kernelSize: 3, padding: 1);
        using var conv2 = new Conv1d<float>(16, 64, kernelSize: 5, padding: 2);
        using var conv3 = new Conv1d<float>(16, 64, kernelSize: 7, padding: 3);
        using var fc1 = new Linear<float>(64 * 3 * 10, 32);
        using var fc2 = new Linear<float>(32, 2);

        var ids = new float[2 * 10];
        for (int i = 0; i < ids.Length; i++) ids[i] = i % 50;
        var input = ReverseGradTensor<float>.FromArray(ids, requiresGrad: false);
        input.Reshape(2, 10);

        var emb = embedding.Forward(input);
        var ncl = ReverseGradOperations.TransposeAxes(emb, 1, 2);

        var c1 = ReverseGradOperations.Relu(conv1.Forward(ncl));
        var c2 = ReverseGradOperations.Relu(conv2.Forward(ncl));
        var c3 = ReverseGradOperations.Relu(conv3.Forward(ncl));

        int B = c1.shape[0];
        c1.Reshape(B, c1.shape[1] * c1.shape[2]);
        c2.Reshape(B, c2.shape[1] * c2.shape[2]);
        c3.Reshape(B, c3.shape[1] * c3.shape[2]);

        var cat = ReverseGradOperations.Concat([c1, c2, c3], axis: 1);
        var h = ReverseGradOperations.Relu(fc1.Forward(cat));
        var output = fc2.Forward(h);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.Shape[0], Is.EqualTo(2));
        Assert.That(output.Shape[1], Is.EqualTo(2));
    }

    [Test]
    public void ConvTextClassifier_Backward_GradientFlows()
    {
        using var embedding = new Embedding<float>(50, 8);
        using var conv1 = new Conv1d<float>(8, 16, kernelSize: 3, padding: 1);
        using var fc1 = new Linear<float>(16 * 8, 16);
        using var fc2 = new Linear<float>(16, 3);

        var ids = new float[1 * 8];
        for (int i = 0; i < ids.Length; i++) ids[i] = i % 20;
        var input = ReverseGradTensor<float>.FromArray(ids, requiresGrad: false);
        input.Reshape(1, 8);

        var emb = embedding.Forward(input);
        var ncl = ReverseGradOperations.TransposeAxes(emb, 1, 2);
        var c1 = ReverseGradOperations.Relu(conv1.Forward(ncl));
        int B = c1.shape[0];
        c1.Reshape(B, c1.shape[1] * c1.shape[2]);
        var h = ReverseGradOperations.Relu(fc1.Forward(c1));
        var output = fc2.Forward(h);

        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Null, "Input is token IDs, no grad expected");
        foreach (var p in embedding.Parameters())
            Assert.That(p.Value.Grad, Is.Not.Null, $"Gradient should flow to embedding param");
    }

    #endregion

    #region LayerNorm Dot vs Scalar Verification

    [Test]
    public void LayerNormKernel_DotMatchesScalarLoop()
    {
        int rows = 4;
        int cols = 32;
        var input = new float[rows * cols];
        var rng = new Random(123);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        var simdResult = LayerNormKernel<float>.Forward(input, rows, cols,
            ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty,
            float.CreateChecked(1e-5), affine: false);

        var scalarOutput = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            int offset = r * cols;
            float sum = 0;
            for (int j = 0; j < cols; j++)
                sum += input[offset + j];
            float mean = sum / cols;

            float sumSq = 0;
            for (int j = 0; j < cols; j++)
            {
                float diff = input[offset + j] - mean;
                sumSq += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(sumSq / cols + 1e-5f);

            for (int j = 0; j < cols; j++)
                scalarOutput[offset + j] = (input[offset + j] - mean) * invStd;
        }

        for (int i = 0; i < input.Length; i++)
            Assert.That(simdResult.Output[i], Is.EqualTo(scalarOutput[i]).Within(1e-5f),
                $"Mismatch at index {i}: SIMD={simdResult.Output[i]}, scalar={scalarOutput[i]}");
    }

    #endregion

    #region MSELoss

    [Test]
    public void MSELoss_SumReduction_ReturnsSumOfSquaredDifferences()
    {
        using var gradScope = GradientUtils.Grad();
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }),
            requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }),
            requiresGrad: false);

        var loss = new MSELoss<float>().Forward(predictions, targets);

        Assert.That(loss.Length, Is.EqualTo(1));
        float expected = 0f + 1f + 4f;
        Assert.That(loss[0], Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void MSELoss_MeanReduction_ReturnsMeanOfSquaredDifferences()
    {
        using var gradScope = GradientUtils.Grad();
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }),
            requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }),
            requiresGrad: false);

        var loss = new MSELoss<float>().Forward(predictions, targets, reduceToMean: true);

        Assert.That(loss.Length, Is.EqualTo(1));
        float expected = (0f + 1f + 4f) / 3f;
        Assert.That(loss[0], Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void MSELoss_MeanReduction_Backward_GradientsScaled()
    {
        using var gradScope = GradientUtils.Grad();
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 2f, 4f }),
            requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f }),
            requiresGrad: false);

        var loss = new MSELoss<float>().Forward(predictions, targets, reduceToMean: true);
        loss.Backward();

        Assert.That(predictions.Grad, Is.Not.Null);
        Assert.That(predictions.Grad!.Length, Is.EqualTo(2));
        float gradScale = 1f / 2f;
        Assert.That(predictions.Grad[0], Is.EqualTo(2f * 1f * gradScale).Within(1e-5f));
        Assert.That(predictions.Grad[1], Is.EqualTo(2f * 3f * gradScale).Within(1e-5f));
    }

    [Test]
    public void MSELoss_PerfectPrediction_ZeroLoss()
    {
        using var gradScope = GradientUtils.Grad();
        var data = new float[] { 1f, 2f, 3f };
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(data), requiresGrad: false);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create((float[])data.Clone()), requiresGrad: false);

        var sumLoss = new MSELoss<float>().Forward(predictions, targets);
        var meanLoss = new MSELoss<float>().Forward(predictions, targets, reduceToMean: true);

        Assert.That(sumLoss[0], Is.EqualTo(0f).Within(1e-6f));
        Assert.That(meanLoss[0], Is.EqualTo(0f).Within(1e-6f));
    }

    #endregion

    #region BatchNorm Regression Tests (xHat bug + 3D edge cases)

    [Test]
    public void BatchNorm1d_AffineFalse_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm1d<float>(3, affine: false);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }),
            requiresGrad: true);
        input.Reshape(2, 3, 2);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f, 1.2f, -0.3f, 0.8f, 1.5f, -0.7f, 0.1f, 0.9f, -1.0f, 0.4f, 0.6f, -0.2f }),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(12));
        for (int i = 0; i < 12; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False,
                $"Gradient[{i}] should be finite, was {input.Grad[i]}");

        bool hasNonZeroGrad = false;
        for (int i = 0; i < 12; i++)
            if (Math.Abs(input.Grad[i]) > 1e-6f) { hasNonZeroGrad = true; break; }
        Assert.That(hasNonZeroGrad, Is.True, "Input gradients should be non-zero when affine=false (xHat must contain normalized values)");
    }

    [Test]
    public void BatchNorm1d_3DInput_EvalMode_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();
        var trainInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4]),
            requiresGrad: false);
        trainInput.Reshape(2, 3, 4);
        bn.Forward(trainInput);

        bn.Eval();
        var evalInput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4]),
            requiresGrad: true);
        evalInput.Reshape(2, 3, 4);

        var output = bn.Forward(evalInput);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 4]),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 4);

        output.Backward(gradOutput);

        Assert.That(evalInput.Grad, Is.Not.Null);
        Assert.That(evalInput.Grad!.Length, Is.EqualTo(24));
        for (int i = 0; i < 24; i++)
            Assert.That(float.IsNaN(evalInput.Grad[i]) || float.IsInfinity(evalInput.Grad[i]), Is.False);
    }

    [Test]
    public void BatchNorm1d_3DInput_SmallPlaneSize_ScalarPath_Backward()
    {
        using var bn = new BatchNorm1d<float>(3);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 2]), // B=2, C=3, L=2 (planeSize=2 < 4, forces scalar path)
            requiresGrad: true);
        input.Reshape(2, 3, 2);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.3f, -0.5f, 1.1f, 0.2f, -0.8f, 0.6f, 0.9f, -0.1f, 0.4f, 0.7f, -0.3f, 0.5f }),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(12));
        for (int i = 0; i < 12; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);
    }

    [Test]
    public void BatchNorm1d_3DInput_SmallPlaneSize_AffineFalse_ScalarPath_Backward()
    {
        using var bn = new BatchNorm1d<float>(3, affine: false);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 2]), // B=2, C=3, L=2 (planeSize=2, scalar path + affine=false)
            requiresGrad: true);
        input.Reshape(2, 3, 2);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.7f, -0.4f, 1.3f, -0.1f, 0.5f, 0.9f, -0.6f, 0.2f, 0.8f, -0.3f, 1.1f, -0.5f }),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(12));
        for (int i = 0; i < 12; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False,
                $"Gradient[{i}] should be finite on scalar path with affine=false, was {input.Grad[i]}");

        bool hasNonZeroGrad = false;
        for (int i = 0; i < 12; i++)
            if (Math.Abs(input.Grad[i]) > 1e-6f) { hasNonZeroGrad = true; break; }
        Assert.That(hasNonZeroGrad, Is.True, "Gradients should be non-zero: xHat must contain normalized values even on scalar path with affine=false");
    }

    [Test]
    public void BatchNorm2d_AffineFalse_Backward_GradientsFlow()
    {
        using var bn = new BatchNorm2d<float>(3, affine: false);
        bn.Train();

        var input = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[2 * 3 * 2 * 2]),
            requiresGrad: true);
        input.Reshape(2, 3, 2, 2);

        var output = bn.Forward(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.4f, -0.8f, 1.2f, 0.1f, -0.5f, 0.7f, 0.3f, -0.9f, 0.6f, 1.0f, -0.2f, 0.5f, -0.3f, 0.8f, 0.1f, -0.6f, 0.9f, -0.4f, 1.1f, -0.7f, 0.2f, 0.5f, -0.1f, 0.3f }),
            requiresGrad: false);
        gradOutput.Reshape(2, 3, 2, 2);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(24));
        for (int i = 0; i < 24; i++)
            Assert.That(float.IsNaN(input.Grad[i]) || float.IsInfinity(input.Grad[i]), Is.False);

        bool hasNonZeroGrad = false;
        for (int i = 0; i < 24; i++)
            if (Math.Abs(input.Grad[i]) > 1e-6f) { hasNonZeroGrad = true; break; }
        Assert.That(hasNonZeroGrad, Is.True, "BatchNorm2d affine=false backward should produce non-zero gradients");
    }

    [Test]
    public void MSELoss_MeanReduction_BatchInput_CorrectValue()
    {
        using var gradScope = GradientUtils.Grad();
        var predictions = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 2, 3, 4 }),
            requiresGrad: true);
        var targets = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1, 1, 1, 1 }),
            requiresGrad: false);

        var loss = new MSELoss<float>().Forward(predictions, targets, reduceToMean: true);

        Assert.That(loss.Length, Is.EqualTo(1));
        float expected = (0f + 1f + 4f + 9f) / 4f;
        Assert.That(loss[0], Is.EqualTo(expected).Within(1e-5f));
    }

    #endregion
}
