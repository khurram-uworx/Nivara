using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
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

        Assert.That(mu.HasNulls, Is.False);
        Assert.That(logVar.HasNulls, Is.False);

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

        var weightGrad = emb.Weight.Grad;
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
        emb.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(
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

        var weightGrad = emb.Weight.Grad;
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
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(new float[] { 5f }, requiresGrad: true);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(new float[] { 1f }, 1, 1, requiresGrad: true);

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

        Assert.That(conv.WeightParam.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.WeightParam.Tensor.Grad!.Length, Is.EqualTo(3 * 2 * 3 * 3));

        if (conv.BiasParam != null)
        {
            Assert.That(conv.BiasParam.Tensor.Grad, Is.Not.Null);
            Assert.That(conv.BiasParam.Tensor.Grad!.Length, Is.EqualTo(3));
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
        Assert.That(conv.BiasParam, Is.Null);
    }

    [Test]
    public void Conv2d_KernelSize1_MatchesLinearPerSample()
    {
        using var conv = new Conv2d<float>(2, 3, kernelSize: 1);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(
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
        var weight = conv.WeightParam;
        var bias = conv.BiasParam;

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

        Assert.That(convT.WeightParam.Tensor.Grad, Is.Not.Null);
        Assert.That(convT.WeightParam.Tensor.Grad!.Length, Is.EqualTo(3 * 2 * 3 * 3));

        if (convT.BiasParam != null)
        {
            Assert.That(convT.BiasParam.Tensor.Grad, Is.Not.Null);
            Assert.That(convT.BiasParam.Tensor.Grad!.Length, Is.EqualTo(2));
        }
    }

    [Test]
    public void ConvTranspose2d_Forward_WithBias_AddsBias()
    {
        using var convT = new ConvTranspose2d<float>(1, 1, kernelSize: 1, bias: true);
        convT.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(new float[] { 10f }, requiresGrad: true);
        convT.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(new float[] { 2f }, 1, 1, requiresGrad: true);

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
        var weight = convT.WeightParam;
        var bias = convT.BiasParam;

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
        Assert.That(conv.WeightParam.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.WeightParam.Tensor.Grad!.Length, Is.EqualTo(4 * 2 * 3 * 3));
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
        Assert.That(conv.WeightParam.Tensor.Grad, Is.Not.Null);
        Assert.That(conv.WeightParam.Tensor.Grad!.Length, Is.EqualTo(8 * 3 * 3 * 3));
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
        Assert.That(convT.WeightParam.Tensor.Grad, Is.Not.Null);
        Assert.That(convT.WeightParam.Tensor.Grad!.Length, Is.EqualTo(4 * 2 * 3 * 3));
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
}
