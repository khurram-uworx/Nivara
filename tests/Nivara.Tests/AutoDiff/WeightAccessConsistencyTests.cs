using System.Reflection;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class WeightAccessConsistencyTests
{
    sealed class ModuleSpec(string name, Type type, Func<Module<float>> createDefault, Func<Module<float>> createOptionalOff)
    {
        public string Name { get; } = name;
        public Type Type { get; } = type;
        public Func<Module<float>> CreateDefault { get; } = createDefault;
        public Func<Module<float>> CreateOptionalOff { get; } = createOptionalOff;

        public bool HasOptionalParameters =>
            Name is not nameof(Embedding<float>) and not nameof(SparseEmbedding<float>);
    }

    static readonly ModuleSpec[] Modules =
    [
        new(nameof(Linear<float>), typeof(Linear<float>),
            () => new Linear<float>(4, 8),
            () => new Linear<float>(4, 8, bias: false)),
        new(nameof(Conv1d<float>), typeof(Conv1d<float>),
            () => new Conv1d<float>(4, 8, 3),
            () => new Conv1d<float>(4, 8, 3, bias: false)),
        new(nameof(Conv2d<float>), typeof(Conv2d<float>),
            () => new Conv2d<float>(4, 8, 3),
            () => new Conv2d<float>(4, 8, 3, bias: false)),
        new(nameof(ConvTranspose2d<float>), typeof(ConvTranspose2d<float>),
            () => new ConvTranspose2d<float>(4, 8, 3),
            () => new ConvTranspose2d<float>(4, 8, 3, bias: false)),
        new(nameof(BatchNorm1d<float>), typeof(BatchNorm1d<float>),
            () => new BatchNorm1d<float>(8),
            () => new BatchNorm1d<float>(8, affine: false)),
        new(nameof(BatchNorm2d<float>), typeof(BatchNorm2d<float>),
            () => new BatchNorm2d<float>(8),
            () => new BatchNorm2d<float>(8, affine: false)),
        new(nameof(LayerNorm<float>), typeof(LayerNorm<float>),
            () => new LayerNorm<float>(8),
            () => new LayerNorm<float>(8, affine: false)),
        new(nameof(Embedding<float>), typeof(Embedding<float>),
            () => new Embedding<float>(10, 8),
            () => new Embedding<float>(10, 8)),
        new(nameof(SparseEmbedding<float>), typeof(SparseEmbedding<float>),
            () => new SparseEmbedding<float>(10, 8),
            () => new SparseEmbedding<float>(10, 8)),
    ];

    static Parameter<float>? GetAccessor(Module<float> module, string name)
        => (Parameter<float>?)module.GetType().GetProperty(name)!.GetValue(module);

    [Test]
    public void WeightAccess_AllModules_ExposeParameterAccessorsOnly()
    {
        foreach (var spec in Modules)
        {
            var weightProp = spec.Type.GetProperty("Weight");
            Assert.That(weightProp, Is.Not.Null, $"{spec.Name} must expose Weight");
            Assert.That(weightProp!.PropertyType, Is.EqualTo(typeof(Parameter<float>)),
                $"{spec.Name}.Weight must be Parameter<float>?");
            Assert.That(new NullabilityInfoContext().Create(weightProp).ReadState, Is.EqualTo(NullabilityState.Nullable),
                $"{spec.Name}.Weight must be declared Parameter<float>?");

            Assert.That(spec.Type.GetProperty("WeightParam"), Is.Null,
                $"{spec.Name} must not expose a legacy WeightParam accessor");
            Assert.That(spec.Type.GetProperty("BiasParam"), Is.Null,
                $"{spec.Name} must not expose a legacy BiasParam accessor");

            if (spec.HasOptionalParameters)
            {
                var biasProp = spec.Type.GetProperty("Bias");
                Assert.That(biasProp, Is.Not.Null, $"{spec.Name} must expose Bias");
                Assert.That(biasProp!.PropertyType, Is.EqualTo(typeof(Parameter<float>)),
                    $"{spec.Name}.Bias must be Parameter<float>?");
                Assert.That(new NullabilityInfoContext().Create(biasProp).ReadState, Is.EqualTo(NullabilityState.Nullable),
                    $"{spec.Name}.Bias must be declared Parameter<float>?");
            }
            else
            {
                Assert.That(spec.Type.GetProperty("Bias"), Is.Null,
                    $"{spec.Name} must not expose Bias");
            }
        }
    }

    [Test]
    public void WeightAccess_DefaultConstruction_ExposesRegisteredParameters()
    {
        foreach (var spec in Modules)
        {
            using var module = spec.CreateDefault();

            var weight = GetAccessor(module, "Weight");
            Assert.That(weight, Is.Not.Null, $"{spec.Name} default construction must have Weight");
            Assert.That(module.GetParameters()["Weight"].Tensor, Is.SameAs(weight!.Tensor),
                $"{spec.Name}.Weight accessor must expose the registered tensor");

            if (spec.HasOptionalParameters)
            {
                var bias = GetAccessor(module, "Bias");
                Assert.That(bias, Is.Not.Null, $"{spec.Name} default construction must have Bias");
                Assert.That(module.GetParameters()["Bias"].Tensor, Is.SameAs(bias!.Tensor),
                    $"{spec.Name}.Bias accessor must expose the registered tensor");
            }
        }
    }

    [Test]
    public void WeightAccess_OptionalParameterDisabled_ExposesNull()
    {
        foreach (var spec in Modules)
        {
            using var module = spec.CreateOptionalOff();

            if (spec.Name is nameof(BatchNorm1d<float>) or nameof(BatchNorm2d<float>) or nameof(LayerNorm<float>))
            {
                Assert.That(GetAccessor(module, "Weight"), Is.Null,
                    $"{spec.Name} with affine:false must expose null Weight");
                Assert.That(GetAccessor(module, "Bias"), Is.Null,
                    $"{spec.Name} with affine:false must expose null Bias");
            }
            else if (spec.HasOptionalParameters)
            {
                Assert.That(GetAccessor(module, "Weight"), Is.Not.Null,
                    $"{spec.Name} must always have Weight");
                Assert.That(GetAccessor(module, "Bias"), Is.Null,
                    $"{spec.Name} with bias:false must expose null Bias");
            }
        }
    }
}
