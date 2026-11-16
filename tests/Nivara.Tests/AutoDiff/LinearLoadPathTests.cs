using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LinearLoadPathTests
{
    [Test]
    public void Uninitialized_WeightIsNull_AndNotRegistered()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        Assert.That(linear.Weight, Is.Null);
        Assert.That(linear.GetParameters().ContainsKey("Weight"), Is.False);
    }

    [Test]
    public void Constructor_Default_InitializesWeight()
    {
        var linear = new Linear<float>(4, 3);
        Assert.That(linear.Weight, Is.Not.Null);
    }

    [Test]
    public void Forward_BeforeBinding_Throws()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        var input = ReverseGradTensor<float>.FromArray(new float[4 * 4]);
        input.Reshape(4, 4);
        var ex = Assert.Throws<InvalidOperationException>(() => linear.Forward(input));
        Assert.That(ex!.Message, Does.Contain("not been loaded"));
    }

    [Test]
    public void StateDict_BeforeBinding_Throws()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        Assert.Throws<InvalidOperationException>(() => linear.StateDict());
    }

    [Test]
    public void ExportToOnnx_BeforeBinding_Throws()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        Assert.Throws<InvalidOperationException>(() =>
            linear.ExportToOnnx(null!, "x"));
    }

    [Test]
    public void BindWeight_IsZeroCopy()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        var data = new float[4 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var tensor = ReverseGradTensor<float>.FromMatrix(data, 3, 4);

        linear.BindWeight(tensor);

        Assert.That(linear.Weight, Is.Not.Null);
        Assert.That(linear.Weight!.Tensor.Data.TryGetSpan(out var span), Is.True);
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(span), ref MemoryMarshal.GetReference(data.AsSpan())), Is.True);
    }

    [Test]
    public void LoadStateDict_OnUninitialized_MixedDtype_DefaultsRequiresGradTrue()
    {
        var linear = Linear<float, Half>.Uninitialized(4, 3);
        var source = ReverseGradTensor<float>.FromMatrix(new float[4 * 3], 3, 4); // requiresGrad defaults to false
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = source,
        });
        Assert.That(linear.Weight, Is.Not.Null);
        Assert.That(linear.Weight!.Tensor.RequiresGrad, Is.True);
    }

    [Test]
    public void BindWeight_ShapeMismatch_Throws()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        var tensor = ReverseGradTensor<float>.FromMatrix(new float[2 * 2], 2, 2);
        Assert.Throws<ArgumentException>(() => linear.BindWeight(tensor));
    }

    [Test]
    public void LoadStateDict_OnUninitialized_MaterializesWeight_SameDtype()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        var source = ReverseGradTensor<float>.FromMatrix(Enumerable.Range(1, 12).Select(i => (float)i).ToArray(), 3, 4);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = source,
        });
        Assert.That(linear.Weight, Is.Not.Null);
        Assert.That(linear.Weight!.Tensor.Shape, Is.EqualTo(new[] { 3, 4 }));
        Assert.That(linear.Weight!.Tensor.Data.ToArray(), Is.EqualTo(source.Data.ToArray()));
    }

    [Test]
    public void LoadStateDict_OnUninitialized_MaterializesWeight_MixedDtype()
    {
        var linear = Linear<float, Half>.Uninitialized(4, 3);
        var source = ReverseGradTensor<float>.FromMatrix(Enumerable.Range(1, 12).Select(i => (float)i).ToArray(), 3, 4);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = source,
        });
        Assert.That(linear.Weight, Is.Not.Null);
        Assert.That(linear.Weight!.Tensor.Shape, Is.EqualTo(new[] { 3, 4 }));
    }

    [Test]
    public void LoadStateDict_ShapeMismatch_Throws()
    {
        var linear = Linear<float>.Uninitialized(4, 3);
        var source = ReverseGradTensor<float>.FromMatrix(new float[2 * 2], 2, 2);
        Assert.Throws<InvalidOperationException>(() =>
            linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
            {
                ["Weight"] = source,
            }));
    }
}
