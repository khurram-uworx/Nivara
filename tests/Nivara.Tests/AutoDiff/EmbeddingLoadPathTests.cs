using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NUnit.Framework;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class EmbeddingLoadPathTests
{
    [Test]
    public void Uninitialized_WeightIsNull_AndNotRegistered()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        Assert.That(emb.Weight, Is.Null);
        Assert.That(emb.GetParameters().ContainsKey("Weight"), Is.False);
    }

    [Test]
    public void Constructor_Default_InitializesWeight()
    {
        var emb = new Embedding<float>(5, 4);
        Assert.That(emb.Weight, Is.Not.Null);
    }

    [Test]
    public void Forward_BeforeBinding_Throws()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        var input = ReverseGradTensor<float>.FromArray(new float[] { 0, 1, 2 });
        var ex = Assert.Throws<InvalidOperationException>(() => emb.Forward(input));
        Assert.That(ex!.Message, Does.Contain("not loaded"));
    }

    [Test]
    public void Forward_IntArray_BeforeBinding_Throws()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        Assert.Throws<InvalidOperationException>(() => emb.Forward(new[] { 0, 1 }));
    }

    [Test]
    public void GatherRows_BeforeBinding_Throws()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        Assert.Throws<InvalidOperationException>(() => emb.GatherRows(new[] { 0 }, new float[4]));
    }

    [Test]
    public void StateDict_BeforeBinding_Throws()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        Assert.Throws<InvalidOperationException>(() => emb.StateDict());
    }

    [Test]
    public void BindWeight_IsZeroCopy()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        var data = new float[5 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var tensor = ReverseGradTensor<float>.FromMatrix(data, 5, 4);

        emb.BindWeight(tensor);

        Assert.That(emb.Weight, Is.Not.Null);
        Assert.That(emb.Weight!.Tensor.Data.TryGetSpan(out var span), Is.True);
        Assert.That(Unsafe.AreSame(ref MemoryMarshal.GetReference(span), ref MemoryMarshal.GetReference(data.AsSpan())), Is.True);
    }

    [Test]
    public void BindWeight_ShapeMismatch_Throws()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        var tensor = ReverseGradTensor<float>.FromMatrix(new float[2 * 2], 2, 2);
        Assert.Throws<ArgumentException>(() => emb.BindWeight(tensor));
    }

    [Test]
    public void LoadStateDict_OnUninitialized_MaterializesWeight()
    {
        var emb = Embedding<float>.Uninitialized(5, 4);
        var source = ReverseGradTensor<float>.FromMatrix(Enumerable.Range(1, 20).Select(i => (float)i).ToArray(), 5, 4);
        emb.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = source,
        });
        Assert.That(emb.Weight, Is.Not.Null);
        Assert.That(emb.Weight!.Tensor.Shape, Is.EqualTo(new[] { 5, 4 }));
    }

    [Test]
    public void LoadStateDict_MixedDtype_DefaultsRequiresGradTrue()
    {
        var emb = Embedding<float, Half>.Uninitialized(5, 4);
        var source = ReverseGradTensor<float>.FromMatrix(new float[5 * 4], 5, 4); // requiresGrad defaults false
        emb.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = source,
        });
        Assert.That(emb.Weight, Is.Not.Null);
        Assert.That(emb.Weight!.Tensor.RequiresGrad, Is.True);
    }
}