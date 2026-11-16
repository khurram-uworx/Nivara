using Nivara.AI;
using Nivara.AutoDiff.Nn;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Covers the lenient (default) vs strict missing-key behaviour of
/// <see cref="StateDictLoader"/> helpers (#1241).
/// </summary>
[TestFixture]
public class StateDictLoaderTests
{
    static float[] Filled(int n, float start = 0f)
    {
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = start + i;
        return data;
    }

    // ── LoadLinear (single-type) ─────────────────────────────────────────────

    [Test]
    public void LoadLinear_MissingWeight_Lenient_DoesNotThrow()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = new Linear<float>(4, 2);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLinear(linear, tensors, "W"));
    }

    [Test]
    public void LoadLinear_MissingWeight_Strict_ThrowsNamingKey()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = new Linear<float>(4, 2);
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinear(linear, tensors, "W", strict: true));
        Assert.That(ex!.Message, Does.Contain("W.weight"));
    }

    [Test]
    public void LoadLinear_WeightPresentBiasMissing_Strict_ThrowsNamingBias()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(8), [2, 4]),
        };
        var linear = new Linear<float>(4, 2); // has bias
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinear(linear, tensors, "W", strict: true));
        Assert.That(ex!.Message, Does.Contain("W.bias"));
    }

    [Test]
    public void LoadLinear_WeightPresentBiasMissing_Lenient_LoadsWeight()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(8), [2, 4]),
        };
        var linear = new Linear<float>(4, 2);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLinear(linear, tensors, "W"));
    }

    [Test]
    public void LoadLinear_BiasFree_Strict_DoesNotRequireBias()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(8), [2, 4]),
        };
        var linear = new Linear<float>(4, 2, bias: false);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLinear(linear, tensors, "W", strict: true));
    }

    // ── LoadLinear (mixed) ───────────────────────────────────────────────────

    [Test]
    public void LoadLinearMixed_MissingWeight_Strict_ThrowsNamingKey()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = Linear<float, float>.Uninitialized(4, 2);
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinear(linear, tensors, "W", strict: true));
        Assert.That(ex!.Message, Does.Contain("W.weight"));
    }

    [Test]
    public void LoadLinearMixed_MissingWeight_Lenient_DoesNotThrow()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = Linear<float, float>.Uninitialized(4, 2);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLinear(linear, tensors, "W"));
    }

    // ── LoadLinearWidened ────────────────────────────────────────────────────

    [Test]
    public void LoadLinearWidened_MissingWeight_Strict_ThrowsNamingKey()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = new Linear<float>(4, 2);
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinearWidened(linear, tensors, "W", strict: true));
        Assert.That(ex!.Message, Does.Contain("W.weight"));
    }

    [Test]
    public void LoadLinearWidened_MissingWeight_Lenient_DoesNotThrow()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = new Linear<float>(4, 2);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLinearWidened(linear, tensors, "W"));
    }

    [Test]
    public void LoadLinearWidened_WeightPresentBiasMissing_Strict_ThrowsNamingBias()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(8), [2, 4]),
        };
        var linear = new Linear<float>(4, 2);
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinearWidened(linear, tensors, "W", strict: true));
        Assert.That(ex!.Message, Does.Contain("W.bias"));
    }

    // ── LoadLayerNorm ────────────────────────────────────────────────────────

    [Test]
    public void LoadLayerNorm_MissingWeight_Strict_ThrowsNamingKey()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var ln = new LayerNorm<float>(4);
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLayerNorm(ln, tensors, "N", strict: true));
        Assert.That(ex!.Message, Does.Contain("N.weight"));
    }

    [Test]
    public void LoadLayerNorm_MissingWeight_Lenient_DoesNotThrow()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var ln = new LayerNorm<float>(4);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLayerNorm(ln, tensors, "N"));
    }

    [Test]
    public void LoadLayerNorm_WeightPresentBiasMissing_Strict_ThrowsNamingBias()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["N.weight"] = (Filled(4), [4]),
        };
        var ln = new LayerNorm<float>(4); // has bias
        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLayerNorm(ln, tensors, "N", strict: true));
        Assert.That(ex!.Message, Does.Contain("N.bias"));
    }

    [Test]
    public void LoadLayerNorm_BiasFree_Strict_DoesNotRequireBias()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["N.weight"] = (Filled(4), [4]),
        };
        var ln = new LayerNorm<float>(4, bias: false);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLayerNorm(ln, tensors, "N", strict: true));
    }

    [Test]
    public void LoadLayerNorm_NonAffine_Strict_DoesNotRequireAnything()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var ln = new LayerNorm<float>(4, affine: false, bias: false);
        Assert.DoesNotThrow(() => StateDictLoader.LoadLayerNorm(ln, tensors, "N", strict: true));
    }
}