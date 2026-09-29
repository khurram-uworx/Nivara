using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Covers <see cref="LayerNorm{T}"/>'s bias-free mode (<c>affine: true, bias: false</c>), the shape
/// modern pre-norm architectures need: a learnable gamma and no beta at all. The whole point is
/// that the bias is <em>absent</em> rather than pinned at zero — an absent parameter is missing
/// from the state dictionary and from an optimizer's list, so nothing can drift off zero and turn
/// the model into a different architecture than the checkpoint defines.
/// </summary>
[TestFixture]
public class LayerNormTests
{
    const int Cols = 8;
    const int Rows = 4;
    const float Eps = 1e-5f;

    static float[] Input() =>
    [
        0.5f, -1.2f, 2.3f, 3.1f, -0.7f, 1.8f, -2.4f, 0.1f,
        1.5f, 2.2f, -3.3f, 0.9f, -1.1f, 0.6f, 2.7f, -0.2f,
        3.5f, -0.8f, 1.2f, -2.1f, 0.4f, 1.9f, -1.6f, 2.8f,
        0.3f, -2.9f, 1.4f, 2.6f, -0.5f, 1.7f, -3.0f, 0.8f,
    ];

    /// <summary>A gamma that is recognisably not all-ones, so "is this weight actually applied" is answerable.</summary>
    static float[] Gamma() => [1.5f, -0.75f, 2.25f, 0.5f, -1.25f, 3f, 0.125f, -2f];

    static float[] GradOutput() =>
    [
        0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
        -0.9f, 1.0f, -1.1f, 1.2f, -1.3f, 1.4f, -1.5f, 1.6f,
        1.7f, -1.8f, 1.9f, -2.0f, 2.1f, -2.2f, 2.3f, -2.4f,
        -2.5f, 2.6f, -2.7f, 2.8f, -2.9f, 3.0f, -3.1f, 3.2f,
    ];

    static void SetGamma(LayerNorm<float> norm, float[] values)
        => norm.Weight!.Tensor = ReverseGradTensor<float>.FromArray(values, requiresGrad: true);

    static float[] Values(ReverseGradTensor<float> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = tensor[i];
        return result;
    }

    static float[] ColumnValues(NivaraColumn<float> column)
    {
        var result = new float[column.Length];
        column.CopyTo(result, 0f);
        return result;
    }

    static ReverseGradTensor<float> Shaped(float[] data, int rows, int cols, bool requiresGrad)
    {
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(rows, cols);
        return tensor;
    }

    //  ── construction ─────────────────────────────────────────────────────────

    [Test]
    public void LayerNorm_BiasFalse_KeepsWeightAndDropsBias()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);

        Assert.Multiple(() =>
        {
            Assert.That(norm.Affine, Is.True, "gamma is still applied, so the affine transform is not off");
            Assert.That(norm.Weight, Is.Not.Null);
            Assert.That(norm.Bias, Is.Null);
            Assert.That(norm.GetParameters().Keys, Is.EqualTo(new[] { "Weight" }));
        });
    }

    [Test]
    public void LayerNorm_BiasFalse_DefaultConstructionIsUnchanged()
    {
        using var norm = new LayerNorm<float>(Cols, Eps);

        Assert.Multiple(() =>
        {
            Assert.That(norm.Weight, Is.Not.Null);
            Assert.That(norm.Bias, Is.Not.Null);
            Assert.That(norm.GetParameters().Keys, Is.EquivalentTo(new[] { "Weight", "Bias" }));
        });
    }

    [Test]
    public void LayerNorm_AffineFalseWithBias_Throws()
    {
        // "No affine parameters" and "do have a beta" cannot both hold. Failing loudly beats
        // silently dropping the beta, which is the exact trap #446 exists to close. bias defaults
        // to true, so plain `affine: false` is the same contradiction and throws too.
        var ex = Assert.Throws<ArgumentException>(
            () => new LayerNorm<float>(Cols, Eps, affine: false, bias: true));
        Assert.That(ex!.Message, Does.Contain("affine"));

        Assert.Throws<ArgumentException>(() => new LayerNorm<float>(Cols, Eps, affine: false));
    }

    [Test]
    public void LayerNorm_AffineFalse_StillOmitsBothParameters()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, affine: false, bias: false);

        Assert.Multiple(() =>
        {
            Assert.That(norm.Affine, Is.False);
            Assert.That(norm.Weight, Is.Null);
            Assert.That(norm.Bias, Is.Null);
            Assert.That(norm.GetParameters(), Is.Empty);
        });
    }

    //  ── state dictionary round-trip (#446 acceptance) ─────────────────────────

    [Test]
    public void LayerNorm_BiasFalse_StateDictEmitsWeightOnly()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);

        Assert.That(norm.StateDict().Keys, Is.EqualTo(new[] { "Weight" }));
    }

    [Test]
    public void LayerNorm_BiasFalse_LoadStateDictWithoutBiasSucceedsWhenStrict()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        var checkpoint = new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(Gamma()),
        };

        Assert.DoesNotThrow(() => norm.LoadStateDict(checkpoint, strict: true));

        Assert.That(Values(norm.Weight!.Tensor), Is.EqualTo(Gamma()));
    }

    [Test]
    public void LayerNorm_BiasFalse_StateDictRoundTripsThroughLoadStateDict()
    {
        using var source = new LayerNorm<float>(Cols, Eps, bias: false);
        SetGamma(source, Gamma());

        using var target = new LayerNorm<float>(Cols, Eps, bias: false);
        target.LoadStateDict(source.StateDict(), strict: true);

        Assert.That(target.StateDict().Keys, Is.EqualTo(new[] { "Weight" }));
        Assert.That(Values(target.Weight!.Tensor), Is.EqualTo(Gamma()));
    }

    [Test]
    public void LayerNorm_BiasFalse_LoadStateDictWithBias_ThrowsNamingTheParameter()
    {
        // A checkpoint that does carry a beta must not be loaded into a bias-free norm silently.
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        var biased = new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(Gamma()),
            ["Bias"] = ReverseGradTensor<float>.FromArray(new float[Cols]),
        };

        var ex = Assert.Throws<InvalidOperationException>(() => norm.LoadStateDict(biased));
        Assert.That(ex!.Message, Does.Contain("Bias"));
    }

    [Test]
    public void LayerNorm_BiasDefault_LoadStateDictMissingBias_ThrowsWhenStrict()
    {
        // The converse direction stays loud too: a biased norm told to load a bias-free
        // checkpoint under strict mode must report the missing Bias rather than leave it at zero.
        using var norm = new LayerNorm<float>(Cols, Eps);
        var weightOnly = new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(Gamma()),
        };

        var ex = Assert.Throws<InvalidOperationException>(
            () => norm.LoadStateDict(weightOnly, strict: true));
        Assert.That(ex!.Message, Does.Contain("Bias"));
    }

    //  ── forward equivalence with the zero-Beta emulation ─────────────────────

    [Test]
    public void LayerNorm_BiasFalse_ForwardMatchesZeroBetaEmulation()
    {
        // The emulation this mode replaces: a normal LayerNorm whose Bias is never loaded and so
        // keeps its zero initialization. The bias-free path performs strictly fewer operations,
        // so the only possible divergence is signed zero — IEEE-754 gives -0.0 + +0.0 == +0.0,
        // and .NET's == treats -0.0f == 0.0f, so exact equality is the right assertion here.
        using var biasFree = new LayerNorm<float>(Cols, Eps, bias: false);
        using var emulated = new LayerNorm<float>(Cols, Eps);
        SetGamma(biasFree, Gamma());
        SetGamma(emulated, Gamma());

        using (GradientUtils.Grad())
        {
            var input = Input();

            var actual = Values(biasFree.Forward(Shaped(input, Rows, Cols, requiresGrad: true)));
            var expected = Values(emulated.Forward(Shaped(input, Rows, Cols, requiresGrad: true)));

            Assert.That(actual, Is.EqualTo(expected));
        }
    }

    [Test]
    public void LayerNorm_BiasFalse_ForwardAppliesGammaButNotBeta()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        SetGamma(norm, Gamma());

        var output = Values(norm.Forward(Shaped(Input(), Rows, Cols, requiresGrad: false)));

        // Zero mean and unit variance survive the gamma multiply per row only when the output is
        // re-centered by gamma; assert the operationally meaningful thing instead: the output is
        // the gamma-scaled normalization, so a gamma of all ones would give a zero-mean row.
        using var unit = new LayerNorm<float>(Cols, Eps, bias: false);
        var unitOutput = Values(unit.Forward(Shaped(Input(), Rows, Cols, requiresGrad: false)));

        for (int i = 0; i < unitOutput.Length; i++)
            Assert.That(output[i], Is.EqualTo(unitOutput[i] * Gamma()[i % Cols]));
    }

    [Test]
    public void LayerNorm_BiasFalse_OutsideGrad_MatchesTrainingForward()
    {
        // The inference fast path is a separate branch of the kernel.
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        SetGamma(norm, Gamma());
        var data = Input();

        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(norm.Forward(Shaped(data, Rows, Cols, requiresGrad: true)));
        }

        var actual = norm.Forward(Shaped(data, Rows, Cols, requiresGrad: false));

        Assert.That(actual.IsLeaf, Is.True, "the fast path must not build a graph node");
        Assert.That(Values(actual), Is.EqualTo(expected));
    }

    //  ── backward (#446 acceptance) ───────────────────────────────────────────

    [Test]
    public void LayerNorm_BiasFalse_BackwardGradientsWeightAndRegistersNoBias()
    {
        using var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        SetGamma(norm, Gamma());
        var input = Shaped(Input(), Rows, Cols, requiresGrad: true);

        using (GradientUtils.Grad())
        {
            var output = norm.Forward(input);
            output.Backward(Shaped(GradOutput(), Rows, Cols, requiresGrad: false));
        }

        Assert.Multiple(() =>
        {
            Assert.That(input.Grad, Is.Not.Null);
            Assert.That(input.Grad!.Length, Is.EqualTo(Rows * Cols));
            Assert.That(norm.Weight!.Tensor.Grad, Is.Not.Null);
            Assert.That(norm.Weight!.Tensor.Grad!.Length, Is.EqualTo(Cols));
            Assert.That(norm.Bias, Is.Null, "there is no beta parameter to receive a gradient");
            Assert.That(norm.GetParameters().Keys, Is.EqualTo(new[] { "Weight" }),
                "an optimizer built from GetParameters() therefore cannot update a beta");
        });
    }

    [Test]
    public void LayerNorm_BiasFalse_BackwardInputMatchesZeroBetaEmulation()
    {
        // gradInput does not depend on beta at all — beta's gradient is the row sum of gradOutput
        // — so the input gradient must be identical, not merely close.
        using var biasFree = new LayerNorm<float>(Cols, Eps, bias: false);
        using var emulated = new LayerNorm<float>(Cols, Eps);
        SetGamma(biasFree, Gamma());
        SetGamma(emulated, Gamma());

        var freeInput = Shaped(Input(), Rows, Cols, requiresGrad: true);
        var emulatedInput = Shaped(Input(), Rows, Cols, requiresGrad: true);

        using (GradientUtils.Grad())
        {
            biasFree.Forward(freeInput).Backward(Shaped(GradOutput(), Rows, Cols, requiresGrad: false));
            emulated.Forward(emulatedInput).Backward(Shaped(GradOutput(), Rows, Cols, requiresGrad: false));
        }

        Assert.That(ColumnValues(freeInput.Grad!), Is.EqualTo(ColumnValues(emulatedInput.Grad!)));
    }

    [Test]
    public void LayerNorm_BiasFalse_WeightGradientMatchesEmulatedWeightGradient()
    {
        // dL/dgamma = sum_rows(gradOutput * xHat), and xHat is identical in both modes, so the
        // gamma gradient must match exactly too.
        using var biasFree = new LayerNorm<float>(Cols, Eps, bias: false);
        using var emulated = new LayerNorm<float>(Cols, Eps);
        SetGamma(biasFree, Gamma());
        SetGamma(emulated, Gamma());

        using (GradientUtils.Grad())
        {
            biasFree.Forward(Shaped(Input(), Rows, Cols, requiresGrad: true))
                .Backward(Shaped(GradOutput(), Rows, Cols, requiresGrad: false));
            emulated.Forward(Shaped(Input(), Rows, Cols, requiresGrad: true))
                .Backward(Shaped(GradOutput(), Rows, Cols, requiresGrad: false));
        }

        Assert.That(
            ColumnValues(biasFree.Weight!.Tensor.Grad!),
            Is.EqualTo(ColumnValues(emulated.Weight!.Tensor.Grad!)));
    }

    //  ── lifetime ─────────────────────────────────────────────────────────────

    [Test]
    public void LayerNorm_BiasFalse_DisposeReleasesTheOnlyParameter()
    {
        var norm = new LayerNorm<float>(Cols, Eps, bias: false);
        var weight = norm.Weight;

        norm.Dispose();

        Assert.Multiple(() =>
        {
            Assert.That(weight, Is.Not.Null);
            Assert.That(() => weight!.Tensor, Throws.InstanceOf<ObjectDisposedException>(),
                "the registered weight is disposed with the module");
            Assert.That(norm.Bias, Is.Null);
        });
    }
}
