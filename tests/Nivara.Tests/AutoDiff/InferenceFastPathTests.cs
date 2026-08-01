using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Verifies that the inference-only fast paths (no saved-state allocations,
/// no input snapshots) produce numerically identical output to the training
/// forward path. Inference inputs carry requiresGrad: false, so the fast paths
/// must not build graph nodes and must still return correct values.
/// </summary>
[TestFixture]
public class InferenceFastPathTests
{
    static float[] Values(ReverseGradTensor<float> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = tensor[i];
        return result;
    }

    [Test]
    public void LayerNorm_OutsideGrad_MatchesTrainingForward()
    {
        using var ln = new LayerNorm<float>(8);
        var data = new float[] { 0.5f, -1.2f, 2.3f, 3.1f, -0.7f, 1.8f, -2.4f, 0.1f,
                                 1.5f, 2.2f, -3.3f, 0.9f, -1.1f, 0.6f, 2.7f, -0.2f,
                                 3.5f, -0.8f, 1.2f, -2.1f, 0.4f, 1.9f, -1.6f, 2.8f,
                                 0.3f, -2.9f, 1.4f, 2.6f, -0.5f, 1.7f, -3.0f, 0.8f };

        var trainInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: true);
        trainInput.Reshape(4, 8);

        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(ln.Forward(trainInput));
        }

        var inferenceInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        inferenceInput.Reshape(4, 8);
        var actual = ln.Forward(inferenceInput);

        Assert.That(actual.IsLeaf, Is.True);
        Assert.That(actual.Shape, Is.EqualTo(new[] { 4, 8 }));
        Assert.That(Values(actual), Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void LayerNorm_AffineFalse_OutsideGrad_MatchesTrainingForward()
    {
        using var ln = new LayerNorm<float>(6, affine: false);
        var data = new float[] { 1.1f, -2.2f, 3.3f, -4.4f, 5.5f, -6.6f,
                                 0.7f, 0.8f, -0.9f, 1.2f, -1.3f, 1.4f,
                                 2.5f, -2.6f, 2.7f, -2.8f, 2.9f, -3.0f };

        var trainInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: true);
        trainInput.Reshape(3, 6);

        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(ln.Forward(trainInput));
        }

        var inferenceInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        inferenceInput.Reshape(3, 6);
        var actual = ln.Forward(inferenceInput);

        Assert.That(actual.IsLeaf, Is.True);
        Assert.That(Values(actual), Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void Gelu_OutsideGrad_MatchesTrainingForward()
    {
        var data = new float[] { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, 3.0f, -0.25f };

        var trainInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: true);
        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(Activation.Gelu(trainInput));
        }

        var inferenceInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        var actual = Activation.Gelu(inferenceInput);

        Assert.That(actual.IsLeaf, Is.True);
        Assert.That(Values(actual), Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void GeluExact_OutsideGrad_MatchesTrainingForward()
    {
        var data = new float[] { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, 3.0f, -0.25f };

        var trainInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: true);
        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(Activation.GeluExact(trainInput));
        }

        var inferenceInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        var actual = Activation.GeluExact(inferenceInput);

        Assert.That(actual.IsLeaf, Is.True);
        Assert.That(Values(actual), Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void TransformerBlock_LayerNormNorm_OutsideGrad_MatchesTrainingForward()
    {
        using var block = new TransformerBlock<float>(nEmbd: 16, nHead: 4, dropout: 0.0, maxSeqLen: 8, normType: NormType.LayerNorm);
        var data = new float[4 * 16];
        for (int i = 0; i < data.Length; i++)
            data[i] = ((i * 7) % 13 - 6) * 0.5f;

        var trainInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: true);
        trainInput.Reshape(4, 16);

        float[] expected;
        using (GradientUtils.Grad())
        {
            expected = Values(block.Forward(trainInput));
        }

        var inferenceInput = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        inferenceInput.Reshape(4, 16);
        var actual = block.Forward(inferenceInput);

        Assert.That(actual.IsLeaf, Is.True);
        Assert.That(actual.Shape, Is.EqualTo(new[] { 4, 16 }));
        Assert.That(Values(actual), Is.EqualTo(expected).Within(1e-4f));
    }
}
