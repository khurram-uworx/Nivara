using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class TokenClassifierModelTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Constructor_SetsPropertiesCorrectly()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 100, embeddingDim: 16, hiddenDim: 32, numClasses: 5, maxSeqLen: 10);

        Assert.That(model.VocabSize, Is.EqualTo(100));
        Assert.That(model.EmbeddingDim, Is.EqualTo(16));
        Assert.That(model.MaxSeqLen, Is.EqualTo(10));
        Assert.That(model.NumClasses, Is.EqualTo(5));
    }

    [Test]
    public void Constructor_RegistersAllSubModules()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 4, maxSeqLen: 5);

        var parameters = model.Parameters();
        Assert.That(parameters.Count, Is.EqualTo(5),
            "Embedding Weight + 2 Linear (Weight+Bias each) = 5 params");
    }

    [Test]
    public void Forward_OutputShape_IsTokensByClasses()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 5, maxSeqLen: 4);

        int batchSize = 2;
        int seqLen = 4;
        var tokenData = new float[batchSize * seqLen];
        for (int i = 0; i < tokenData.Length; i++)
            tokenData[i] = i % 10;
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, batchSize, seqLen, requiresGrad: false);

        var output = model.Forward(input);

        int totalTokens = batchSize * seqLen;
        Assert.That(output.Shape, Is.EqualTo(new[] { totalTokens, 5 }));
        Assert.That(output.Length, Is.EqualTo(totalTokens * 5));
    }

    [Test]
    public void Forward_OutputContainsNoNaNOrInfinity()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);

        var tokenData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 2, 4, requiresGrad: false);

        var output = model.Forward(input);

        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False,
                $"Output[{i}] should not be NaN or Infinity");
    }

    [Test]
    public void Forward_NoMeanPool_PreservesSequenceLength()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 6);

        int seqLen = 6;
        var tokenData = new float[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokenData[i] = i + 1;
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, seqLen, requiresGrad: false);

        var output = model.Forward(input);

        int expectedTokens = 1 * seqLen;
        Assert.That(output.Length, Is.EqualTo(expectedTokens * 3),
            "TokenClassifier should produce one prediction per token (no MeanPool)");
    }

    [Test]
    public void Predict_ReturnsPerTokenClassIndices()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 4, maxSeqLen: 5);

        int batchSize = 2;
        int seqLen = 5;
        var tokenIds = new int[batchSize * seqLen];
        for (int i = 0; i < tokenIds.Length; i++)
            tokenIds[i] = (i % 10) + 1;

        var result = model.Predict(tokenIds);

        Assert.That(result.Length, Is.EqualTo(tokenIds.Length),
            "TokenClassifier should return one prediction per token");
        for (int i = 0; i < result.Length; i++)
            Assert.That(result[i], Is.InRange(0, 3),
                $"Prediction[{i}] should be a valid class index (0-3)");
    }

    [Test]
    public void Predict_SingleToken_ReturnsOnePrediction()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 1);

        var tokenIds = new int[] { 5 };

        var result = model.Predict(tokenIds);

        Assert.That(result.Length, Is.EqualTo(1));
        Assert.That(result[0], Is.InRange(0, 2));
    }

    [Test]
    public void Backward_GradientsFlowToAllParameters()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);

        var tokenData = new float[] { 1, 2, 3, 4 };
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, 4, requiresGrad: false);

        var output = model.Forward(input);
        var grad = new float[output.Length];
        for (int i = 0; i < grad.Length; i++) grad[i] = 1f;
        var gradTensor = ReverseGradTensor<float>.FromArray(grad);
        gradTensor.Reshape(output.Shape);

        output.Backward(gradTensor);

        foreach (var (name, param) in model.GetParameters())
        {
            Assert.That(param.Tensor.Grad, Is.Not.Null,
                $"Parameter '{name}' should have gradient after Backward");
            Assert.That(param.Tensor.Grad!.Length, Is.EqualTo(param.Tensor.Length));
            for (int i = 0; i < param.Tensor.Grad.Length; i++)
                Assert.That(float.IsNaN(param.Tensor.Grad[i]) || float.IsInfinity(param.Tensor.Grad[i]), Is.False,
                    $"Gradient for '{name}[{i}]' should not be NaN or Infinity");
        }
    }

    [Test]
    public void Serialization_RoundTrip()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 5, maxSeqLen: 4);
        var path = Path.Combine(Path.GetTempPath(),
            $"tokenclassifier_test_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(model, path);

            using var loaded = new TokenClassifierModel<float>(
                vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 5, maxSeqLen: 4);
            ModelSerializer.Load(loaded, path);

            var originalParams = model.Parameters();
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
    public void Serialization_LoadedModel_ProducesSameOutput()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);
        var path = Path.Combine(Path.GetTempPath(),
            $"tokenclassifier_parity_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(model, path);

            var tokenData = new float[] { 1, 2, 3, 4 };
            var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, 4, requiresGrad: false);
            var originalOutput = model.Forward(input);

            using var loaded = new TokenClassifierModel<float>(
                vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);
            ModelSerializer.Load(loaded, path);
            var loadedOutput = loaded.Forward(input);

            Assert.That(loadedOutput.Length, Is.EqualTo(originalOutput.Length));
            for (int i = 0; i < originalOutput.Length; i++)
                Assert.That(loadedOutput[i], Is.EqualTo(originalOutput[i]).Within(1e-6f));
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Test]
    public void Forward_NullInput_ThrowsArgumentNullException()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);

        Assert.Throws<ArgumentNullException>(() => model.Forward(null!));
    }

    [Test]
    public void Predict_NullTokenIds_ThrowsArgumentNullException()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);

        Assert.Throws<ArgumentNullException>(() => model.Predict(null!));
    }

    [Test]
    public void Predict_InvalidLength_ThrowsArgumentException()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 4, maxSeqLen: 5);

        var tokenIds = new int[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => model.Predict(tokenIds));
    }

    [Test]
    public void TrainEval_TogglesTraining()
    {
        using var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);

        Assert.That(model.IsTraining, Is.True);
        model.Eval();
        Assert.That(model.IsTraining, Is.False);
        model.Train();
        Assert.That(model.IsTraining, Is.True);
    }

    [Test]
    public void Dispose_DisposesSubModules()
    {
        var model = new TokenClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 4);
        var param = model.GetParameters().First().Value;
        var tensor = param.Tensor;

        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = tensor.Length);
    }
}
