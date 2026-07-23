using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class TextClassifierModelTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Constructor_SetsPropertiesCorrectly()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 100, embeddingDim: 16, hiddenDim: 32, numClasses: 3, maxSeqLen: 10);

        Assert.That(model.VocabSize, Is.EqualTo(100));
        Assert.That(model.EmbeddingDim, Is.EqualTo(16));
        Assert.That(model.MaxSeqLen, Is.EqualTo(10));
    }

    [Test]
    public void Constructor_RegistersAllSubModules()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 2, maxSeqLen: 5);

        var parameters = model.Parameters();
        Assert.That(parameters.Count, Is.EqualTo(5),
            "Embedding Weight + 2 Linear (Weight+Bias each) = 5 params");
    }

    [Test]
    public void Forward_OutputShape_IsBatchByClasses()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        int batchSize = 2;
        int seqLen = 5;
        var tokenData = new float[batchSize * seqLen];
        for (int i = 0; i < tokenData.Length; i++)
            tokenData[i] = i % 10;
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, batchSize, seqLen, requiresGrad: false);

        var output = model.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { batchSize, 3 }));
        Assert.That(output.Length, Is.EqualTo(batchSize * 3));
    }

    [Test]
    public void Forward_OutputContainsNoNaNOrInfinity()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 4, maxSeqLen: 5);

        var tokenData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 2, 5, requiresGrad: false);

        var output = model.Forward(input);

        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsNaN(output[i]) || float.IsInfinity(output[i]), Is.False,
                $"Output[{i}] should not be NaN or Infinity");
    }

    [Test]
    public void Forward_SingleBatch_ProducesCorrectShape()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 20, embeddingDim: 8, hiddenDim: 16, numClasses: 2, maxSeqLen: 4);

        var tokenData = new float[] { 1, 2, 3, 4 };
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, 4, requiresGrad: false);

        var output = model.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 2 }));
    }

    [Test]
    public void Predict_ReturnsValidClassIndices()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        int batchSize = 2;
        int seqLen = 5;
        var tokenIds = new int[batchSize * seqLen];
        for (int i = 0; i < tokenIds.Length; i++)
            tokenIds[i] = (i % 10) + 1;

        var result = model.Predict(tokenIds);

        Assert.That(result.Length, Is.EqualTo(batchSize));
        for (int i = 0; i < result.Length; i++)
            Assert.That(result[i], Is.InRange(0, 2),
                $"Prediction[{i}] should be a valid class index (0-2)");
    }

    [Test]
    public void Predict_DifferentInputs_CanProduceDifferentOutputs()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        var tokens1 = new int[] { 1, 2, 3, 4, 5 };
        var tokens2 = new int[] { 10, 11, 12, 13, 14 };

        var result1 = model.Predict(tokens1);
        var result2 = model.Predict(tokens2);

        Assert.That(result1.Length, Is.EqualTo(1));
        Assert.That(result2.Length, Is.EqualTo(1));
    }

    [Test]
    public void Backward_GradientsFlowToAllParameters()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        var tokenData = new float[] { 1, 2, 3, 4, 5 };
        var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, 5, requiresGrad: false);

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
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);
        var path = Path.Combine(Path.GetTempPath(),
            $"textclassifier_test_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(model, path);

            using var loaded = new TextClassifierModel<float>(
                vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);
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
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);
        var path = Path.Combine(Path.GetTempPath(),
            $"textclassifier_parity_{Guid.NewGuid()}.json");

        try
        {
            ModelSerializer.Save(model, path);

            var tokenData = new float[] { 1, 2, 3, 4, 5 };
            var input = ReverseGradTensor<float>.FromMatrix(tokenData, 1, 5, requiresGrad: false);
            var originalOutput = model.Forward(input);

            using var loaded = new TextClassifierModel<float>(
                vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);
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
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        Assert.Throws<ArgumentNullException>(() => model.Forward(null!));
    }

    [Test]
    public void Predict_NullTokenIds_ThrowsArgumentNullException()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        Assert.Throws<ArgumentNullException>(() => model.Predict(null!));
    }

    [Test]
    public void Predict_InvalidLength_ThrowsArgumentException()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        var tokenIds = new int[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => model.Predict(tokenIds));
    }

    [Test]
    public void TrainEval_TogglesTraining()
    {
        using var model = new TextClassifierModel<float>(
            vocabSize: 50, embeddingDim: 8, hiddenDim: 16, numClasses: 3, maxSeqLen: 5);

        Assert.That(model.IsTraining, Is.True);
        model.Eval();
        Assert.That(model.IsTraining, Is.False);
        model.Train();
        Assert.That(model.IsTraining, Is.True);
    }
}
