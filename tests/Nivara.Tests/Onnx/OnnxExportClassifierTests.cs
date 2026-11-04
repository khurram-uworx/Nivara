using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NivaraClassifier;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Onnx;

public class OnnxExportClassifierTests
{
    [Test]
    public void TextClassifierExport_RoundTrip_MatchesNivara()
    {
        const int vocabSize = 100;
        const int embeddingDim = 16;
        const int hiddenDim = 32;
        const int numClasses = 2;
        const int maxSeqLen = 8;
        const int batchSize = 2;

        var model = new TextClassifierModel<float>(vocabSize, embeddingDim, hiddenDim, numClasses, maxSeqLen);
        model.Eval();

        var tokenIds = new int[batchSize * maxSeqLen];
        for (int i = 0; i < tokenIds.Length; i++)
            tokenIds[i] = i % vocabSize;

        var inputData = new float[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            inputData[i] = tokenIds[i];

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(inputData), false, new[] { batchSize, maxSeqLen });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { batchSize, maxSeqLen } });
        var bytes = ms.ToArray();

        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-text-classifier.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(inputData, new[] { batchSize, maxSeqLen });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void ConvTextClassifierExport_RoundTrip_MatchesNivara()
    {
        const int vocabSize = 100;
        const int embeddingDim = 16;
        const int hiddenDim = 32;
        const int numClasses = 2;
        const int maxSeqLen = 8;
        const int batchSize = 2;

        var model = new ConvTextClassifierModel<float>(vocabSize, embeddingDim, hiddenDim, numClasses, maxSeqLen, dropout: 0.0f);
        model.Eval();

        var tokenIds = new int[batchSize * maxSeqLen];
        for (int i = 0; i < tokenIds.Length; i++)
            tokenIds[i] = (i * 7) % vocabSize;

        var inputData = new float[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            inputData[i] = tokenIds[i];

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(inputData), false, new[] { batchSize, maxSeqLen });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { batchSize, maxSeqLen } });
        var bytes = ms.ToArray();

        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-conv-text-classifier.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(inputData, new[] { batchSize, maxSeqLen });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }
}
