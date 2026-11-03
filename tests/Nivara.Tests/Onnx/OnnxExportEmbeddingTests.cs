using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;

namespace Nivara.Tests.Onnx;

public sealed class EmbeddingExportModel : Module<float>
{
    readonly Embedding<float> embedding;

    public EmbeddingExportModel(int numEmbeddings, int embeddingDim) : base()
    {
        embedding = new Embedding<float>(numEmbeddings, embeddingDim);
        RegisterModules(embedding);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => embedding.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.Embedding(inputName, embedding);
}

public class OnnxExportEmbeddingTests
{
    [Test]
    public void EmbeddingExport_RoundTrip_MatchesNivara()
    {
        const int Vocab = 7;
        const int Dim = 4;
        const int Rows = 2;
        const int Cols = 3;
        var input = new float[Rows * Cols];
        for (int i = 0; i < input.Length; i++) input[i] = i % Vocab;

        var model = new EmbeddingExportModel(Vocab, Dim);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { Rows, Cols });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { Rows, Cols } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-embedding.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { Rows, Cols });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-5f));
    }
}
