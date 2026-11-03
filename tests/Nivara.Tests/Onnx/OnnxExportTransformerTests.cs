using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;

namespace Nivara.Tests.Onnx;

public class OnnxExportTransformerTests
{
    [Test]
    public void TransformerBlockExport_RoundTrip_MatchesNivara()
    {
        const int L = 3;
        const int D = 4;
        const int H = 2;
        var block = new TransformerBlock<float>(D, H, dropout: 0.0, maxSeqLen: 8);
        block.Eval();

        var input = new float[L * D];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.05f - 0.1f;

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { L, D });
        var expectedTensor = block.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(block, ms, new OnnxExportOptions { InputShape = new[] { L, D } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-transformer.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { L, D });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(5e-4f));
    }
}
