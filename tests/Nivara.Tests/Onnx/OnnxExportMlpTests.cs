using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Onnx;

public sealed class LrlModel : Module<float>
{
    readonly Linear<float> fc1;
    readonly Linear<float> fc2;

    public LrlModel() : base()
    {
        fc1 = new Linear<float>(4, 8);
        fc2 = new Linear<float>(8, 3);
        RegisterModules(fc1, fc2);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var h = fc1.Forward(input);
        h = Activation.Relu(h);
        return fc2.Forward(h);
    }

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
    {
        var h = context.Linear(inputName, fc1);
        h = context.Relu(h);
        return context.Linear(h, fc2);
    }
}

public class OnnxExportMlpTests
{
    [Test]
    public void MlpExport_RoundTrip_MatchesNivara()
    {
        var model = new LrlModel();
        model.Eval();

        const int rows = 2;
        const int features = 4;
        var input = new float[rows * features];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.5f;

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { rows, features });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { rows, features } });
        var bytes = ms.ToArray();

        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-mlp.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { rows, features });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }
}
