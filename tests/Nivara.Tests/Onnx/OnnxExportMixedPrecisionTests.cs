using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Onnx;

public sealed class MixedPrecisionLinearModel : Module<float>
{
    readonly Linear<float, System.Numerics.BFloat16> fc;

    public MixedPrecisionLinearModel() : base()
    {
        fc = new Linear<float, System.Numerics.BFloat16>(4, 3);
        RegisterModules(fc);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        => fc.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.Linear(inputName, fc);
}

public class OnnxExportMixedPrecisionTests
{
    [Test]
    public void MixedPrecisionLinearExport_RoundTrip_MatchesNivara()
    {
        var model = new MixedPrecisionLinearModel();
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

        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-mixed-linear.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { rows, features });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-3f));
    }
}
