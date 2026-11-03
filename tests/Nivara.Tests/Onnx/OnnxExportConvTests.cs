using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;

namespace Nivara.Tests.Onnx;

public sealed class Conv1dExportModel : Module<float>
{
    readonly Conv1d<float> conv;

    public Conv1dExportModel(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true)
        : base()
    {
        conv = new Conv1d<float>(inChannels, outChannels, kernelSize, stride, padding, bias: bias);
        RegisterModules(conv);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => conv.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.Conv1d(inputName, conv);
}

public sealed class Conv2dExportModel : Module<float>
{
    readonly Conv2d<float> conv;

    public Conv2dExportModel(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true)
        : base()
    {
        conv = new Conv2d<float>(inChannels, outChannels, kernelSize, stride, padding, bias: bias);
        RegisterModules(conv);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => conv.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.Conv2d(inputName, conv);
}

public class OnnxExportConvTests
{
    [Test]
    public void Conv1dExport_RoundTrip_MatchesNivara()
    {
        const int B = 2, C = 2, L = 5;
        var input = new float[B * C * L];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.2f;

        var model = new Conv1dExportModel(C, 3, 3, padding: 1);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { B, C, L });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { B, C, L } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-conv1d.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { B, C, L });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void Conv1dExport_WithoutBias_Strided_RoundTrip_MatchesNivara()
    {
        const int B = 2, C = 2, L = 6;
        var input = new float[B * C * L];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.3f;

        var model = new Conv1dExportModel(C, 3, 2, stride: 2, padding: 0, bias: false);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { B, C, L });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { B, C, L } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-conv1d-nobias.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { B, C, L });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void Conv2dExport_RoundTrip_MatchesNivara()
    {
        const int B = 2, C = 2, H = 4, W = 4;
        var input = new float[B * C * H * W];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.05f - 0.1f;

        var model = new Conv2dExportModel(C, 3, 3, padding: 1);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { B, C, H, W });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { B, C, H, W } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-conv2d.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { B, C, H, W });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void Conv2dExport_StridedPadded_WithoutBias_RoundTrip_MatchesNivara()
    {
        const int B = 1, C = 2, H = 5, W = 5;
        var input = new float[B * C * H * W];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.05f - 0.2f;

        var model = new Conv2dExportModel(C, 2, 3, stride: 2, padding: 1, bias: false);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { B, C, H, W });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { B, C, H, W } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-conv2d-stride.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { B, C, H, W });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }
}