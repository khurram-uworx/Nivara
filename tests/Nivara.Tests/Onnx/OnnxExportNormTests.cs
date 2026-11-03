using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;

namespace Nivara.Tests.Onnx;

public sealed class BatchNorm1dExportModel : Module<float>
{
    readonly BatchNorm1d<float> bn;

    public BatchNorm1dExportModel(int features, bool affine = true) : base()
    {
        bn = new BatchNorm1d<float>(features, affine: affine);
        RegisterModules(bn);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => bn.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.BatchNorm1d(inputName, bn);
}

public sealed class BatchNorm2dExportModel : Module<float>
{
    readonly BatchNorm2d<float> bn;

    public BatchNorm2dExportModel(int features, bool affine = true) : base()
    {
        bn = new BatchNorm2d<float>(features, affine: affine);
        RegisterModules(bn);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => bn.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.BatchNorm2d(inputName, bn);
}

public sealed class LayerNormExportModel : Module<float>
{
    readonly LayerNorm<float> ln;

    public LayerNormExportModel(int size, bool affine = true, bool bias = true) : base()
    {
        ln = new LayerNorm<float>(size, affine: affine, bias: bias);
        RegisterModules(ln);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => ln.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.LayerNorm(inputName, ln);
}

public sealed class RMSNormExportModel : Module<float>
{
    readonly RMSNorm<float> rms;

    public RMSNormExportModel(int size) : base()
    {
        rms = new RMSNorm<float>(size);
        RegisterModules(rms);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input) => rms.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.RMSNorm(inputName, rms);
}

public class OnnxExportNormTests
{
    [Test]
    public void BatchNorm1dExport_RoundTrip_MatchesNivara()
    {
        const int N = 4, C = 4;
        var input = new float[N * C];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.2f;

        var model = new BatchNorm1dExportModel(C);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, C });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, C } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-bn1d.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, C });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void BatchNorm1dExport_NonAffine_RoundTrip_MatchesNivara()
    {
        const int N = 4, C = 4;
        var input = new float[N * C];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.25f;

        var model = new BatchNorm1dExportModel(C, affine: false);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, C });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, C } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-bn1d-nonaffine.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, C });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void BatchNorm2dExport_RoundTrip_MatchesNivara()
    {
        const int N = 2, C = 3, H = 4, W = 4;
        var input = new float[N * C * H * W];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.05f - 0.1f;

        var model = new BatchNorm2dExportModel(C);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, C, H, W });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, C, H, W } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-bn2d.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, C, H, W });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void LayerNormExport_RoundTrip_MatchesNivara()
    {
        const int N = 2, D = 4;
        var input = new float[N * D];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.3f;

        var model = new LayerNormExportModel(D);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, D });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, D } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-ln.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, D });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void LayerNormExport_BiasFree_RoundTrip_MatchesNivara()
    {
        const int N = 2, D = 4;
        var input = new float[N * D];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.3f;

        var model = new LayerNormExportModel(D, affine: true, bias: false);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, D });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, D } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-ln-biasfree.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, D });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void LayerNormExport_NonAffine_RoundTrip_MatchesNivara()
    {
        const int N = 2, D = 4;
        var input = new float[N * D];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.3f;

        var model = new LayerNormExportModel(D, affine: false, bias: false);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, D });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, D } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-ln-nonaffine.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, D });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }

    [Test]
    public void RMSNormExport_RoundTrip_MatchesNivara()
    {
        const int N = 2, D = 4;
        var input = new float[N * D];
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.1f - 0.2f;

        var model = new RMSNormExportModel(D);
        model.Eval();

        var inTensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(input), false, new[] { N, D });
        var expectedTensor = model.Forward(inTensor);
        var expected = new float[expectedTensor.Length];
        expectedTensor.Data.CopyTo(expected, 0f);

        var ms = new System.IO.MemoryStream();
        OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { N, D } });
        var bytes = ms.ToArray();
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nivara-rms.onnx");
        System.IO.File.WriteAllBytes(path, bytes);

        using var session = new InferenceSession(path);
        var ortInput = new DenseTensor<float>(input, new[] { N, D });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortInput) });
        var ortOutput = results.First().AsTensor<float>().ToArray();

        Assert.That(ortOutput.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ortOutput[i], Is.EqualTo(expected[i]).Within(1e-4f));
    }
}