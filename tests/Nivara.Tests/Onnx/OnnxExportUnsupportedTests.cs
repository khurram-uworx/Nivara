using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Onnx;
using NUnit.Framework;

namespace Nivara.Tests.Onnx;

public sealed class UnsupportedMultiheadAttentionModel : Module<float>
{
    readonly MultiheadAttention<float> mha;

    public UnsupportedMultiheadAttentionModel() : base()
    {
        mha = new MultiheadAttention<float>(4, 2);
        RegisterModules(mha);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        => mha.Forward(input);

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
        => context.MultiheadAttention(inputName, mha);
}

public sealed class UnsupportedConvTransposeModel : Module<float>
{
    readonly ConvTranspose2d<float> conv;

    public UnsupportedConvTransposeModel() : base()
    {
        conv = new ConvTranspose2d<float>(2, 2, 3);
        RegisterModules(conv);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        => conv.Forward(input);
}

public sealed class UnimplementedExportModel : Module<float>
{
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        => input;
}

public class OnnxExportUnsupportedTests
{
    [Test]
    public void MultiheadAttentionExport_ThrowsNotSupportedException()
    {
        var model = new UnsupportedMultiheadAttentionModel();
        model.Eval();
        var ms = new System.IO.MemoryStream();
        Assert.Throws<System.NotSupportedException>(() =>
            OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { 2, 4 } }));
    }

    [Test]
    public void ConvTranspose2dExport_ThrowsNotSupportedException()
    {
        var model = new UnsupportedConvTransposeModel();
        model.Eval();
        var ms = new System.IO.MemoryStream();
        Assert.Throws<System.NotSupportedException>(() =>
            OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { 1, 2, 4, 4 } }));
    }

    [Test]
    public void UnimplementedModuleExport_ThrowsNotSupportedException()
    {
        var model = new UnimplementedExportModel();
        model.Eval();
        var ms = new System.IO.MemoryStream();
        Assert.Throws<System.NotSupportedException>(() =>
            OnnxExporter.Export(model, ms, new OnnxExportOptions { InputShape = new[] { 1, 4 } }));
    }
}
