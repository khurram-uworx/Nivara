using Nivara.Silk;
using Nivara.Tests.NivaraTorch;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the Silk (OpenCL) vision kernels against the PyTorch reference fixtures generated
/// by <c>samples/NivaraTorch/gen_reference.py</c> (stored under
/// <c>samples/data/torch-comparison/</c>). Complements the existing CPU Torch-parity
/// suites and the kernel-level Silk vs ILGPU gate: these tests exercise the Silk path
/// end-to-end against an independent reference rather than against our own other backend.
/// </summary>
[TestFixture]
public sealed class SilkVisionTorchParityTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk vision Torch parity."); throw; }
    }

    static nint Upload(SilkRuntime silk, float[] data)
    {
        var buf = silk.Allocate(data.Length);
        silk.Upload(buf, data);
        return buf;
    }

    static void ReleaseAll(SilkRuntime silk, params nint[] bufs)
    {
        foreach (var b in bufs) silk.Release(b);
    }

    [Test]
    public void Conv2d_3x3Stride1Pad1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_3x3_s1_p1_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_3x3_s1_p1_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_3x3_s1_p1_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_3x3_s1_p1_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sw = Upload(silk, weight), sb = Upload(silk, bias);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Conv2d(silk, si, sw, sb, so, 1, 3, 7, 7, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, 7, 7);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Conv2d_3x3_s1_p1");
        ReleaseAll(silk, si, sw, sb, so);
    }

    [Test]
    public void Conv2d_1x1Stride1Pad0_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_1x1_s1_p0_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_1x1_s1_p0_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_1x1_s1_p0_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_1x1_s1_p0_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sw = Upload(silk, weight), sb = Upload(silk, bias);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Conv2d(silk, si, sw, sb, so, 1, 3, 7, 7, 32, 1, 1, 1, 1, 0, 0, 0, 0, 1, true, 7, 7);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Conv2d_1x1_s1_p0");
        ReleaseAll(silk, si, sw, sb, so);
    }

    [Test]
    public void Conv2d_Depthwise_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_depthwise_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_depthwise_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_depthwise_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_depthwise_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sw = Upload(silk, weight), sb = Upload(silk, bias);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Conv2d(silk, si, sw, sb, so, 1, 16, 5, 5, 16, 3, 3, 1, 1, 1, 1, 1, 1, 16, true, 5, 5);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Conv2d_depthwise");
        ReleaseAll(silk, si, sw, sb, so);
    }

    [Test]
    public void Conv2d_Stride2_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_stride2_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_stride2_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_stride2_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_stride2_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sw = Upload(silk, weight), sb = Upload(silk, bias);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Conv2d(silk, si, sw, sb, so, 1, 3, 14, 14, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, true, 7, 7);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Conv2d_stride2");
        ReleaseAll(silk, si, sw, sb, so);
    }

    [Test]
    public void Conv2d_WithBias_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_with_bias_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_with_bias_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_with_bias_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_with_bias_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sw = Upload(silk, weight), sb = Upload(silk, bias);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Conv2d(silk, si, sw, sb, so, 1, 3, 4, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, 4, 4);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk Conv2d_with_bias");
        ReleaseAll(silk, si, sw, sb, so);
    }

    [Test]
    public void BatchNorm2d_Eval_16ch_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_16ch_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_16ch_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_16ch_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_16ch_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_16ch_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_16ch_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sg = Upload(silk, gamma), sbt = Upload(silk, beta);
        nint sme = Upload(silk, runningMean), sv = Upload(silk, runningVar);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.BatchNormEval(silk, si, sme, sv, sg, sbt, so, 1, 16, 5, 5, 1e-5f, true);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk BatchNorm2d_16ch_eval");
        ReleaseAll(silk, si, sg, sbt, sme, sv, so);
    }

    [Test]
    public void BatchNorm2d_Eval_3ch_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_3ch_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_3ch_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_3ch_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_3ch_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_3ch_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_3ch_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sg = Upload(silk, gamma), sbt = Upload(silk, beta);
        nint sme = Upload(silk, runningMean), sv = Upload(silk, runningVar);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.BatchNormEval(silk, si, sme, sv, sg, sbt, so, 1, 3, 7, 7, 1e-5f, true);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk BatchNorm2d_3ch_eval");
        ReleaseAll(silk, si, sg, sbt, sme, sv, so);
    }

    [Test]
    public void BatchNorm2d_Eval_Batch4_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_batch4_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_batch4_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_batch4_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_batch4_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_batch4_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_batch4_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input), sg = Upload(silk, gamma), sbt = Upload(silk, beta);
        nint sme = Upload(silk, runningMean), sv = Upload(silk, runningVar);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.BatchNormEval(silk, si, sme, sv, sg, sbt, so, 4, 16, 8, 8, 1e-5f, true);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk BatchNorm2d_batch4_eval");
        ReleaseAll(silk, si, sg, sbt, sme, sv, so);
    }

    [Test]
    public void ReLU_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_1d_input.bin");
        var expected = TestHelpers.LoadBin("relu_1d_relu_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Relu(silk, si, so, expected.Length);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk ReLU_1D");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void ReLU_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_4d_input.bin");
        var expected = TestHelpers.LoadBin("relu_4d_relu_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Relu(silk, si, so, expected.Length);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk ReLU_4D");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void ReLU6_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_1d_input.bin");
        var expected = TestHelpers.LoadBin("relu_1d_relu6_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Relu6(silk, si, so, expected.Length);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk ReLU6_1D");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void ReLU6_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_4d_input.bin");
        var expected = TestHelpers.LoadBin("relu_4d_relu6_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.Relu6(silk, si, so, expected.Length);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk ReLU6_4D");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void MaxPool2d_3x3Stride2Pad1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("maxpool_3x3_s2_p1_input.bin");
        var expected = TestHelpers.LoadBin("maxpool_3x3_s2_p1_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.MaxPool2d(silk, si, so, 1, 16, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 7, 7);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk MaxPool2d_3x3_s2_p1");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void MaxPool2d_2x2Stride2Pad0_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("maxpool_2x2_s2_p0_input.bin");
        var expected = TestHelpers.LoadBin("maxpool_2x2_s2_p0_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.MaxPool2d(silk, si, so, 1, 32, 28, 28, 2, 2, 2, 2, 0, 0, 0, 0, 14, 14);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk MaxPool2d_2x2_s2_p0");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void AdaptiveAvgPool2d_1x1_Large_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("adaptiveavgpool_1x1_input.bin");
        var expected = TestHelpers.LoadBin("adaptiveavgpool_1x1_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.AdaptiveAvgPool2d(silk, si, so, 1, 512, 7, 7);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk AdaptiveAvgPool2d_1x1_large");
        ReleaseAll(silk, si, so);
    }

    [Test]
    public void AdaptiveAvgPool2d_1x1_Small_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("adaptiveavgpool_1x1_sm_input.bin");
        var expected = TestHelpers.LoadBin("adaptiveavgpool_1x1_sm_output.bin");

        using var silk = Device();
        nint si = Upload(silk, input);
        nint so = silk.Allocate(expected.Length);
        SilkLauncher.AdaptiveAvgPool2d(silk, si, so, 1, 32, 14, 14);
        silk.Finish();
        var actual = silk.Download(so, expected.Length);
        TestHelpers.AssertTensorEqual(expected, actual, label: "Silk AdaptiveAvgPool2d_1x1_small");
        ReleaseAll(silk, si, so);
    }
}
