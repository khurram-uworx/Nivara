using System.Text.Json;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class PyTorchReferenceTests
{
    IDisposable? gradScope;
    static string TestDataDir => Path.Combine(
        TestContext.CurrentContext.TestDirectory,
        "..", "..", "..", "..", "..", "samples", "data", "torch-comparison");

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static float[] LoadBin(string name)
    {
        var path = Path.Combine(TestDataDir, name);
        Assert.That(File.Exists(path), Is.True, $"Missing reference file: {path}");
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    static void AssertTensorEqual(float[] expected, float[] actual, float absTol = 1e-5f, float relTol = 1e-4f, string? label = null)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length),
            $"{label}: length mismatch {actual.Length} vs {expected.Length}");

        int failCount = 0;
        float maxDiff = 0f;
        int maxDiffIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float threshold = absTol + relTol * MathF.Abs(expected[i]);
            if (diff > threshold)
            {
                failCount++;
                if (diff > maxDiff)
                {
                    maxDiff = diff;
                    maxDiffIdx = i;
                }
            }
        }

        if (failCount > 0)
        {
            int showCount = Math.Min(5, failCount);
            var diffs = new List<string>();
            int shown = 0;
            for (int i = 0; i < expected.Length && shown < showCount; i++)
            {
                float diff = MathF.Abs(expected[i] - actual[i]);
                float threshold = absTol + relTol * MathF.Abs(expected[i]);
                if (diff > threshold)
                {
                    diffs.Add($"  [{i}] expected={expected[i]:G7} actual={actual[i]:G7} diff={diff:G7}");
                    shown++;
                }
            }
            Assert.Fail($"{label}: {failCount} elements differ (max diff={maxDiff:G7} at [{maxDiffIdx}]).\n" +
                        string.Join("\n", diffs));
        }
    }

    static float[] ExtractOutput(ReverseGradTensor<float> tensor)
    {
        var arr = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
            arr[i] = tensor[i];
        return arr;
    }

    // =========================================================================
    // Conv2d tests
    // =========================================================================
    [Test]
    public void Conv2d_3x3Stride1Pad1_MatchesPyTorch()
    {
        var input = LoadBin("conv2d_3x3_s1_p1_input.bin");
        var weight = LoadBin("conv2d_3x3_s1_p1_weight.bin");
        var bias = LoadBin("conv2d_3x3_s1_p1_bias.bin");
        var expected = LoadBin("conv2d_3x3_s1_p1_output.bin");

        using var conv = new Conv2d<float>(3, 16, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 1);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 27, requiresGrad: false);
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 7, 7 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Conv2d_3x3_s1_p1");
    }

    [Test]
    public void Conv2d_1x1Stride1Pad0_MatchesPyTorch()
    {
        var input = LoadBin("conv2d_1x1_s1_p0_input.bin");
        var weight = LoadBin("conv2d_1x1_s1_p0_weight.bin");
        var bias = LoadBin("conv2d_1x1_s1_p0_bias.bin");
        var expected = LoadBin("conv2d_1x1_s1_p0_output.bin");

        using var conv = new Conv2d<float>(3, 32, kernelSize: 1, stride: 1, padding: 0, bias: true, groups: 1);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 32, 3, requiresGrad: false);
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 7, 7 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Conv2d_1x1_s1_p0");
    }

    [Test]
    public void Conv2d_Depthwise_MatchesPyTorch()
    {
        var input = LoadBin("conv2d_depthwise_input.bin");
        var weight = LoadBin("conv2d_depthwise_weight.bin");
        var bias = LoadBin("conv2d_depthwise_bias.bin");
        var expected = LoadBin("conv2d_depthwise_output.bin");

        using var conv = new Conv2d<float>(16, 16, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 16);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 9, requiresGrad: false);
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 5, 5);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 5, 5 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Conv2d_depthwise");
    }

    [Test]
    public void Conv2d_Stride2_MatchesPyTorch()
    {
        var input = LoadBin("conv2d_stride2_input.bin");
        var weight = LoadBin("conv2d_stride2_weight.bin");
        var bias = LoadBin("conv2d_stride2_bias.bin");
        var expected = LoadBin("conv2d_stride2_output.bin");

        using var conv = new Conv2d<float>(3, 32, kernelSize: 3, stride: 2, padding: 1, bias: true, groups: 1);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 32, 27, requiresGrad: false);
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 14, 14);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 7, 7 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Conv2d_stride2");
    }

    [Test]
    public void Conv2d_WithBias_MatchesPyTorch()
    {
        var input = LoadBin("conv2d_with_bias_input.bin");
        var weight = LoadBin("conv2d_with_bias_weight.bin");
        var bias = LoadBin("conv2d_with_bias_bias.bin");
        var expected = LoadBin("conv2d_with_bias_output.bin");

        using var conv = new Conv2d<float>(3, 8, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 1);
        conv.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 8, 27, requiresGrad: false);
        conv.BiasParam!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 4, 4);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 4, 4 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Conv2d_with_bias");
    }

    // =========================================================================
    // BatchNorm2d tests (eval mode — these test the bug fix)
    // =========================================================================
    [Test]
    public void BatchNorm2d_Eval_16ch_MatchesPyTorch()
    {
        var input = LoadBin("bn2d_16ch_input.bin");
        var gamma = LoadBin("bn2d_16ch_gamma.bin");
        var beta = LoadBin("bn2d_16ch_beta.bin");
        var runningMean = LoadBin("bn2d_16ch_running_mean.bin");
        var runningVar = LoadBin("bn2d_16ch_running_var.bin");
        var expected = LoadBin("bn2d_16ch_output.bin");

        using var bn = new BatchNorm2d<float>(16);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 5, 5);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 5, 5 }));
        AssertTensorEqual(expected, ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_16ch_eval");
    }

    [Test]
    public void BatchNorm2d_Eval_3ch_MatchesPyTorch()
    {
        var input = LoadBin("bn2d_3ch_input.bin");
        var gamma = LoadBin("bn2d_3ch_gamma.bin");
        var beta = LoadBin("bn2d_3ch_beta.bin");
        var runningMean = LoadBin("bn2d_3ch_running_mean.bin");
        var runningVar = LoadBin("bn2d_3ch_running_var.bin");
        var expected = LoadBin("bn2d_3ch_output.bin");

        using var bn = new BatchNorm2d<float>(3);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 7, 7 }));
        AssertTensorEqual(expected, ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_3ch_eval");
    }

    [Test]
    public void BatchNorm2d_Eval_Batch4_MatchesPyTorch()
    {
        var input = LoadBin("bn2d_batch4_input.bin");
        var gamma = LoadBin("bn2d_batch4_gamma.bin");
        var beta = LoadBin("bn2d_batch4_beta.bin");
        var runningMean = LoadBin("bn2d_batch4_running_mean.bin");
        var runningVar = LoadBin("bn2d_batch4_running_var.bin");
        var expected = LoadBin("bn2d_batch4_output.bin");

        using var bn = new BatchNorm2d<float>(16);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 16, 8, 8);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16, 8, 8 }));
        AssertTensorEqual(expected, ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_batch4_eval");
    }

    // =========================================================================
    // ReLU tests
    // =========================================================================
    [Test]
    public void ReLU_1D_MatchesPyTorch()
    {
        var input = LoadBin("relu_1d_input.bin");
        var expected = LoadBin("relu_1d_relu_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.Relu(inputTensor);

        AssertTensorEqual(expected, ExtractOutput(output), label: "ReLU_1D");
    }

    [Test]
    public void ReLU_4D_MatchesPyTorch()
    {
        var input = LoadBin("relu_4d_input.bin");
        var expected = LoadBin("relu_4d_relu_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 8, 8);
        var output = Activation.Relu(inputTensor);

        AssertTensorEqual(expected, ExtractOutput(output), label: "ReLU_4D");
    }

    [Test]
    public void ReLU6_1D_MatchesPyTorch()
    {
        var input = LoadBin("relu_1d_input.bin");
        var expected = LoadBin("relu_1d_relu6_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = ReverseGradOperations.Clip(Activation.Relu(inputTensor), 0f, 6f);

        AssertTensorEqual(expected, ExtractOutput(output), label: "ReLU6_1D");
    }

    [Test]
    public void ReLU6_4D_MatchesPyTorch()
    {
        var input = LoadBin("relu_4d_input.bin");
        var expected = LoadBin("relu_4d_relu6_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 8, 8);
        var output = ReverseGradOperations.Clip(Activation.Relu(inputTensor), 0f, 6f);

        AssertTensorEqual(expected, ExtractOutput(output), label: "ReLU6_4D");
    }

    // =========================================================================
    // MaxPool2d tests
    // =========================================================================
    [Test]
    public void MaxPool2d_3x3Stride2Pad1_MatchesPyTorch()
    {
        var input = LoadBin("maxpool_3x3_s2_p1_input.bin");
        var expected = LoadBin("maxpool_3x3_s2_p1_output.bin");

        using var pool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 14, 14);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 7, 7 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "MaxPool2d_3x3_s2_p1");
    }

    [Test]
    public void MaxPool2d_2x2Stride2Pad0_MatchesPyTorch()
    {
        var input = LoadBin("maxpool_2x2_s2_p0_input.bin");
        var expected = LoadBin("maxpool_2x2_s2_p0_output.bin");

        using var pool = new MaxPool2d<float>(kernelSize: 2, stride: 2, padding: 0);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 32, 28, 28);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 14, 14 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "MaxPool2d_2x2_s2_p0");
    }

    // =========================================================================
    // AdaptiveAvgPool2d tests
    // =========================================================================
    [Test]
    public void AdaptiveAvgPool2d_1x1_Large_MatchesPyTorch()
    {
        var input = LoadBin("adaptiveavgpool_1x1_input.bin");
        var expected = LoadBin("adaptiveavgpool_1x1_output.bin");

        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 512, 7, 7);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 512, 1, 1 }));
        AssertTensorEqual(expected, ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "AdaptiveAvgPool2d_1x1_large");
    }

    [Test]
    public void AdaptiveAvgPool2d_1x1_Small_MatchesPyTorch()
    {
        var input = LoadBin("adaptiveavgpool_1x1_sm_input.bin");
        var expected = LoadBin("adaptiveavgpool_1x1_sm_output.bin");

        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 32, 14, 14);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 1, 1 }));
        AssertTensorEqual(expected, ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "AdaptiveAvgPool2d_1x1_small");
    }

    // =========================================================================
    // Linear tests
    // =========================================================================
    [Test]
    public void Linear_128_64_MatchesPyTorch()
    {
        var input = LoadBin("linear_128_64_input.bin");
        var weight = LoadBin("linear_128_64_weight.bin");
        var biasData = LoadBin("linear_128_64_bias.bin");
        var expected = LoadBin("linear_128_64_output.bin");

        using var linear = new Linear<float>(128, 64, bias: true);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weight, 64, 128, requiresGrad: false),
            ["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, 64, requiresGrad: false),
        });

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 128);

        var output = linear.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 64 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Linear_128_64");
    }

    [Test]
    public void Linear_512_1000_MatchesPyTorch()
    {
        var input = LoadBin("linear_512_1000_input.bin");
        var weight = LoadBin("linear_512_1000_weight.bin");
        var biasData = LoadBin("linear_512_1000_bias.bin");
        var expected = LoadBin("linear_512_1000_output.bin");

        using var linear = new Linear<float>(512, 1000, bias: true);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weight, 1000, 512, requiresGrad: false),
            ["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, 1000, requiresGrad: false),
        });

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 512);

        var output = linear.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1000 }));
        AssertTensorEqual(expected, ExtractOutput(output), label: "Linear_512_1000");
    }
}

