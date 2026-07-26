using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace NivaraInference;

internal sealed class MobileNetV2 : Module<float>
{
    readonly Module<float> stem;
    readonly Module<float>[] blocks;
    readonly Conv2d<float> headConv;
    readonly BatchNorm2d<float> headBn;
    readonly AdaptiveAvgPool2d<float> avgPool;
    readonly Linear<float> classifier;

    const int NumBlocks = 16;
    static readonly (int expand, int outChannels, int stride)[] BlockConfigs =
    [
        (96, 24, 1),
        (144, 24, 1),
        (144, 32, 2),
        (192, 32, 1),
        (192, 32, 1),
        (192, 64, 2),
        (384, 64, 1),
        (384, 64, 1),
        (384, 64, 1),
        (384, 96, 1),
        (576, 96, 1),
        (576, 96, 2),
        (576, 160, 1),
        (960, 160, 1),
        (960, 160, 1),
        (960, 320, 1)
    ];

    public MobileNetV2(int numClasses = 1001)
    {
        stem = BuildStem();
        RegisterModules(stem);

        blocks = new Module<float>[NumBlocks];
        int inCh = 16;
        for (int i = 0; i < NumBlocks; i++)
        {
            var (expand, outCh, stride) = BlockConfigs[i];
            blocks[i] = new InvertedResidualBlock(inCh, expand, outCh, stride);
            inCh = outCh;
        }
        RegisterModules([.. blocks]);

        headConv = new Conv2d<float>(320, 1280, 1);
        headBn = new BatchNorm2d<float>(1280);
        RegisterModules(headConv, headBn);

        avgPool = new AdaptiveAvgPool2d<float>(1);
        RegisterModules(avgPool);

        classifier = new Linear<float>(1280, numClasses, bias: true);
        RegisterModules(classifier);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var x = stem.Forward(input);
        foreach (var block in blocks)
            x = block.Forward(x);

        x = headConv.Forward(x);
        x = headBn.Forward(x);
        x = Relu6(x);

        x = avgPool.Forward(x);

        int n = x.Shape[0], c = x.Shape[1];
        x.Reshape(n, c);

        x = classifier.Forward(x);
        return x;
    }

    static ReverseGradTensor<float> Relu6(ReverseGradTensor<float> x)
        => ReverseGradOperations.Clip(ReverseGradOperations.Relu(x), 0f, 6f);

    static Module<float> BuildStem()
    {
        var firstConv = new Conv2d<float>(3, 32, 3, stride: 2, padding: 1, bias: false);
        var firstBn = new BatchNorm2d<float>(32);
        var dwConv = new Conv2d<float>(32, 32, 3, padding: 1, groups: 32, bias: false);
        var dwBn = new BatchNorm2d<float>(32);
        var pwConv = new Conv2d<float>(32, 16, 1, bias: false);
        var pwBn = new BatchNorm2d<float>(16);
        return new StemBlock(firstConv, firstBn, dwConv, dwBn, pwConv, pwBn);
    }

    static void LoadConv(Module<float> conv, float[] data, int[] shape)
    {
        var tensor = ReverseGradTensor<float>.FromMatrix(data, shape[0], shape[1] * shape[2] * shape[3]);
        conv.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>> { ["Weight"] = tensor });
    }

    static void LoadBn(BatchNorm2d<float> bn,
        float[]? weight, float[]? bias, float[]? runningMean, float[]? runningVar)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>();
        if (weight != null) dict["Weight"] = ReverseGradTensor<float>.FromArray(weight);
        if (bias != null) dict["Bias"] = ReverseGradTensor<float>.FromArray(bias);
        if (runningMean != null) dict["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean);
        if (runningVar != null) dict["running_var"] = ReverseGradTensor<float>.FromArray(runningVar);
        if (dict.Count > 0) bn.LoadStateDict(dict);
    }

    static void LoadLinear(Linear<float> linear, float[] weightData, int[] weightShape, float[]? biasData)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weightData, weightShape[0], weightShape[1])
        };
        if (biasData != null) dict["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, biasData.Length);
        linear.LoadStateDict(dict);
    }

    sealed class StemBlock : Module<float>
    {
        internal readonly Conv2d<float> firstConv;
        internal readonly BatchNorm2d<float> firstBn;
        internal readonly Conv2d<float> dwConv;
        internal readonly BatchNorm2d<float> dwBn;
        internal readonly Conv2d<float> pwConv;
        internal readonly BatchNorm2d<float> pwBn;

        public StemBlock(Conv2d<float> firstConv, BatchNorm2d<float> firstBn,
            Conv2d<float> dwConv, BatchNorm2d<float> dwBn,
            Conv2d<float> pwConv, BatchNorm2d<float> pwBn)
        {
            this.firstConv = firstConv;
            this.firstBn = firstBn;
            this.dwConv = dwConv;
            this.dwBn = dwBn;
            this.pwConv = pwConv;
            this.pwBn = pwBn;
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        {
            var x = firstConv.Forward(input);
            x = firstBn.Forward(x);
            x = Relu6(x);

            x = dwConv.Forward(x);
            x = dwBn.Forward(x);
            x = Relu6(x);

            x = pwConv.Forward(x);
            x = pwBn.Forward(x);
            x = Relu6(x);
            return x;
        }
    }

    sealed class InvertedResidualBlock : Module<float>
    {
        internal readonly Conv2d<float>? expandConv;
        internal readonly BatchNorm2d<float>? expandBn;
        internal readonly Conv2d<float> depthwiseConv;
        internal readonly BatchNorm2d<float> depthwiseBn;
        internal readonly Conv2d<float> projectConv;
        internal readonly BatchNorm2d<float> projectBn;
        internal readonly bool hasExpansion;
        internal readonly bool useResidual;

        public InvertedResidualBlock(int inChannels, int expandChannels, int outChannels, int stride)
        {
            hasExpansion = inChannels != expandChannels;
            useResidual = stride == 1 && inChannels == outChannels;

            if (hasExpansion)
            {
                expandConv = new Conv2d<float>(inChannels, expandChannels, 1, bias: false);
                expandBn = new BatchNorm2d<float>(expandChannels);
            }

            depthwiseConv = new Conv2d<float>(expandChannels, expandChannels, 3, stride: stride, padding: 1, groups: expandChannels, bias: false);
            depthwiseBn = new BatchNorm2d<float>(expandChannels);

            projectConv = new Conv2d<float>(expandChannels, outChannels, 1, bias: false);
            projectBn = new BatchNorm2d<float>(outChannels);
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        {
            var x = input;
            if (hasExpansion)
            {
                x = expandConv!.Forward(x);
                x = expandBn!.Forward(x);
                x = Relu6(x);
            }

            x = depthwiseConv.Forward(x);
            x = depthwiseBn.Forward(x);
            x = Relu6(x);

            x = projectConv.Forward(x);
            x = projectBn.Forward(x);

            if (useResidual)
                x = x + input;
            return x;
        }
    }

    public static MobileNetV2 LoadWeights(Dictionary<string, (float[] Data, int[] Shape)> tensors, int numClasses = 1001)
    {
        var model = new MobileNetV2(numClasses);
        model.Eval();

        LoadStem((StemBlock)model.stem, tensors);

        for (int i = 0; i < NumBlocks; i++)
            LoadBlock((InvertedResidualBlock)model.blocks[i], i, tensors);

        LoadConv(model.headConv,
            tensors["mobilenet_v2.conv_1x1.convolution.weight"].Data,
            tensors["mobilenet_v2.conv_1x1.convolution.weight"].Shape);
        LoadBn(model.headBn,
            tensors.TryGetValue("mobilenet_v2.conv_1x1.normalization.weight", out var h1) ? h1.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_1x1.normalization.bias", out var h2) ? h2.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_1x1.normalization.running_mean", out var h3) ? h3.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_1x1.normalization.running_var", out var h4) ? h4.Data : null);

        LoadLinear(model.classifier,
            tensors["classifier.weight"].Data,
            tensors["classifier.weight"].Shape,
            tensors.TryGetValue("classifier.bias", out var bias) ? bias.Data : null);

        return model;
    }

    public static int CountParameters(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        return tensors.Values.Sum(v => v.Data.Length);
    }

    static void LoadStem(StemBlock stem, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        LoadConv(stem.firstConv,
            tensors["mobilenet_v2.conv_stem.first_conv.convolution.weight"].Data,
            tensors["mobilenet_v2.conv_stem.first_conv.convolution.weight"].Shape);
        LoadBn(stem.firstBn,
            tensors.TryGetValue("mobilenet_v2.conv_stem.first_conv.normalization.weight", out var sw1) ? sw1.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.first_conv.normalization.bias", out var sb1) ? sb1.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.first_conv.normalization.running_mean", out var sm1) ? sm1.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.first_conv.normalization.running_var", out var sv1) ? sv1.Data : null);

        LoadConv(stem.dwConv,
            tensors["mobilenet_v2.conv_stem.conv_3x3.convolution.weight"].Data,
            tensors["mobilenet_v2.conv_stem.conv_3x3.convolution.weight"].Shape);
        LoadBn(stem.dwBn,
            tensors.TryGetValue("mobilenet_v2.conv_stem.conv_3x3.normalization.weight", out var sw2) ? sw2.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.conv_3x3.normalization.bias", out var sb2) ? sb2.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.conv_3x3.normalization.running_mean", out var sm2) ? sm2.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.conv_3x3.normalization.running_var", out var sv2) ? sv2.Data : null);

        LoadConv(stem.pwConv,
            tensors["mobilenet_v2.conv_stem.reduce_1x1.convolution.weight"].Data,
            tensors["mobilenet_v2.conv_stem.reduce_1x1.convolution.weight"].Shape);
        LoadBn(stem.pwBn,
            tensors.TryGetValue("mobilenet_v2.conv_stem.reduce_1x1.normalization.weight", out var sw3) ? sw3.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.reduce_1x1.normalization.bias", out var sb3) ? sb3.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.reduce_1x1.normalization.running_mean", out var sm3) ? sm3.Data : null,
            tensors.TryGetValue("mobilenet_v2.conv_stem.reduce_1x1.normalization.running_var", out var sv3) ? sv3.Data : null);
    }

    static void LoadBlock(InvertedResidualBlock block, int index, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        string prefix = $"mobilenet_v2.layer.{index}";

        if (block.hasExpansion)
        {
            LoadConv(block.expandConv!,
                tensors[$"{prefix}.expand_1x1.convolution.weight"].Data,
                tensors[$"{prefix}.expand_1x1.convolution.weight"].Shape);
            LoadBn(block.expandBn!,
                tensors.TryGetValue($"{prefix}.expand_1x1.normalization.weight", out var ew) ? ew.Data : null,
                tensors.TryGetValue($"{prefix}.expand_1x1.normalization.bias", out var eb) ? eb.Data : null,
                tensors.TryGetValue($"{prefix}.expand_1x1.normalization.running_mean", out var em) ? em.Data : null,
                tensors.TryGetValue($"{prefix}.expand_1x1.normalization.running_var", out var ev) ? ev.Data : null);
        }

        LoadConv(block.depthwiseConv,
            tensors[$"{prefix}.conv_3x3.convolution.weight"].Data,
            tensors[$"{prefix}.conv_3x3.convolution.weight"].Shape);
        LoadBn(block.depthwiseBn,
            tensors.TryGetValue($"{prefix}.conv_3x3.normalization.weight", out var dw) ? dw.Data : null,
            tensors.TryGetValue($"{prefix}.conv_3x3.normalization.bias", out var db) ? db.Data : null,
            tensors.TryGetValue($"{prefix}.conv_3x3.normalization.running_mean", out var dm) ? dm.Data : null,
            tensors.TryGetValue($"{prefix}.conv_3x3.normalization.running_var", out var dv) ? dv.Data : null);

        LoadConv(block.projectConv,
            tensors[$"{prefix}.reduce_1x1.convolution.weight"].Data,
            tensors[$"{prefix}.reduce_1x1.convolution.weight"].Shape);
        LoadBn(block.projectBn,
            tensors.TryGetValue($"{prefix}.reduce_1x1.normalization.weight", out var pw) ? pw.Data : null,
            tensors.TryGetValue($"{prefix}.reduce_1x1.normalization.bias", out var pb) ? pb.Data : null,
            tensors.TryGetValue($"{prefix}.reduce_1x1.normalization.running_mean", out var pm) ? pm.Data : null,
            tensors.TryGetValue($"{prefix}.reduce_1x1.normalization.running_var", out var pv) ? pv.Data : null);
    }
}
