using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace NivaraInference;

internal sealed class ResNet18 : Module<float>
{
    readonly Conv2d<float> stemConv;
    readonly BatchNorm2d<float> stemBn;
    readonly MaxPool2d<float> stemPool;
    readonly Module<float> stage0Layer0;
    readonly Module<float> stage0Layer1;
    readonly Module<float> stage1Layer0;
    readonly Module<float> stage1Layer1;
    readonly Module<float> stage2Layer0;
    readonly Module<float> stage2Layer1;
    readonly Module<float> stage3Layer0;
    readonly Module<float> stage3Layer1;
    readonly AdaptiveAvgPool2d<float> avgPool;
    readonly Linear<float> fc;

    public ResNet18(int numClasses = 1000)
    {
        stemConv = new Conv2d<float>(3, 64, 7, stride: 2, padding: 3, bias: false);
        stemBn = new BatchNorm2d<float>(64);
        stemPool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);

        stage0Layer0 = new BasicBlock(64, 64);
        stage0Layer1 = new BasicBlock(64, 64);

        stage1Layer0 = new BasicBlock(64, 128, stride: 2);
        stage1Layer1 = new BasicBlock(128, 128);

        stage2Layer0 = new BasicBlock(128, 256, stride: 2);
        stage2Layer1 = new BasicBlock(256, 256);

        stage3Layer0 = new BasicBlock(256, 512, stride: 2);
        stage3Layer1 = new BasicBlock(512, 512);

        avgPool = new AdaptiveAvgPool2d<float>(1);
        fc = new Linear<float>(512, numClasses, bias: true);

        RegisterModules(stemConv, stemBn, stemPool);
        RegisterModules(stage0Layer0, stage0Layer1);
        RegisterModules(stage1Layer0, stage1Layer1);
        RegisterModules(stage2Layer0, stage2Layer1);
        RegisterModules(stage3Layer0, stage3Layer1);
        RegisterModules(avgPool, fc);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var x = stemConv.Forward(input);
        x = stemBn.Forward(x);
        x = ReverseGradOperations.Relu(x);
        x = stemPool.Forward(x);

        x = stage0Layer0.Forward(x);
        x = stage0Layer1.Forward(x);

        x = stage1Layer0.Forward(x);
        x = stage1Layer1.Forward(x);

        x = stage2Layer0.Forward(x);
        x = stage2Layer1.Forward(x);

        x = stage3Layer0.Forward(x);
        x = stage3Layer1.Forward(x);

        x = avgPool.Forward(x);

        int n = x.Shape[0], c = x.Shape[1];
        x.Reshape(n, c);

        x = fc.Forward(x);
        return x;
    }

    sealed class BasicBlock : Module<float>
    {
        internal readonly Conv2d<float> conv1;
        internal readonly BatchNorm2d<float> bn1;
        internal readonly Conv2d<float> conv2;
        internal readonly BatchNorm2d<float> bn2;
        internal readonly Conv2d<float>? downsampleConv;
        internal readonly BatchNorm2d<float>? downsampleBn;
        internal readonly bool hasDownsample;

        public BasicBlock(int inChannels, int outChannels, int stride = 1)
        {
            hasDownsample = inChannels != outChannels || stride != 1;

            conv1 = new Conv2d<float>(inChannels, outChannels, 3, stride: stride, padding: 1, bias: false);
            bn1 = new BatchNorm2d<float>(outChannels);

            conv2 = new Conv2d<float>(outChannels, outChannels, 3, padding: 1, bias: false);
            bn2 = new BatchNorm2d<float>(outChannels);

            if (hasDownsample)
            {
                downsampleConv = new Conv2d<float>(inChannels, outChannels, 1, stride: stride, bias: false);
                downsampleBn = new BatchNorm2d<float>(outChannels);
                RegisterModules(conv1, bn1, conv2, bn2, downsampleConv, downsampleBn);
            }
            else
            {
                RegisterModules(conv1, bn1, conv2, bn2);
            }
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
        {
            var x = conv1.Forward(input);
            x = bn1.Forward(x);
            x = ReverseGradOperations.Relu(x);

            x = conv2.Forward(x);
            x = bn2.Forward(x);

            var residual = hasDownsample
                ? downsampleBn!.Forward(downsampleConv!.Forward(input))
                : input;

            x = x + residual;
            x = ReverseGradOperations.Relu(x);
            return x;
        }
    }

    internal static void LoadConv(Module<float> conv, float[] data, int[] shape)
    {
        var tensor = ReverseGradTensor<float>.FromMatrix(data, shape[0], shape[1] * shape[2] * shape[3]);
        conv.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>> { ["Weight"] = tensor });
    }

    internal static void LoadBn(BatchNorm2d<float> bn,
        float[]? weight, float[]? bias, float[]? runningMean, float[]? runningVar)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>();
        if (weight != null) dict["Weight"] = ReverseGradTensor<float>.FromArray(weight);
        if (bias != null) dict["Bias"] = ReverseGradTensor<float>.FromArray(bias);
        if (runningMean != null) dict["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean);
        if (runningVar != null) dict["running_var"] = ReverseGradTensor<float>.FromArray(runningVar);
        if (dict.Count > 0) bn.LoadStateDict(dict);
    }

    internal static void LoadLinear(Linear<float> linear, float[] weightData, int[] weightShape, float[]? biasData)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weightData, weightShape[0], weightShape[1])
        };
        if (biasData != null) dict["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, biasData.Length);
        linear.LoadStateDict(dict);
    }

    public static int CountParameters(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        return tensors.Values.Sum(v => v.Data.Length);
    }

    static void LoadBasicBlock(Module<float> block, string prefix,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        var bb = (BasicBlock)block;

        LoadConv(bb.conv1, tensors[$"{prefix}.layer.0.convolution.weight"].Data,
            tensors[$"{prefix}.layer.0.convolution.weight"].Shape);
        LoadBn(bb.bn1,
            tensors.TryGetValue($"{prefix}.layer.0.normalization.weight", out var w1) ? w1.Data : null,
            tensors.TryGetValue($"{prefix}.layer.0.normalization.bias", out var b1) ? b1.Data : null,
            tensors.TryGetValue($"{prefix}.layer.0.normalization.running_mean", out var m1) ? m1.Data : null,
            tensors.TryGetValue($"{prefix}.layer.0.normalization.running_var", out var v1) ? v1.Data : null);

        LoadConv(bb.conv2, tensors[$"{prefix}.layer.1.convolution.weight"].Data,
            tensors[$"{prefix}.layer.1.convolution.weight"].Shape);
        LoadBn(bb.bn2,
            tensors.TryGetValue($"{prefix}.layer.1.normalization.weight", out var w2) ? w2.Data : null,
            tensors.TryGetValue($"{prefix}.layer.1.normalization.bias", out var b2) ? b2.Data : null,
            tensors.TryGetValue($"{prefix}.layer.1.normalization.running_mean", out var m2) ? m2.Data : null,
            tensors.TryGetValue($"{prefix}.layer.1.normalization.running_var", out var v2) ? v2.Data : null);

        if (bb.hasDownsample && bb.downsampleConv != null && bb.downsampleBn != null)
        {
            LoadConv(bb.downsampleConv, tensors[$"{prefix}.shortcut.convolution.weight"].Data,
                tensors[$"{prefix}.shortcut.convolution.weight"].Shape);
            LoadBn(bb.downsampleBn,
                tensors.TryGetValue($"{prefix}.shortcut.normalization.weight", out var sw) ? sw.Data : null,
                tensors.TryGetValue($"{prefix}.shortcut.normalization.bias", out var sb) ? sb.Data : null,
                tensors.TryGetValue($"{prefix}.shortcut.normalization.running_mean", out var sm) ? sm.Data : null,
                tensors.TryGetValue($"{prefix}.shortcut.normalization.running_var", out var sv) ? sv.Data : null);
        }
    }

    public static ResNet18 LoadWeights(Dictionary<string, (float[] Data, int[] Shape)> tensors, int numClasses = 1000)
    {
        var model = new ResNet18(numClasses);
        model.Eval();

        LoadConv(model.stemConv,
            tensors["resnet.embedder.embedder.convolution.weight"].Data,
            tensors["resnet.embedder.embedder.convolution.weight"].Shape);
        LoadBn(model.stemBn,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.weight", out var sw0) ? sw0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.bias", out var sb0) ? sb0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_mean", out var sm0) ? sm0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_var", out var sv0) ? sv0.Data : null);

        LoadBasicBlock(model.stage0Layer0, "resnet.encoder.stages.0.layers.0", tensors);
        LoadBasicBlock(model.stage0Layer1, "resnet.encoder.stages.0.layers.1", tensors);
        LoadBasicBlock(model.stage1Layer0, "resnet.encoder.stages.1.layers.0", tensors);
        LoadBasicBlock(model.stage1Layer1, "resnet.encoder.stages.1.layers.1", tensors);
        LoadBasicBlock(model.stage2Layer0, "resnet.encoder.stages.2.layers.0", tensors);
        LoadBasicBlock(model.stage2Layer1, "resnet.encoder.stages.2.layers.1", tensors);
        LoadBasicBlock(model.stage3Layer0, "resnet.encoder.stages.3.layers.0", tensors);
        LoadBasicBlock(model.stage3Layer1, "resnet.encoder.stages.3.layers.1", tensors);

        LoadLinear(model.fc,
            tensors["classifier.1.weight"].Data,
            tensors["classifier.1.weight"].Shape,
            tensors.TryGetValue("classifier.1.bias", out var bias) ? bias.Data : null);

        return model;
    }
}
