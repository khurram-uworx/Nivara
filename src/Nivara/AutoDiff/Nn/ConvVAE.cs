using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class ConvVAE<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv2d<T>[] encoderConvs;
    readonly Conv2d<T> muConv;
    readonly Conv2d<T> logVarConv;
    readonly ConvTranspose2d<T>[] decoderConvs;
    readonly ConvTranspose2d<T> reconConv;
    readonly int latentChannels;
    readonly int spatialSize;

    public int LatentChannels => latentChannels;
    public int SpatialSize => spatialSize;

    public ConvVAE(
        int inputChannels,
        int[] encoderChannels,
        int latentChannels,
        int spatialSize,
        int kernelSize = 3,
        int stride = 2,
        int padding = 1)
    {
        if (encoderChannels == null || encoderChannels.Length == 0)
            throw new ArgumentException("Encoder channels must have at least one entry.", nameof(encoderChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (spatialSize <= 0) throw new ArgumentOutOfRangeException(nameof(spatialSize));

        this.latentChannels = latentChannels;
        this.spatialSize = spatialSize;

        encoderConvs = new Conv2d<T>[encoderChannels.Length];
        for (int i = 0; i < encoderChannels.Length; i++)
        {
            int inC = i == 0 ? inputChannels : encoderChannels[i - 1];
            encoderConvs[i] = new Conv2d<T>(inC, encoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        int lastEncCh = encoderChannels[^1];
        muConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);
        logVarConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);

        int decoderSteps = encoderChannels.Length;
        var decoderChannels = new int[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
            decoderChannels[i] = encoderChannels[decoderSteps - 1 - i];

        decoderConvs = new ConvTranspose2d<T>[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
        {
            int inC = i == 0 ? latentChannels : decoderChannels[i - 1];
            decoderConvs[i] = new ConvTranspose2d<T>(inC, decoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        reconConv = new ConvTranspose2d<T>(decoderChannels[^1], inputChannels, 1, bias: true);

        RegisterModules(encoderConvs);
        RegisterModules(muConv, logVarConv);
        RegisterModules(decoderConvs);
        RegisterModules(reconConv);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var (mu, logVar) = Encode(input);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> input)
    {
        var h = input;
        for (int i = 0; i < encoderConvs.Length; i++)
        {
            h = encoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        var mu = muConv.Forward(h);
        var logVar = logVarConv.Forward(h);
        return (mu, logVar);
    }

    public ReverseGradTensor<T> Reparameterize(ReverseGradTensor<T> mu, ReverseGradTensor<T> logVar, int? seed = null)
        => ModuleHelpers<T>.Reparameterize(mu, logVar, IsTraining, seed);

    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        var h = z;
        for (int i = 0; i < decoderConvs.Length; i++)
        {
            h = decoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        return reconConv.Forward(h);
    }

    public ReverseGradTensor<T> ElboLoss(
        ReverseGradTensor<T> recon,
        ReverseGradTensor<T> original,
        ReverseGradTensor<T> mu,
        ReverseGradTensor<T> logVar)
    {
        var diff = ReverseGradOperations.Subtract(recon, original);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        var reconLoss = ReverseGradOperations.Sum(squared);

        var kl = ReverseGradOperations.KlDivergence(mu, logVar);

        return ReverseGradOperations.Add(reconLoss, kl);
    }
}
