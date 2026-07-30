using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class ConvVAE<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv2d<T>[] _encoderConvs;
    readonly Conv2d<T> _muConv;
    readonly Conv2d<T> _logVarConv;
    readonly ConvTranspose2d<T>[] _decoderConvs;
    readonly ConvTranspose2d<T> _reconConv;
    readonly int _latentChannels;
    readonly int _spatialSize;

    public int LatentChannels => _latentChannels;
    public int SpatialSize => _spatialSize;

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

        _latentChannels = latentChannels;
        _spatialSize = spatialSize;

        _encoderConvs = new Conv2d<T>[encoderChannels.Length];
        for (int i = 0; i < encoderChannels.Length; i++)
        {
            int inC = i == 0 ? inputChannels : encoderChannels[i - 1];
            _encoderConvs[i] = new Conv2d<T>(inC, encoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        int lastEncCh = encoderChannels[^1];
        _muConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);
        _logVarConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);

        int decoderSteps = encoderChannels.Length;
        var decoderChannels = new int[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
            decoderChannels[i] = encoderChannels[decoderSteps - 1 - i];

        _decoderConvs = new ConvTranspose2d<T>[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
        {
            int inC = i == 0 ? latentChannels : decoderChannels[i - 1];
            _decoderConvs[i] = new ConvTranspose2d<T>(inC, decoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        _reconConv = new ConvTranspose2d<T>(decoderChannels[^1], inputChannels, 1, bias: true);

        RegisterModules(_encoderConvs);
        RegisterModules(_muConv, _logVarConv);
        RegisterModules(_decoderConvs);
        RegisterModules(_reconConv);
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
        for (int i = 0; i < _encoderConvs.Length; i++)
        {
            h = _encoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        var mu = _muConv.Forward(h);
        var logVar = _logVarConv.Forward(h);
        return (mu, logVar);
    }

    public ReverseGradTensor<T> Reparameterize(ReverseGradTensor<T> mu, ReverseGradTensor<T> logVar, int? seed = null)
        => ModuleHelpers<T>.Reparameterize(mu, logVar, IsTraining, seed);

    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        var h = z;
        for (int i = 0; i < _decoderConvs.Length; i++)
        {
            h = _decoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        return _reconConv.Forward(h);
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
