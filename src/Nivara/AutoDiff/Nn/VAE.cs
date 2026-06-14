using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class VAE<T> : Module<T> where T : struct, INumber<T>
{
    readonly Linear<T> _encoderLayer1;
    readonly Linear<T> _encoderLayer2;
    readonly Linear<T> _muHead;
    readonly Linear<T> _logVarHead;
    readonly Linear<T> _decoderLayer1;
    readonly Linear<T> _decoderLayer2;
    readonly Parameter<T> _beta;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>> _activation;

    public VAE(
        int inputDim,
        int latentDim,
        int hiddenDim,
        int? decoderHiddenDim = null,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>>? activation = null,
        float beta = 1.0f)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (latentDim <= 0) throw new ArgumentOutOfRangeException(nameof(latentDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));

        _encoderLayer1 = new Linear<T>(inputDim, hiddenDim);
        _encoderLayer2 = new Linear<T>(hiddenDim, hiddenDim);
        _muHead = new Linear<T>(hiddenDim, latentDim);
        _logVarHead = new Linear<T>(hiddenDim, latentDim);

        var decHidden = decoderHiddenDim ?? hiddenDim;
        _decoderLayer1 = new Linear<T>(latentDim, decHidden);
        _decoderLayer2 = new Linear<T>(decHidden, inputDim);

        var betaData = new T[] { T.CreateChecked(beta) };
        _beta = new Parameter<T>("Beta", betaData, requiresGrad: false);

        _activation = activation ?? (x => Activation.Relu(x));

        RegisterModules(
            _encoderLayer1, _encoderLayer2,
            _muHead, _logVarHead,
            _decoderLayer1, _decoderLayer2);
        RegisterParameters(_beta);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> x)
    {
        var (mu, logVar) = Encode(x);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> x)
    {
        var h = _activation(_encoderLayer1.Forward(x));
        h = _activation(_encoderLayer2.Forward(h));
        var mu = _muHead.Forward(h);
        var logVar = _logVarHead.Forward(h);
        return (mu, logVar);
    }

    public ReverseGradTensor<T> Reparameterize(ReverseGradTensor<T> mu, ReverseGradTensor<T> logVar, int? seed = null)
    {
        if (!IsTraining)
            return mu;
        return ReverseGradOperations.SampleNormal(mu, logVar, seed);
    }

    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        var h = _activation(_decoderLayer1.Forward(z));
        return _decoderLayer2.Forward(h);
    }

    public ReverseGradTensor<T> ElboLoss(
        ReverseGradTensor<T> recon,
        ReverseGradTensor<T> original,
        ReverseGradTensor<T> mu,
        ReverseGradTensor<T> logVar,
        ElboLossType lossType = ElboLossType.KldBeta)
    {
        var diff = ReverseGradOperations.Subtract(recon, original);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        var reconLoss = ReverseGradOperations.Sum(squared);

        var kl = ReverseGradOperations.KlDivergence(mu, logVar);

        var betaResult = lossType switch
        {
            ElboLossType.KldBeta => ReverseGradOperations.Multiply(kl, _beta.Tensor),
            ElboLossType.KldAnnealing => kl,
            _ => throw new ArgumentException($"Unknown ElboLossType: {lossType}", nameof(lossType))
        };

        return ReverseGradOperations.Add(reconLoss, betaResult);
    }
}
