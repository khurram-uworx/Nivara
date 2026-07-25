using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class VAE<T> : Module<T> where T : struct, INumber<T>
{
    readonly Linear<T> encoderLayer1;
    readonly Linear<T> encoderLayer2;
    readonly Linear<T> muHead;
    readonly Linear<T> logVarHead;
    readonly Linear<T> decoderLayer1;
    readonly Linear<T> decoderLayer2;
    readonly Parameter<T> beta;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>> activation;
    readonly int inputDim;
    readonly int latentDim;
    readonly int hiddenDim;
    readonly int conditionDim;

    public VAE(
        int inputDim,
        int latentDim,
        int hiddenDim,
        int? decoderHiddenDim = null,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>>? activation = null,
        float beta = 1.0f,
        int conditionDim = 0)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (latentDim <= 0) throw new ArgumentOutOfRangeException(nameof(latentDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (conditionDim < 0) throw new ArgumentOutOfRangeException(nameof(conditionDim));

        this.inputDim = inputDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;
        this.conditionDim = conditionDim;

        var encoderInputDim = inputDim + conditionDim;
        var decoderInputDim = latentDim + conditionDim;

        encoderLayer1 = new Linear<T>(encoderInputDim, hiddenDim);
        encoderLayer2 = new Linear<T>(hiddenDim, hiddenDim);
        muHead = new Linear<T>(hiddenDim, latentDim);
        logVarHead = new Linear<T>(hiddenDim, latentDim);

        var decHidden = decoderHiddenDim ?? hiddenDim;
        decoderLayer1 = new Linear<T>(decoderInputDim, decHidden);
        decoderLayer2 = new Linear<T>(decHidden, inputDim);

        var betaData = new T[] { T.CreateChecked(beta) };
        this.beta = new Parameter<T>("Beta", betaData, requiresGrad: false);

        this.activation = activation ?? (x => Activation.Relu(x));

        RegisterModules(
            encoderLayer1, encoderLayer2,
            muHead, logVarHead,
            decoderLayer1, decoderLayer2);
        RegisterParameters(this.beta);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> x)
    {
        var (mu, logVar) = Encode(x);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    public new ReverseGradTensor<T> Forward(ReverseGradTensor<T> x, ReverseGradTensor<T>? condition)
    {
        var (mu, logVar) = Encode(x, condition);
        var z = Reparameterize(mu, logVar);
        return Decode(z, condition);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> x)
    {
        return Encode(x, condition: null);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(
        ReverseGradTensor<T> x,
        ReverseGradTensor<T>? condition = null)
    {
        var input = condition != null
            ? ReverseGradOperations.Concat(new[] { x, condition }, axis: 1)
            : x;

        var h = activation(encoderLayer1.Forward(input));
        h = activation(encoderLayer2.Forward(h));
        var mu = muHead.Forward(h);
        var logVar = logVarHead.Forward(h);
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
        return Decode(z, condition: null);
    }

    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z, ReverseGradTensor<T>? condition = null)
    {
        var input = condition != null
            ? ReverseGradOperations.Concat(new[] { z, condition }, axis: 1)
            : z;

        var h = activation(decoderLayer1.Forward(input));
        return decoderLayer2.Forward(h);
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
            ElboLossType.KldBeta => ReverseGradOperations.Multiply(kl, beta.Tensor),
            ElboLossType.KldAnnealing => kl,
            _ => throw new ArgumentException($"Unknown ElboLossType: {lossType}", nameof(lossType))
        };

        return ReverseGradOperations.Add(reconLoss, betaResult);
    }
}
