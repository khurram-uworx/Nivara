using System.Numerics;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace NivaraVAE;

public sealed class VaeModel<T> : Module<T> where T : struct, INumber<T>
{
    readonly Linear<T> _enc1;
    readonly Linear<T> _enc2;
    readonly Linear<T> _muHead;
    readonly Linear<T> _logVarHead;
    readonly Linear<T> _dec1;
    readonly Linear<T> _dec2;
    readonly Dropout<T> _drop1;
    readonly Dropout<T> _drop2;
    readonly Dropout<T> _drop3;
    readonly Dropout<T> _drop4;
    readonly int _latentDim;

    public VaeModel(int inputDim, int hiddenDim, int latentDim, float dropout = 0.2f)
    {
        _latentDim = latentDim;

        _enc1 = new Linear<T>(inputDim, hiddenDim);
        _enc2 = new Linear<T>(hiddenDim, hiddenDim);
        _muHead = new Linear<T>(hiddenDim, latentDim);
        _logVarHead = new Linear<T>(hiddenDim, latentDim);
        _dec1 = new Linear<T>(latentDim, hiddenDim);
        _dec2 = new Linear<T>(hiddenDim, inputDim);
        _drop1 = new Dropout<T>(dropout);
        _drop2 = new Dropout<T>(dropout);
        _drop3 = new Dropout<T>(dropout);
        _drop4 = new Dropout<T>(dropout);

        RegisterModules(
            _enc1, _enc2, _muHead, _logVarHead,
            _dec1, _dec2,
            _drop1, _drop2, _drop3, _drop4);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var (mu, logVar) = Encode(input);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> input)
    {
        var h = Activation.LeakyRelu(_enc1.Forward(input));
        h = _drop1.Forward(h);
        h = Activation.LeakyRelu(_enc2.Forward(h));
        h = _drop2.Forward(h);
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
        var h = Activation.LeakyRelu(_dec1.Forward(z));
        h = _drop3.Forward(h);
        h = _dec2.Forward(h);
        return h;
    }
}
