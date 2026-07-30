using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace NivaraTimeSeries;

public sealed class TimeSeriesModel<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv1d<T> _conv1;
    readonly Conv1d<T> _conv2;
    readonly Conv1d<T> _conv3;
    readonly BatchNorm1d<T> _bn1;
    readonly BatchNorm1d<T> _bn2;
    readonly BatchNorm1d<T> _bn3;
    readonly Linear<T> _muHead;
    readonly Linear<T> _logVarHead;
    readonly Linear<T> _dec1;
    readonly Linear<T> _dec2;
    readonly Linear<T> _dec3;
    readonly Dropout<T> _drop1;
    readonly Dropout<T> _drop2;
    readonly int _windowSize;
    readonly int _latentDim;
    readonly int _convOutputSize;

    public TimeSeriesModel(
        int numChannels = 4,
        int windowSize = 64,
        int latentDim = 16,
        int hiddenDim = 128,
        float dropout = 0.2f)
    {
        _windowSize = windowSize;
        _latentDim = latentDim;

        _conv1 = new Conv1d<T>(numChannels, 32, kernelSize: 7, padding: 3);
        _bn1 = new BatchNorm1d<T>(32);
        _conv2 = new Conv1d<T>(32, 64, kernelSize: 5, padding: 2);
        _bn2 = new BatchNorm1d<T>(64);
        _conv3 = new Conv1d<T>(64, 32, kernelSize: 3, padding: 1);
        _bn3 = new BatchNorm1d<T>(32);

        _convOutputSize = 32 * windowSize;

        _muHead = new Linear<T>(_convOutputSize, latentDim);
        _logVarHead = new Linear<T>(_convOutputSize, latentDim);

        _dec1 = new Linear<T>(latentDim, hiddenDim);
        _dec2 = new Linear<T>(hiddenDim, hiddenDim);
        _dec3 = new Linear<T>(hiddenDim, numChannels * windowSize);

        _drop1 = new Dropout<T>(dropout);
        _drop2 = new Dropout<T>(dropout);

        RegisterModules(_conv1, _bn1, _conv2, _bn2, _conv3, _bn3,
            _muHead, _logVarHead, _dec1, _dec2, _dec3, _drop1, _drop2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var (mu, logVar) = Encode(input);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> input)
    {
        var x = _conv1.Forward(input);
        x = _bn1.Forward(x);
        x = ReverseGradOperations.LeakyRelu(x, T.CreateChecked(0.01f));

        x = _conv2.Forward(x);
        x = _bn2.Forward(x);
        x = ReverseGradOperations.LeakyRelu(x, T.CreateChecked(0.01f));

        x = _conv3.Forward(x);
        x = _bn3.Forward(x);
        x = ReverseGradOperations.LeakyRelu(x, T.CreateChecked(0.01f));

        int B = x.Shape[0];
        x.Reshape(B, _convOutputSize);

        var mu = _muHead.Forward(x);
        var logVar = _logVarHead.Forward(x);
        return (mu, logVar);
    }

    public ReverseGradTensor<T> Reparameterize(ReverseGradTensor<T> mu, ReverseGradTensor<T> logVar, int? seed = null)
    {
        if (!IsTraining) return mu;
        return ReverseGradOperations.SampleNormal(mu, logVar, seed);
    }

    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        var x = _dec1.Forward(z);
        x = ReverseGradOperations.LeakyRelu(x, T.CreateChecked(0.01f));
        x = _drop1.Forward(x);

        x = _dec2.Forward(x);
        x = ReverseGradOperations.LeakyRelu(x, T.CreateChecked(0.01f));
        x = _drop2.Forward(x);

        x = _dec3.Forward(x);
        return x;
    }

    public float ReconstructError(ReverseGradTensor<T> input)
    {
        var recon = Forward(input);
        float sumSq = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float diff = float.CreateChecked(recon[i] - input[i]);
            sumSq += diff * diff;
        }
        return sumSq / input.Length;
    }
}
