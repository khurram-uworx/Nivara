using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class MultiheadAttention<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int _embedDim;
    readonly int _numHeads;
    readonly int _headDim;
    readonly T _attnScale;
    readonly bool _causal;

    readonly Linear<T> _qProj;
    readonly Linear<T> _kProj;
    readonly Linear<T> _vProj;
    readonly Linear<T> _oProj;

    readonly Dropout<T>? _attnDropout;

    public int EmbedDim => _embedDim;
    public int NumHeads => _numHeads;
    public int HeadDim => _headDim;

    public MultiheadAttention(
        int embedDim,
        int numHeads,
        bool causal = false,
        double dropout = 0.0,
        double initStd = 0.02)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads}).");

        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _attnScale = T.CreateChecked(1.0 / Math.Sqrt(_headDim));
        _causal = causal;

        var weightInit = new NormalInitializer<T>(T.Zero, T.CreateChecked(initStd));
        var outInit = new NormalInitializer<T>(T.Zero, T.Zero);

        _qProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        _kProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        _vProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        _oProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: outInit);

        RegisterModules(_qProj, _kProj, _vProj, _oProj);

        if (dropout > 0.0)
        {
            _attnDropout = new Dropout<T>(dropout);
            RegisterModules(_attnDropout);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"MultiheadAttention expects 2D input [L, D], got {input.Rank}D");

        int L = input.shape[0];
        int D = input.shape[1];

        var Q = _qProj.Forward(input);
        var K = _kProj.Forward(input);
        var V = _vProj.Forward(input);

        var xAttn = ComputeAttention(Q, K, V, L);
        var xProj = _oProj.Forward(xAttn);

        return _attnDropout != null ? _attnDropout.Forward(xProj) : xProj;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T> paddingMask)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"MultiheadAttention expects 2D input [L, D], got {input.Rank}D");

        int L = input.shape[0];

        var Q = _qProj.Forward(input);
        var K = _kProj.Forward(input);
        var V = _vProj.Forward(input);

        var xAttn = ComputeAttention(Q, K, V, L, paddingMask: paddingMask);
        var xProj = _oProj.Forward(xAttn);

        return _attnDropout != null ? _attnDropout.Forward(xProj) : xProj;
    }

    public ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> query,
        ReverseGradTensor<T> key,
        ReverseGradTensor<T> value,
        bool causal = false,
        ReverseGradTensor<T>? paddingMask = null)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));

        int L = query.shape[0];

        var Q = _qProj.Forward(query);
        var K = _kProj.Forward(key);
        var V = _vProj.Forward(value);

        var xAttn = ComputeAttention(Q, K, V, L, causal, paddingMask);
        var xProj = _oProj.Forward(xAttn);

        return _attnDropout != null ? _attnDropout.Forward(xProj) : xProj;
    }

    ReverseGradTensor<T> ComputeAttention(
        ReverseGradTensor<T> Q,
        ReverseGradTensor<T> K,
        ReverseGradTensor<T> V,
        int qLen,
        bool? overrideCausal = null,
        ReverseGradTensor<T>? paddingMask = null)
    {
        bool useCausal = overrideCausal ?? _causal;
        int kvLen = K.shape[0];

        ReverseGradTensor<T>? mask = null;
        if (useCausal)
            mask = ModuleHelpers<T>.CreateCausalMask(qLen, kvLen);
        else if (paddingMask != null)
            mask = CreatePaddingMask(paddingMask, qLen, kvLen);

        return ReverseGradOperations.MultiHeadAttention(Q, K, V, _numHeads, _attnScale, mask);
    }

    ReverseGradTensor<T> CreatePaddingMask(ReverseGradTensor<T> paddingMask, int qLen, int kvLen)
    {
        int maskLen = paddingMask.Length;
        var maskData = new T[qLen * kvLen];
        var negInf = T.CreateChecked(double.NegativeInfinity);
        for (int j = 0; j < maskLen; j++)
        {
            if (paddingMask.Data[j] == T.Zero)
            {
                for (int i = 0; i < qLen; i++)
                    maskData[i * kvLen + j] = negInf;
            }
        }
        var col = NivaraColumn<T>.Create(maskData);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(qLen, kvLen);
        return tensor;
    }
}
