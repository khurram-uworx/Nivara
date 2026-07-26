using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class MultiheadAttention<T> : Module<T> where T : struct, INumber<T>
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

    public ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> query,
        ReverseGradTensor<T> key,
        ReverseGradTensor<T> value,
        bool causal = false)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));

        int L = query.shape[0];

        var Q = _qProj.Forward(query);
        var K = _kProj.Forward(key);
        var V = _vProj.Forward(value);

        var xAttn = ComputeAttention(Q, K, V, L, causal);
        var xProj = _oProj.Forward(xAttn);

        return _attnDropout != null ? _attnDropout.Forward(xProj) : xProj;
    }

    ReverseGradTensor<T> ComputeAttention(
        ReverseGradTensor<T> Q,
        ReverseGradTensor<T> K,
        ReverseGradTensor<T> V,
        int qLen,
        bool? overrideCausal = null)
    {
        bool useCausal = overrideCausal ?? _causal;
        int kvLen = K.shape[0];
        var heads = new ReverseGradTensor<T>[_numHeads];

        var scaleTensor = GradientUtils.Full(qLen * kvLen, _attnScale);
        scaleTensor.Reshape(qLen, kvLen);

        ReverseGradTensor<T>? mask = null;
        if (useCausal)
            mask = CreateCausalMask(qLen);

        for (int h = 0; h < _numHeads; h++)
        {
            int hs = h * _headDim;

            var Q_h = ReverseGradOperations.Slice(Q, hs, _headDim);
            var K_h = ReverseGradOperations.Slice(K, hs, _headDim);
            var V_h = ReverseGradOperations.Slice(V, hs, _headDim);

            var K_h_T = ReverseGradOperations.Transpose(K_h);
            var scores = ReverseGradOperations.MatMul(Q_h, K_h_T);

            scores = ReverseGradOperations.Multiply(scores, scaleTensor);

            if (mask != null)
                scores = ReverseGradOperations.Add(scores, mask);

            var weights = ReverseGradOperations.Softmax(scores);

            heads[h] = ReverseGradOperations.MatMul(weights, V_h);
        }

        return ReverseGradOperations.Concat(heads, axis: 1);
    }

    ReverseGradTensor<T> CreateCausalMask(int L)
    {
        var maskData = new T[L * L];
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (j > i)
                    maskData[i * L + j] = T.CreateChecked(double.NegativeInfinity);

        var col = NivaraColumn<T>.Create(maskData);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(L, L);
        return tensor;
    }
}
