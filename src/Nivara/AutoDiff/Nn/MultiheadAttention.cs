using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class MultiheadAttention<T> : Module<T>, IMultipleInputModule<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int embedDim;
    readonly int numHeads;
    readonly int headDim;
    readonly T attnScale;
    readonly bool causal;

    readonly Linear<T> qProj;
    readonly Linear<T> kProj;
    readonly Linear<T> vProj;
    readonly Linear<T> oProj;

    readonly Dropout<T>? attnDropout;

    public int EmbedDim => embedDim;
    public int NumHeads => numHeads;
    public int HeadDim => headDim;

    public MultiheadAttention(
        int embedDim,
        int numHeads,
        bool causal = false,
        double dropout = 0.0,
        double initStd = 0.02)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads}).");

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        headDim = embedDim / numHeads;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(headDim));
        this.causal = causal;

        var weightInit = new NormalInitializer<T>(T.Zero, T.CreateChecked(initStd));
        var outInit = new NormalInitializer<T>(T.Zero, T.Zero);

        qProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        kProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        vProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: weightInit);
        oProj = new Linear<T>(embedDim, embedDim, bias: false, weightInitializer: outInit);

        RegisterModules(qProj, kProj, vProj, oProj);

        if (dropout > 0.0)
        {
            attnDropout = new Dropout<T>(dropout);
            RegisterModules(attnDropout);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"MultiheadAttention expects 2D input [L, D], got {input.Rank}D");

        int L = input.shape[0];
        int D = input.shape[1];

        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        var xAttn = ComputeAttention(Q, K, V, L);
        var xProj = oProj.Forward(xAttn);

        return attnDropout != null ? attnDropout.Forward(xProj) : xProj;
    }

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T> paddingMask)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"MultiheadAttention expects 2D input [L, D], got {input.Rank}D");

        int L = input.shape[0];

        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        var xAttn = ComputeAttention(Q, K, V, L, paddingMask: paddingMask);
        var xProj = oProj.Forward(xAttn);

        return attnDropout != null ? attnDropout.Forward(xProj) : xProj;
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

        var Q = qProj.Forward(query);
        var K = kProj.Forward(key);
        var V = vProj.Forward(value);

        var xAttn = ComputeAttention(Q, K, V, L, causal, paddingMask);
        var xProj = oProj.Forward(xAttn);

        return attnDropout != null ? attnDropout.Forward(xProj) : xProj;
    }

    ReverseGradTensor<T> ComputeAttention(
        ReverseGradTensor<T> Q,
        ReverseGradTensor<T> K,
        ReverseGradTensor<T> V,
        int qLen,
        bool? overrideCausal = null,
        ReverseGradTensor<T>? paddingMask = null)
    {
        bool useCausal = overrideCausal ?? causal;
        int kvLen = K.shape[0];

        ReverseGradTensor<T>? mask = null;
        if (useCausal)
            mask = ModuleHelpers<T>.CreateCausalMask(qLen, kvLen);
        else if (paddingMask != null)
            mask = CreatePaddingMask(paddingMask, qLen, kvLen);

        return ReverseGradOperations.MultiHeadAttention(Q, K, V, numHeads, attnScale, mask);
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
