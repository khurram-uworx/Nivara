using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Standalone multi-head self-/cross-attention. Projects a 2D input <c>[L, D]</c> into query,
/// key, and value spaces, splits them into <c>numHeads</c> heads of size <c>D / numHeads</c>,
/// scales attention logits by <c>1 / sqrt(headDim)</c>, and re-projects the concatenated result.
/// Supports optional causal masking, padding masks, and attention dropout.
/// </summary>
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

    /// <summary>Gets the embedding dimension of the input/output.</summary>
    public int EmbedDim => embedDim;
    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => numHeads;
    /// <summary>Gets the per-head dimension (<c>embedDim / numHeads</c>).</summary>
    public int HeadDim => headDim;

    /// <summary>
    /// Creates a multi-head attention module.
    /// </summary>
    /// <param name="embedDim">Embedding dimension of the input/output (must be divisible by numHeads)</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="causal">Whether a causal (upper-triangular) mask is applied by default</param>
    /// <param name="dropout">Attention/output dropout probability</param>
    /// <param name="initStd">Standard deviation for the normal initialization of the Q/K/V projections</param>
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

    /// <summary>
    /// Runs self-attention over a 2D input <c>[L, D]</c> using the default causal setting.
    /// </summary>
    /// <param name="input">The input tensor (rank 2)</param>
    /// <returns>The attention output</returns>
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

    /// <summary>
    /// Runs self-attention over a 2D input <c>[L, D]</c>, masking positions where the padding
    /// mask is zero.
    /// </summary>
    /// <param name="input">The input tensor (rank 2)</param>
    /// <param name="paddingMask">Per-position mask of length <c>L</c>; zero entries are masked</param>
    /// <returns>The attention output</returns>
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

    /// <summary>
    /// Runs cross-attention with separate query, key, and value tensors, optionally overriding
    /// the causal setting and providing a padding mask.
    /// </summary>
    /// <param name="query">The query tensor (rank 2)</param>
    /// <param name="key">The key tensor (rank 2)</param>
    /// <param name="value">The value tensor (rank 2)</param>
    /// <param name="causal">Whether to apply a causal mask for this call</param>
    /// <param name="paddingMask">Per-key mask; zero entries are masked</param>
    /// <returns>The attention output</returns>
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
