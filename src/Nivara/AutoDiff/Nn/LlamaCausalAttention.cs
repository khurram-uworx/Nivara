using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Llama-family causal self-attention with grouped-query attention (GQA), rotary position
/// embeddings (RoPE), and a fused causal masked head loop. Query uses
/// <c>numHeads</c> heads and key/value share <c>numKeyValueHeads</c>; the 3 (or N) key/value
/// heads are repeated so all heads align on the shared-head attention path. This mirrors
/// the <c>LlamaAttention</c> structure used by SmolLM and the Llama family.
/// </summary>
public sealed class LlamaCausalAttention<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int hiddenSize;
    readonly int numHeads;
    readonly int numKeyValueHeads;
    readonly int headDim;

    /// <summary>Gets the query projection linear.</summary>
    public Linear<T> QProj { get; }
    /// <summary>Gets the key projection linear.</summary>
    public Linear<T> KProj { get; }
    /// <summary>Gets the value projection linear.</summary>
    public Linear<T> VProj { get; }
    /// <summary>Gets the output projection linear.</summary>
    public Linear<T> OProj { get; }
    readonly RotaryEmbedding<T> rotary;

    readonly T attnScale;

    /// <summary>Gets the hidden size (embedding dimension).</summary>
    public int HiddenSize => hiddenSize;
    /// <summary>Gets the number of query heads.</summary>
    public int NumHeads => numHeads;
    /// <summary>Gets the number of key/value heads (shared across K and V).</summary>
    public int NumKeyValueHeads => numKeyValueHeads;
    /// <summary>Gets the per-head dimension.</summary>
    public int HeadDim => headDim;

    /// <summary>
    /// Creates a Llama causal self-attention module.
    /// </summary>
    /// <param name="hiddenSize">Hidden (embedding) dimension</param>
    /// <param name="numHeads">Number of query heads</param>
    /// <param name="numKeyValueHeads">Number of key/value heads (must divide numHeads)</param>
    /// <param name="maxPositionEmbeddings">Maximum position for RoPE tables</param>
    /// <param name="ropeTheta">RoPE inverse-frequency base</param>
    public LlamaCausalAttention(
        int hiddenSize,
        int numHeads,
        int numKeyValueHeads,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKeyValueHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKeyValueHeads));
        if (numHeads % numKeyValueHeads != 0)
            throw new ArgumentException($"{nameof(numHeads)} ({numHeads}) must be divisible by {nameof(numKeyValueHeads)} ({numKeyValueHeads}).");
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException($"{nameof(hiddenSize)} ({hiddenSize}) must be divisible by {nameof(numHeads)} ({numHeads}).");

        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.numKeyValueHeads = numKeyValueHeads;
        headDim = hiddenSize / numHeads;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(headDim));

        QProj = new Linear<T>(hiddenSize, numHeads * headDim, bias: false);
        KProj = new Linear<T>(hiddenSize, numKeyValueHeads * headDim, bias: false);
        VProj = new Linear<T>(hiddenSize, numKeyValueHeads * headDim, bias: false);
        OProj = new Linear<T>(hiddenSize, hiddenSize, bias: false);
        rotary = new RotaryEmbedding<T>(headDim, maxPositionEmbeddings, ropeTheta);

        RegisterModules(QProj, KProj, VProj, OProj, rotary);
    }

    /// <summary>
    /// Runs Llama causal self-attention over a <c>[L, hiddenSize]</c> input, applying a
    /// causal (upper-triangular) mask.
    /// </summary>
    /// <param name="input">The input tensor (rank 2)</param>
    /// <returns>The attention output with shape <c>[L, hiddenSize]</c></returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"LlamaCausalAttention expects 2D input [L, D], got {input.Rank}D");
        if (input.shape[1] != hiddenSize)
            throw new ArgumentException($"Expected input width {hiddenSize}, got {input.shape[1]}.");

        int qLen = input.shape[0];

        var Q = QProj.Forward(input);
        var K = KProj.Forward(input);
        var V = VProj.Forward(input);

        // Apply RoPE before splitting/repeating.
        Q = rotary.Forward(Q);
        K = rotary.Forward(K);

        // GQA: repeat key/value heads to the query head count.
        K = ReverseGradOperations.GqaRepeatKV(K, numHeads, numKeyValueHeads);
        V = ReverseGradOperations.GqaRepeatKV(V, numHeads, numKeyValueHeads);

        var mask = ModuleHelpers<T>.CreateCausalMask(qLen, qLen);
        var attn = ReverseGradOperations.MultiHeadAttention(Q, K, V, numHeads, attnScale, mask);
        return OProj.Forward(attn);
    }
}
