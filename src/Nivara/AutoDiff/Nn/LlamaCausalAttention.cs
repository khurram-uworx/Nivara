using Nivara.AutoDiff.Operations;
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

    /// <summary>
    /// Runs Llama causal self-attention for a <em>single new token</em> during cached inference.
    /// Computes Q/K/V for the one position, applies RoPE at its absolute position
    /// (<paramref name="positionOffset"/>), appends the per-KV-head K/V into
    /// <paramref name="kCache"/>/<paramref name="vCache"/>, and attends the new query against the
    /// full cached prefix. The cache holds all positions seen so far (inclusive of the new token),
    /// with the parent incrementing <paramref name="cacheLen"/> across calls. This mirrors
    /// <see cref="Forward(ReverseGradTensor{T})"/> numerically but avoids re-running projections
    /// over the whole prefix. Inference-only (no graph nodes built).
    /// </summary>
    /// <param name="input">The new-token hidden state <c>[1, hiddenSize]</c></param>
    /// <param name="positionOffset">Absolute position of the new token</param>
    /// <param name="kCache">Buffer of RoPE'd per-KV-head keys, row-major <c>[kvLen, numKeyValueHeads * headDim]</c></param>
    /// <param name="vCache">Buffer of per-KV-head values, row-major <c>[kvLen, numKeyValueHeads * headDim]</c></param>
    /// <param name="cacheLen">Number of tokens already cached before this call</param>
    /// <returns>The attention output <c>[1, hiddenSize]</c> after the output projection</returns>
    public ReverseGradTensor<T> ForwardCached(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[] kCache,
        T[] vCache,
        int cacheLen)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Shape[0] != 1) throw new ArgumentException($"ForwardCached expects a single-token input [1, D], got [.., {input.Shape[0]}].", nameof(input));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (cacheLen < 0) throw new ArgumentOutOfRangeException(nameof(cacheLen));

        int kvWidth = numKeyValueHeads * headDim;
        int newLen = cacheLen + 1;
        int needed = newLen * kvWidth;
        if (kCache.Length < needed || vCache.Length < needed)
            throw new ArgumentException("Cache buffers must have capacity for the new token row.");

        var Q = QProj.Forward(input);                     // [1, numHeads * headDim]
        var K = KProj.Forward(input);                     // [1, numKeyValueHeads * headDim]
        var V = VProj.Forward(input);                     // [1, numKeyValueHeads * headDim]

        Q = rotary.Forward(Q, positionOffset);
        K = rotary.Forward(K, positionOffset);

        K.AsSpan().CopyTo(kCache.AsSpan(cacheLen * kvWidth, kvWidth));
        V.AsSpan().CopyTo(vCache.AsSpan(cacheLen * kvWidth, kvWidth));

        // Build exact-size per-KV-head K/V matrices from the used cache region.
        var kData = new T[needed];
        var vData = new T[needed];
        Buffer.BlockCopy(kCache, 0, kData, 0, needed * System.Runtime.CompilerServices.Unsafe.SizeOf<T>());
        Buffer.BlockCopy(vCache, 0, vData, 0, needed * System.Runtime.CompilerServices.Unsafe.SizeOf<T>());

        var kCol = NivaraColumn<T>.CreateFromOwnedArray(kData);
        var vCol = NivaraColumn<T>.CreateFromOwnedArray(vData);
        var kTensor = new ReverseGradTensor<T>(kCol, requiresGrad: false);
        var vTensor = new ReverseGradTensor<T>(vCol, requiresGrad: false);
        kTensor.Reshape(newLen, kvWidth);
        vTensor.Reshape(newLen, kvWidth);

        // GQA: repeat KV heads to the query head count across the full prefix.
        var KFull = ReverseGradOperations.GqaRepeatKV(kTensor, numHeads, numKeyValueHeads);
        var VFull = ReverseGradOperations.GqaRepeatKV(vTensor, numHeads, numKeyValueHeads);

        // Fully-open mask: the new token attends to every cached position.
        var openMaskData = new T[newLen];
        var maskCol = NivaraColumn<T>.CreateFromOwnedArray(openMaskData);
        var openMask = new ReverseGradTensor<T>(maskCol, requiresGrad: false);
        openMask.Reshape(1, newLen);

        var attn = ReverseGradOperations.MultiHeadAttention(Q, KFull, VFull, numHeads, attnScale, openMask);
        return OProj.Forward(attn);
    }
}
