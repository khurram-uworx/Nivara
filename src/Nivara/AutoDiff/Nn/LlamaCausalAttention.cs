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
    /// <summary>Gets the rotary position embedding tables (fused-kernel path).</summary>
    internal RotaryEmbedding<T> Rotary => rotary;

    /// <summary>
    /// Creates a Llama causal self-attention module.
    /// </summary>
    /// <param name="hiddenSize">Hidden (embedding) dimension</param>
    /// <param name="numHeads">Number of query heads</param>
    /// <param name="numKeyValueHeads">Number of key/value heads (must divide numHeads)</param>
    /// <param name="maxPositionEmbeddings">Maximum position for RoPE tables</param>
    /// <param name="ropeTheta">RoPE inverse-frequency base</param>
    /// <param name="qkvBias">Whether the query/key/value projections carry a bias vector
    /// (Qwen2-style models do; canonical Llama does not). Backlog #384: public-docs coverage
    /// for this option.</param>
    public LlamaCausalAttention(
        int hiddenSize,
        int numHeads,
        int numKeyValueHeads,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f,
        bool qkvBias = false)
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

        QProj = new Linear<T>(hiddenSize, numHeads * headDim, bias: qkvBias);
        KProj = new Linear<T>(hiddenSize, numKeyValueHeads * headDim, bias: qkvBias);
        VProj = new Linear<T>(hiddenSize, numKeyValueHeads * headDim, bias: qkvBias);
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

        return ForwardCore(input, 0, null, null);
    }

    /// <summary>
    /// Runs Llama causal self-attention over a batched <c>[L, hiddenSize]</c> prompt during
    /// cached-inference prefill, capturing the per-KV-head key/value rows into
    /// <paramref name="kCache"/>/<paramref name="vCache"/> exactly as the per-token
    /// <see cref="ForwardCached"/> walk would. Computes Q/K/V for all L positions at once,
    /// applies RoPE at absolute positions <c>[positionOffset, positionOffset + L)</c>, copies the
    /// RoPE'd pre-repeat K/V (<c>[L, kvWidth]</c>, the cache's row-major per-KV-head layout) into
    /// cache rows <c>[positionOffset, positionOffset + L)</c>, then runs the same GQA repeat,
    /// causal attention, and output projection as <see cref="Forward(ReverseGradTensor{T})"/>.
    /// One batched pass reads the weights once instead of L per-token walks. Inference and
    /// grad-enabled paths follow <see cref="Forward(ReverseGradTensor{T})"/> exactly (the capture
    /// is a plain array write; no graph node is built for it).
    /// </summary>
    /// <param name="input">The prompt hidden states <c>[L, hiddenSize]</c></param>
    /// <param name="positionOffset">Absolute position of the first prompt row (0 for a fresh prefill)</param>
    /// <param name="kCache">Per-layer RoPE'd key cache, row-major <c>[kvLen, numKeyValueHeads * headDim]</c></param>
    /// <param name="vCache">Per-layer value cache, row-major <c>[kvLen, numKeyValueHeads * headDim]</c></param>
    /// <returns>The attention output with shape <c>[L, hiddenSize]</c></returns>
    public ReverseGradTensor<T> ForwardPrefill(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[] kCache,
        T[] vCache)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"LlamaCausalAttention expects 2D input [L, D], got {input.Rank}D");
        if (input.shape[1] != hiddenSize)
            throw new ArgumentException($"Expected input width {hiddenSize}, got {input.shape[1]}.");
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));

        int kvWidth = numKeyValueHeads * headDim;
        int needed = (positionOffset + input.shape[0]) * kvWidth;
        if (kCache.Length < needed || vCache.Length < needed)
            throw new ArgumentException($"Cache buffers must have capacity for {(positionOffset + input.shape[0])} rows of width {kvWidth}.");

        return ForwardCore(input, positionOffset, kCache, vCache);
    }

    ReverseGradTensor<T> ForwardCore(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[]? kCache,
        T[]? vCache)
    {
        int qLen = input.shape[0];

        var Q = QProj.Forward(input);
        var K = KProj.Forward(input);
        var V = VProj.Forward(input);

        // Apply RoPE before splitting/repeating (batched with an absolute offset; row p is
        // rotated by positionOffset + p — the same per-row math as ForwardCached).
        Q = rotary.Forward(Q, positionOffset);
        K = rotary.Forward(K, positionOffset);

        // Prefill: capture the pre-repeat per-KV-head K/V into the cache rows the fused
        // ports read (row-major [kvLen, numKeyValueHeads * headDim]), then — on the
        // inference-only path — attend this chunk's rows with the virtual GQA mapping:
        // no GqaRepeatKV expansion, no head repack, no [numHeads, L, L] score block, no
        // mask allocation, no graph nodes (exactly the contract DecodeAttention established
        // on the decode side).
        if (kCache is not null && vCache is not null)
        {
            int kvWidth = numKeyValueHeads * headDim;
            int rowBytes = qLen * kvWidth;
            K.AsSpan().CopyTo(kCache.AsSpan(positionOffset * kvWidth, rowBytes));
            V.AsSpan().CopyTo(vCache.AsSpan(positionOffset * kvWidth, rowBytes));

            if (!GradientUtils.IsGradEnabled)
            {
                var outData = new T[qLen * numHeads * headDim];
                AttentionKernels<T>.BatchedAttention(
                    Q.AsSpan(),
                    kCache.AsSpan(positionOffset * kvWidth, rowBytes),
                    vCache.AsSpan(positionOffset * kvWidth, rowBytes),
                    outData, qLen, numHeads, numKeyValueHeads, headDim, attnScale);
                return OProj.Forward(new ReverseGradTensor<T>(
                    NivaraColumn<T>.CreateFromOwnedArray(outData), requiresGrad: false,
                    new[] { qLen, numHeads * headDim }));
            }
        }

        // Grad-enabled (or cache-less full Forward) path — GQA repeat + causal mask + MHA
        // exactly as before so backward flows through the same graph nodes.
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

        ReverseGradTensor<T> attn;
        if (!GradientUtils.IsGradEnabled)
        {
            // Inference-only fused path: attend the single query row against the cached KV
            // prefix with virtual GQA head mapping — zero-copy cache reads (no BlockCopy), no
            // GqaRepeatKV expansion across the prefix, no head repack, no mask allocation.
            var outData = new T[numHeads * headDim];
            AttentionKernels<T>.DecodeAttention(
                Q.AsSpan(), kCache.AsSpan(0, needed), vCache.AsSpan(0, needed),
                outData, newLen, numHeads, numKeyValueHeads, headDim, attnScale);
            attn = new ReverseGradTensor<T>(
                NivaraColumn<T>.CreateFromOwnedArray(outData), requiresGrad: false, new[] { 1, numHeads * headDim });
        }
        else
        {
            // Grad-enabled fallback: keep the exact multi-step path so backward flows through
            // the same graph nodes as before (the fused path is inference-only).
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

            attn = ReverseGradOperations.MultiHeadAttention(Q, KFull, VFull, numHeads, attnScale, openMask);
        }

        return OProj.Forward(attn);
    }
}
