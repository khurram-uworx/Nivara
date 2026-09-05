using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// A single Llama-family transformer decoder block: pre-norm self-attention with a residual
/// add, followed by a pre-norm gated SiLU feed-forward network with a second residual add.
/// This mirrors the <c>LlamaDecoderLayer</c> used by SmolLM and the Llama family. Output
/// shape equals the input shape <c>[L, hiddenSize]</c>.
/// </summary>
public sealed class LlamaDecoderBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int hiddenSize;

    /// <summary>Gets the pre-attention RMS norm.</summary>
    public RMSNorm<T> InputNorm { get; }
    /// <summary>Gets the attention module.</summary>
    public LlamaCausalAttention<T> Attention { get; }
    /// <summary>Gets the post-attention RMS norm.</summary>
    public RMSNorm<T> PostNorm { get; }
    /// <summary>Gets the SiLU gated-projection linear.</summary>
    public Linear<T> GateProj { get; }
    /// <summary>Gets the up-projection linear.</summary>
    public Linear<T> UpProj { get; }
    /// <summary>Gets the down-projection linear.</summary>
    public Linear<T> DownProj { get; }

    /// <summary>Gets the hidden (embedding) dimension.</summary>
    public int HiddenSize => hiddenSize;

    /// <summary>
    /// Creates a Llama decoder block.
    /// </summary>
    /// <param name="hiddenSize">Hidden (embedding) dimension</param>
    /// <param name="numHeads">Number of query attention heads</param>
    /// <param name="numKeyValueHeads">Number of key/value attention heads</param>
    /// <param name="intermediateSize">Feed-forward hidden size</param>
    /// <param name="rmsNormEps">RMS normalization stability term</param>
    /// <param name="maxPositionEmbeddings">Maximum position for RoPE tables</param>
    /// <param name="ropeTheta">RoPE inverse-frequency base</param>
    /// <param name="qkvBias">Whether the self-attention query/key/value projections carry a bias
    /// (Qwen2-style models; backlog #384).</param>
    public LlamaDecoderBlock(
        int hiddenSize,
        int numHeads,
        int numKeyValueHeads,
        int intermediateSize,
        float rmsNormEps = 1e-5f,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f,
        bool qkvBias = false)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (intermediateSize <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateSize));

        this.hiddenSize = hiddenSize;

        InputNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        Attention = new LlamaCausalAttention<T>(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta, qkvBias);
        PostNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        GateProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        UpProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        DownProj = new Linear<T>(intermediateSize, hiddenSize, bias: false);

        RegisterModules(InputNorm, Attention, PostNorm, GateProj, UpProj, DownProj);
    }

    /// <summary>
    /// Runs one Llama decoder block over a <c>[L, hiddenSize]</c> input.
    /// </summary>
    /// <param name="input">The input tensor (rank 2)</param>
    /// <returns>The block output with shape <c>[L, hiddenSize]</c></returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"LlamaDecoderBlock expects 2D input [L, D], got {input.Rank}D");
        if (input.shape[1] != hiddenSize)
            throw new ArgumentException($"Expected input width {hiddenSize}, got {input.shape[1]}.");

        // Pre-norm self-attention with residual add.
        var attnOut = Attention.Forward(InputNorm.Forward(input));
        var h = ReverseGradOperations.Add(input, attnOut);

        // Pre-norm gated SiLU feed-forward with residual add.
        var ffnIn = PostNorm.Forward(h);
        var gate = Activation.Silu(GateProj.Forward(ffnIn));
        var up = UpProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = DownProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }

    /// <summary>
    /// Runs one Llama decoder block over a single new token during cached inference. Mirrors
    /// <see cref="Forward(ReverseGradTensor{T})"/> (same residual/FFN structure) but routes the
    /// attention through <see cref="LlamaCausalAttention{T}.ForwardCached"/> so projections are
    /// only computed for the one position.
    /// </summary>
    /// <param name="input">The new-token hidden state <c>[1, hiddenSize]</c></param>
    /// <param name="positionOffset">Absolute position of the new token</param>
    /// <param name="kCache">Per-layer RoPE'd key cache (row-major per-KV-head)</param>
    /// <param name="vCache">Per-layer value cache (row-major per-KV-head)</param>
    /// <param name="cacheLen">Number of tokens already cached before this call</param>
    /// <returns>The block output <c>[1, hiddenSize]</c></returns>
    public ReverseGradTensor<T> ForwardCached(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[] kCache,
        T[] vCache,
        int cacheLen)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var attnOut = Attention.ForwardCached(InputNorm.Forward(input), positionOffset, kCache, vCache, cacheLen);
        var h = ReverseGradOperations.Add(input, attnOut);

        var ffnIn = PostNorm.Forward(h);
        var gate = Activation.Silu(GateProj.Forward(ffnIn));
        var up = UpProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = DownProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }
}
