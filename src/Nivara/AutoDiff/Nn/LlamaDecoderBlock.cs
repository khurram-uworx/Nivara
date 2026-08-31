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

    readonly RMSNorm<T> inputNorm;
    readonly LlamaCausalAttention<T> attention;
    readonly RMSNorm<T> postNorm;
    readonly Linear<T> gateProj;
    readonly Linear<T> upProj;
    readonly Linear<T> downProj;

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
    public LlamaDecoderBlock(
        int hiddenSize,
        int numHeads,
        int numKeyValueHeads,
        int intermediateSize,
        float rmsNormEps = 1e-5f,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (intermediateSize <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateSize));

        this.hiddenSize = hiddenSize;

        inputNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        attention = new LlamaCausalAttention<T>(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta);
        postNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        gateProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        upProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        downProj = new Linear<T>(intermediateSize, hiddenSize, bias: false);

        RegisterModules(inputNorm, attention, postNorm, gateProj, upProj, downProj);
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
        var attnOut = attention.Forward(inputNorm.Forward(input));
        var h = ReverseGradOperations.Add(input, attnOut);

        // Pre-norm gated SiLU feed-forward with residual add.
        var ffnIn = postNorm.Forward(h);
        var gate = Activation.Silu(gateProj.Forward(ffnIn));
        var up = upProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = downProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }
}
