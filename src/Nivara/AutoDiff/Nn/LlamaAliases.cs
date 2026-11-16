using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class LlamaCausalAttention<T> : LlamaCausalAttention<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public LlamaCausalAttention(int hiddenSize, int numHeads, int numKeyValueHeads, int maxPositionEmbeddings = 2048, float ropeTheta = 10000f, bool qkvBias = false)
        : base(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta, qkvBias) { }

    internal LlamaCausalAttention(int hiddenSize, int numHeads, int numKeyValueHeads, int maxPositionEmbeddings, float ropeTheta, bool qkvBias, bool skipWeightInitialization)
        : base(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta, qkvBias, skipWeightInitialization) { }

    public static LlamaCausalAttention<T> Uninitialized(int hiddenSize, int numHeads, int numKeyValueHeads, int maxPositionEmbeddings = 2048, float ropeTheta = 10000f, bool qkvBias = false)
        => new LlamaCausalAttention<T>(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta, qkvBias, true);
}

public sealed class LlamaDecoderBlock<T> : LlamaDecoderBlock<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public LlamaDecoderBlock(int hiddenSize, int numHeads, int numKeyValueHeads, int intermediateSize, float rmsNormEps = 1e-5f, int maxPositionEmbeddings = 2048, float ropeTheta = 10000f, bool qkvBias = false)
        : base(hiddenSize, numHeads, numKeyValueHeads, intermediateSize, rmsNormEps, maxPositionEmbeddings, ropeTheta, qkvBias) { }

    internal LlamaDecoderBlock(int hiddenSize, int numHeads, int numKeyValueHeads, int intermediateSize, float rmsNormEps, int maxPositionEmbeddings, float ropeTheta, bool qkvBias, bool skipWeightInitialization)
        : base(hiddenSize, numHeads, numKeyValueHeads, intermediateSize, rmsNormEps, maxPositionEmbeddings, ropeTheta, qkvBias, skipWeightInitialization) { }

    public static LlamaDecoderBlock<T> Uninitialized(int hiddenSize, int numHeads, int numKeyValueHeads, int intermediateSize, float rmsNormEps = 1e-5f, int maxPositionEmbeddings = 2048, float ropeTheta = 10000f, bool qkvBias = false)
        => new LlamaDecoderBlock<T>(hiddenSize, numHeads, numKeyValueHeads, intermediateSize, rmsNormEps, maxPositionEmbeddings, ropeTheta, qkvBias, true);
}
