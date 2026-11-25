namespace Nivara.Silk;

using Nivara.AutoDiff.Nn;

/// <summary>
/// Host-side conversion from F32/BF16 weight tensors to the Silk INT8 block-32 quantized format.
/// Delegates to <see cref="Nivara.AutoDiff.Nn.Int8Block32Quantizer"/> (single source of truth).
/// See docs/SILK.md.
/// </summary>
internal static class SilkInt8Quantizer
{
    public const int BlockSize = Int8Block32Quantizer.BlockSize;

    public static (sbyte[] Weights, float[] Scales) Quantize(ReadOnlySpan<float> transposedWeights, int inDim, int outDim)
        => Int8Block32Quantizer.Quantize(transposedWeights, inDim, outDim);

    public static (sbyte[] Weights, float[] Scales) QuantizeFromCheckpoint(ReadOnlySpan<float> checkpointWeights, int outDim, int inDim)
        => Int8Block32Quantizer.QuantizeFromCheckpoint(checkpointWeights, outDim, inDim);

    public static (sbyte[] Weights, float[] Scales) QuantizeFromBf16(ReadOnlySpan<ushort> checkpointWeights, int outDim, int inDim)
        => Int8Block32Quantizer.QuantizeFromBf16(checkpointWeights, outDim, inDim);
}
