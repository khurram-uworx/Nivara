namespace Nivara.Silk;

/// <summary>
/// Host-side conversion from F32/BF16 weight tensors to the Silk INT8 block-32 quantized format
/// (symmetric, no zero point, one F32 scale per 32-element block along the reduction dimension).
/// Format: <c>SilkInt8Block32 v1</c>. See docs/SILK.md.
/// </summary>
internal static class SilkInt8Quantizer
{
    public const int BlockSize = 32;

    /// <summary>
    /// Quantizes a transposed weight matrix <c>[inDim, outDim]</c> (F32) into INT8 blocks + F32 scales.
    /// Returns (int8Weights of length inDim*outDim, scales of length outDim*numBlocks).
    /// </summary>
    public static (sbyte[] Weights, float[] Scales) Quantize(ReadOnlySpan<float> transposedWeights, int inDim, int outDim)
    {
        if (transposedWeights.Length != inDim * outDim)
            throw new ArgumentException("transposedWeights length must equal inDim*outDim", nameof(transposedWeights));
        if (inDim <= 0 || outDim <= 0)
            throw new ArgumentOutOfRangeException("dimensions must be positive");

        int numBlocks = (inDim + BlockSize - 1) / BlockSize;
        var weights = new sbyte[inDim * outDim];
        var scales = new float[outDim * numBlocks];

        for (int col = 0; col < outDim; col++)
        {
            for (int b = 0; b < numBlocks; b++)
            {
                int kStart = b * BlockSize;
                int kEnd = Math.Min(kStart + BlockSize, inDim);
                // max abs
                float maxAbs = 0f;
                for (int k = kStart; k < kEnd; k++)
                {
                    float v = transposedWeights[k * outDim + col];
                    float a = v < 0 ? -v : v;
                    if (a > maxAbs) maxAbs = a;
                }
                float scale = maxAbs > 0f ? maxAbs / 127f : 1f;
                scales[col * numBlocks + b] = scale;
                float inv = 1f / scale;
                for (int k = kStart; k < kEnd; k++)
                {
                    float v = transposedWeights[k * outDim + col];
                    int q = (int)MathF.Round(v * inv);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    weights[k * outDim + col] = (sbyte)q;
                }
                // padding zeros for partial tail already default 0
            }
        }
        return (weights, scales);
    }

    /// <summary>
    /// Convenience: quantize from a checkpoint-style <c>[out, in]</c> F32 matrix by transposing first.
    /// </summary>
    public static (sbyte[] Weights, float[] Scales) QuantizeFromCheckpoint(ReadOnlySpan<float> checkpointWeights, int outDim, int inDim)
    {
        if (checkpointWeights.Length != outDim * inDim)
            throw new ArgumentException("checkpointWeights length must equal outDim*inDim", nameof(checkpointWeights));
        var transposed = new float[inDim * outDim];
        for (int r = 0; r < outDim; r++)
            for (int c = 0; c < inDim; c++)
                transposed[c * outDim + r] = checkpointWeights[r * inDim + c];
        return Quantize(transposed, inDim, outDim);
    }

    /// <summary>
    /// Quantizes from BF16 (as ushort bit patterns) checkpoint weights.
    /// </summary>
    public static (sbyte[] Weights, float[] Scales) QuantizeFromBf16(ReadOnlySpan<ushort> checkpointWeights, int outDim, int inDim)
    {
        if (checkpointWeights.Length != outDim * inDim)
            throw new ArgumentException("checkpointWeights length must equal outDim*inDim", nameof(checkpointWeights));
        var f = new float[outDim * inDim];
        for (int i = 0; i < f.Length; i++)
            f[i] = BitConverter.Int32BitsToSingle(checkpointWeights[i] << 16);
        return QuantizeFromCheckpoint(f, outDim, inDim);
    }
}
