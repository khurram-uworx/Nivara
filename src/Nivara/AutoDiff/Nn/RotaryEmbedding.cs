using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Rotary position embeddings (RoPE) for transformer attention. Precomputes the cosine and
/// sine frequency tables once from <c>inv_freq = theta^(-2j/dim)</c> and rotates every
/// adjacent pair of a <c>[L, headDim]</c> query/key by its absolute position, making
/// attention positions relative. This is the positional-encoding scheme used by the
/// Llama family of models. No learnable parameters are involved.
/// </summary>
public sealed class RotaryEmbedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int headDim;
    readonly int maxPositionEmbeddings;
    readonly T ropeTheta;

    T[] cosCache;
    T[] sinCache;
    bool cosCacheValid;

    /// <summary>Gets the head dimension (must be even).</summary>
    public int HeadDim => headDim;

    /// <summary>Gets the maximum absolute position supported.</summary>
    public int MaxPositionEmbeddings => maxPositionEmbeddings;

    /// <summary>Gets the base of the rope theta frequency scaling.</summary>
    public T RopeTheta => ropeTheta;

    /// <summary>
    /// Creates a rotary position embedding layer.
    /// </summary>
    /// <param name="headDim">Head dimension (must be even and positive)</param>
    /// <param name="maxPositionEmbeddings">Maximum absolute position (default 2048)</param>
    /// <param name="ropeTheta">Base for the inverse-frequency scaling (default 10000)</param>
    public RotaryEmbedding(int headDim, int maxPositionEmbeddings = 2048, float ropeTheta = 10000f)
    {
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (headDim % 2 != 0) throw new ArgumentException("headDim must be even for pairwise rotary rotation.", nameof(headDim));
        if (maxPositionEmbeddings <= 0) throw new ArgumentOutOfRangeException(nameof(maxPositionEmbeddings));
        if (ropeTheta <= 0) throw new ArgumentOutOfRangeException(nameof(ropeTheta));

        this.headDim = headDim;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.ropeTheta = T.CreateChecked(ropeTheta);

        cosCache = Array.Empty<T>();
        sinCache = Array.Empty<T>();
        cosCacheValid = false;
    }

    /// <summary>
    /// Rotates the last dimension of a tensor by its absolute position. Each row is treated
    /// as one or more contiguous <c>headDim</c> blocks; every block is rotated by the row's
    /// position. The leading dimension is the sequence length. A tensor of width
    /// <c>headDim</c> (one head per row) or <c>numHeads * headDim</c> (all heads combined,
    /// head-major) is supported.
    /// </summary>
    /// <param name="input">The query or key tensor with shape <c>[L, headDim]</c> or <c>[L, numHeads * headDim]</c></param>
    /// <returns>The rotated tensor with the same shape</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"RotaryEmbedding expects a 2D tensor, got rank {input.Rank}.");
        int seqLen = input.shape[0];
        int width = input.shape[1];
        if (width % headDim != 0)
            throw new ArgumentException($"Width ({width}) must be a multiple of headDim ({headDim}).");
        if (seqLen > maxPositionEmbeddings)
            throw new ArgumentException($"Sequence length {seqLen} exceeds max_position_embeddings {maxPositionEmbeddings}.");

        int numBlocks = width / headDim;

        EnsureCache(seqLen);
        var cos = cosCache.AsSpan(0, seqLen * (headDim / 2));
        var sin = sinCache.AsSpan(0, seqLen * (headDim / 2));

        var inputData = ModuleHelpers<T>.GetSpan(input);
        bool trackGrad = GradientUtils.ShouldTrackGrad(input);

        var outputData = new T[input.Length];
        for (int p = 0; p < seqLen; p++)
        {
            var rowC = cos.Slice(p * (headDim / 2), headDim / 2);
            var rowS = sin.Slice(p * (headDim / 2), headDim / 2);
            var rowIn = inputData.Slice(p * width, width);
            var rowOut = outputData.AsSpan(p * width, width);
            for (int b = 0; b < numBlocks; b++)
            {
                GradKernels.RotaryForward(
                    rowIn.Slice(b * headDim, headDim), rowC, rowS,
                    rowOut.Slice(b * headDim, headDim));
            }
        }

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.CreateFromOwnedArray(outputData), trackGrad, input.shape);

        if (trackGrad)
        {
            var savedInput = ModuleHelpers<T>.GetSpan(input).ToArray();
            var savedCos = cos.ToArray();
            var savedSin = sin.ToArray();
            int savedSeq = seqLen;
            int savedHeadDim = headDim;
            int savedBlocks = numBlocks;

            var gradFn = new OpNode<T>("RotaryEmbedding", [input], (typedGradOutput) =>
            {
                var gradArr = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradArr, default(T)!);
                for (int p = 0; p < savedSeq; p++)
                {
                    var rowC = savedCos.AsSpan(p * (savedHeadDim / 2), savedHeadDim / 2);
                    var rowS = savedSin.AsSpan(p * (savedHeadDim / 2), savedHeadDim / 2);
                    var rowG = gradArr.AsSpan(p * width, width);
                    for (int b = 0; b < savedBlocks; b++)
                    {
                        GradKernels.RotaryBackward(
                            rowG.Slice(b * savedHeadDim, savedHeadDim), rowC, rowS,
                            rowG.Slice(b * savedHeadDim, savedHeadDim));
                    }
                }
                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    void EnsureCache(int seqLen)
    {
        int halfDim = headDim / 2;
        int needed = seqLen * halfDim;
        if (cosCacheValid && cosCache.Length >= needed)
            return;

        var newCos = new T[needed];
        var newSin = new T[needed];
        var invFreq = new T[halfDim];
        T two = T.CreateChecked(2);
        T dimT = T.CreateChecked(headDim);
        for (int j = 0; j < halfDim; j++)
        {
            T exponent = -T.CreateChecked(2 * j) / dimT;
            invFreq[j] = T.Pow(ropeTheta, exponent);
        }

        for (int p = 0; p < seqLen; p++)
        {
            int offset = p * halfDim;
            for (int j = 0; j < halfDim; j++)
            {
                T theta = T.CreateChecked(p) * invFreq[j];
                newCos[offset + j] = T.Cos(theta);
                newSin[offset + j] = T.Sin(theta);
            }
        }

        cosCache = newCos;
        sinCache = newSin;
        cosCacheValid = true;
    }
}
