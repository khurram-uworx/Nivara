using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace NivaraChatClient;

/// <summary>
/// Causal batched transformer with weight tying and a fixed sinusoidal position
/// encoding. Input is a rank-2 token tensor [B, L]; output is a rank-2 logits
/// tensor [B*L, V] (positions flattened row-major), matching the TensorDataset /
/// CrossEntropyLoss training layout used by NivaraGpt.
/// </summary>
public sealed class BatchedTransformer<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int vocabSize;
    readonly int nEmbd;
    readonly int nHead;
    readonly int maxSeqLen;
    readonly T attnScale;

    readonly Embedding<T> tokenEmb;
    readonly BatchedTransformerBlock<T>[] blocks;
    readonly LayerNorm<T> finalNorm;

    public int MaxSeqLen => maxSeqLen;
    public int VocabSize => vocabSize;
    public Embedding<T> TokenEmbedding => tokenEmb;

    public BatchedTransformer(
        int vocabSize,
        int nEmbd,
        int nLayer,
        int nHead,
        int maxSeqLen,
        double dropout = 0.0)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (nEmbd <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbd));
        if (nLayer <= 0) throw new ArgumentOutOfRangeException(nameof(nLayer));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.nHead = nHead;
        this.maxSeqLen = maxSeqLen;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(nHead > 0 ? nEmbd / nHead : 1));

        tokenEmb = new Embedding<T>(vocabSize, nEmbd);

        blocks = new BatchedTransformerBlock<T>[nLayer];
        for (int i = 0; i < nLayer; i++)
            blocks[i] = new BatchedTransformerBlock<T>(nEmbd, nHead, maxSeqLen, dropout);

        finalNorm = new LayerNorm<T>(nEmbd);

        RegisterModules(tokenEmb);
        foreach (var block in blocks) RegisterModules(block);
        RegisterModules(finalNorm);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2)
            throw new ArgumentException($"Expected [B, L] input, got {input.Rank}D.");
        if (input.Shape[1] > maxSeqLen)
            throw new ArgumentException($"Sequence length {input.Shape[1]} exceeds maxSeqLen {maxSeqLen}.");

        int batch = input.Shape[0];
        int length = input.Shape[1];

        var x = tokenEmb.Forward(input);                    // [B, L, D]
        var positions = PositionEncoding.Build<T>(batch, length, nEmbd);
        x = ReverseGradOperations.Add(x, positions);

        var mask = BuildCausalMask(batch, length);
        foreach (var block in blocks)
            x = block.ForwardBlock(x, mask);                    // [B, L, D]

        x = finalNorm.Forward(x);                           // [B, L, D]
        x.Reshape(batch * length, nEmbd);                   // [B*L, D]

        var wteT = ReverseGradOperations.Transpose(tokenEmb.Weight); // [D, V]
        return ReverseGradOperations.MatMul(x, wteT);               // [B*L, V]
    }

    internal static ReverseGradTensor<T> BuildCausalMask(int batch, int length)
    {
        var data = new T[batch * length * length];
        int stride = length * length;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < length; i++)
            {
                int rowStart = b * stride + i * length;
                for (int j = 0; j < length; j++)
                    data[rowStart + j] = j > i ? T.NegativeInfinity : T.Zero;
            }
        }

        var column = NivaraColumn<T>.Create(data);
        var tensor = new ReverseGradTensor<T>(column, requiresGrad: false);
        tensor.Reshape(batch, length, length);
        return tensor;
    }
}

/// <summary>
/// Pre-norm transformer block operating on rank-3 [B, L, D] tensors. Attention is
/// computed for the whole batch in a single call to
/// <see cref="ReverseGradOperations.BatchedMultiHeadAttention{T}"/>.
/// </summary>
public sealed class BatchedTransformerBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int nEmbd;
    readonly int nHead;
    readonly T attnScale;

    readonly LayerNorm<T> ln1;
    readonly LayerNorm<T> ln2;
    readonly Linear<T> qProj;
    readonly Linear<T> kProj;
    readonly Linear<T> vProj;
    readonly Linear<T> oProj;
    readonly Linear<T> mlpFc1;
    readonly Linear<T> mlpFc2;

    public BatchedTransformerBlock(int nEmbd, int nHead, int maxSeqLen, double dropout = 0.0)
    {
        if (nEmbd <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbd));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        this.nEmbd = nEmbd;
        this.nHead = nHead;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(nEmbd / nHead));

        ln1 = new LayerNorm<T>(nEmbd);
        ln2 = new LayerNorm<T>(nEmbd);
        qProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        kProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        vProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        oProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        mlpFc1 = new Linear<T>(nEmbd, 4 * nEmbd);
        mlpFc2 = new Linear<T>(4 * nEmbd, nEmbd);

        RegisterModules(ln1, ln2, qProj, kProj, vProj, oProj, mlpFc1, mlpFc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input) =>
        ForwardBlock(input, null);

    public ReverseGradTensor<T> ForwardBlock(ReverseGradTensor<T> input, ReverseGradTensor<T>? causalMask)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException($"Expected [B, L, D] input, got {input.Rank}D.");

        int batch = input.Shape[0];
        int length = input.Shape[1];

        var mask = causalMask ?? BatchedTransformer<T>.BuildCausalMask(batch, length);

        var residual = input;

        var normed = ln1.Forward(input);                    // [B, L, D]
        var flat = normed;
        flat.Reshape(batch * length, nEmbd);                // [B*L, D]

        var q = qProj.Forward(flat);                        // [B*L, D]
        var k = kProj.Forward(flat);
        var v = vProj.Forward(flat);
        q.Reshape(batch, length, nEmbd);
        k.Reshape(batch, length, nEmbd);
        v.Reshape(batch, length, nEmbd);

        var attn = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, nHead, attnScale, mask);
        attn.Reshape(batch * length, nEmbd);                // [B*L, D]
        var attnOut = oProj.Forward(attn);
        attnOut.Reshape(batch, length, nEmbd);              // [B, L, D]

        var x = ReverseGradOperations.Add(attnOut, residual);

        residual = x;
        var normed2 = ln2.Forward(x);
        var hidden = normed2;
        hidden.Reshape(batch * length, nEmbd);              // [B*L, D]
        hidden = mlpFc1.Forward(hidden);
        hidden = Activation.Gelu(hidden);
        hidden = mlpFc2.Forward(hidden);
        hidden.Reshape(batch, length, nEmbd);               // [B, L, D]

        return ReverseGradOperations.Add(hidden, residual);
    }
}
