using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace NivaraGpt;

public sealed class NivaraGptModel<T> : Module<T> where T : struct, INumber<T>
{
    readonly int vocabSize;
    readonly int nEmbd;
    readonly int nLayer;
    readonly int blockSize;
    readonly int nHead;
    readonly bool weightTying;

    readonly Embedding<T> tokenEmb;
    readonly Embedding<T> posEmb;
    readonly Dropout<T>? embDropout;
    readonly TransformerBlock<T>[] blocks;
    readonly Dropout<T>? finalDropout;
    readonly Linear<T>? lmHead;

    public NivaraGptModel(
        int vocabSize, int nEmbd, int nLayer, int blockSize, int nHead,
        double dropout = 0.1, bool weightTying = true, double initStd = 0.02)
    {
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.blockSize = blockSize;
        this.nHead = nHead;
        this.weightTying = weightTying;

        tokenEmb = new Embedding<T>(vocabSize, nEmbd);
        posEmb = new Embedding<T>(blockSize, nEmbd);
        embDropout = dropout > 0.0 ? new Dropout<T>(dropout) : null;

        blocks = new TransformerBlock<T>[nLayer];
        for (int i = 0; i < nLayer; i++)
            blocks[i] = new TransformerBlock<T>(nEmbd, nHead, dropout, blockSize, initStd);

        finalDropout = dropout > 0.0 ? new Dropout<T>(dropout) : null;

        if (!weightTying)
            lmHead = new Linear<T>(nEmbd, vocabSize, bias: false,
                weightInitializer: new Nivara.AutoDiff.Nn.Initializers.NormalInitializer<T>(
                    T.Zero, T.CreateChecked(initStd)));

        if (embDropout != null) RegisterModules(embDropout);
        RegisterModules(tokenEmb, posEmb);
        foreach (var block in blocks) RegisterModules(block);
        if (finalDropout != null) RegisterModules(finalDropout);
        if (lmHead != null) RegisterModules(lmHead);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        int totalTokens = input.Length;
        int L = blockSize;
        int B = totalTokens / L;

        var tokEmb = tokenEmb.Forward(input);
        var posIds = CreatePositionIds(B, L);
        var posEmbTensor = posEmb.Forward(posIds);

        var x = ReverseGradOperations.Add(tokEmb, posEmbTensor);
        x = embDropout != null ? embDropout.Forward(x) : x;

        var blockOutputs = new ReverseGradTensor<T>[B];
        for (int b = 0; b < B; b++)
        {
            var seq = SliceSequence(x, b, L);
            foreach (var block in blocks)
                seq = block.Forward(seq);
            blockOutputs[b] = seq;
        }
        x = ReverseGradOperations.Concat(blockOutputs, axis: 0);

        x = ReverseGradOperations.PerRowRMSNorm(x, B * L, nEmbd);
        x = finalDropout != null ? finalDropout.Forward(x) : x;

        if (weightTying)
        {
            var wteWeight = tokenEmb.Weight;
            var wteT = ReverseGradOperations.Transpose(wteWeight);
            return ReverseGradOperations.MatMul(x, wteT);
        }
        else
        {
            return lmHead!.Forward(x);
        }
    }

    ReverseGradTensor<T> SliceSequence(ReverseGradTensor<T> x, int batchIdx, int L)
    {
        var indices = new int[L];
        for (int i = 0; i < L; i++)
            indices[i] = batchIdx * L + i;
        return ReverseGradOperations.Gather(x, indices, axis: 0);
    }

    static ReverseGradTensor<T> CreatePositionIds(int B, int L)
    {
        var data = new T[B * L];
        for (int b = 0; b < B; b++)
            for (int l = 0; l < L; l++)
                data[b * L + l] = T.CreateChecked(l);

        var col = NivaraColumn<T>.Create(data);
        return new ReverseGradTensor<T>(col, requiresGrad: false);
    }

}
