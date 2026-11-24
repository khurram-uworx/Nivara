using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkQwenGpuRunner
{
    public SilkQwenGpuRunner(
        SilkRuntime runtime,
        QwenSilkConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        SilkWeightQuantization quantization = SilkWeightQuantization.None)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        this.config = config;
        int8Weights = quantization == SilkWeightQuantization.Int8Block32;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        numKvHeads = config.NumKeyValueHeads;
        headDim = hiddenDim / numHeads;
        qDim = numHeads * headDim;
        kvDim = numKvHeads * headDim;
        eps = config.RmsNormEps;
        scale = 1f / MathF.Sqrt(headDim);
        maxPos = config.MaxPositionEmbeddings;
        ropeTheta = config.RopeTheta;
        seqCap = maxPos;
        vocabSize = config.VocabSize;
        bf16Weights = !int8Weights;
        runtime.EnsureAttentionLocalMemory(headDim);

        static float[] Widen(BFloat16[] src)
        {
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = (float)src[i];
            return dst;
        }

        try
        {
            int h = hiddenDim;
            var tok = WeightLayout.Req(tensors, "model.embed_tokens.weight", "Silk");
            // BFloat16 and ushort share an identical 2-byte layout, so this is a
            // bit-preserving reinterpretation (no conversion) for device storage.
            var tokUs = System.Runtime.InteropServices.MemoryMarshal.Cast<System.Numerics.BFloat16, ushort>(tok.Data.AsSpan()).ToArray();
            tokEmb = runtime.AllocateBytes((ulong)tokUs.Length * sizeof(ushort));
            runtime.Upload(tokEmb, tokUs.AsSpan());

            int vocab = config.VocabSize;
            var tokTF = new ushort[vocab * h];
            for (int r = 0; r < vocab; r++)
                for (int c = 0; c < h; c++)
                    tokTF[c * vocab + r] = tokUs[r * h + c];
            tokEmbTransposed = runtime.AllocateBytes((ulong)tokTF.Length * sizeof(ushort));
            runtime.Upload(tokEmbTransposed, tokTF.AsSpan());

            if (int8Weights)
            {
                var (qw, sc) = SilkInt8Quantizer.QuantizeFromBf16(tokTF.AsSpan(), vocab, h);
                tokEmbInt8 = runtime.Allocate(qw.Length);
                runtime.Upload(tokEmbInt8, qw.AsSpan());
                tokEmbInt8Scale = runtime.Allocate(sc.Length);
                runtime.Upload(tokEmbInt8Scale, sc.AsSpan());
                tokEmbInt8NumBlocks = (h + SilkInt8Quantizer.BlockSize - 1) / SilkInt8Quantizer.BlockSize;
            }

            var fn = WeightLayout.Req(tensors, "model.norm.weight", "Silk");
            finalNormW = runtime.Allocate(fn.Data.Length);
            runtime.Upload(finalNormW, Widen(fn.Data));

            layers = new LayerWeights[config.NumHiddenLayers];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, i);

            kCache = new nint[config.NumHiddenLayers];
            vCache = new nint[config.NumHiddenLayers];
            for (int i = 0; i < config.NumHiddenLayers; i++)
            {
                kCache[i] = runtime.Allocate(maxPos * kvDim);
                vCache[i] = runtime.Allocate(maxPos * kvDim);
            }

            rotary = new RotaryEmbedding<float>(headDim, maxPos, ropeTheta);

            argmaxGroups = (vocabSize + 255) / 256;
            partialArgmaxIdx = runtime.Allocate(argmaxGroups);
            partialArgmaxVal = runtime.Allocate(argmaxGroups);
        }
        catch
        {
            Dispose();
            throw;
        }
    }
}
