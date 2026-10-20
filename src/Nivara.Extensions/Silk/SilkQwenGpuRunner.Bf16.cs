using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkQwenGpuRunner
{
    public SilkQwenGpuRunner(
        SilkRuntime runtime,
        QwenSilkConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        this.config = config;
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
        bf16Weights = true;
        runtime.EnsureAttentionLocalMemory(headDim);

        static float[] Widen(BFloat16[] src)
        {
            var u = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan());
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = BitConverter.UInt32BitsToSingle((uint)u[i] << 16);
            return dst;
        }

        try
        {
            int h = hiddenDim;
            var tok = WeightLayout.Req(tensors, "model.embed_tokens.weight", "Silk");
            var tokF = Widen(tok.Data);
            tokEmb = runtime.Allocate(tokF.Length);
            runtime.Upload(tokEmb, tokF);

            int vocab = config.VocabSize;
            var tokTF = new float[vocab * h];
            for (int r = 0; r < vocab; r++)
                for (int c = 0; c < h; c++)
                    tokTF[c * vocab + r] = tokF[r * h + c];
            tokEmbTransposed = runtime.Allocate(tokTF.Length);
            runtime.Upload(tokEmbTransposed, tokTF);

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
