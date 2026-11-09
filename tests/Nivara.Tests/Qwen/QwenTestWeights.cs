using Nivara.Samples;
using Nivara.Silk;
using System.Numerics;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Test-owned synthetic Qwen smoke weights for Silk/OpenCL parity and future GPU
/// probe harnesses. Uses a small, deterministic configuration (2 layers, 256 vocab)
/// so smoke parity tests do not depend on <see cref="NivaraInference.Qwen"/>
/// sample internals or a multi-GB checkpoint. Precision/scale choices appropriate
/// for test scale are owned here.
/// </summary>
internal static class QwenTestWeights
{
    public static LlamaConfig SmokeConfig
    {
        get
        {
            return new LlamaConfig
            {
                HiddenSize = 896,
                NumHiddenLayers = 2,
                NumAttentionHeads = 14,
                NumKeyValueHeads = 2,
                IntermediateSize = 4864,
                VocabSize = 256,
                MaxPositionEmbeddings = 16,
                RmsNormEps = 1e-6f,
                RopeTheta = 1000000f,
                TieWordEmbeddings = true,
                HiddenAct = "silu",
            };
        }
    }

    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234)
    {
        var config = SmokeConfig;
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();

        var rng = new Random((int)seed);
        void Add(string name, params int[] dims)
        {
            int n = 1;
            for (int i = 0; i < dims.Length; i++) n *= dims[i];
            var data = new float[n];
            for (int i = 0; i < n; i++)
                data[i] = (float)((rng.NextDouble() - 0.5) * 0.05);
            tensors[name] = (data, dims);
        }

        int h = config.HiddenSize;
        int kvH = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        int inter = config.IntermediateSize;
        int vocab = config.VocabSize;

        Add("model.embed_tokens.weight", vocab, h);
        Add("model.norm.weight", h);

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            string p = "model.layers." + i;
            Add(p + ".input_layernorm.weight", h);
            Add(p + ".post_attention_layernorm.weight", h);

            Add(p + ".self_attn.q_proj.weight", h, h);
            Add(p + ".self_attn.q_proj.bias", h);
            Add(p + ".self_attn.k_proj.weight", kvH, h);
            Add(p + ".self_attn.k_proj.bias", kvH);
            Add(p + ".self_attn.v_proj.weight", kvH, h);
            Add(p + ".self_attn.v_proj.bias", kvH);
            Add(p + ".self_attn.o_proj.weight", h, h);

            Add(p + ".mlp.gate_proj.weight", inter, h);
            Add(p + ".mlp.up_proj.weight", inter, h);
            Add(p + ".mlp.down_proj.weight", h, inter);
        }

        return (tensors, config);
    }

    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config)
        Synthesize<TWeight>(uint seed = 1234)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var (floatTensors, config) = Synthesize(seed);
        var tensors = new Dictionary<string, (TWeight[] Data, int[] Shape)>();
        foreach (var kv in floatTensors)
        {
            var data = new TWeight[kv.Value.Data.Length];
            for (int i = 0; i < data.Length; i++)
                data[i] = TWeight.CreateChecked(kv.Value.Data[i]);
            tensors[kv.Key] = (data, kv.Value.Shape);
        }
        return (tensors, config);
    }

    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config)
        SynthesizeFloat(uint seed = 1234) => Synthesize(seed);

    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config)
        SynthesizeBFloat16(uint seed = 1234) => Synthesize<BFloat16>(seed);

    public static Dictionary<string, (BFloat16[] Data, int[] Shape)>
        ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (BFloat16[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var data = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < data.Length; i++)
                data[i] = (BFloat16)kv.Value.Data[i];
            result[kv.Key] = (data, kv.Value.Shape);
        }
        return result;
    }

    public static Dictionary<string, (float[] Data, int[] Shape)>
        ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var data = new float[kv.Value.Data.Length];
            for (int i = 0; i < data.Length; i++)
                data[i] = (float)kv.Value.Data[i];
            result[kv.Key] = (data, kv.Value.Shape);
        }
        return result;
    }

    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos)
    {
        return new QwenSilkConfig
        {
            HiddenSize = config.HiddenSize,
            NumHiddenLayers = config.NumHiddenLayers,
            NumAttentionHeads = config.NumAttentionHeads,
            NumKeyValueHeads = config.NumKeyValueHeads,
            IntermediateSize = config.IntermediateSize,
            VocabSize = config.VocabSize,
            MaxPositionEmbeddings = maxPos,
            RmsNormEps = config.RmsNormEps,
            RopeTheta = config.RopeTheta,
        };
    }
}
