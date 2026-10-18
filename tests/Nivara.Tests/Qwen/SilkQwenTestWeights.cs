using Nivara.Samples;

namespace Nivara.Tests.Qwen;

internal static class SilkQwenTestWeights
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

    public static Nivara.Silk.QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos)
    {
        return new Nivara.Silk.QwenSilkConfig
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
