using Nivara.Silk;
using System.Numerics;

namespace Nivara.Tests.SmolLM;

/// <summary>
/// Test-owned synthetic SmolLM smoke weights for Silk/OpenCL BF16 parity. Uses a
/// shallow, deterministic configuration (2 layers) so the cached-decode parity test
/// does not depend on <see cref="NivaraInference.Qwen"/> sample internals or a
/// multi-GB checkpoint. Precision/scale choices are owned here, mirroring
/// <see cref="Nivara.Tests.Qwen.QwenTestWeights"/>. Geometry keeps the real SmolLM-135M widths
/// (hidden 576 / 9 heads / 3 kv heads / inter 1536) so the non-whole-group gemv
/// tail path is exercised; vocab is reduced to 1024 to keep the fixture cheap.
/// </summary>
internal static class SmolLMTestWeights
{
    public static SmolLMConfig SmokeConfig => new()
    {
        HiddenSize = 576,
        NumHiddenLayers = 2,
        NumAttentionHeads = 9,
        NumKeyValueHeads = 3,
        IntermediateSize = 1536,
        VocabSize = 1024,
        MaxPositionEmbeddings = 16,
        RmsNormEps = 1e-5f,
        RopeTheta = 10000f,
        AttentionBias = false,
    };

    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, SmolLMConfig config) Synthesize(uint seed = 1234)
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
            Add(p + ".self_attn.k_proj.weight", kvH, h);
            Add(p + ".self_attn.v_proj.weight", kvH, h);
            Add(p + ".self_attn.o_proj.weight", h, h);

            Add(p + ".mlp.gate_proj.weight", inter, h);
            Add(p + ".mlp.up_proj.weight", inter, h);
            Add(p + ".mlp.down_proj.weight", h, inter);
        }

        return (tensors, config);
    }

    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors)
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
}
