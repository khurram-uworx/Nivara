using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.Samples;

public sealed record DistilBertConfig
{
    public int Dim { get; init; } = 768;
    public int NLayers { get; init; } = 6;
    public int NHeads { get; init; } = 12;
    public int HiddenDim { get; init; } = 3072;
    public int MaxPosition { get; init; } = 512;
    public float Eps { get; init; } = 1e-12f;
    public int VocabSize { get; init; } = 30522;

    public BertConfig ToBertConfig() => new()
    {
        HiddenSize = Dim,
        NumAttentionHeads = NHeads,
        NumHiddenLayers = NLayers,
        IntermediateSize = HiddenDim,
        MaxPositionEmbeddings = MaxPosition,
        VocabSize = VocabSize,
        LayerNormEps = Eps,
    };

    public static DistilBertConfig FromJson(string json)
    {
        var config = new DistilBertConfig();
        var props = typeof(DistilBertConfig).GetProperties();
        foreach (var prop in props)
        {
            var key = SnakeKey(prop.Name);
            var idx = json.IndexOf($"\"{key}\"", StringComparison.OrdinalIgnoreCase);
            if (idx < 0) continue;
            var colonIdx = json.IndexOf(':', idx);
            var endIdx = json.IndexOfAny([',', '}'], colonIdx);
            var valStr = json.Substring(colonIdx + 1, endIdx - colonIdx - 1).Trim();
            if (prop.PropertyType == typeof(int) && int.TryParse(valStr, out int intVal))
                prop.SetValue(config, intVal);
            else if (prop.PropertyType == typeof(float) && float.TryParse(valStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float floatVal))
                prop.SetValue(config, floatVal);
        }
        return config;
    }

    static string SnakeKey(string pascal)
    {
        if (pascal == "NLayers") return "n_layers";
        if (pascal == "NHeads") return "n_heads";
        if (pascal == "MaxPosition") return "max_position_embeddings";
        if (pascal == "HiddenDim") return "hidden_dim";
        if (pascal == "VocabSize") return "vocab_size";
        return pascal.ToLowerInvariant();
    }
}

public static class DistilBertLoader
{
    public static BertEncoder<float> LoadEncoder(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, BertConfig config)
    {
        var encoder = new BertEncoder<float>(config, includeTokenTypeEmbedding: false);
        LoadEncoderWeights<float, float>(encoder, tensors, "distilbert");
        encoder.Eval();
        return encoder;
    }

    public static void LoadEncoderWeights<TModel, TWeight>(
        BertEncoder<TModel> encoder,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string prefix = "distilbert")
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        StateDictLoader.LoadEmbed(encoder.wordEmbed, tensors, $"{prefix}.embeddings.word_embeddings.weight");
        StateDictLoader.LoadEmbed(encoder.posEmbed, tensors, $"{prefix}.embeddings.position_embeddings.weight");
        StateDictLoader.LoadLayerNorm(encoder.embedLn, tensors, $"{prefix}.embeddings.LayerNorm");

        for (int i = 0; i < encoder.layers.Length; i++)
        {
            var layer = encoder.layers[i];
            var p = $"{prefix}.transformer.layer.{i}";
            StateDictLoader.LoadLinear(layer.attn.qProj, tensors, $"{p}.attention.q_lin");
            StateDictLoader.LoadLinear(layer.attn.kProj, tensors, $"{p}.attention.k_lin");
            StateDictLoader.LoadLinear(layer.attn.vProj, tensors, $"{p}.attention.v_lin");
            StateDictLoader.LoadLinear(layer.attn.oProj, tensors, $"{p}.attention.out_lin");
            StateDictLoader.LoadLayerNorm(layer.ln1, tensors, $"{p}.sa_layer_norm");
            StateDictLoader.LoadLinear(layer.fc1, tensors, $"{p}.ffn.lin1");
            StateDictLoader.LoadLinear(layer.fc2, tensors, $"{p}.ffn.lin2");
            StateDictLoader.LoadLayerNorm(layer.ln2, tensors, $"{p}.output_layer_norm");
        }
    }
}
