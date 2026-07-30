using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using System.Numerics;

namespace NivaraFineTuning;

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

public sealed class DistilBertForSequenceClassification<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    internal readonly BertEncoder<T> encoder;
    internal readonly Linear<T> preClassifier;
    internal readonly Linear<T> classifier;
    readonly int hiddenDim;

    public int NumClasses { get; }

    public DistilBertForSequenceClassification(BertConfig config, int numClasses)
    {
        NumClasses = numClasses;
        hiddenDim = config.HiddenSize;
        encoder = new BertEncoder<T>(config);
        preClassifier = new Linear<T>(hiddenDim, hiddenDim, bias: true);
        classifier = new Linear<T>(hiddenDim, numClasses, bias: true);
        RegisterModules(encoder, preClassifier, classifier);
    }

    public ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> inputIds,
        ReverseGradTensor<T> attentionMask,
        int batchSize,
        int seqLen)
    {
        var encoded = encoder.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
        var clsTokens = ExtractClsTokens(encoded, batchSize, seqLen);
        var h = preClassifier.Forward(clsTokens);
        h = ReverseGradOperations.Gelu(h);
        var logits = classifier.Forward(h);
        return logits;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => throw new NotImplementedException("Use Forward(inputIds, attentionMask, batchSize, seqLen).");

    static ReverseGradTensor<T> ExtractClsTokens(ReverseGradTensor<T> encoded, int batchSize, int seqLen)
    {
        var indices = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
            indices[b] = b * seqLen;
        return ReverseGradOperations.Gather(encoded, indices, axis: 0);
    }

    public void LoadWeights<TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var enc = encoder;

        LoadEmbed<TWeight>(enc.wordEmbed, tensors, "distilbert.embeddings.word_embeddings.weight");
        LoadEmbed<TWeight>(enc.posEmbed, tensors, "distilbert.embeddings.position_embeddings.weight");
        LoadLayerNorm<TWeight>(enc.embedLn, tensors, "distilbert.embeddings.LayerNorm");

        for (int i = 0; i < enc.layers.Length; i++)
        {
            var layer = enc.layers[i];
            var prefix = $"distilbert.transformer.layer.{i}";
            LoadLinear<TWeight>(layer.attn.qProj, tensors, $"{prefix}.attention.q_lin");
            LoadLinear<TWeight>(layer.attn.kProj, tensors, $"{prefix}.attention.k_lin");
            LoadLinear<TWeight>(layer.attn.vProj, tensors, $"{prefix}.attention.v_lin");
            LoadLinear<TWeight>(layer.attn.oProj, tensors, $"{prefix}.attention.out_lin");
            LoadLayerNorm<TWeight>(layer.ln1, tensors, $"{prefix}.sa_layer_norm");
            LoadLinear<TWeight>(layer.fc1, tensors, $"{prefix}.ffn.lin1");
            LoadLinear<TWeight>(layer.fc2, tensors, $"{prefix}.ffn.lin2");
            LoadLayerNorm<TWeight>(layer.ln2, tensors, $"{prefix}.output_layer_norm");
        }

        if (tensors.ContainsKey("pre_classifier.weight") || tensors.ContainsKey("pre_classifier.bias"))
            LoadLinear<TWeight>(preClassifier, tensors, "pre_classifier");

        if (tensors.ContainsKey("classifier.weight") || tensors.ContainsKey("classifier.bias"))
            LoadLinear<TWeight>(classifier, tensors, "classifier");

        Eval();
    }

    static void LoadEmbed<TWeight>(Embedding<T> embed, Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, string key)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        if (!tensors.TryGetValue(key, out var t)) return;
        var tensor = TypeConverter.Convert<TWeight, T>(
            ReverseGradTensor<TWeight>.FromMatrix(t.Data, t.Shape[0], t.Shape[1]));
        embed.LoadStateDict(new Dictionary<string, ReverseGradTensor<T>> { ["Weight"] = tensor });
    }

    static void LoadLinear<TWeight>(Linear<T> linear, Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, string prefix)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var dict = new Dictionary<string, ReverseGradTensor<T>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = TypeConverter.Convert<TWeight, T>(
                ReverseGradTensor<TWeight>.FromMatrix(w.Data, w.Shape[0], w.Shape[1]));
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = TypeConverter.Convert<TWeight, T>(
                ReverseGradTensor<TWeight>.FromMatrix(b.Data, 1, b.Shape[0]));
        if (dict.Count > 0) linear.LoadStateDict(dict);
    }

    static void LoadLayerNorm<TWeight>(LayerNorm<T> ln, Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, string prefix)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var dict = new Dictionary<string, ReverseGradTensor<T>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = TypeConverter.Convert<TWeight, T>(
                ReverseGradTensor<TWeight>.FromArray(w.Data));
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = TypeConverter.Convert<TWeight, T>(
                ReverseGradTensor<TWeight>.FromArray(b.Data));
        if (dict.Count > 0) ln.LoadStateDict(dict);
    }
}
