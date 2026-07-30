using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.Samples;
using System.Numerics;
using System.Runtime.CompilerServices;

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

    public void LoadWeights(
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        if (typeof(T) != typeof(float))
            return;

        var enc = Unsafe.As<BertEncoder<float>>(encoder);

        LoadEmbed(enc.wordEmbed, tensors, "distilbert.embeddings.word_embeddings.weight");
        LoadEmbed(enc.posEmbed, tensors, "distilbert.embeddings.position_embeddings.weight");
        LoadLayerNorm(enc.embedLn, tensors, "distilbert.embeddings.LayerNorm");

        for (int i = 0; i < enc.layers.Length; i++)
        {
            var layer = enc.layers[i];
            var prefix = $"distilbert.transformer.layer.{i}";
            LoadLinear(layer.attn.qProj, tensors, $"{prefix}.attention.q_lin");
            LoadLinear(layer.attn.kProj, tensors, $"{prefix}.attention.k_lin");
            LoadLinear(layer.attn.vProj, tensors, $"{prefix}.attention.v_lin");
            LoadLinear(layer.attn.oProj, tensors, $"{prefix}.attention.out_lin");
            LoadLayerNorm(layer.ln1, tensors, $"{prefix}.sa_layer_norm");
            LoadLinear(layer.fc1, tensors, $"{prefix}.ffn.lin1");
            LoadLinear(layer.fc2, tensors, $"{prefix}.ffn.lin2");
            LoadLayerNorm(layer.ln2, tensors, $"{prefix}.output_layer_norm");
        }

        var preCls = Unsafe.As<Linear<float>>(preClassifier);
        if (tensors.ContainsKey("pre_classifier.weight") || tensors.ContainsKey("pre_classifier.bias"))
            LoadLinear(preCls, tensors, "pre_classifier");

        var cls = Unsafe.As<Linear<float>>(classifier);
        if (tensors.ContainsKey("classifier.weight") || tensors.ContainsKey("classifier.bias"))
            LoadLinear(cls, tensors, "classifier");

        Eval();
    }

    static void LoadEmbed(Embedding<float> embed, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        if (!tensors.TryGetValue(key, out var t)) return;
        var tensor = ReverseGradTensor<float>.FromMatrix(t.Data, t.Shape[0], t.Shape[1]);
        embed.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>> { ["Weight"] = tensor });
    }

    static void LoadLinear(Linear<float> linear, Dictionary<string, (float[] Data, int[] Shape)> tensors, string prefix)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = ReverseGradTensor<float>.FromMatrix(w.Data, w.Shape[0], w.Shape[1]);
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = ReverseGradTensor<float>.FromMatrix(b.Data, 1, b.Shape[0]);
        if (dict.Count > 0) linear.LoadStateDict(dict);
    }

    static void LoadLayerNorm(LayerNorm<float> ln, Dictionary<string, (float[] Data, int[] Shape)> tensors, string prefix)
    {
        var dict = new Dictionary<string, ReverseGradTensor<float>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = ReverseGradTensor<float>.FromArray(w.Data);
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = ReverseGradTensor<float>.FromArray(b.Data);
        if (dict.Count > 0) ln.LoadStateDict(dict);
    }
}
