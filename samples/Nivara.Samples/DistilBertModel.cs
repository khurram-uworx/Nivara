using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.Samples;

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
        encoder = new BertEncoder<T>(config, includeTokenTypeEmbedding: false);
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
        h = ReverseGradOperations.Relu(h);
        var logits = classifier.Forward(h);
        return logits;
    }

    /// <summary>
    /// Exact-integer-index variant of <see cref="Forward(ReverseGradTensor{T}, ReverseGradTensor{T}, int, int)"/>.
    /// Token IDs are passed as <see cref="int"/> so they survive narrow-precision compute
    /// dtypes (BFloat16 / Half) that cannot represent typical vocabularies.
    /// </summary>
    public ReverseGradTensor<T> Forward(
        int[] inputIds,
        ReverseGradTensor<T> attentionMask,
        int batchSize,
        int seqLen)
    {
        var encoded = encoder.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
        var clsTokens = ExtractClsTokens(encoded, batchSize, seqLen);
        var h = preClassifier.Forward(clsTokens);
        h = ReverseGradOperations.Relu(h);
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
        DistilBertLoader.LoadEncoderWeights<T, TWeight>(encoder, tensors, "distilbert");

        if (tensors.ContainsKey("pre_classifier.weight") || tensors.ContainsKey("pre_classifier.bias"))
            StateDictLoader.LoadLinearWidened(preClassifier, tensors, "pre_classifier");

        if (tensors.ContainsKey("classifier.weight") || tensors.ContainsKey("classifier.bias"))
            StateDictLoader.LoadLinearWidened(classifier, tensors, "classifier");

        Eval();
    }

    /// <summary>
    /// Walks this model's parameters in registration order and produces the Silk/OpenCL
    /// checkpoint-shaped tensor dictionary (HF-style <c>[out, in]</c> arrays) consumed by
    /// <see cref="Nivara.Silk.SilkBertEncoderRunner"/>. Registration order is the contract:
    /// encoder (word/pos embeddings, embed LayerNorm, per-layer norms/attention/ffn) then
    /// <c>pre_classifier</c> then <c>classifier</c>.
    /// </summary>
    public Dictionary<string, (float[] Data, int[] Shape)> ToSilkTensors()
    {
        if (typeof(T) != typeof(float))
            throw new InvalidOperationException("ToSilkTensors is only supported for float models.");

        var result = new Dictionary<string, (float[] Data, int[] Shape)>();
        var parameters = GetParameters().Values.ToList();
        int idx = 0;

        Func<Parameter<T>, (float[] Data, int[] Shape)> conv = param =>
        {
            var tensor = param.Tensor;
            var src = new T[tensor.Length];
            tensor.Data.CopyTo(src, default(T)!);
            var data = new float[src.Length];
            for (int j = 0; j < src.Length; j++)
                data[j] = float.CreateChecked(src[j]);
            // Linear biases are stored as [1, outFeatures] in the AutoDiff tensor but the
            // checkpoint (and Silk) convention is a flat [outFeatures] vector; squeeze so the
            // device upload and LoadWeights round-trip agree.
            var shape = (int[])tensor.Shape.Clone();
            if (shape.Length == 2 && shape[0] == 1)
                shape = new[] { shape[1] };
            return (data, shape);
        };

        string emb = "distilbert.embeddings";
        result[$"{emb}.word_embeddings.weight"] = conv(parameters[idx++]);
        result[$"{emb}.position_embeddings.weight"] = conv(parameters[idx++]);
        result[$"{emb}.LayerNorm.weight"] = conv(parameters[idx++]);
        result[$"{emb}.LayerNorm.bias"] = conv(parameters[idx++]);

        for (int l = 0; l < encoder.layers.Length; l++)
        {
            string layer = $"distilbert.transformer.layer.{l}";
            result[$"{layer}.sa_layer_norm.weight"] = conv(parameters[idx++]);
            result[$"{layer}.sa_layer_norm.bias"] = conv(parameters[idx++]);
            foreach (var proj in new[] { "q_lin", "k_lin", "v_lin", "out_lin" })
            {
                result[$"{layer}.attention.{proj}.weight"] = conv(parameters[idx++]);
                result[$"{layer}.attention.{proj}.bias"] = conv(parameters[idx++]);
            }
            result[$"{layer}.output_layer_norm.weight"] = conv(parameters[idx++]);
            result[$"{layer}.output_layer_norm.bias"] = conv(parameters[idx++]);
            result[$"{layer}.ffn.lin1.weight"] = conv(parameters[idx++]);
            result[$"{layer}.ffn.lin1.bias"] = conv(parameters[idx++]);
            result[$"{layer}.ffn.lin2.weight"] = conv(parameters[idx++]);
            result[$"{layer}.ffn.lin2.bias"] = conv(parameters[idx++]);
        }

        result["pre_classifier.weight"] = conv(parameters[idx++]);
        result["pre_classifier.bias"] = conv(parameters[idx++]);
        result["classifier.weight"] = conv(parameters[idx++]);
        result["classifier.bias"] = conv(parameters[idx++]);

        if (idx != parameters.Count)
            throw new InvalidOperationException($"Unexpected parameter count: consumed {idx} of {parameters.Count}.");
        return result;
    }
}

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
        StateDictLoader.LoadEmbedWidened(encoder.wordEmbed, tensors, $"{prefix}.embeddings.word_embeddings.weight");
        StateDictLoader.LoadEmbedWidened(encoder.posEmbed, tensors, $"{prefix}.embeddings.position_embeddings.weight");
        StateDictLoader.LoadLayerNorm(encoder.embedLn, tensors, $"{prefix}.embeddings.LayerNorm");

        for (int i = 0; i < encoder.layers.Length; i++)
        {
            var layer = encoder.layers[i];
            var p = $"{prefix}.transformer.layer.{i}";
            StateDictLoader.LoadLinearWidened(layer.attn.qProj, tensors, $"{p}.attention.q_lin");
            StateDictLoader.LoadLinearWidened(layer.attn.kProj, tensors, $"{p}.attention.k_lin");
            StateDictLoader.LoadLinearWidened(layer.attn.vProj, tensors, $"{p}.attention.v_lin");
            StateDictLoader.LoadLinearWidened(layer.attn.oProj, tensors, $"{p}.attention.out_lin");
            StateDictLoader.LoadLayerNorm(layer.ln1, tensors, $"{p}.sa_layer_norm");
            StateDictLoader.LoadLinearWidened(layer.fc1, tensors, $"{p}.ffn.lin1");
            StateDictLoader.LoadLinearWidened(layer.fc2, tensors, $"{p}.ffn.lin2");
            StateDictLoader.LoadLayerNorm(layer.ln2, tensors, $"{p}.output_layer_norm");
        }
    }
}
