using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.Samples;

public sealed record BertConfig
{
    public int HiddenSize { get; init; } = 384;
    public int NumAttentionHeads { get; init; } = 12;
    public int NumHiddenLayers { get; init; } = 6;
    public int IntermediateSize { get; init; } = 1536;
    public int MaxPositionEmbeddings { get; init; } = 512;
    public int VocabSize { get; init; } = 30522;
    public float LayerNormEps { get; init; } = 1e-12f;

    public static BertConfig FromJson(string json)
    {
        var config = new BertConfig();
        var props = typeof(BertConfig).GetProperties();
        foreach (var prop in props)
        {
            var key = $"\"{JsonSnakeCase(prop.Name)}\"";
            var idx = json.IndexOf(key, StringComparison.OrdinalIgnoreCase);
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

    static string JsonSnakeCase(string pascalCase)
    {
        return string.Concat(pascalCase.Select((c, i) =>
            i > 0 && char.IsUpper(c) ? "_" + char.ToLower(c) : char.ToLower(c).ToString()));
    }
}

internal sealed class BertSelfAttention<T> : Module<T> where T : struct, INumber<T>
{
    internal readonly Linear<T> qProj;
    internal readonly Linear<T> kProj;
    internal readonly Linear<T> vProj;
    internal readonly Linear<T> oProj;
    readonly int _numHeads;
    readonly int _headDim;
    readonly T _scale;

    public BertSelfAttention(int embedDim, int numHeads)
    {
        qProj = new Linear<T>(embedDim, embedDim);
        kProj = new Linear<T>(embedDim, embedDim);
        vProj = new Linear<T>(embedDim, embedDim);
        oProj = new Linear<T>(embedDim, embedDim);
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _scale = T.CreateChecked(1.0 / Math.Sqrt(_headDim));
        RegisterModules(qProj, kProj, vProj, oProj);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);
        return oProj.Forward(MultiHeadAttention(Q, K, V, null));
    }

    public ReverseGradTensor<T> ForwardWithMask(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        ReverseGradTensor<T>? mask = null;
        if (paddingMask != null)
        {
            int L = Q.Shape[0];
            var maskData = new T[L * L];
            for (int j = 0; j < paddingMask.Length && j < L; j++)
                if (paddingMask.Data[j] is T pv && float.CreateChecked(pv) < 0.5f)
                    for (int i = 0; i < L; i++)
                        maskData[i * L + j] = T.CreateChecked(float.NegativeInfinity);
            mask = ReverseGradTensor<T>.FromMatrix(maskData, L, L, requiresGrad: false);
        }

        return oProj.Forward(MultiHeadAttention(Q, K, V, mask));
    }

    ReverseGradTensor<T> MultiHeadAttention(
        ReverseGradTensor<T> Q, ReverseGradTensor<T> K, ReverseGradTensor<T> V,
        ReverseGradTensor<T>? mask)
    {
        int L = Q.Shape[0];
        var scaleData = new T[L * L];
        Array.Fill(scaleData, _scale);
        var scaleTensor = ReverseGradTensor<T>.FromArray(scaleData, requiresGrad: false);
        scaleTensor.Reshape(L, L);

        var heads = new ReverseGradTensor<T>[_numHeads];

        for (int h = 0; h < _numHeads; h++)
        {
            int hs = h * _headDim;

            var Q_h = ReverseGradOperations.Slice(Q, hs, _headDim);
            var K_h = ReverseGradOperations.Slice(K, hs, _headDim);
            var V_h = ReverseGradOperations.Slice(V, hs, _headDim);

            var K_h_T = ReverseGradOperations.Transpose(K_h);
            var scores = ReverseGradOperations.MatMul(Q_h, K_h_T);
            scores = ReverseGradOperations.Multiply(scores, scaleTensor);

            if (mask != null)
                scores = ReverseGradOperations.Add(scores, mask);

            var weights = ReverseGradOperations.Softmax(scores);
            heads[h] = ReverseGradOperations.MatMul(weights, V_h);
        }

        return ReverseGradOperations.Concat(heads, axis: 1);
    }
}

internal sealed class BertLayer<T> : Module<T> where T : struct, INumber<T>
{
    internal readonly LayerNorm<T> ln1;
    internal readonly BertSelfAttention<T> attn;
    internal readonly LayerNorm<T> ln2;
    internal readonly Linear<T> fc1;
    internal readonly Linear<T> fc2;

    public BertLayer(int hiddenSize, int intermediateSize, int numHeads, float eps)
    {
        ln1 = new LayerNorm<T>(hiddenSize, eps);
        attn = new BertSelfAttention<T>(hiddenSize, numHeads);
        ln2 = new LayerNorm<T>(hiddenSize, eps);
        fc1 = new Linear<T>(hiddenSize, intermediateSize, bias: true);
        fc2 = new Linear<T>(intermediateSize, hiddenSize, bias: true);
        RegisterModules(ln1, attn, ln2, fc1, fc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var h = ln1.Forward(input);
        h = attn.Forward(h);
        h = ReverseGradOperations.Add(h, input);

        var h2 = ln2.Forward(h);
        h2 = fc1.Forward(h2);
        h2 = ReverseGradOperations.Gelu(h2);
        h2 = fc2.Forward(h2);
        h2 = ReverseGradOperations.Add(h2, h);
        return h2;
    }

    public ReverseGradTensor<T> ForwardWithMask(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        var h = ln1.Forward(input);
        h = attn.ForwardWithMask(h, paddingMask);
        h = ReverseGradOperations.Add(h, input);

        var h2 = ln2.Forward(h);
        h2 = fc1.Forward(h2);
        h2 = ReverseGradOperations.Gelu(h2);
        h2 = fc2.Forward(h2);
        h2 = ReverseGradOperations.Add(h2, h);
        return h2;
    }
}

internal sealed class BertEncoder<T> : Module<T> where T : struct, INumber<T>
{
    internal readonly Embedding<T> wordEmbed;
    internal readonly Embedding<T> posEmbed;
    internal readonly LayerNorm<T> embedLn;
    internal readonly BertLayer<T>[] layers;

    public BertEncoder(BertConfig config)
    {
        wordEmbed = new Embedding<T>(config.VocabSize, config.HiddenSize);
        posEmbed = new Embedding<T>(config.MaxPositionEmbeddings, config.HiddenSize);
        embedLn = new LayerNorm<T>(config.HiddenSize, config.LayerNormEps);
        layers = new BertLayer<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            layers[i] = new BertLayer<T>(config.HiddenSize, config.IntermediateSize, config.NumAttentionHeads, config.LayerNormEps);

        RegisterModules(wordEmbed, posEmbed, embedLn);
        foreach (var layer in layers)
            RegisterModules(layer);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        int seqLen = input.Shape[0];

        var posIds = new T[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = T.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<T>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(seqLen);

        var wordEmb = wordEmbed.Forward(input);
        var posEmb = posEmbed.Forward(posEmbInput);
        var hidden = ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        foreach (var layer in layers)
            hidden = layer.Forward(hidden);

        return hidden;
    }

    public ReverseGradTensor<T> ForwardWithMask(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        int seqLen = input.Shape[0];

        var posIds = new T[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = T.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<T>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(seqLen);

        var wordEmb = wordEmbed.Forward(input);
        var posEmb = posEmbed.Forward(posEmbInput);
        var hidden = ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        foreach (var layer in layers)
            hidden = layer.ForwardWithMask(hidden, paddingMask);

        return hidden;
    }
}

public sealed class MiniLMDistilled<T> : Module<T> where T : struct, INumber<T>
{
    internal readonly BertEncoder<T> encoder;
    internal readonly BertConfig config;

    public MiniLMDistilled(BertConfig config)
    {
        this.config = config;
        encoder = new BertEncoder<T>(config);
        RegisterModules(encoder);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var hidden = encoder.Forward(input);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    public ReverseGradTensor<T> ForwardWithMask(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        var hidden = encoder.ForwardWithMask(input, paddingMask);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    static ReverseGradTensor<T> ExtractRow(ReverseGradTensor<T> matrix, int row, int cols)
    {
        int offset = row * cols;
        var data = new T[cols];
        matrix.Data.TryGetSpan(out var span);
        if (!span.IsEmpty)
            span.Slice(offset, cols).CopyTo(data);
        else
        {
            var full = new T[matrix.Length];
            matrix.Data.CopyTo(full, default(T)!);
            Array.Copy(full, offset, data, 0, cols);
        }
        var result = ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
        result.Reshape(cols);
        return result;
    }

    static ReverseGradTensor<T> L2Normalize(ReverseGradTensor<T> vec, int hiddenSize)
    {
        int n = vec.Length;
        var data = new T[n];
        vec.Data.CopyTo(data.AsSpan(), default(T)!);
        float norm = 0f;
        for (int i = 0; i < n; i++)
        {
            float val = float.CreateChecked(data[i]);
            norm += val * val;
        }
        norm = MathF.Sqrt(norm);
        if (norm > 1e-12f)
        {
            float invNorm = 1f / norm;
            for (int i = 0; i < n; i++)
                data[i] = T.CreateChecked(float.CreateChecked(data[i]) * invNorm);
        }
        var result = ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
        result.Reshape(hiddenSize);
        return result;
    }

    public static MiniLMDistilled<float> LoadWeights(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertConfig config)
    {
        var model = new MiniLMDistilled<float>(config);
        var enc = model.encoder;

        LoadEmbed(enc.wordEmbed, tensors, "embeddings.word_embeddings.weight");
        LoadEmbed(enc.posEmbed, tensors, "embeddings.position_embeddings.weight");
        LoadLayerNorm(enc.embedLn, tensors, "embeddings.LayerNorm");

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var layer = enc.layers[i];
            LoadLayerNorm(layer.ln1, tensors, $"encoder.layers.{i}.attention.output.LayerNorm");
            LoadLinear(layer.attn.qProj, tensors, $"encoder.layers.{i}.attention.self.query");
            LoadLinear(layer.attn.kProj, tensors, $"encoder.layers.{i}.attention.self.key");
            LoadLinear(layer.attn.vProj, tensors, $"encoder.layers.{i}.attention.self.value");
            LoadLinear(layer.attn.oProj, tensors, $"encoder.layers.{i}.attention.output.dense");
            LoadLayerNorm(layer.ln2, tensors, $"encoder.layers.{i}.output.LayerNorm");
            LoadLinear(layer.fc1, tensors, $"encoder.layers.{i}.intermediate.dense");
            LoadLinear(layer.fc2, tensors, $"encoder.layers.{i}.output.dense");
        }

        model.Eval();
        return model;
    }

    static void LoadEmbed(Embedding<float> embed, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        if (!tensors.TryGetValue(key, out var t)) throw new KeyNotFoundException($"Missing tensor: {key}");
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

public static class MiniLMTokenizer
{
    public static Microsoft.ML.Tokenizers.BertTokenizer Load(string vocabPath)
    {
        return Microsoft.ML.Tokenizers.BertTokenizer.Create(vocabPath);
    }

    public static (float[] TokenIds, float[] AttentionMask, int SeqLen) Encode(
        Microsoft.ML.Tokenizers.BertTokenizer tokenizer,
        string text,
        int maxLen = 128)
    {
        var ids = tokenizer.EncodeToIds(text, addSpecialTokens: true);
        int seqLen = Math.Min(ids.Count, maxLen);
        var tokenIds = new float[maxLen];
        var attentionMask = new float[maxLen];

        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = ids[i];
            attentionMask[i] = 1f;
        }
        for (int i = seqLen; i < maxLen; i++)
            tokenIds[i] = tokenizer.PaddingTokenId;

        return (tokenIds, attentionMask, seqLen);
    }

    public static ReverseGradTensor<float> Tokenize(
        Microsoft.ML.Tokenizers.BertTokenizer tokenizer,
        string text,
        int maxLen = 128)
    {
        var (tokenIds, _, _) = Encode(tokenizer, text, maxLen);
        var input = ReverseGradTensor<float>.FromArray(tokenIds, requiresGrad: false);
        input.Reshape(tokenIds.Length);
        return input;
    }
}
