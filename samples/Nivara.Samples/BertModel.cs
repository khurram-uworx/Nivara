using Nivara.AI;
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

public sealed class BertSelfAttention<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly Linear<T> qProj;
    public readonly Linear<T> kProj;
    public readonly Linear<T> vProj;
    public readonly Linear<T> oProj;
    readonly int embedDim;
    readonly int _numHeads;
    readonly T _scale;

    public BertSelfAttention(int embedDim, int numHeads)
    {
        qProj = new Linear<T>(embedDim, embedDim);
        kProj = new Linear<T>(embedDim, embedDim);
        vProj = new Linear<T>(embedDim, embedDim);
        oProj = new Linear<T>(embedDim, embedDim);
        this.embedDim = embedDim;
        _numHeads = numHeads;
        _scale = T.CreateChecked(1.0 / Math.Sqrt(embedDim / numHeads));
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

    public ReverseGradTensor<T> ForwardBatched(
        ReverseGradTensor<T> input, ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        Q.Reshape(batchSize, seqLen, embedDim);
        K.Reshape(batchSize, seqLen, embedDim);
        V.Reshape(batchSize, seqLen, embedDim);

        var mask = BuildBatchedPaddingMask(attentionMask, batchSize, seqLen);
        var attn = ReverseGradOperations.BatchedMultiHeadAttention(Q, K, V, _numHeads, _scale, mask);
        attn.Reshape(batchSize * seqLen, embedDim);

        return oProj.Forward(attn);
    }

    static ReverseGradTensor<T> BuildBatchedPaddingMask(
        ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        var maskData = new T[batchSize * seqLen * seqLen];
        var negInf = T.CreateChecked(double.NegativeInfinity);

        for (int b = 0; b < batchSize; b++)
        {
            int rowBase = b * seqLen * seqLen;
            int maskBase = b * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                if (attentionMask.Data[maskBase + j] == T.Zero)
                {
                    for (int i = 0; i < seqLen; i++)
                        maskData[rowBase + i * seqLen + j] = negInf;
                }
            }
        }

        var tensor = ReverseGradTensor<T>.FromArray(maskData, requiresGrad: false);
        tensor.Reshape(batchSize, seqLen, seqLen);
        return tensor;
    }

    ReverseGradTensor<T> MultiHeadAttention(
        ReverseGradTensor<T> Q, ReverseGradTensor<T> K, ReverseGradTensor<T> V,
        ReverseGradTensor<T>? mask)
    {
        return ReverseGradOperations.MultiHeadAttention(Q, K, V, _numHeads, _scale, mask);
    }
}

public sealed class BertLayer<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly LayerNorm<T> ln1;
    public readonly BertSelfAttention<T> attn;
    public readonly LayerNorm<T> ln2;
    public readonly Linear<T> fc1;
    public readonly Linear<T> fc2;

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
        var h = attn.Forward(input);
        h = ReverseGradOperations.Add(h, input);
        h = ln1.Forward(h);

        var h2 = fc1.Forward(h);
        h2 = ReverseGradOperations.GeluExact(h2);
        h2 = fc2.Forward(h2);
        h2 = ReverseGradOperations.Add(h2, h);
        h2 = ln2.Forward(h2);
        return h2;
    }

    public ReverseGradTensor<T> ForwardWithMask(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        var h = attn.ForwardWithMask(input, paddingMask);
        h = ReverseGradOperations.Add(h, input);
        h = ln1.Forward(h);

        var h2 = fc1.Forward(h);
        h2 = ReverseGradOperations.GeluExact(h2);
        h2 = fc2.Forward(h2);
        h2 = ReverseGradOperations.Add(h2, h);
        h2 = ln2.Forward(h2);
        return h2;
    }

    public ReverseGradTensor<T> ForwardBatched(
        ReverseGradTensor<T> input, ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        var h = attn.ForwardBatched(input, attentionMask, batchSize, seqLen);
        h = ReverseGradOperations.Add(h, input);
        h = ln1.Forward(h);

        var h2 = fc1.Forward(h);
        h2 = ReverseGradOperations.GeluExact(h2);
        h2 = fc2.Forward(h2);
        h2 = ReverseGradOperations.Add(h2, h);
        h2 = ln2.Forward(h2);
        return h2;
    }
}

public sealed class BertEncoder<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly Embedding<T> wordEmbed;
    public readonly Embedding<T> posEmbed;
    public readonly Embedding<T>? tokenTypeEmbed;
    public readonly LayerNorm<T> embedLn;
    public readonly BertLayer<T>[] layers;
    readonly bool _includeTokenTypeEmbedding;

    public BertEncoder(BertConfig config, bool includeTokenTypeEmbedding = true)
    {
        _includeTokenTypeEmbedding = includeTokenTypeEmbedding;
        wordEmbed = new Embedding<T>(config.VocabSize, config.HiddenSize);
        posEmbed = new Embedding<T>(config.MaxPositionEmbeddings, config.HiddenSize);
        if (includeTokenTypeEmbedding)
            tokenTypeEmbed = new Embedding<T>(2, config.HiddenSize);
        embedLn = new LayerNorm<T>(config.HiddenSize, config.LayerNormEps);
        layers = new BertLayer<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            layers[i] = new BertLayer<T>(config.HiddenSize, config.IntermediateSize, config.NumAttentionHeads, config.LayerNormEps);

        RegisterModules(wordEmbed, posEmbed, embedLn);
        if (includeTokenTypeEmbedding)
            RegisterModules(tokenTypeEmbed!);
        foreach (var layer in layers)
            RegisterModules(layer);
    }

    ReverseGradTensor<T> TokenTypeEmb(int len)
    {
        var ttIds = new T[len];
        Array.Fill(ttIds, T.Zero);
        var ttInput = ReverseGradTensor<T>.FromArray(ttIds, requiresGrad: false);
        ttInput.Reshape(len);
        return tokenTypeEmbed!.Forward(ttInput);
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
        var hidden = _includeTokenTypeEmbedding
            ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(seqLen)))
            : ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        for (int i = 0; i < layers.Length; i++)
            hidden = layers[i].Forward(hidden);

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
        var hidden = _includeTokenTypeEmbedding
            ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(seqLen)))
            : ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        foreach (var layer in layers)
            hidden = layer.ForwardWithMask(hidden, paddingMask);

        return hidden;
    }

    public ReverseGradTensor<T> ForwardBatched(
        ReverseGradTensor<T> input, ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        var posIds = new T[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < seqLen; i++)
                posIds[b * seqLen + i] = T.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<T>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(batchSize * seqLen);

        var wordEmb = wordEmbed.Forward(input);
        var posEmb = posEmbed.Forward(posEmbInput);
        var hidden = _includeTokenTypeEmbedding
            ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(batchSize * seqLen)))
            : ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        foreach (var layer in layers)
            hidden = layer.ForwardBatched(hidden, attentionMask, batchSize, seqLen);

        return hidden;
    }
}

public sealed class MiniLMDistilled<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly BertEncoder<T> encoder;
    public readonly BertConfig config;

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

    public ReverseGradTensor<T> ForwardBatched(
        ReverseGradTensor<T> input, ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        return encoder.ForwardBatched(input, attentionMask, batchSize, seqLen);
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

    public static MiniLMDistilled<TModel> LoadWeights<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        BertConfig config)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var model = new MiniLMDistilled<TModel>(config);
        var enc = model.encoder;

        StateDictLoader.LoadEmbed(enc.wordEmbed, tensors, "embeddings.word_embeddings.weight");
        StateDictLoader.LoadEmbed(enc.posEmbed, tensors, "embeddings.position_embeddings.weight");
        if (enc.tokenTypeEmbed != null)
            StateDictLoader.LoadEmbed(enc.tokenTypeEmbed, tensors, "embeddings.token_type_embeddings.weight");
        StateDictLoader.LoadLayerNorm(enc.embedLn, tensors, "embeddings.LayerNorm");

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var layer = enc.layers[i];
            StateDictLoader.LoadLayerNorm(layer.ln1, tensors, $"encoder.layer.{i}.attention.output.LayerNorm");
            StateDictLoader.LoadLinear(layer.attn.qProj, tensors, $"encoder.layer.{i}.attention.self.query");
            StateDictLoader.LoadLinear(layer.attn.kProj, tensors, $"encoder.layer.{i}.attention.self.key");
            StateDictLoader.LoadLinear(layer.attn.vProj, tensors, $"encoder.layer.{i}.attention.self.value");
            StateDictLoader.LoadLinear(layer.attn.oProj, tensors, $"encoder.layer.{i}.attention.output.dense");
            StateDictLoader.LoadLayerNorm(layer.ln2, tensors, $"encoder.layer.{i}.output.LayerNorm");
            StateDictLoader.LoadLinear(layer.fc1, tensors, $"encoder.layer.{i}.intermediate.dense");
            StateDictLoader.LoadLinear(layer.fc2, tensors, $"encoder.layer.{i}.output.dense");
        }

        model.Eval();
        return model;
    }

    public static MiniLMDistilled<float> LoadWeights(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertConfig config)
        => LoadWeights<float, float>(tensors, config);
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

    public static (ReverseGradTensor<float> Input, ReverseGradTensor<float>? Mask) TokenizeWithMask(
        Microsoft.ML.Tokenizers.BertTokenizer tokenizer,
        string text,
        int maxLen = 128)
    {
        var (tokenIds, attnMask, _) = Encode(tokenizer, text, maxLen);
        var input = ReverseGradTensor<float>.FromArray(tokenIds, requiresGrad: false);
        input.Reshape(tokenIds.Length);
        ReverseGradTensor<float>? mask = null;
        if (attnMask.Any(m => m < 0.5f))
        {
            mask = ReverseGradTensor<float>.FromArray(attnMask, requiresGrad: false);
            mask.Reshape(attnMask.Length);
        }
        return (input, mask);
    }
}

public static class MiniLMEmbeddingGenerator
{
    public static NivaraEmbeddingGenerator<string> Create(
        string modelDir,
        int maxLen = 128,
        string providerName = "Nivara-MiniLM")
    {
        var tensors = SafeTensorsLoader.Read(Path.Combine(modelDir, "model.safetensors"));
        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));

        Func<string, float[]> embeddingFactory = text =>
        {
            var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, text, maxLen);
            var output = mask != null
                ? model.ForwardWithMask(input, mask)
                : model.Forward(input);

            var result = new float[output.Length];
            output.Data.TryGetSpan(out var span);
            if (!span.IsEmpty)
                span.Slice(0, output.Length).CopyTo(result);
            else
                output.Data.CopyTo(result, 0f);
            return result;
        };

        return new NivaraEmbeddingGenerator<string>(
            embeddingFactory,
            config.HiddenSize,
            providerName,
            defaultModelId: "all-minilm-l6-v2");
    }
}
