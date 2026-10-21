using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AI;

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

public class BertSelfAttention<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly Linear<TModel, TWeight> qProj;
    public readonly Linear<TModel, TWeight> kProj;
    public readonly Linear<TModel, TWeight> vProj;
    public readonly Linear<TModel, TWeight> oProj;
    readonly int embedDim;
    readonly int _numHeads;
    readonly TModel _scale;

    public BertSelfAttention(int embedDim, int numHeads)
    {
        qProj = new Linear<TModel, TWeight>(embedDim, embedDim);
        kProj = new Linear<TModel, TWeight>(embedDim, embedDim);
        vProj = new Linear<TModel, TWeight>(embedDim, embedDim);
        oProj = new Linear<TModel, TWeight>(embedDim, embedDim);
        this.embedDim = embedDim;
        _numHeads = numHeads;
        _scale = TModel.CreateChecked(1.0 / Math.Sqrt(embedDim / numHeads));
        RegisterModules(qProj, kProj, vProj, oProj);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);
        return oProj.Forward(MultiHeadAttention(Q, K, V, null));
    }

    public ReverseGradTensor<TModel> ForwardWithValidLength(ReverseGradTensor<TModel> input, int validLength)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        return oProj.Forward(ReverseGradOperations.MultiHeadAttention(Q, K, V, _numHeads, _scale, null, -1, validLength));
    }

    public ReverseGradTensor<TModel> ForwardWithMask(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? paddingMask)
    {
        var Q = qProj.Forward(input);
        var K = kProj.Forward(input);
        var V = vProj.Forward(input);

        ReverseGradTensor<TModel>? mask = null;
        if (paddingMask != null)
        {
            int L = Q.Shape[0];
            var maskData = new TModel[L * L];
            for (int j = 0; j < paddingMask.Length && j < L; j++)
                if (paddingMask.Data[j] is TModel pv && float.CreateChecked(pv) < 0.5f)
                    for (int i = 0; i < L; i++)
                        maskData[i * L + j] = TModel.CreateChecked(float.NegativeInfinity);
            mask = ReverseGradTensor<TModel>.FromMatrix(maskData, L, L, requiresGrad: false);
        }

        return oProj.Forward(MultiHeadAttention(Q, K, V, mask));
    }

    public ReverseGradTensor<TModel> ForwardBatched(
        ReverseGradTensor<TModel> input, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
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

    static ReverseGradTensor<TModel> BuildBatchedPaddingMask(
        ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
    {
        var maskData = new TModel[batchSize * seqLen * seqLen];
        var negInf = TModel.CreateChecked(double.NegativeInfinity);

        for (int b = 0; b < batchSize; b++)
        {
            int rowBase = b * seqLen * seqLen;
            int maskBase = b * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                if (attentionMask.Data[maskBase + j] == TModel.Zero)
                {
                    for (int i = 0; i < seqLen; i++)
                        maskData[rowBase + i * seqLen + j] = negInf;
                }
            }
        }

        var tensor = ReverseGradTensor<TModel>.FromArray(maskData, requiresGrad: false);
        tensor.Reshape(batchSize, seqLen, seqLen);
        return tensor;
    }

    ReverseGradTensor<TModel> MultiHeadAttention(
        ReverseGradTensor<TModel> Q, ReverseGradTensor<TModel> K, ReverseGradTensor<TModel> V,
        ReverseGradTensor<TModel>? mask)
    {
        return ReverseGradOperations.MultiHeadAttention(Q, K, V, _numHeads, _scale, mask);
    }
}

public class BertLayer<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly LayerNorm<TModel> ln1;
    public readonly BertSelfAttention<TModel, TWeight> attn;
    public readonly LayerNorm<TModel> ln2;
    public readonly Linear<TModel, TWeight> fc1;
    public readonly Linear<TModel, TWeight> fc2;

    public BertLayer(int hiddenSize, int intermediateSize, int numHeads, float eps)
    {
        ln1 = new LayerNorm<TModel>(hiddenSize, eps);
        attn = new BertSelfAttention<TModel, TWeight>(hiddenSize, numHeads);
        ln2 = new LayerNorm<TModel>(hiddenSize, eps);
        fc1 = new Linear<TModel, TWeight>(hiddenSize, intermediateSize, bias: true);
        fc2 = new Linear<TModel, TWeight>(intermediateSize, hiddenSize, bias: true);
        RegisterModules(ln1, attn, ln2, fc1, fc2);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
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

    public ReverseGradTensor<TModel> ForwardWithMask(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? paddingMask)
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

    public ReverseGradTensor<TModel> ForwardBatched(
        ReverseGradTensor<TModel> input, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
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

public class BertEncoder<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly Embedding<TModel, TWeight> wordEmbed;
    public readonly Embedding<TModel, TWeight> posEmbed;
    public readonly Embedding<TModel, TWeight>? tokenTypeEmbed;
    public readonly LayerNorm<TModel> embedLn;
    public readonly BertLayer<TModel, TWeight>[] layers;
    readonly bool _includeTokenTypeEmbedding;

    public BertEncoder(BertConfig config, bool includeTokenTypeEmbedding = true)
    {
        _includeTokenTypeEmbedding = includeTokenTypeEmbedding;
        wordEmbed = new Embedding<TModel, TWeight>(config.VocabSize, config.HiddenSize);
        posEmbed = new Embedding<TModel, TWeight>(config.MaxPositionEmbeddings, config.HiddenSize);
        if (includeTokenTypeEmbedding)
            tokenTypeEmbed = new Embedding<TModel, TWeight>(2, config.HiddenSize);
        embedLn = new LayerNorm<TModel>(config.HiddenSize, config.LayerNormEps);
        layers = new BertLayer<TModel, TWeight>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            layers[i] = new BertLayer<TModel, TWeight>(config.HiddenSize, config.IntermediateSize, config.NumAttentionHeads, config.LayerNormEps);

        RegisterModules(wordEmbed, posEmbed, embedLn);
        if (includeTokenTypeEmbedding)
            RegisterModules(tokenTypeEmbed!);
        foreach (var layer in layers)
            RegisterModules(layer);
    }

    ReverseGradTensor<TModel> TokenTypeEmb(int len)
    {
        var ttIds = new TModel[len];
        Array.Fill(ttIds, TModel.Zero);
        var ttInput = ReverseGradTensor<TModel>.FromArray(ttIds, requiresGrad: false);
        ttInput.Reshape(len);
        return tokenTypeEmbed!.Forward(ttInput);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        int seqLen = input.Shape[0];

        var posIds = new TModel[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
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

    public ReverseGradTensor<TModel> ForwardWithMask(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? paddingMask)
    {
        int seqLen = input.Shape[0];

        var posIds = new TModel[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
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

    public ReverseGradTensor<TModel> ForwardBatched(
        ReverseGradTensor<TModel> input, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
    {
        var posIds = new TModel[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < seqLen; i++)
                posIds[b * seqLen + i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
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

    /// <summary>
    /// Exact-integer-index variant of <see cref="Forward(ReverseGradTensor{TModel})"/>. Token IDs
    /// are supplied as <see cref="int"/> and never round-tripped through the compute dtype
    /// <typeparamref name="TModel"/>, so this is the correct overload for narrow-precision dtypes
    /// (BFloat16 / Half) whose integer range cannot represent typical vocabularies.
    /// </summary>
    public ReverseGradTensor<TModel> Forward(int[] inputIds)
    {
        int seqLen = inputIds.Length;

        var posIds = new TModel[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(seqLen);

        var wordEmb = wordEmbed.Forward(inputIds);
        var posEmb = posEmbed.Forward(posEmbInput);
        var hidden = _includeTokenTypeEmbedding
            ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(seqLen)))
            : ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        for (int i = 0; i < layers.Length; i++)
            hidden = layers[i].Forward(hidden);

        return hidden;
    }

    /// <summary>
    /// Exact-integer-index variant of <see cref="ForwardWithMask(ReverseGradTensor{TModel}, ReverseGradTensor{TModel}?)"/>.
    /// See <see cref="Forward(int[])"/> for why token IDs are passed as <see cref="int"/>.
    /// </summary>
    public ReverseGradTensor<TModel> ForwardWithMask(int[] inputIds, ReverseGradTensor<TModel>? paddingMask)
    {
        int seqLen = inputIds.Length;

        var posIds = new TModel[seqLen];
        for (int i = 0; i < seqLen; i++) posIds[i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(seqLen);

        var wordEmb = wordEmbed.Forward(inputIds);
        var posEmb = posEmbed.Forward(posEmbInput);
        var hidden = _includeTokenTypeEmbedding
            ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(seqLen)))
            : ReverseGradOperations.Add(wordEmb, posEmb);
        hidden = embedLn.Forward(hidden);

        foreach (var layer in layers)
            hidden = layer.ForwardWithMask(hidden, paddingMask);

        return hidden;
    }

    /// <summary>
    /// Exact-integer-index variant of
    /// <see cref="ForwardBatched(ReverseGradTensor{TModel}, ReverseGradTensor{TModel}, int, int)"/>.
    /// See <see cref="Forward(int[])"/> for why token IDs are passed as <see cref="int"/>.
    /// </summary>
    public ReverseGradTensor<TModel> ForwardBatched(int[] inputIds, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
    {
        var posIds = new TModel[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < seqLen; i++)
                posIds[b * seqLen + i] = TModel.CreateChecked(i);
        var posEmbInput = ReverseGradTensor<TModel>.FromArray(posIds, requiresGrad: false);
        posEmbInput.Reshape(batchSize * seqLen);

        var wordEmb = wordEmbed.Forward(inputIds);
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

public class MiniLMDistilled<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly BertEncoder<TModel, TWeight> encoder;
    public readonly BertConfig config;

    public MiniLMDistilled(BertConfig config)
    {
        this.config = config;
        encoder = new BertEncoder<TModel, TWeight>(config);
        RegisterModules(encoder);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        var hidden = encoder.Forward(input);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    public ReverseGradTensor<TModel> ForwardWithMask(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? paddingMask)
    {
        var hidden = encoder.ForwardWithMask(input, paddingMask);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    public ReverseGradTensor<TModel> ForwardBatched(
        ReverseGradTensor<TModel> input, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
    {
        return encoder.ForwardBatched(input, attentionMask, batchSize, seqLen);
    }

    /// <summary>
    /// Exact-integer-index variant of <see cref="Forward(ReverseGradTensor{TModel})"/>. Token IDs
    /// are passed as <see cref="int"/> (see <see cref="BertEncoder{TModel}.Forward(int[])"/>).
    /// </summary>
    public ReverseGradTensor<TModel> Forward(int[] inputIds)
    {
        var hidden = encoder.Forward(inputIds);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    /// <summary>
    /// Exact-integer-index variant of <see cref="ForwardWithMask(ReverseGradTensor{TModel}, ReverseGradTensor{TModel}?)"/>.
    /// </summary>
    public ReverseGradTensor<TModel> ForwardWithMask(int[] inputIds, ReverseGradTensor<TModel>? paddingMask)
    {
        var hidden = encoder.ForwardWithMask(inputIds, paddingMask);
        var clsToken = ExtractRow(hidden, 0, config.HiddenSize);
        return L2Normalize(clsToken, config.HiddenSize);
    }

    /// <summary>
    /// Exact-integer-index variant of
    /// <see cref="ForwardBatched(ReverseGradTensor{TModel}, ReverseGradTensor{TModel}, int, int)"/>.
    /// </summary>
    public ReverseGradTensor<TModel> ForwardBatched(int[] inputIds, ReverseGradTensor<TModel> attentionMask, int batchSize, int seqLen)
    {
        return encoder.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
    }

    static ReverseGradTensor<TModel> ExtractRow(ReverseGradTensor<TModel> matrix, int row, int cols)
    {
        int offset = row * cols;
        var data = new TModel[cols];
        matrix.Data.TryGetSpan(out var span);
        if (!span.IsEmpty)
            span.Slice(offset, cols).CopyTo(data);
        else
        {
            var full = new TModel[matrix.Length];
            matrix.Data.CopyTo(full, default(TModel)!);
            Array.Copy(full, offset, data, 0, cols);
        }
        var result = ReverseGradTensor<TModel>.FromArray(data, requiresGrad: false);
        result.Reshape(cols);
        return result;
    }

    static ReverseGradTensor<TModel> L2Normalize(ReverseGradTensor<TModel> vec, int hiddenSize)
    {
        int n = vec.Length;
        var data = new TModel[n];
        vec.Data.CopyTo(data.AsSpan(), default(TModel)!);
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
                data[i] = TModel.CreateChecked(float.CreateChecked(data[i]) * invNorm);
        }
        var result = ReverseGradTensor<TModel>.FromArray(data, requiresGrad: false);
        result.Reshape(hiddenSize);
        return result;
    }

    public static MiniLMDistilled<TModel, TWeight> LoadWeights(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        BertConfig config)
    {
        var model = new MiniLMDistilled<TModel, TWeight>(config);
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

public sealed class BertSelfAttention<T> : BertSelfAttention<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public BertSelfAttention(int embedDim, int numHeads) : base(embedDim, numHeads) { }
}

public sealed class BertLayer<T> : BertLayer<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public BertLayer(int hiddenSize, int intermediateSize, int numHeads, float eps) : base(hiddenSize, intermediateSize, numHeads, eps) { }
}

public sealed class BertEncoder<T> : BertEncoder<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public BertEncoder(BertConfig config, bool includeTokenTypeEmbedding = true) : base(config, includeTokenTypeEmbedding) { }
}

public sealed class MiniLMDistilled<T> : MiniLMDistilled<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public MiniLMDistilled(BertConfig config) : base(config) { }
}
