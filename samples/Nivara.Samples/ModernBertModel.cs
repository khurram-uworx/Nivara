using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;
using System.Text.Json;

namespace Nivara.Samples;

/// <summary>
/// Configuration for a ModernBERT encoder. Understands both the 4.47-era Hub layout
/// (<c>global_rope_theta</c> / <c>local_rope_theta</c> with <c>layer_types</c> derived from
/// <c>global_attn_every_n_layers</c>) and the newer explicit layout (<c>layer_types</c> array plus
/// <c>rope_parameters</c>), so the stock backbone and fine-tuned copies that nest the encoder under
/// a different tensor prefix share one parser.
/// </summary>
public sealed record ModernBertConfig
{
    const string FullAttentionType = "full_attention";
    const string SlidingAttentionType = "sliding_attention";

    public int HiddenSize { get; init; } = 1024;
    public int NumAttentionHeads { get; init; } = 16;
    public int NumHiddenLayers { get; init; } = 28;
    public int IntermediateSize { get; init; } = 2624;
    public int VocabSize { get; init; } = 50368;
    public int MaxPositionEmbeddings { get; init; } = 8192;
    public float NormEps { get; init; } = 1e-5f;
    public int LocalAttention { get; init; } = 128;
    public int GlobalAttnEveryNLayers { get; init; } = 3;

    /// <summary>
    /// Gets or sets the per-layer attention type exactly as the checkpoint declared it, or null when
    /// it did not. <see cref="FromJson(string)"/> sets this when the checkpoint ships a
    /// <c>layer_types</c> array.
    /// </summary>
    public IReadOnlyList<string>? ExplicitLayerTypes { get; init; }

    /// <summary>
    /// Gets the per-layer attention type. Left unset, it is derived the way HuggingFace derives it — a
    /// layer attends globally when its index is a multiple of
    /// <see cref="GlobalAttnEveryNLayers"/> — so a hand-built config is usable without repeating
    /// the pattern.
    /// </summary>
    /// <remarks>
    /// Deliberately computed rather than lazily cached in a private field. A record's synthesized
    /// equality compares every instance field, so a cache field would make two equal configs compare
    /// unequal depending on whether <c>LayerTypes</c> had been read, and <c>with</c> would carry a
    /// stale array across a change to <see cref="NumHiddenLayers"/>. Deriving 28 interned strings per
    /// call is negligible — it happens once per layer during construction.
    /// </remarks>
    public IReadOnlyList<string> LayerTypes
        => ExplicitLayerTypes ?? DeriveLayerTypes(GlobalAttnEveryNLayers, NumHiddenLayers);

    public float RopeThetaFull { get; init; } = 160000f;
    public float RopeThetaSliding { get; init; } = 10000f;
    public string HiddenActivation { get; init; } = "gelu";
    public int PadTokenId { get; init; } = 50283;
    public int ClsTokenId { get; init; } = 50281;
    public int SepTokenId { get; init; } = 50282;
    public int MaskTokenId { get; init; } = 50284;

    /// <summary>Gets the per-head dimension (<c>hidden_size / num_attention_heads</c>).</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>
    /// Gets the sliding-window half-width. ModernBERT treats <c>local_attention</c> as the
    /// <em>total</em> window, so the inclusive distance bound is half of it.
    /// </summary>
    public int SlidingWindow => LocalAttention / 2;

    /// <summary>Gets a value indicating whether the layer at <paramref name="layer"/> attends globally.</summary>
    public bool IsFullAttention(int layer) => LayerTypes[layer] == FullAttentionType;

    /// <summary>Gets the rope theta for the layer at <paramref name="layer"/>, which is per layer type.</summary>
    public float RopeTheta(int layer) => IsFullAttention(layer) ? RopeThetaFull : RopeThetaSliding;

    /// <summary>
    /// Gets the inclusive attention band for the layer at <paramref name="layer"/>: the largest
    /// <c>abs(i - j)</c> a query may reach, or -1 for unlimited (full-attention) layers.
    /// </summary>
    public int Band(int layer) => IsFullAttention(layer) ? -1 : SlidingWindow;

    public static ModernBertConfig FromJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var numLayers = ReadInt(root, "num_hidden_layers", 28);
        var config = new ModernBertConfig
        {
            HiddenSize = ReadInt(root, "hidden_size", 1024),
            NumAttentionHeads = ReadInt(root, "num_attention_heads", 16),
            NumHiddenLayers = numLayers,
            IntermediateSize = ReadInt(root, "intermediate_size", 2624),
            VocabSize = ReadInt(root, "vocab_size", 50368),
            MaxPositionEmbeddings = ReadInt(root, "max_position_embeddings", 8192),
            NormEps = ReadFloat(root, "norm_eps") ?? ReadFloat(root, "layer_norm_eps") ?? 1e-5f,
            LocalAttention = ReadInt(root, "local_attention", 128),
            GlobalAttnEveryNLayers = ReadInt(root, "global_attn_every_n_layers", 3),
            HiddenActivation = ReadString(root, "hidden_activation") ?? "gelu",
            PadTokenId = ReadInt(root, "pad_token_id", 50283),
            ClsTokenId = ReadInt(root, "cls_token_id", 50281),
            SepTokenId = ReadInt(root, "sep_token_id", 50282),
            MaskTokenId = ReadInt(root, "mask_token_id", 50284),
        };

        if (config.HiddenActivation != "gelu")
            throw new NotSupportedException(
                $"ModernBERT hidden_activation '{config.HiddenActivation}' is not supported; " +
                "only exact-erf 'gelu' is implemented.");
        if (config.HiddenSize % config.NumAttentionHeads != 0)
            throw new InvalidOperationException(
                $"hidden_size {config.HiddenSize} is not a multiple of num_attention_heads {config.NumAttentionHeads}.");

        return config with
        {
            ExplicitLayerTypes = ReadLayerTypes(root, config.GlobalAttnEveryNLayers, numLayers),
            RopeThetaFull = ReadRopeTheta(root, FullAttentionType, "global_rope_theta", 160000f),
            RopeThetaSliding = ReadRopeTheta(root, SlidingAttentionType, "local_rope_theta", 10000f),
        };
    }

    static IReadOnlyList<string> DeriveLayerTypes(int globalEveryN, int numLayers)
    {
        if (globalEveryN <= 0)
            throw new InvalidOperationException($"global_attn_every_n_layers must be positive, got {globalEveryN}.");
        if (numLayers <= 0)
            throw new InvalidOperationException($"num_hidden_layers must be positive, got {numLayers}.");

        var derived = new string[numLayers];
        for (int i = 0; i < numLayers; i++)
            derived[i] = i % globalEveryN == 0 ? FullAttentionType : SlidingAttentionType;
        return derived;
    }

    /// <summary>
    /// Reads the explicit <c>layer_types</c> array when present, otherwise reproduces the
    /// HuggingFace derivation: a layer attends globally when its index is a multiple of
    /// <c>global_attn_every_n_layers</c>.
    /// </summary>
    static IReadOnlyList<string> ReadLayerTypes(JsonElement root, int globalEveryN, int numLayers)
    {
        if (root.TryGetProperty("layer_types", out var layerTypes)
            && layerTypes.ValueKind == JsonValueKind.Array)
        {
            var declared = layerTypes.EnumerateArray().Select(t => t.GetString() ?? SlidingAttentionType).ToArray();
            if (declared.Length != numLayers)
                throw new InvalidOperationException(
                    $"layer_types has {declared.Length} entries but num_hidden_layers is {numLayers}. " +
                    "HuggingFace validates this too; a short array would otherwise surface as an " +
                    "IndexOutOfRangeException from IsFullAttention, and a long one would be " +
                    "silently ignored.");
            return declared;
        }

        return DeriveLayerTypes(globalEveryN, numLayers);
    }

    /// <summary>
    /// Reads the rope theta for a layer type from the newer <c>rope_parameters</c> object, falling
    /// back to the flat <c>global_rope_theta</c> / <c>local_rope_theta</c> keys.
    /// </summary>
    static float ReadRopeTheta(JsonElement root, string layerType, string legacyKey, float fallback)
    {
        if (root.TryGetProperty("rope_parameters", out var rope)
            && rope.ValueKind == JsonValueKind.Object
            && rope.TryGetProperty(layerType, out var layerParams)
            && layerParams.ValueKind == JsonValueKind.Object
            && layerParams.TryGetProperty("rope_theta", out var theta)
            && theta.ValueKind == JsonValueKind.Number
            && theta.TryGetSingle(out float nested))
        {
            return nested;
        }

        return ReadFloat(root, legacyKey) ?? fallback;
    }

    static int ReadInt(JsonElement root, string name, int fallback)
        => root.TryGetProperty(name, out var value)
           && value.ValueKind == JsonValueKind.Number
           && value.TryGetInt32(out int parsed)
            ? parsed
            : fallback;

    static float? ReadFloat(JsonElement root, string name)
        => root.TryGetProperty(name, out var value)
           && value.ValueKind == JsonValueKind.Number
           && value.TryGetSingle(out float parsed)
            ? parsed
            : null;

    static string? ReadString(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
}

/// <summary>
/// Builds the additive attention masks ModernBERT needs: a symmetric inclusive sliding-window band
/// (<c>abs(i - j) &lt;= local_attention / 2</c>, matching HuggingFace's
/// <c>sliding_window_bidirectional_overlay</c>) intersected with the padding mask. Suppressed
/// positions carry negative infinity, so their softmax weight is exactly zero.
/// </summary>
/// <remarks>
/// Attention is bidirectional, so a padding row far enough from the valid region can end up with
/// every key suppressed. What such a row produces is an artifact of the mask constant rather than a
/// meaningful value, and the two implementations do <em>not</em> agree on it: this mask is additive
/// <c>-inf</c>, so the row max is <c>-inf</c> and the safe-softmax clamp returns zeros, while
/// HuggingFace masks with <c>torch.finfo(dtype).min</c>, so its row max is finite and it returns a
/// uniform distribution. Both are finite. Either way it never leaks into the valid positions,
/// because a valid query never reads a padding key, so compare only the valid positions against a
/// reference.
/// </remarks>
public static class ModernBertMasks
{
    /// <summary>
    /// The largest sequence length for which a dense <c>[L, L]</c> mask is built. At the model's
    /// full 8192-token context one dense mask is 67M elements (268 MB in F32), so longer inputs
    /// throw instead of silently allocating. A banded attention kernel is the proper fix and is
    /// tracked as a follow-up.
    /// </summary>
    public const int MaxDenseLength = 2048;

    /// <summary>
    /// Builds a single <c>[seqLen, seqLen]</c> additive mask. <paramref name="band"/> is the largest
    /// inclusive <c>abs(i - j)</c> a query may reach, or -1 for no distance limit.
    /// <paramref name="validLength"/> is the number of leading positions that are real tokens; the
    /// rest are padding and are suppressed for every query.
    /// </summary>
    public static ReverseGradTensor<T> Build<T>(int seqLen, int band, int validLength)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen), $"seqLen must be positive, got {seqLen}.");
        if (seqLen > MaxDenseLength)
            throw new ArgumentOutOfRangeException(
                nameof(seqLen),
                $"seqLen {seqLen} exceeds the dense-mask limit of {MaxDenseLength} " +
                $"(a [L, L] mask at the model's 8192-token context would be 268 MB). " +
                "A banded attention kernel is required for longer sequences.");
        if (validLength <= 0 || validLength > seqLen)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength {validLength} must be within [1, {seqLen}].");

        var data = new T[seqLen * seqLen];
        var negInf = T.CreateChecked(double.NegativeInfinity);

        for (int i = 0; i < seqLen; i++)
        {
            int rowBase = i * seqLen;
            int firstAllowed = band < 0 ? 0 : Math.Max(0, i - band);
            int lastAllowed = band < 0 ? validLength - 1 : Math.Min(validLength - 1, i + band);

            for (int j = 0; j < seqLen; j++)
                data[rowBase + j] = j < firstAllowed || j > lastAllowed ? negInf : default;
        }

        return ReverseGradTensor<T>.FromMatrix(data, seqLen, seqLen, requiresGrad: false);
    }
}

/// <summary>
/// ModernBERT self-attention: bias-free projections fed from a fused QKV weight, rotary position
/// embeddings at the layer type's theta, and bidirectional (non-causal) scaled dot-product
/// attention under an additive mask.
/// </summary>
public sealed class ModernBertAttention<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly Linear<T> qProj;
    public readonly Linear<T> kProj;
    public readonly Linear<T> vProj;
    public readonly Linear<T> oProj;

    readonly RotaryEmbedding<T> rope;
    readonly int numHeads;
    readonly T scale;

    public ModernBertAttention(ModernBertConfig config, float ropeTheta)
    {
        qProj = new Linear<T>(config.HiddenSize, config.HiddenSize, bias: false);
        kProj = new Linear<T>(config.HiddenSize, config.HiddenSize, bias: false);
        vProj = new Linear<T>(config.HiddenSize, config.HiddenSize, bias: false);
        oProj = new Linear<T>(config.HiddenSize, config.HiddenSize, bias: false);

        rope = new RotaryEmbedding<T>(config.HeadDim, config.MaxPositionEmbeddings, ropeTheta);
        numHeads = config.NumAttentionHeads;
        scale = T.CreateChecked(1.0 / Math.Sqrt(config.HeadDim));
        RegisterModules(qProj, kProj, vProj, oProj);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input) => Forward(input, null);

    /// <summary>
    /// Projects, applies rotary embeddings to the queries and keys, then attends under
    /// <paramref name="mask"/> — an additive <c>[seqLen, seqLen]</c> tensor, or null for no mask.
    /// </summary>
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T>? mask)
    {
        ArgumentNullException.ThrowIfNull(input);

        var query = rope.Forward(qProj.Forward(input));
        var key = rope.Forward(kProj.Forward(input));
        var value = vProj.Forward(input);

        var context = ReverseGradOperations.MultiHeadAttention(query, key, value, numHeads, scale, mask);
        return oProj.Forward(context);
    }
}

/// <summary>
/// ModernBERT's gated feed-forward network. The fused <c>Wi</c> weight produces two halves side by
/// side and HuggingFace names them <c>input, gate = Wi(x).chunk(2, dim=-1)</c> — but it is the
/// <em>first</em> half, <c>input</c>, that carries the exact-erf GELU, and the product is
/// <c>act(input) * gate</c>. So the activated half is rows <c>[0, intermediate)</c> and the linear
/// "gate" is rows <c>[intermediate, 2 * intermediate)</c>, which is why the loader hands the
/// <em>first</em> row block to <see cref="inputProj"/> and the second to <see cref="gateProj"/>.
/// </summary>
public sealed class ModernBertMlp<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly Linear<T> inputProj;
    public readonly Linear<T> gateProj;
    public readonly Linear<T> downProj;

    public ModernBertMlp(ModernBertConfig config)
    {
        inputProj = new Linear<T>(config.HiddenSize, config.IntermediateSize, bias: false);
        gateProj = new Linear<T>(config.HiddenSize, config.IntermediateSize, bias: false);
        downProj = new Linear<T>(config.IntermediateSize, config.HiddenSize, bias: false);
        RegisterModules(inputProj, gateProj, downProj);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var activated = ReverseGradOperations.GeluExact(inputProj.Forward(input));
        var gate = gateProj.Forward(input);
        return downProj.Forward(ReverseGradOperations.Multiply(activated, gate));
    }
}

/// <summary>
/// One pre-norm ModernBERT encoder layer: a residual add around attention, then a residual add
/// around the gated MLP. The norms are bias-free LayerNorms — HuggingFace's <c>norm_bias: false</c>
/// — emulated by leaving the Beta at its zero initialization, which is exact for inference. The
/// first layer's attention norm is an identity and so has no weights in the checkpoint.
/// </summary>
public sealed class ModernBertLayer<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly LayerNorm<T>? attnNorm;
    public readonly ModernBertAttention<T> attn;
    public readonly LayerNorm<T> mlpNorm;
    public readonly ModernBertMlp<T> mlp;

    public ModernBertLayer(ModernBertConfig config, int layerIndex)
    {
        if (layerIndex != 0)
            attnNorm = new LayerNorm<T>(config.HiddenSize, config.NormEps);
        attn = new ModernBertAttention<T>(config, config.RopeTheta(layerIndex));
        mlpNorm = new LayerNorm<T>(config.HiddenSize, config.NormEps);
        mlp = new ModernBertMlp<T>(config);

        if (attnNorm != null)
            RegisterModules(attnNorm);
        RegisterModules(attn, mlpNorm, mlp);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input) => Forward(input, null);

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T>? mask)
    {
        ArgumentNullException.ThrowIfNull(input);

        var normed = attnNorm != null ? attnNorm.Forward(input) : input;
        var hidden = ReverseGradOperations.Add(input, attn.Forward(normed, mask));
        return ReverseGradOperations.Add(hidden, mlp.Forward(mlpNorm.Forward(hidden)));
    }
}

/// <summary>
/// A ModernBERT encoder: token embeddings, a bias-free embedding norm, the pre-norm layer stack,
/// and a final bias-free norm. There are no position or token-type embeddings — positions come
/// entirely from rotary embeddings.
/// </summary>
public sealed class ModernBertEncoder<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly ModernBertConfig config;
    public readonly Embedding<T> tokenEmbedding;
    public readonly LayerNorm<T> embedNorm;
    public readonly ModernBertLayer<T>[] layers;
    public readonly LayerNorm<T> finalNorm;

    public ModernBertEncoder(ModernBertConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        this.config = config;
        tokenEmbedding = new Embedding<T>(config.VocabSize, config.HiddenSize);
        embedNorm = new LayerNorm<T>(config.HiddenSize, config.NormEps);
        layers = new ModernBertLayer<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            layers[i] = new ModernBertLayer<T>(config, i);
        finalNorm = new LayerNorm<T>(config.HiddenSize, config.NormEps);

        RegisterModules(tokenEmbedding, embedNorm, finalNorm);
        foreach (var layer in layers)
            RegisterModules(layer);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var hidden = embedNorm.Forward(tokenEmbedding.Forward(input));
        foreach (var layer in layers)
            hidden = layer.Forward(hidden);
        return finalNorm.Forward(hidden);
    }

    /// <summary>
    /// Encodes <paramref name="tokenIds"/> and returns the last hidden state of shape
    /// <c>[seqLen, hidden]</c>. <paramref name="validLength"/> is the number of leading positions
    /// that are real tokens; the rest are padding and are masked out of every attention. Token IDs
    /// are passed as <see cref="int"/> so narrow-precision dtypes (Half / BFloat16) are never asked
    /// to represent vocabulary indices.
    /// </summary>
    public ReverseGradTensor<T> Forward(int[] tokenIds, int validLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        int seqLen = tokenIds.Length;
        if (validLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength must be positive, got {validLength}.");

        var fullMask = ModernBertMasks.Build<T>(seqLen, -1, validLength);
        var slidingMask = ModernBertMasks.Build<T>(seqLen, config.SlidingWindow, validLength);

        var hidden = embedNorm.Forward(tokenEmbedding.Forward(tokenIds));
        for (int i = 0; i < layers.Length; i++)
        {
            var mask = config.IsFullAttention(i) ? fullMask : slidingMask;
            hidden = layers[i].Forward(hidden, mask);
        }
        return finalNorm.Forward(hidden);
    }

    /// <summary>
    /// Loads encoder weights from a safetensors dictionary. <paramref name="prefix"/> selects the
    /// checkpoint layout: <c>"model"</c> for the stock HuggingFace backbone and <c>"encoder"</c> for
    /// Laya, which nests a copy of the same architecture under that name. Fused QKV and gate/up
    /// weights are split into contiguous row blocks.
    /// </summary>
    public static ModernBertEncoder<TModel> LoadWeights<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        ModernBertConfig config,
        string prefix = "model")
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(tensors);

        var encoder = new ModernBertEncoder<TModel>(config);

        StateDictLoader.LoadEmbed(encoder.tokenEmbedding, tensors, $"{prefix}.embeddings.tok_embeddings.weight");
        StateDictLoader.LoadLayerNorm(encoder.embedNorm, tensors, $"{prefix}.embeddings.norm");
        StateDictLoader.LoadLayerNorm(encoder.finalNorm, tensors, $"{prefix}.final_norm");

        int hidden = config.HiddenSize;
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var layer = encoder.layers[i];
            string layerPrefix = $"{prefix}.layers.{i}";

            if (layer.attnNorm != null)
                StateDictLoader.LoadLayerNorm(layer.attnNorm, tensors, $"{layerPrefix}.attn_norm");

            StateDictLoader.LoadLinearSlice(layer.attn.qProj, tensors, $"{layerPrefix}.attn.Wqkv", 0, hidden);
            StateDictLoader.LoadLinearSlice(layer.attn.kProj, tensors, $"{layerPrefix}.attn.Wqkv", hidden, hidden);
            StateDictLoader.LoadLinearSlice(layer.attn.vProj, tensors, $"{layerPrefix}.attn.Wqkv", 2 * hidden, hidden);
            StateDictLoader.LoadLinear(layer.attn.oProj, tensors, $"{layerPrefix}.attn.Wo");

            StateDictLoader.LoadLinearSlice(layer.mlp.inputProj, tensors, $"{layerPrefix}.mlp.Wi", 0, config.IntermediateSize);
            StateDictLoader.LoadLinearSlice(layer.mlp.gateProj, tensors, $"{layerPrefix}.mlp.Wi", config.IntermediateSize, config.IntermediateSize);
            StateDictLoader.LoadLinear(layer.mlp.downProj, tensors, $"{layerPrefix}.mlp.Wo");

            StateDictLoader.LoadLayerNorm(layer.mlpNorm, tensors, $"{layerPrefix}.mlp_norm");
        }

        encoder.Eval();
        return encoder;
    }

    /// <summary>Loads F32 weights from a safetensors dictionary.</summary>
    public static ModernBertEncoder<float> LoadWeights(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        ModernBertConfig config,
        string prefix = "model")
        => LoadWeights<float, float>(tensors, config, prefix);
}
