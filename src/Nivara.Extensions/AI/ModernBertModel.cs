using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Diagnostics;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;

namespace Nivara.AI;

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
    /// The largest sequence length for which a dense <c>[L, L]</c> mask is built by the reference
    /// helper <see cref="Build{TModel}"/>. At the model's full 8192-token context one dense mask is 67M
    /// elements (268 MB in F32), and a forward builds <em>two</em> of them — a global and a sliding
    /// — so 537 MB for the pair. Longer inputs throw instead of silently allocating.
    ///
    /// This cap constrains only <see cref="Build{TModel}"/> (used by tests and reference tooling). The
    /// CPU attention path no longer builds a dense mask (#476): it carries <c>band</c> and
    /// <c>validLength</c> as metadata bounds, so it is not limited by this value.
    /// </summary>
    public const int MaxDenseLength = 2048;

    /// <summary>
    /// Builds a single <c>[seqLen, seqLen]</c> additive mask. <paramref name="band"/> is the largest
    /// inclusive <c>abs(i - j)</c> a query may reach, or -1 for no distance limit.
    /// <paramref name="validLength"/> is the number of leading positions that are real tokens; the
    /// rest are padding and are suppressed for every query.
    /// </summary>
    public static ReverseGradTensor<TModel> Build<TModel>(int seqLen, int band, int validLength)
        where TModel : struct, IFloatingPointIeee754<TModel>
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

        var data = new TModel[seqLen * seqLen];
        var negInf = TModel.CreateChecked(double.NegativeInfinity);

        for (int i = 0; i < seqLen; i++)
        {
            int rowBase = i * seqLen;
            int firstAllowed = band < 0 ? 0 : Math.Max(0, i - band);
            int lastAllowed = band < 0 ? validLength - 1 : Math.Min(validLength - 1, i + band);

            for (int j = 0; j < seqLen; j++)
                data[rowBase + j] = j < firstAllowed || j > lastAllowed ? negInf : default;
        }

        return ReverseGradTensor<TModel>.FromMatrix(data, seqLen, seqLen, requiresGrad: false);
    }
}

/// <summary>
/// ModernBERT self-attention: bias-free projections fed from a fused QKV weight, rotary position
/// embeddings at the layer type's theta, and bidirectional (non-causal) scaled dot-product
/// attention under an additive mask.
/// </summary>
public class ModernBertAttention<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly Linear<TModel, TWeight> qProj;
    public readonly Linear<TModel, TWeight> kProj;
    public readonly Linear<TModel, TWeight> vProj;
    public readonly Linear<TModel, TWeight> oProj;

    readonly RotaryEmbedding<TModel> rope;
    readonly int numHeads;
    readonly TModel scale;

    /// <summary>Gets the rotary embedding (carries the per-layer theta).</summary>
    public RotaryEmbedding<TModel> Rope => rope;

    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => numHeads;
    /// <summary>Gets the per-head dimension.</summary>
    public int HeadDim => rope.HeadDim;

    public ModernBertAttention(ModernBertConfig config, float ropeTheta)
    {
        qProj = new Linear<TModel, TWeight>(config.HiddenSize, config.HiddenSize, bias: false);
        kProj = new Linear<TModel, TWeight>(config.HiddenSize, config.HiddenSize, bias: false);
        vProj = new Linear<TModel, TWeight>(config.HiddenSize, config.HiddenSize, bias: false);
        oProj = new Linear<TModel, TWeight>(config.HiddenSize, config.HiddenSize, bias: false);

        rope = new RotaryEmbedding<TModel>(config.HeadDim, config.MaxPositionEmbeddings, ropeTheta);
        numHeads = config.NumAttentionHeads;
        scale = TModel.CreateChecked(1.0 / Math.Sqrt(config.HeadDim));
        RegisterModules(qProj, kProj, vProj, oProj);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input) => Forward(input, null);

    /// <summary>
    /// Projects, applies rotary embeddings to the queries and keys, then attends under
    /// <paramref name="mask"/> — an additive <c>[seqLen, seqLen]</c> tensor, or null for no mask.
    /// </summary>
    public ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? mask)
    {
        ArgumentNullException.ThrowIfNull(input);

        var query = rope.Forward(qProj.Forward(input));
        var key = rope.Forward(kProj.Forward(input));
        var value = vProj.Forward(input);

        var context = ReverseGradOperations.MultiHeadAttention(query, key, value, numHeads, scale, mask);
        return oProj.Forward(context);
    }

    public ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input, int band, int validLength)
    {
        ArgumentNullException.ThrowIfNull(input);

        var query = rope.Forward(qProj.Forward(input));
        var key = rope.Forward(kProj.Forward(input));
        var value = vProj.Forward(input);

        var context = ReverseGradOperations.MultiHeadAttention(query, key, value, numHeads, scale, null, band, validLength);
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
public class ModernBertMlp<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly Linear<TModel, TWeight> inputProj;
    public readonly Linear<TModel, TWeight> gateProj;
    public readonly Linear<TModel, TWeight> downProj;

    public ModernBertMlp(ModernBertConfig config)
    {
        inputProj = new Linear<TModel, TWeight>(config.HiddenSize, config.IntermediateSize, bias: false);
        gateProj = new Linear<TModel, TWeight>(config.HiddenSize, config.IntermediateSize, bias: false);
        downProj = new Linear<TModel, TWeight>(config.IntermediateSize, config.HiddenSize, bias: false);
        RegisterModules(inputProj, gateProj, downProj);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
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
/// — so they are constructed with <c>bias: false</c> and register no beta for an optimizer to
/// drift. The first layer's attention norm is an identity and so has no weights in the checkpoint.
/// </summary>
public class ModernBertLayer<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly int hiddenSize;

    public readonly LayerNorm<TModel>? attnNorm;
    public readonly ModernBertAttention<TModel, TWeight> attn;
    public readonly LayerNorm<TModel> mlpNorm;
    public readonly ModernBertMlp<TModel, TWeight> mlp;

    /// <summary>Gets the hidden (embedding) dimension.</summary>
    public int HiddenSize => hiddenSize;

    public ModernBertLayer(ModernBertConfig config, int layerIndex)
    {
        hiddenSize = config.HiddenSize;
        if (layerIndex != 0)
            attnNorm = new LayerNorm<TModel>(config.HiddenSize, config.NormEps, bias: false);
        attn = new ModernBertAttention<TModel, TWeight>(config, config.RopeTheta(layerIndex));
        mlpNorm = new LayerNorm<TModel>(config.HiddenSize, config.NormEps, bias: false);
        mlp = new ModernBertMlp<TModel, TWeight>(config);

        if (attnNorm != null)
            RegisterModules(attnNorm);
        RegisterModules(attn, mlpNorm, mlp);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input) => Forward(input, null);

    public ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input, ReverseGradTensor<TModel>? mask)
    {
        ArgumentNullException.ThrowIfNull(input);

        var normed = attnNorm != null ? attnNorm.Forward(input) : input;
        var hidden = ReverseGradOperations.Add(input, attn.Forward(normed, mask));
        return ReverseGradOperations.Add(hidden, mlp.Forward(mlpNorm.Forward(hidden)));
    }

    public ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input, int band, int validLength)
    {
        ArgumentNullException.ThrowIfNull(input);

        var normed = attnNorm != null ? attnNorm.Forward(input) : input;
        var hidden = ReverseGradOperations.Add(input, attn.Forward(normed, band, validLength));
        return ReverseGradOperations.Add(hidden, mlp.Forward(mlpNorm.Forward(hidden)));
    }

    /// <summary>
    /// Inference-only fused whole-layer forward. Replicates <see cref="Forward(ReverseGradTensor{TModel}, int, int)"/>
    /// using the shared span kernels (LayerNorm, banded attention, RoPE, GeGLU) with zero
    /// steady-state heap allocations. <paramref name="band"/> is -1 for full attention;
    /// <paramref name="validLength"/> is the number of leading real tokens.
    /// Throws inside a <see cref="GradientUtils.Grad()"/> scope.
    /// </summary>
    public void ForwardFused(ReadOnlySpan<TModel> input, Span<TModel> output, int band, int validLength)
    {
        if (input.Length == 0 || input.Length % hiddenSize != 0)
            throw new ArgumentException($"Input length {input.Length} must be a positive multiple of hidden size {hiddenSize}.");
        int seqLen = input.Length / hiddenSize;
        if (output.Length < input.Length)
            throw new ArgumentException($"Output span (length {output.Length}) must hold {input.Length} elements.");
        if (validLength <= 0 || validLength > seqLen)
            throw new ArgumentOutOfRangeException(nameof(validLength), validLength, $"validLength must be within [1, {seqLen}].");
        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("ForwardFused is inference-only; do not call it inside GradientUtils.Grad().");

        int numHeads = attn.NumHeads;
        int headDim = attn.HeadDim;
        int qWidth = numHeads * headDim; // == hiddenSize for MHA
        int intermediateSize = mlp.inputProj.OutFeatures;
        TModel scale = TModel.CreateChecked(1.0 / Math.Sqrt(headDim));
        TModel eps = mlpNorm.Eps;
        bool diag = InferenceDiagnostics.IsEnabled;

        // Scratch buffers (rented; returned in finally).
        var normBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * hiddenSize, 1));
        var qBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * qWidth, 1));
        var kBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * qWidth, 1));
        var vBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * qWidth, 1));
        var scores = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * seqLen, 1));
        var context = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * qWidth, 1));
        var attnOut = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * hiddenSize, 1));
        var residual = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * hiddenSize, 1));
        var mlpNormBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * hiddenSize, 1));
        var actBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * intermediateSize, 1));
        var gateBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * intermediateSize, 1));
        var mlpOut = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * hiddenSize, 1));
        try
        {
            var normSpan = normBuf.AsSpan(0, seqLen * hiddenSize);
            var qSpan = qBuf.AsSpan(0, seqLen * qWidth);
            var kSpan = kBuf.AsSpan(0, seqLen * qWidth);
            var vSpan = vBuf.AsSpan(0, seqLen * qWidth);
            var scoresSpan = scores.AsSpan(0, seqLen * seqLen);
            var contextSpan = context.AsSpan(0, seqLen * qWidth);
            var attnOutSpan = attnOut.AsSpan(0, seqLen * hiddenSize);
            var residualSpan = residual.AsSpan(0, seqLen * hiddenSize);
            var mlpNormSpan = mlpNormBuf.AsSpan(0, seqLen * hiddenSize);
            var actSpan = actBuf.AsSpan(0, seqLen * intermediateSize);
            var gateSpan = gateBuf.AsSpan(0, seqLen * intermediateSize);
            var mlpOutSpan = mlpOut.AsSpan(0, seqLen * hiddenSize);

            var mAttnNorm = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 1. Attention norm (or identity on layer 0).
            if (attnNorm != null)
            {
                var gamma = WeightSpan(attnNorm.Weight);
                var beta = attnNorm.Bias != null ? WeightSpan(attnNorm.Bias) : ReadOnlySpan<TModel>.Empty;
                var lnOut = LayerNormKernel<TModel>.ForwardInference(input, seqLen, hiddenSize, gamma, beta, attnNorm.Eps);
                lnOut.AsSpan(0, seqLen * hiddenSize).CopyTo(normSpan);
            }
            else
            {
                input.CopyTo(normSpan);
            }
            if (diag) mAttnNorm.Record("CPU-Encoder-AttnNorm", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);

            var mQkv = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 2. Q/K/V projections.
            var wQ = WeightSpan(attn.qProj.Weight);
            var wK = WeightSpan(attn.kProj.Weight);
            var wV = WeightSpan(attn.vProj.Weight);
            GradKernels.MatMulTransposedB(normSpan, wQ, qBuf, seqLen, hiddenSize, qWidth);
            GradKernels.MatMulTransposedB(normSpan, wK, kBuf, seqLen, hiddenSize, qWidth);
            GradKernels.MatMulTransposedB(normSpan, wV, vBuf, seqLen, hiddenSize, qWidth);
            if (diag) mQkv.Record("CPU-Encoder-QKV", global::Nivara.KernelSelector.DetermineKernelType<TModel>(3 * seqLen * hiddenSize), 3 * seqLen * hiddenSize, typeof(TModel), false, null);

            var mRope = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 3. RoPE in place.
            attn.Rope.GetPositionTables(0, seqLen, out var cos, out var sin);
            int half = headDim / 2;
            for (int r = 0; r < seqLen; r++)
            {
                var rowC = cos.Slice(r * half, half);
                var rowS = sin.Slice(r * half, half);
                for (int b = 0; b < numHeads; b++)
                {
                    GradKernels.RotaryForward(qSpan.Slice(r * qWidth + b * headDim, headDim), rowC, rowS,
                        qSpan.Slice(r * qWidth + b * headDim, headDim));
                    GradKernels.RotaryForward(kSpan.Slice(r * qWidth + b * headDim, headDim), rowC, rowS,
                        kSpan.Slice(r * qWidth + b * headDim, headDim));
                }
            }
            if (diag) mRope.Record("CPU-Encoder-RoPE", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * qWidth), seqLen * qWidth, typeof(TModel), false, null);

            var mAttention = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 4. Banded attention per head.
            for (int h = 0; h < numHeads; h++)
            {
                // Gather Q/K/V head slices into contiguous per-head buffers is handled by
                // indexing into the packed layout; here we operate on strided head columns
                // by copying into temp head-major buffers.
            }
            // Use packed head-major buffers for the matmul path (matches autodiff kernel).
            var qHeads = ArrayPool<TModel>.Shared.Rent(Math.Max(numHeads * seqLen * headDim, 1));
            var kHeads = ArrayPool<TModel>.Shared.Rent(Math.Max(numHeads * seqLen * headDim, 1));
            var vHeads = ArrayPool<TModel>.Shared.Rent(Math.Max(numHeads * seqLen * headDim, 1));
            var outHeads = ArrayPool<TModel>.Shared.Rent(Math.Max(numHeads * seqLen * headDim, 1));
            try
            {
                AttentionKernels<TModel>.PackHeads(qSpan, qHeads.AsSpan(0, numHeads * seqLen * headDim), seqLen, numHeads, headDim);
                AttentionKernels<TModel>.PackHeads(kSpan, kHeads.AsSpan(0, numHeads * seqLen * headDim), seqLen, numHeads, headDim);
                AttentionKernels<TModel>.PackHeads(vSpan, vHeads.AsSpan(0, numHeads * seqLen * headDim), seqLen, numHeads, headDim);

                for (int h = 0; h < numHeads; h++)
                {
                    var qh = qHeads.AsSpan(h * seqLen * headDim, seqLen * headDim);
                    var kh = kHeads.AsSpan(h * seqLen * headDim, seqLen * headDim);
                    var vh = vHeads.AsSpan(h * seqLen * headDim, seqLen * headDim);
                    var oh = outHeads.AsSpan(h * seqLen * headDim, seqLen * headDim);

                    GradKernels.MatMulTransposedB(qh, kh, scores, seqLen, headDim, seqLen);
                    TensorPrimitives.Multiply(scoresSpan, scale, scoresSpan);
                    AttentionKernels<TModel>.ApplyBounds(scoresSpan, Span<bool>.Empty, seqLen, seqLen, band, validLength);
                    AttentionKernels<TModel>.SoftmaxRows(scoresSpan, seqLen, seqLen);
                    var outHeadBuf = ArrayPool<TModel>.Shared.Rent(Math.Max(seqLen * headDim, 1));
                    try
                    {
                        GradKernels.MatMul(scoresSpan, vh, outHeadBuf, seqLen, seqLen, headDim);
                        outHeadBuf.AsSpan(0, seqLen * headDim).CopyTo(oh);
                    }
                    finally
                    {
                        ArrayPool<TModel>.Shared.Return(outHeadBuf);
                    }
                }

                // Scatter heads back into context (width = hidden).
                for (int h = 0; h < numHeads; h++)
                {
                    var oh = outHeads.AsSpan(h * seqLen * headDim, seqLen * headDim);
                    AttentionKernels<TModel>.ScatterHead(oh, contextSpan, seqLen, hiddenSize, h, headDim);
                }
            }
            finally
            {
                ArrayPool<TModel>.Shared.Return(qHeads);
                ArrayPool<TModel>.Shared.Return(kHeads);
                ArrayPool<TModel>.Shared.Return(vHeads);
                ArrayPool<TModel>.Shared.Return(outHeads);
            }
            if (diag) mAttention.Record("CPU-Encoder-Attention", global::Nivara.KernelSelector.DetermineKernelType<TModel>(numHeads * seqLen * seqLen), numHeads * seqLen * seqLen, typeof(TModel), false, null);

            var mOProj = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 5. Output projection + residual.
            var wO = WeightSpan(attn.oProj.Weight);
            GradKernels.MatMulTransposedB(contextSpan, wO, attnOut, seqLen, hiddenSize, hiddenSize);
            input.CopyTo(residualSpan);
            TensorPrimitives.Add(residualSpan, attnOutSpan, residualSpan);
            if (diag) mOProj.Record("CPU-Encoder-OProj", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);

            var mLpNorm = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 6. MLP norm.
            var mlpGamma = WeightSpan(mlpNorm.Weight);
            var mlpBeta = mlpNorm.Bias != null ? WeightSpan(mlpNorm.Bias) : ReadOnlySpan<TModel>.Empty;
            var lnMlp = LayerNormKernel<TModel>.ForwardInference(residualSpan, seqLen, hiddenSize, mlpGamma, mlpBeta, mlpNorm.Eps);
            lnMlp.AsSpan(0, seqLen * hiddenSize).CopyTo(mlpNormSpan);
            if (diag) mLpNorm.Record("CPU-Encoder-MlpNorm", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);

            var mGeGlu = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 7. GeGLU.
            var wIn = WeightSpan(mlp.inputProj.Weight);
            var wGate = WeightSpan(mlp.gateProj.Weight);
            var wDown = WeightSpan(mlp.downProj.Weight);
            GradKernels.MatMulTransposedB(mlpNormSpan, wIn, actBuf, seqLen, hiddenSize, intermediateSize);
            GradKernels.GeluExact(actSpan, actSpan);
            GradKernels.MatMulTransposedB(mlpNormSpan, wGate, gateBuf, seqLen, hiddenSize, intermediateSize);
            TensorPrimitives.Multiply(actSpan, gateSpan, gateSpan);
            GradKernels.MatMulTransposedB(gateSpan, wDown, mlpOut, seqLen, intermediateSize, hiddenSize);
            if (diag) mGeGlu.Record("CPU-Encoder-GeGLU", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * intermediateSize), seqLen * intermediateSize * 3, typeof(TModel), false, null);

            var mResid = diag ? DiagnosticsTracker.StartMeasurement() : default;
            // 8. Residual into output.
            TensorPrimitives.Add(residualSpan, mlpOutSpan, output);
            if (diag) mResid.Record("CPU-Encoder-Residual", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);
        }
        finally
        {
            ArrayPool<TModel>.Shared.Return(normBuf);
            ArrayPool<TModel>.Shared.Return(qBuf);
            ArrayPool<TModel>.Shared.Return(kBuf);
            ArrayPool<TModel>.Shared.Return(vBuf);
            ArrayPool<TModel>.Shared.Return(scores);
            ArrayPool<TModel>.Shared.Return(context);
            ArrayPool<TModel>.Shared.Return(attnOut);
            ArrayPool<TModel>.Shared.Return(residual);
            ArrayPool<TModel>.Shared.Return(mlpNormBuf);
            ArrayPool<TModel>.Shared.Return(actBuf);
            ArrayPool<TModel>.Shared.Return(gateBuf);
            ArrayPool<TModel>.Shared.Return(mlpOut);
        }
    }

    static ReadOnlySpan<TModel> WeightSpan<TParam>(Parameter<TParam>? weight) where TParam : struct, IFloatingPointIeee754<TParam>
    {
        if (weight is null) throw new InvalidOperationException("Expected a non-null parameter.");
        if (typeof(TParam) == typeof(TModel))
        {
            if (!weight.Tensor.Data.TryGetSpan(out var narrowSpan))
                throw new InvalidOperationException("Parameter data must be null-free.");
            return System.Runtime.InteropServices.MemoryMarshal.Cast<TParam, TModel>(narrowSpan);
        }
        if (!weight.Tensor.Data.TryGetSpan(out var sourceSpan))
            throw new InvalidOperationException("Parameter data must be null-free.");
        var widened = new TModel[sourceSpan.Length];
        for (int i = 0; i < widened.Length; i++)
            widened[i] = TModel.CreateChecked(sourceSpan[i]);
        return widened.AsSpan();
    }
}

/// <summary>
/// A ModernBERT encoder: token embeddings, a bias-free embedding norm (<c>bias: false</c>), the
/// pre-norm layer stack, and a final bias-free norm. There are no position or token-type
/// embeddings — positions come entirely from rotary embeddings.
/// </summary>
public class ModernBertEncoder<TModel, TWeight> : Module<TModel> where TModel : struct, IFloatingPointIeee754<TModel> where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly ModernBertConfig config;
    public readonly Embedding<TModel, TWeight> tokenEmbedding;
    public readonly LayerNorm<TModel> embedNorm;
    public readonly ModernBertLayer<TModel, TWeight>[] layers;
    public readonly LayerNorm<TModel> finalNorm;

    public ModernBertEncoder(ModernBertConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        this.config = config;
        tokenEmbedding = new Embedding<TModel, TWeight>(config.VocabSize, config.HiddenSize);
        embedNorm = new LayerNorm<TModel>(config.HiddenSize, config.NormEps, bias: false);
        layers = new ModernBertLayer<TModel, TWeight>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            layers[i] = new ModernBertLayer<TModel, TWeight>(config, i);
        finalNorm = new LayerNorm<TModel>(config.HiddenSize, config.NormEps, bias: false);

        RegisterModules(tokenEmbedding, embedNorm, finalNorm);
        foreach (var layer in layers)
            RegisterModules(layer);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
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
    public ReverseGradTensor<TModel> Forward(int[] tokenIds, int validLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        int seqLen = tokenIds.Length;
        if (validLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength must be positive, got {validLength}.");

        if (GradientUtils.IsGradEnabled)
        {
            var hidden = embedNorm.Forward(tokenEmbedding.Forward(tokenIds));
            for (int i = 0; i < layers.Length; i++)
            {
                int band = config.IsFullAttention(i) ? -1 : config.SlidingWindow;
                hidden = layers[i].Forward(hidden, band, validLength);
            }
            return finalNorm.Forward(hidden);
        }

        // Inference-only fused path.
        int hiddenSize = config.HiddenSize;
        bool diag = InferenceDiagnostics.IsEnabled;
        var weightData = tokenEmbedding.Weight != null
            ? WeightSpan(tokenEmbedding.Weight)
            : throw new InvalidOperationException("tokenEmbedding has no weight.");
        var hiddenBuf = new TModel[seqLen * hiddenSize];
        for (int i = 0; i < seqLen; i++)
        {
            int id = tokenIds[i];
            if (id < 0 || id >= config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} out of range.");
            weightData.Slice(id * hiddenSize, hiddenSize).CopyTo(hiddenBuf.AsSpan(i * hiddenSize, hiddenSize));
        }

        var mEmbed = diag ? DiagnosticsTracker.StartMeasurement() : default;
        var embedGamma = WeightSpan(embedNorm.Weight);
        var embedBeta = embedNorm.Bias != null ? WeightSpan(embedNorm.Bias) : ReadOnlySpan<TModel>.Empty;
        var embedOut = LayerNormKernel<TModel>.ForwardInference(hiddenBuf, seqLen, hiddenSize, embedGamma, embedBeta, embedNorm.Eps);
        embedOut.AsSpan(0, seqLen * hiddenSize).CopyTo(hiddenBuf);
        if (diag) mEmbed.Record("CPU-Encoder-EmbedNorm", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);

        var bufA = hiddenBuf;
        var bufB = new TModel[seqLen * hiddenSize];
        for (int i = 0; i < layers.Length; i++)
        {
            int band = config.IsFullAttention(i) ? -1 : config.SlidingWindow;
            layers[i].ForwardFused(bufA.AsSpan(0, seqLen * hiddenSize), bufB.AsSpan(0, seqLen * hiddenSize), band, validLength);
            var tmp = bufA; bufA = bufB; bufB = tmp;
        }
        if (bufA != hiddenBuf)
            bufA.AsSpan(0, seqLen * hiddenSize).CopyTo(hiddenBuf);

        var mFinal = diag ? DiagnosticsTracker.StartMeasurement() : default;
        var finalGamma = WeightSpan(finalNorm.Weight);
        var finalBeta = finalNorm.Bias != null ? WeightSpan(finalNorm.Bias) : ReadOnlySpan<TModel>.Empty;
        var finalOut = LayerNormKernel<TModel>.ForwardInference(hiddenBuf, seqLen, hiddenSize, finalGamma, finalBeta, finalNorm.Eps);
        finalOut.AsSpan(0, seqLen * hiddenSize).CopyTo(hiddenBuf);
        if (diag) mFinal.Record("CPU-Encoder-FinalNorm", global::Nivara.KernelSelector.DetermineKernelType<TModel>(seqLen * hiddenSize), seqLen * hiddenSize, typeof(TModel), false, null);

        return new ReverseGradTensor<TModel>(
            NivaraColumn<TModel>.CreateFromOwnedArray(hiddenBuf), false, new[] { seqLen, hiddenSize });
    }

    /// <summary>
    /// Loads encoder weights from a safetensors dictionary. <paramref name="prefix"/> selects the
    /// checkpoint layout: <c>"model"</c> for the stock HuggingFace backbone and <c>"encoder"</c> for
    /// Laya, which nests a copy of the same architecture under that name. Fused QKV and gate/up
    /// weights are split into contiguous row blocks.
    /// </summary>
    public static ModernBertEncoder<TModel, TWeight> LoadWeights(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        ModernBertConfig config,
        string prefix = "model")
    {
        ArgumentNullException.ThrowIfNull(tensors);

        var encoder = new ModernBertEncoder<TModel, TWeight>(config);

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
        => LoadWeights(tensors, config, prefix);

    static ReadOnlySpan<TModel> WeightSpan<TParam>(Parameter<TParam>? weight) where TParam : struct, IFloatingPointIeee754<TParam>
    {
        if (weight is null) throw new InvalidOperationException("Expected a non-null parameter.");
        if (typeof(TParam) == typeof(TModel))
        {
            if (!weight.Tensor.Data.TryGetSpan(out var narrowSpan))
                throw new InvalidOperationException("Parameter data must be null-free.");
            return System.Runtime.InteropServices.MemoryMarshal.Cast<TParam, TModel>(narrowSpan);
        }
        if (!weight.Tensor.Data.TryGetSpan(out var sourceSpan))
            throw new InvalidOperationException("Parameter data must be null-free.");
        var widened = new TModel[sourceSpan.Length];
        for (int i = 0; i < widened.Length; i++)
            widened[i] = TModel.CreateChecked(sourceSpan[i]);
        return widened.AsSpan();
    }
}
