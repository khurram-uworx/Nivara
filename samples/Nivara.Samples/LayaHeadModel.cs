using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.Samples;

/// <summary>
/// One pre-norm transformer encoder layer: <c>LayerNorm → attention → residual → LayerNorm →
/// ReLU FFN → residual</c>, with the key-padding mask supplied per call.
/// </summary>
/// <remarks>
/// <para>
/// This is <c>nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
/// norm_first=True)</c> with its default <c>activation=ReLU</c>, at inference. Three consequences
/// of being the default that are worth stating, because each is a way to get this subtly wrong:
/// </para>
/// <list type="bullet">
/// <item><description>
/// The activation is <b>ReLU</b>, not GELU. <c>nn.TransformerEncoderLayer</c> defaults to ReLU
/// while every other block in the stack uses GELU, so reading the checkpoint's
/// <c>linear1</c>/<c>linear2</c> shapes does not disambiguate and the default is easy to carry over
/// from a neighbouring model.
/// </description></item>
/// <item><description>
/// The FFN is plain, not gated. There is no gate projection to split, which is why the checkpoint
/// has exactly two FFN matrices per layer where ModernBERT has a fused gate/up pair.
/// </description></item>
/// <item><description>
/// Dropout is 0.1 and is not emulated. It is identity at inference, exactly as leaving ModernBERT's
/// optional parameter initializations alone is exact for inference.
/// </description></item>
/// </list>
/// <para>
/// Attention is the reused <see cref="BertSelfAttention{T}"/> rather than a second implementation.
/// It already has biased q/k/v, a biased output projection, the dense <c>[L, L]</c> additive
/// key-padding mask, and the <c>1 / sqrt(embedDim / numHeads)</c> scale that PyTorch's
/// <c>nn.MultiheadAttention</c> applies. The checkpoint's fused <c>in_proj_weight</c> /
/// <c>in_proj_bias</c> are a load-time concern only: <see cref="StateDictLoader"/> slices the fused
/// rows into the three projections.
/// </para>
/// <para>
/// <b>No final norm on the stack.</b> <c>nn.TransformerEncoder</c> applies none, and the checkpoint
/// has no <c>head.norm</c> — the last thing to touch the hidden state is the second layer's FFN
/// residual.
/// </para>
/// </remarks>
public sealed class LayaHeadLayer<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    public readonly LayerNorm<T> norm1;
    public readonly BertSelfAttention<T> attn;
    public readonly LayerNorm<T> norm2;
    public readonly Linear<T> linear1;
    public readonly Linear<T> linear2;

    public LayaHeadLayer(int hiddenSize, int intermediateSize, int numHeads, float eps)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive");
        if (intermediateSize <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(intermediateSize), intermediateSize, "intermediateSize must be positive");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), numHeads, "numHeads must be positive");
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException(
                $"hiddenSize {hiddenSize} does not divide into {numHeads} heads");

        norm1 = new LayerNorm<T>(hiddenSize, eps);
        attn = new BertSelfAttention<T>(hiddenSize, numHeads);
        norm2 = new LayerNorm<T>(hiddenSize, eps);
        linear1 = new Linear<T>(hiddenSize, intermediateSize, bias: true);
        linear2 = new Linear<T>(intermediateSize, hiddenSize, bias: true);
        RegisterModules(norm1, attn, norm2, linear1, linear2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input) => Forward(input, null);

    /// <param name="input">Hidden state of shape <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="paddingMask">
    /// One value per position, where a value below 0.5 marks a padded key to suppress. This is the
    /// reference's <c>src_key_padding_mask</c> (True = ignore) expressed in the additive form:
    /// <c>~attention_mask</c> suppressed is the same predicate as a 0 in <c>attention_mask</c>
    /// suppressed, so no inversion is needed at the call site.
    /// </param>
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T>? paddingMask)
    {
        ArgumentNullException.ThrowIfNull(input);

        var attended = attn.ForwardWithMask(norm1.Forward(input), paddingMask);
        var hidden = ReverseGradOperations.Add(input, attended);
        var ffn = linear2.Forward(ReverseGradOperations.Relu(linear1.Forward(norm2.Forward(hidden))));
        return ReverseGradOperations.Add(hidden, ffn);
    }
}

/// <summary>What the decision head produced for one question.</summary>
public readonly record struct LayaHeadOutput
{
    /// <summary>Index into <see cref="Features"/> holding the top marker probability.</summary>
    public const int FeatureTop1 = 0;

    /// <summary>Index into <see cref="Features"/> holding top-1 minus top-2.</summary>
    public const int FeatureGap = 1;

    /// <summary>Index into <see cref="Features"/> holding the normalized entropy.</summary>
    public const int FeatureEntropy = 2;

    /// <summary>Index into <see cref="Features"/> holding the option count over 255.</summary>
    public const int FeatureOptionCount = 3;

    /// <summary>Number of features the act head consumes, and its input width above the pooled row.</summary>
    public const int FeatureCount = 4;

    /// <summary>Marker-scorer logits, one per surviving marker, in marker order.</summary>
    public required float[] Logits { get; init; }

    /// <summary>
    /// Softmax of <see cref="Logits"/> at temperature 1. This is the reference's <c>p</c>, which
    /// feeds the act head's features. It is deliberately <b>not</b> the temperature-softened
    /// distribution the answer is decoded from: the reference divides by the temperature only in the
    /// decode step, after the head has already consumed the untempered features.
    /// </summary>
    public required double[] MarkerProbabilities { get; init; }

    /// <summary>
    /// The four values concatenated onto the pooled row: top-1 probability, top-1 minus top-2,
    /// entropy normalized by <c>log(max(2, k))</c>, and <c>max(2, k) / 255</c>. Reported because the
    /// option-count feature is the least obvious of the four and the gap is what the act head reads
    /// for decisiveness.
    /// </summary>
    public required double[] Features { get; init; }

    /// <summary>
    /// Softmax of the act head's logits, class 0. Reproduced faithfully; see
    /// <see cref="LayaCalibration.Decode"/> for why it must not be gated on.
    /// </summary>
    public required double ActionProbability { get; init; }

    /// <summary>Width of the act head's output, 2 on the shipped checkpoint.</summary>
    public required int ActionClassCount { get; init; }
}

/// <summary>
/// Laya's typed decision head, ported from <c>laya.common.DecisionModel</c> (PyPI <c>laya</c>
/// 0.3.20). Consumes the encoder's last hidden state and produces one score per option marker,
/// plus the act head's action reading.
/// </summary>
/// <remarks>
/// <para>The composition, in the reference's order:</para>
/// <code>
/// h = encoder(...)                                    # supplied by the caller
/// h = h + type_emb[qtype]                            # broadcast over positions
/// h = head.layers[i](h, src_key_padding_mask)        # 0, 1 or 2 layers, no final norm
/// logits = scorer(h[marker_pos]).squeeze(-1)         # [k]
/// p = softmax(logits)                                # untempered; feeds the features
/// feats = [top1, top1 - top2, ent, max(2, k) / 255]
/// act  = act_head(cat([h[0], feats]))                # [n_act]
/// </code>
/// <para>
/// <b>Temperature is not applied here.</b> The checkpoint carries a <c>temperature [3]</c> buffer,
/// and the reference registers it, but <c>DecisionModel.forward</c> never reads it — the caller
/// divides the logits after <c>masked_fill</c>. The buffer duplicates the agent config's
/// <c>temperature</c> array and would be a second, silently divergent source of the same numbers if
/// it were read, so it is loaded and exposed (see <see cref="CheckpointTemperature"/>) and not
/// applied. Calibration lives in <see cref="LayaCalibration"/>.
/// </para>
/// <para>
/// <b>Marker masking is a no-op here.</b> The reference's <c>logits.masked_fill(~marker_mask, -1e4)</c>
/// exists because a batch pads every row to the batch-wide maximum option count. This head runs one
/// question at a time, so <c>k == markerPositions.Length</c> and there is no padding slot to mask.
/// The fill is therefore not reimplemented; the option-count feature still applies the same
/// <c>clamp(min=2)</c>, which is load-bearing for the one-option question.
/// </para>
/// </remarks>
public sealed class LayaDecisionHead<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Question types the type embedding has rows for; matches <see cref="LayaQuestionType"/>.</summary>
    public const int QuestionTypeCount = 3;

    /// <summary>Hidden width of the act head's first projection, from the checkpoint's <c>act_head.0</c>.</summary>
    public const int ActHeadHidden = 256;

    /// <summary>The reference's head width per type: <c>nhead = max(1, d // 64)</c>.</summary>
    public const int HeadDimTarget = 64;

    readonly LayaHeadLayer<T>[] headLayers;

    /// <summary>
    /// The pre-norm head stack, empty when <c>head_layers</c> is 0. Exposed so a backend that runs
    /// the head itself — the GPU path — can drive the same layers rather than reimplementing them.
    /// </summary>
    public IReadOnlyList<LayaHeadLayer<T>> HeadLayers => headLayers;

    /// <summary>The question-type embedding, added to the hidden state before the head stack.</summary>
    public readonly Embedding<T> typeEmb;

    // The two <c>nn.Sequential</c> blocks, held as parts so the load sites name the checkpoint keys
    // directly. Both default activations are nn.GELU (exact), unlike the head layers' ReLU.
    public readonly LayerNorm<T> scorerNorm;
    public readonly Linear<T> scorerHidden;
    public readonly Linear<T> scorerOut;
    public readonly Linear<T> actInput;
    public readonly Linear<T> actOut;

    readonly T[] checkpointTemperature;

    public int HiddenSize => typeEmb.EmbeddingDim;
    public int HeadLayerCount => headLayers.Length;
    public int NumHeads { get; }

    /// <summary>
    /// The per-type temperatures the checkpoint stores in its <c>temperature</c> buffer. Loaded for
    /// completeness and cross-checking against the agent config; never applied.
    /// </summary>
    public IReadOnlyList<T> CheckpointTemperature => checkpointTemperature;

    public LayaDecisionHead(int hiddenSize, int headLayerCount = 2, int actClassCount = 2, float eps = 1e-5f)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive");
        if (headLayerCount < 0)
            throw new ArgumentOutOfRangeException(
                nameof(headLayerCount), headLayerCount, "headLayerCount cannot be negative");
        if (actClassCount <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(actClassCount), actClassCount, "actClassCount must be positive");

        // max(1, d // 64), the reference's own expression. Integer division is the point: a
        // headDim chosen as a divisor of d is what keeps the attention reshape exact.
        NumHeads = Math.Max(1, hiddenSize / HeadDimTarget);
        if (hiddenSize % NumHeads != 0)
            throw new ArgumentException(
                $"hiddenSize {hiddenSize} does not divide into {NumHeads} heads at {HeadDimTarget} wide");

        typeEmb = new Embedding<T>(QuestionTypeCount, hiddenSize);
        scorerNorm = new LayerNorm<T>(hiddenSize, eps);
        scorerHidden = new Linear<T>(hiddenSize, hiddenSize, bias: true);
        scorerOut = new Linear<T>(hiddenSize, 1, bias: true);
        actInput = new Linear<T>(hiddenSize + LayaHeadOutput.FeatureCount, ActHeadHidden, bias: true);
        actOut = new Linear<T>(ActHeadHidden, actClassCount, bias: true);

        headLayers = new LayaHeadLayer<T>[headLayerCount];
        for (int i = 0; i < headLayerCount; i++)
            headLayers[i] = new LayaHeadLayer<T>(hiddenSize, 4 * hiddenSize, NumHeads, eps);

        // The reference registers torch.ones(3); a checkpoint overrides it. The value is kept
        // unloaded-as-one rather than left at a silent identity.
        checkpointTemperature = new T[QuestionTypeCount];
        for (int i = 0; i < QuestionTypeCount; i++)
            checkpointTemperature[i] = T.One;

        RegisterModules(typeEmb, scorerNorm, scorerHidden, scorerOut, actInput, actOut);
        foreach (var layer in headLayers)
            RegisterModules(layer);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => throw new NotSupportedException(
            "The decision head needs the question type, the marker positions and the valid length. " +
            "Use Forward(hidden, type, markerPositions, validLength).");

    /// <summary>Runs the head over one question's encoder output.</summary>
    /// <param name="hidden">
    /// The encoder's last hidden state, shape <c>[seqLen, hiddenSize]</c>, as produced by
    /// <see cref="ModernBertEncoder{TModel}.Forward"/>.
    /// </param>
    /// <param name="type">The question type, selecting the type-embedding row.</param>
    /// <param name="markerPositions">
    /// Indices of the <c>[MASK]</c> tokens, one per option, as returned by
    /// <see cref="LayaPromptBuilder.BuildSequence"/>. Fewer than the question's option count is
    /// reachable: the builder drops markers the length budget pushed out.
    /// </param>
    /// <param name="validLength">
    /// The number of leading positions that are real tokens. The reference's collate step makes the
    /// real tokens a prefix of the row, so the padding mask is this and not an arbitrary vector.
    /// </param>
    public LayaHeadOutput Forward(
        ReverseGradTensor<T> hidden,
        LayaQuestionType type,
        IReadOnlyList<int> markerPositions,
        int validLength)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(markerPositions);

        int seqLen = hidden.Shape[0];
        int hiddenSize = hidden.Shape[1];
        if (hiddenSize != HiddenSize)
            throw new ArgumentException(
                $"The encoder returned {hiddenSize} features but the head expects {HiddenSize}.",
                nameof(hidden));
        if (validLength <= 0 || validLength > seqLen)
            throw new ArgumentOutOfRangeException(
                nameof(validLength), validLength, $"validLength must be within [1, {seqLen}].");
        if (markerPositions.Count == 0)
            throw new ArgumentException(
                "No marker positions survived. Every option's [MASK] was pushed out by the length " +
                "budget, so the question does not fit head_max_len.", nameof(markerPositions));
        if (seqLen > ModernBertMasks.MaxDenseLength)
            throw new ArgumentOutOfRangeException(
                nameof(hidden),
                $"seqLen {seqLen} exceeds the dense-mask limit of {ModernBertMasks.MaxDenseLength}. " +
                "The head's attention builds a [L, L] key-padding mask, exactly as the encoder's does.");

        // h = h + type_emb[qtype][:, None, :] — the type embedding is one row, broadcast over every
        // position. BroadcastAdd is the op for that (Add requires equal lengths), and it keeps the
        // gradient edge back to type_emb rather than materialising a detached copy of the row.
        var typeRow = typeEmb.Forward((int)type);
        typeRow.Reshape(typeRow.Data.Length);
        var withType = ReverseGradOperations.BroadcastAdd(hidden, typeRow);

        if (headLayers.Length > 0)
        {
            var paddingMask = BuildPrefixMask(seqLen, validLength);
            for (int i = 0; i < headLayers.Length; i++)
                withType = headLayers[i].Forward(withType, paddingMask);
        }

        // torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, d)) — a row gather, because the
        // batch dimension is 1. Gather validates each index against the row count.
        var markerHidden = ReverseGradOperations.Gather(withType, [.. markerPositions]);

        // scorer: Sequential(LayerNorm, Linear, GELU, Linear) → [k, 1]
        var scored = scorerOut.Forward(ReverseGradOperations.GeluExact(
            scorerHidden.Forward(scorerNorm.Forward(markerHidden))));

        int k = markerPositions.Count;
        var logits = new float[k];
        int idx = 0;
        if (scored.Data.TryGetSpan(out var scoredSpan))
            for (; idx < k && idx < scoredSpan.Length; idx++)
                logits[idx] = float.CreateChecked(scoredSpan[idx]);
        for (; idx < k; idx++)
            logits[idx] = 0.0f;

        var probabilities = Softmax(logits);
        double clampedK = Math.Max(2, k);
        double entropy = 0.0;
        for (int i = 0; i < k; i++)
        {
            double p = Math.Max(probabilities[i], 1e-9);
            entropy -= p * Math.Log(p);
        }
        entropy /= Math.Log(clampedK);

        // p.topk(2) raises on a one-element distribution, so the reference special-cases it by
        // padding the missing second entry with 0.0. Softmax over one logit is 1.0 whatever the logit
        // is, so top1 - top2 == 1.0, the same "fully decided" signal an unambiguous question gives.
        double top1 = probabilities[0];
        double top2 = k >= 2 ? SecondHighest(probabilities) : 0.0;

        var features = new double[LayaHeadOutput.FeatureCount];
        features[LayaHeadOutput.FeatureTop1] = top1;
        features[LayaHeadOutput.FeatureGap] = top1 - top2;
        features[LayaHeadOutput.FeatureEntropy] = entropy;
        features[LayaHeadOutput.FeatureOptionCount] = clampedK / 255.0;

        // pooled = h[:, 0] — the post-head-row 0, i.e. the [CLS] position after the head stack.
        var pooled = ReverseGradOperations.Gather(withType, [0]);
        var featureRow = ReverseGradTensor<T>.FromMatrix(ToTensor(features), 1, LayaHeadOutput.FeatureCount);
        var actInputTensor = ReverseGradOperations.Concat([pooled, featureRow], axis: 1);

        // act_head: Sequential(Linear, GELU, Linear)
        var actLogits = actOut.Forward(ReverseGradOperations.GeluExact(actInput.Forward(actInputTensor)));
        var actionProbabilities = Softmax(ToFloats(actLogits));

        return new LayaHeadOutput
        {
            Logits = logits,
            MarkerProbabilities = probabilities,
            Features = features,
            ActionProbability = actionProbabilities[0],
            ActionClassCount = actionProbabilities.Length
        };
    }

    /// <summary>
    /// Loads head weights from a safetensors dictionary. The encoder is not touched: pass the same
    /// dictionary to <see cref="ModernBertEncoder{TModel}.LoadWeights{TModel, TWeight}"/> with
    /// prefix <c>"encoder"</c>.
    /// </summary>
    public static LayaDecisionHead<TModel> LoadWeights<TModel, TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(tensors);

        if (!tensors.TryGetValue("type_emb.weight", out var typeWeight))
            throw new KeyNotFoundException(
                "Missing tensor: type_emb.weight. A stock encoder checkpoint is not a Laya decision " +
                "model; the head is what makes it one.");

        var head = new LayaDecisionHead<TModel>(typeWeight.Shape[1], headLayerCount, actClassCount);
        int hidden = head.HiddenSize;

        StateDictLoader.LoadEmbed(head.typeEmb, tensors, "type_emb.weight");

        for (int i = 0; i < head.headLayers.Length; i++)
        {
            var layer = head.headLayers[i];
            string prefix = $"head.layers.{i}";

            StateDictLoader.LoadLayerNorm(layer.norm1, tensors, $"{prefix}.norm1");
            StateDictLoader.LoadLayerNorm(layer.norm2, tensors, $"{prefix}.norm2");

            // nn.MultiheadAttention fuses q, k and v into one [3 * d, d] weight and one [3 * d] bias,
            // named in_proj_weight / in_proj_bias. The blocks are contiguous and in q, k, v order.
            string attnPrefix = $"{prefix}.self_attn";
            StateDictLoader.LoadFusedLinearSlice(layer.attn.qProj, tensors,
                $"{attnPrefix}.in_proj_weight", $"{attnPrefix}.in_proj_bias", 0, hidden);
            StateDictLoader.LoadFusedLinearSlice(layer.attn.kProj, tensors,
                $"{attnPrefix}.in_proj_weight", $"{attnPrefix}.in_proj_bias", hidden, hidden);
            StateDictLoader.LoadFusedLinearSlice(layer.attn.vProj, tensors,
                $"{attnPrefix}.in_proj_weight", $"{attnPrefix}.in_proj_bias", 2 * hidden, hidden);
            StateDictLoader.LoadLinear(layer.attn.oProj, tensors, $"{attnPrefix}.out_proj");

            StateDictLoader.LoadLinear(layer.linear1, tensors, $"{prefix}.linear1");
            StateDictLoader.LoadLinear(layer.linear2, tensors, $"{prefix}.linear2");
        }

        // nn.Sequential indices: scorer = [LayerNorm, Linear, GELU, Linear], so the two Linear layers
        // are at 1 and 3 and the GELU occupies the gap at 2.
        StateDictLoader.LoadLayerNorm(head.scorerNorm, tensors, "scorer.0");
        StateDictLoader.LoadLinear(head.scorerHidden, tensors, "scorer.1");
        StateDictLoader.LoadLinear(head.scorerOut, tensors, "scorer.3");

        // act_head = [Linear, GELU, Linear] → indices 0 and 2.
        StateDictLoader.LoadLinear(head.actInput, tensors, "act_head.0");
        StateDictLoader.LoadLinear(head.actOut, tensors, "act_head.2");

        if (tensors.TryGetValue("temperature", out var temperature))
        {
            for (int i = 0; i < head.checkpointTemperature.Length && i < temperature.Data.Length; i++)
                head.checkpointTemperature[i] = TModel.CreateChecked(double.CreateChecked(temperature.Data[i]));
        }

        head.Eval();
        return head;
    }

    /// <summary>
    /// A 1-D key-padding mask of length <paramref name="seqLen"/>: 1 for the real tokens, 0 for the
    /// padding. Suppressing a key is expressed as a value below 0.5, which is what the reference's
    /// <c>~attention_mask</c> means once read as an additive mask.
    /// </summary>
    /// <remarks>
    /// The mask must be one entry per <em>position</em>, not per valid token. A shorter mask leaves
    /// the trailing positions unconstrained, which is not a weaker mask but the wrong one: nothing
    /// after it gets suppressed and the padding is attended to normally.
    /// </remarks>
    static ReverseGradTensor<T> BuildPrefixMask(int seqLen, int validLength)
    {
        var data = new T[seqLen];
        for (int i = 0; i < seqLen; i++)
            data[i] = i < validLength ? T.One : T.Zero;
        return ReverseGradTensor<T>.FromArray(data);
    }

    static double[] Softmax(ReadOnlySpan<float> logits)
    {
        var p = new double[logits.Length];
        if (logits.Length == 0)
            return p;

        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            max = Math.Max(max, logits[i]);

        double sum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            p[i] = Math.Exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < logits.Length; i++)
            p[i] /= sum;
        return p;
    }

    static double SecondHighest(double[] values)
    {
        double top = double.NegativeInfinity, second = double.NegativeInfinity;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] > top)
            {
                second = top;
                top = values[i];
            }
            else if (values[i] > second)
            {
                second = values[i];
            }
        }
        return second;
    }

    static T[] ToTensor(double[] values)
    {
        var converted = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
            converted[i] = T.CreateChecked(values[i]);
        return converted;
    }

    static float[] ToFloats(ReverseGradTensor<T> tensor)
    {
        var values = new float[tensor.Data.Length];
        if (tensor.Data.TryGetSpan(out var span))
            for (int i = 0; i < values.Length && i < span.Length; i++)
                values[i] = float.CreateChecked(span[i]);
        return values;
    }
}
