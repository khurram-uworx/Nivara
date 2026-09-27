namespace Nivara.Samples;

/// <summary>Question types the Laya head discriminates between.</summary>
public enum LayaQuestionType
{
    Choice = 0,
    Score = 1,
    Noul = 2
}

/// <summary>One option of a <c>choice</c> question: a label plus an optional description.</summary>
/// <param name="Key">The option's own label. Becomes the reported answer when selected.</param>
/// <param name="Description">
/// The description, or <see langword="null"/> / <see cref="string.Empty"/> for a bare label.
/// Only those two mean "no description" — a description of "0" or "false" is a real value.
/// </param>
public readonly record struct LayaChoiceOption(string Key, string? Description);

/// <summary>
/// A Laya question, in the shape the prompt builder needs. This mirrors the reference's
/// <c>q</c> dict, restricted to the string-valued domain (see <see cref="LayaPromptBuilder"/>).
/// </summary>
public sealed record LayaQuestion
{
    public required string Id { get; init; }
    public required LayaQuestionType Type { get; init; }

    /// <summary>The question text. The reference stringifies it, so it is a string here.</summary>
    public required string Instructions { get; init; }

    /// <summary>Ordered options for a <see cref="LayaQuestionType.Choice"/> question.</summary>
    /// <remarks>
    /// Order is the contract, exactly as a Python dict's insertion order is: the reported answer is
    /// the key at the argmax index, and the per-option probabilities zip against these keys in order.
    /// </remarks>
    public IReadOnlyList<LayaChoiceOption> Options { get; init; } = [];

    /// <summary>Ordered criteria for a <see cref="LayaQuestionType.Score"/> question.</summary>
    /// <remarks>Rendered as "level {i}: {criterion}", so the index is the reported score.</remarks>
    public IReadOnlyList<string?> ScoreCriteria { get; init; } = [];

    /// <summary>The "false" label for a <see cref="LayaQuestionType.Noul"/> question.</summary>
    public string? NoulFalseLabel { get; init; }

    /// <summary>The "true" label for a <see cref="LayaQuestionType.Noul"/> question.</summary>
    public string? NoulTrueLabel { get; init; }

    /// <summary>Optional criterion text shown against the "false" option.</summary>
    public string? NoulFalseCriterion { get; init; }

    /// <summary>Optional criterion text shown against the "true" option.</summary>
    public string? NoulTrueCriterion { get; init; }
}

/// <summary>One built prompt: the token ids and the per-option marker positions within them.</summary>
public readonly record struct LayaSequence(int[] Ids, int[] Markers)
{
    public int Length => Ids.Length;
}

/// <summary>
/// Builds Laya prompts as a port of <c>build_sequence</c> / <c>render_options</c> / <c>render_criterion</c>
/// in <c>laya/common.py</c> (PyPI <c>laya</c> 0.3.20).
/// </summary>
/// <remarks>
/// <para>Format: <c>[CLS] &lt;type&gt; question: &lt;ins&gt; [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]</c>.</para>
/// <para>
/// The layout: <c>[CLS]</c>, the tokenized <c>"&lt;type&gt; question: &lt;ins&gt;"</c> head, a <c>[SEP]</c>,
/// then each option preceded by its own <c>[MASK]</c> — that mask is where the head scores the
/// option — a <c>[SEP]</c>, the truncated state, and a closing <c>[SEP]</c>. The head gathers
/// hidden states at the mask positions, so the marker indices are part of the contract, not an
/// implementation detail.
/// </para>
/// <para>
/// <b>Domain restriction, deliberate.</b> The reference accepts dict and list states and
/// structured criterion values, serializing them with Python's <c>json.dumps</c> using
/// <c>ensure_ascii=False, separators=(", ", ": ")</c>. Matching that byte-for-byte is a separate
/// port with its own float- and exponent-formatting edge cases, so this one takes
/// <see cref="string"/> states and <see cref="string"/> criteria only. That covers every input the
/// gate exercises; a structured state is rejected rather than silently serialized differently.
/// </para>
/// </remarks>
public static class LayaPromptBuilder
{
    /// <summary>Default total sequence budget, from the agent config.</summary>
    public const int DefaultMaxLen = 512;

    /// <summary>Default budget for the head (question + options) portion, from the agent config.</summary>
    public const int DefaultHeadMaxLen = 192;

    /// <summary>Per-option token cap. Matches the reference's <c>max_length=48</c>.</summary>
    internal const int OptionTokenCap = 48;

    /// <summary>
    /// The head is given at least this many instruction tokens after the option shrink, so a
    /// question with very many options still has a readable stem.
    /// </summary>
    internal const int MinimumHeadTokens = 8;

    /// <summary>
    /// If the options leave fewer than this many instruction tokens, every option is shrunk
    /// evenly instead.
    /// </summary>
    internal const int MinimumOptionBudget = 16;

    internal const string MaskToken = "[MASK]";
    internal const string ClsToken = "[CLS]";
    internal const string SepToken = "[SEP]";

    /// <summary>
    /// Part of the token set but never emitted by <see cref="BuildSequence"/>: a built prompt has
    /// its exact length, and padding is the collate step's business.
    /// </summary>
    public const string PadToken = "[PAD]";

    internal const string DefaultNoulFalseLabel = "false";
    internal const string DefaultNoulTrueLabel = "true";
    internal const string DefaultNoulFalseCriterion = "no, the statement does not hold";
    internal const string DefaultNoulTrueCriterion = "yes, the statement holds";

    /// <summary>The reference's <c>QTYPES</c> name for a question type, used in the prompt text.</summary>
    public static string TypeName(LayaQuestionType type) => type switch
    {
        LayaQuestionType.Choice => "choice",
        LayaQuestionType.Score => "score",
        LayaQuestionType.Noul => "noul",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
    };

    /// <summary>
    /// Renders one criterion as prompt text. Strings pass through unchanged, which is the whole of
    /// the in-scope behaviour: the reference's structured branch is the JSON path this port
    /// deliberately excludes.
    /// </summary>
    public static string RenderCriterion(string? value) => value ?? string.Empty;

    /// <summary>
    /// Resolves the two <c>noul</c> option labels, defaulting to "false" / "true" and rejecting
    /// anything that is not two distinct non-empty strings. The reference raises here, and a port
    /// that quietly accepted a degenerate pair would produce a prompt whose two options are
    /// indistinguishable.
    /// </summary>
    public static (string False, string True) ResolveNoulLabels(string? falseLabel, string? trueLabel)
    {
        var f = (falseLabel ?? DefaultNoulFalseLabel).Trim();
        var t = (trueLabel ?? DefaultNoulTrueLabel).Trim();
        if (f.Length == 0 || t.Length == 0 || f == t)
            throw new ArgumentException(
                "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
        return (f, t);
    }

    /// <summary>Renders the option texts in label-index order.</summary>
    /// <remarks>
    /// <c>noul</c>'s semantic order is always [false, true], so <c>p[1]</c> is the noul probability
    /// regardless of how the caller phrased the labels.
    /// </remarks>
    public static IReadOnlyList<string> RenderOptions(LayaQuestion question)
    {
        ArgumentNullException.ThrowIfNull(question);

        switch (question.Type)
        {
            case LayaQuestionType.Choice:
                if (question.NoulFalseLabel is not null || question.NoulTrueLabel is not null)
                    throw new ArgumentException(
                        $"{question.Id}: labels are only supported for noul questions",
                        nameof(question));
                return RenderChoiceOptions(question);

            case LayaQuestionType.Score:
                if (question.NoulFalseLabel is not null || question.NoulTrueLabel is not null)
                    throw new ArgumentException(
                        $"{question.Id}: labels are only supported for noul questions",
                        nameof(question));
                return RenderScoreOptions(question);

            case LayaQuestionType.Noul:
                return RenderNoulOptions(question);

            default:
                throw new ArgumentOutOfRangeException(nameof(question), question.Type, null);
        }
    }

    static IReadOnlyList<string> RenderChoiceOptions(LayaQuestion question)
    {
        var rendered = new List<string>(question.Options.Count);
        foreach (var option in question.Options)
            rendered.Add(IsAbsent(option.Description)
                ? option.Key
                : string.Concat(option.Key, ": ", RenderCriterion(option.Description)));
        return rendered;
    }

    static IReadOnlyList<string> RenderScoreOptions(LayaQuestion question)
    {
        var rendered = new List<string>(question.ScoreCriteria.Count);
        for (int i = 0; i < question.ScoreCriteria.Count; i++)
            rendered.Add(string.Concat("level ", i.ToString(), ": ",
                RenderCriterion(question.ScoreCriteria[i])));
        return rendered;
    }

    static IReadOnlyList<string> RenderNoulOptions(LayaQuestion question)
    {
        var (falseLabel, trueLabel) = ResolveNoulLabels(question.NoulFalseLabel, question.NoulTrueLabel);
        var falseCriterion = IsAbsent(question.NoulFalseCriterion)
            ? DefaultNoulFalseCriterion
            : RenderCriterion(question.NoulFalseCriterion);
        var trueCriterion = IsAbsent(question.NoulTrueCriterion)
            ? DefaultNoulTrueCriterion
            : RenderCriterion(question.NoulTrueCriterion);
        return
        [
            string.Concat(falseLabel, ": ", falseCriterion),
            string.Concat(trueLabel, ": ", trueCriterion)
        ];
    }

    /// <summary>
    /// Only <see langword="null"/> and the empty string mean "absent". A criterion of <c>"0"</c> or
    /// <c>"false"</c> is a real value and must be rendered — this is where the stale reference
    /// copy got it wrong, dropping a legitimate criterion.
    /// </summary>
    static bool IsAbsent(string? value) => value is null || value.Length == 0;

    /// <summary>
    /// Builds the token sequence and marker positions for one question against one state.
    /// </summary>
    /// <param name="tokenizer">The Laya/ModernBERT byte-level tokenizer.</param>
    /// <param name="stateText">The state, as text. Structured states are out of scope.</param>
    /// <param name="question">The question to score.</param>
    /// <param name="maxLen">Total sequence budget, including both <c>[SEP]</c> boundaries.</param>
    /// <param name="headMaxLen">Budget for the question stem plus all options.</param>
    /// <param name="optionOrder">
    /// Index order to emit options in. Defaults to natural order. The reference exposes this for
    /// presenting options in a different display order than their score order.
    /// </param>
    /// <param name="truncateLeft">
    /// Whether to keep the state's *last* tokens rather than its first. The reference defaults to
    /// false and only sets it for chronological conversation lists, where right-truncation would
    /// drop the newest turn. The right slice is <c>state[max(0, len - room):]</c> — never
    /// <c>state[-room:]</c>, because with <c>room == 0</c> Python's <c>-0</c> is <c>0</c> and
    /// <c>state[-0:]</c> is the entire state, silently injecting it and displacing the final
    /// <c>[SEP]</c>.
    /// </param>
    /// <param name="stateIds">
    /// Pre-tokenized state ids, so one state can be tokenized once and reused across every
    /// question. When null the state is tokenized here.
    /// </param>
    public static LayaSequence BuildSequence(
        Gpt2BpeTokenizer tokenizer,
        string stateText,
        LayaQuestion question,
        int maxLen = DefaultMaxLen,
        int headMaxLen = DefaultHeadMaxLen,
        IReadOnlyList<int>? optionOrder = null,
        bool truncateLeft = false,
        IReadOnlyList<int>? stateIds = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(stateText);
        ArgumentNullException.ThrowIfNull(question);

        int maskId = RequireToken(tokenizer, MaskToken);
        int clsId = RequireToken(tokenizer, ClsToken);
        int sepId = RequireToken(tokenizer, SepToken);

        var options = RenderOptions(question);
        var order = optionOrder ?? Enumerable.Range(0, options.Count).ToArray();
        if (order.Count != options.Count)
            throw new ArgumentException(
                $"{question.Id}: optionOrder has {order.Count} entries for {options.Count} options",
                nameof(optionOrder));

        var instructions = question.Instructions.Replace(MaskToken, " ", StringComparison.Ordinal);
        var headIds = tokenizer
            .Encode(string.Concat(TypeName(question.Type), " question: ", instructions))
            .ToArray();

        var optionIdGroups = new List<int[]>(order.Count);
        foreach (int i in order)
        {
            if ((uint)i >= (uint)options.Count)
                throw new ArgumentOutOfRangeException(nameof(optionOrder), i, "option index out of range");

            var text = string.Concat(" ", options[i].Replace(MaskToken, " ", StringComparison.Ordinal));

            // The reference caps inside the tokenizer call (truncation=True, max_length=48) rather
            // than slicing afterwards. Both yield byte-identical ids — HF truncates the token list
            // to 48, which is exactly what [:48] does — so slicing here is equivalent. The
            // difference is only that the reference avoids tokenizing a long description tail,
            // which is a throughput concern this port does not have: it always tokenizes in full.
            var tokens = tokenizer.Encode(text);
            var group = new int[Math.Min(tokens.Count, OptionTokenCap) + 1];
            group[0] = maskId;
            for (int t = 0; t + 1 < group.Length; t++)
                group[t + 1] = tokens[t];
            optionIdGroups.Add(group);
        }

        int optBudget = headMaxLen - optionIdGroups.Sum(g => g.Length);
        if (optBudget < MinimumOptionBudget)
        {
            int per = Math.Max(4, (headMaxLen - MinimumOptionBudget) / Math.Max(1, optionIdGroups.Count));
            optionIdGroups = optionIdGroups.Select(g => g[..Math.Min(g.Length, per)]).ToList();

            // Recomputed from the shrunk groups, not carried over: the reference's second
            // assignment is what the head-token clamp below consumes.
            optBudget = headMaxLen - optionIdGroups.Sum(g => g.Length);
        }

        headIds = headIds[..Math.Min(headIds.Length, Math.Max(MinimumHeadTokens, optBudget))];

        var ids = new List<int> { clsId };
        ids.AddRange(headIds);
        ids.Add(sepId);

        var markers = new List<int>(optionIdGroups.Count);
        foreach (var group in optionIdGroups)
        {
            markers.Add(ids.Count);
            ids.AddRange(group);
        }
        ids.Add(sepId);

        // The -1 pays for the closing [SEP]; without it a full-length prompt would push the final
        // separator out of the [:maxLen] slice.
        int room = Math.Max(0, maxLen - ids.Count - 1);

        // No length cap on the state. HF's tokenizer warns when a sequence exceeds
        // model_max_length (8192) but does not truncate, so the state is bounded solely by room,
        // i.e. by maxLen. Capping here would diverge from the reference on long states.
        var state = stateIds ?? tokenizer.Encode(stateText.Replace(MaskToken, " ", StringComparison.Ordinal));

        int take = Math.Min(room, state.Count);
        int offset = truncateLeft ? state.Count - take : 0;
        for (int i = 0; i < take; i++)
            ids.Add(state[offset + i]);
        ids.Add(sepId);

        if (ids.Count > maxLen)
            ids.RemoveRange(maxLen, ids.Count - maxLen);

        // A marker beyond the truncation point has no [MASK] left to gather, so it is dropped.
        // This is what makes len(markers) != len(options) reachable; the reference's agent checks
        // for it and rejects questions whose options overflow head_max_len.
        return new LayaSequence(
            [.. ids],
            [.. markers.Where(m => m < maxLen)]);
    }

    static int RequireToken(Gpt2BpeTokenizer tokenizer, string token)
    {
        int id = tokenizer.TokenId(token);
        // A silent -1 would splice a garbage id into every prompt and still "run", so this fails
        // at the boundary instead.
        if (id < 0)
            throw new InvalidOperationException(
                $"The tokenizer does not define {token}. Load it from the model's tokenizer directory, not the model root.");
        return id;
    }
}
