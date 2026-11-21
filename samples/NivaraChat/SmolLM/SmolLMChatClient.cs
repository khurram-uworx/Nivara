using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using NivaraChat.Shared;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NivaraChat.SmolLM;

/// <summary>
/// An <see cref="IChatClient"/> backed by a pretrained SmolLM/Llama causal LM.
/// </summary>
internal sealed class SmolLMChatClient<TModel, TWeight> : CausalLmChatClientBase
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    static readonly string[] SpecialTokens =
    [
        Nivara.Samples.SmollmChatTemplate.ImStart,
        Nivara.Samples.SmollmChatTemplate.ImEnd,
        "<|endoftext|>",
    ];

    readonly LlamaForCausalLM<TModel, TWeight> model;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly TokenizePrefixCache tokenizeCache = new();
    readonly int numLayers;
    readonly int kvWidth;

    public SmolLMChatClient(
        LlamaForCausalLM<TModel, TWeight> model,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        int maxNewTokens = 64,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = false)
        : base(temperature ?? 0f, Math.Clamp(topP ?? 1f, 0f, 1f), maxNewTokens, useKvCache, new Random(seed ?? 0))
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.config = config;
        this.numLayers = config.NumHiddenLayers;
        this.kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "SmolLM-135M-Instruct (LlamaForCausalLM + GPT-2 BPE)");

    public int TokenizeCacheHits => tokenizeCache.Hits;

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => Nivara.Samples.SmollmChatTemplate.Render(messages, addGenerationPrompt: true);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizeCache.Encode(prompt, p => EncodePrompt(p));

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode(new[] { tokenId });

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => string.Concat(ids.Select(id => tokenizer.Decode(new[] { id })));

    protected override CausalLmOps BuildOps()
    {
        LlamaKVCache<TModel>? cache = null;
        return new CausalLmOps(
            Prefill: w =>
            {
                cache ??= new LlamaKVCache<TModel>(numLayers, kvWidth);
                return ToFloat(model.ForwardPrefill(w, cache));
            },
            Cached: (t, p) => ToFloat(model.ForwardCached(t, p, cache!)),
            LastLogits: w => ToFloat(model.Forward(w)),
            ResetCache: () => cache = new LlamaKVCache<TModel>(numLayers, kvWidth),
            IsStopToken: id => id == config.EosTokenId,
            VocabSize: config.VocabSize,
            EosTokenId: config.EosTokenId,
            MaxSequenceLength: config.MaxPositionEmbeddings);
    }

    static float[] ToFloat(ReverseGradTensor<TModel> logits)
    {
        var span = LastRow(logits);
        var arr = new float[span.Length];
        for (int i = 0; i < arr.Length; i++)
            arr[i] = float.CreateChecked(span[i]);
        return arr;
    }

    static ReadOnlySpan<TModel> LastRow(ReverseGradTensor<TModel> logits)
    {
        logits.Data.TryGetSpan(out var span);
        int vocab = logits.Shape[^1];
        int rows = logits.Shape[0];
        int offset = (rows - 1) * vocab;
        return offset >= 0 && span.Length >= offset + vocab ? span.Slice(offset, vocab) : span;
    }

    IReadOnlyList<int> EncodePrompt(string prompt)
    {
        var ids = new List<int>();
        int pos = 0;
        while (pos < prompt.Length)
        {
            int special = -1;
            int bestIndex = prompt.Length;
            for (int i = 0; i < SpecialTokens.Length; i++)
            {
                int idx = prompt.IndexOf(SpecialTokens[i], pos, StringComparison.Ordinal);
                if (idx >= 0 && idx < bestIndex)
                {
                    bestIndex = idx;
                    special = i;
                }
            }

            if (special < 0)
            {
                ids.AddRange(tokenizer.Encode(prompt[pos..]));
                break;
            }

            if (bestIndex > pos)
                ids.AddRange(tokenizer.Encode(prompt[pos..bestIndex]));

            int tokenId = tokenizer.TokenId(SpecialTokens[special]);
            if (tokenId >= 0)
                ids.Add(tokenId);

            pos = bestIndex + SpecialTokens[special].Length;
        }
        return ids;
    }
}
