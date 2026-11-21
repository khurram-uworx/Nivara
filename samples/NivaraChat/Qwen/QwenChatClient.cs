using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using NivaraChat.Shared;
using System.Numerics;

namespace NivaraChat.Qwen;

/// <summary>
/// An <see cref="IChatClient"/> backed by Qwen2.5-0.5B-Instruct.
/// </summary>
internal sealed class QwenChatClient<TModel, TWeight> : CausalLmChatClientBase
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly LlamaForCausalLM<TModel, TWeight> model;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly TokenizePrefixCache tokenizeCache = new();
    readonly IReadOnlyList<string>? knownToolNames;
    readonly Action<string>? turnCallback;
    readonly int numLayers;
    readonly int kvWidth;

    public QwenChatClient(
        LlamaForCausalLM<TModel, TWeight> model,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        int maxNewTokens = 128,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = true,
        IReadOnlyList<string>? knownToolNames = null,
        Action<string>? turnCallback = null)
        : base(temperature ?? 0f, Math.Clamp(topP ?? 1f, 0f, 1f), maxNewTokens, useKvCache, new Random(seed ?? 0))
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.config = config;
        this.knownToolNames = knownToolNames;
        this.turnCallback = turnCallback;
        this.numLayers = config.NumHiddenLayers;
        this.kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "Qwen2.5-0.5B-Instruct (LlamaForCausalLM + GPT-2 BPE)");

    public int TokenizeCacheHits => tokenizeCache.Hits;

    public override Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var generated = new List<int>();
        lock (rng)
        {
            foreach (var id in GenerateIds(messages, cancellationToken))
                generated.Add(id);
        }

        var text = DecodeAll(generated);
        turnCallback?.Invoke(text);

        var calls = QwenToolCallParser.Parse(text, knownToolNames);
        var responseMessage = new ChatMessage(ChatRole.Assistant, (string?)null);
        if (calls.Count > 0)
            foreach (var call in calls)
                responseMessage.Contents.Add(call);
        else
            responseMessage.Contents.Add(new TextContent(text));

        return Task.FromResult(new ChatResponse(responseMessage));
    }

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => Nivara.Samples.QwenChatTemplate.Render(messages, addGenerationPrompt: true);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizeCache.Encode(prompt, p => tokenizer.Encode(p));

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode(new[] { tokenId });

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => tokenizer.Decode(ids);

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
            IsStopToken: id => Nivara.Samples.QwenIds.StopIds.Contains(id),
            VocabSize: config.VocabSize,
            EosTokenId: config.EosTokenId,
            MaxSequenceLength: config.MaxPositionEmbeddings);
    }

    protected override bool BreakOnContextLimit => true;

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
}
