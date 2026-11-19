using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Shared;

namespace NivaraChat.Qwen;

internal sealed class SilkQwenChatClient : CausalLmChatClientBase, IDisposable
{
    readonly SilkQwenSession session;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly TokenizePrefixCache tokenizeCache = new();
    readonly IReadOnlyList<string>? knownToolNames;
    readonly Action<string>? turnCallback;

    public SilkQwenChatClient(
        SilkQwenSession session,
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
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        this.session = session;
        this.tokenizer = tokenizer;
        this.config = config;
        this.knownToolNames = knownToolNames;
        this.turnCallback = turnCallback;
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "Qwen2.5-0.5B-Instruct (Silk/OpenCL causal LM + GPT-2 BPE)");

    public int TokenizeCacheHits => tokenizeCache.Hits;

    public override void Dispose() => session.Dispose();

    public override Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var ids = GenerateIds(messages, cancellationToken).ToList();
        var text = tokenizer.Decode(ids);

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

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var ids = GenerateIds(messages, cancellationToken).ToList();
        foreach (var id in ids)
            yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode(new[] { id }));
    }

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => QwenChatTemplate.Render(messages, addGenerationPrompt: true);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizeCache.Encode(prompt, p => tokenizer.Encode(p));

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode(new[] { tokenId });

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => tokenizer.Decode(ids);

    protected override CausalLmOps BuildOps() => new(
        Prefill: w => LastVocab(session.ForwardPrefill(w)),
        Cached: (t, p) => LastVocab(session.ForwardCached(t, p)),
        LastLogits: w => LastVocab(session.Forward(w)),
        ResetCache: session.ResetCache,
        IsStopToken: id => QwenIds.StopIds.Contains(id),
        VocabSize: config.VocabSize,
        EosTokenId: config.EosTokenId,
        MaxSequenceLength: config.MaxPositionEmbeddings);

    protected override bool BreakOnContextLimit => true;

    float[] LastVocab(float[] logits)
    {
        int vocab = config.VocabSize;
        int start = logits.Length - vocab;
        return (start >= 0 && logits.Length >= start + vocab
            ? logits.AsSpan(start, vocab)
            : logits.AsSpan(0, Math.Min(vocab, logits.Length))).ToArray();
    }
}
