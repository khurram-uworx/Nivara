using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Shared;

namespace NivaraChat.SmolLM;

internal sealed class SilkSmolLMChatClient : CausalLmChatClientBase
{
    readonly SilkSmolLMSession session;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly TokenizePrefixCache tokenizeCache = new();

    public SilkSmolLMChatClient(
        SilkSmolLMSession session,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        int maxNewTokens = 64,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = true)
        : base(temperature ?? 0f, Math.Clamp(topP ?? 1f, 0f, 1f), maxNewTokens, useKvCache, new Random(seed ?? 0))
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        this.session = session;
        this.tokenizer = tokenizer;
        this.config = config;
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "SmolLM-135M-Instruct (Silk/OpenCL causal LM + GPT-2 BPE)");

    public int TokenizeCacheHits => tokenizeCache.Hits;

    public override void Dispose() => session.Dispose();

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => Nivara.Samples.SmollmChatTemplate.Render(messages, addGenerationPrompt: true);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizeCache.Encode(prompt, p => SmolLMChatEncoding.EncodePrompt(tokenizer, p));

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode(new[] { tokenId });

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => string.Concat(ids.Select(id => tokenizer.Decode(new[] { id })));

    protected override CausalLmOps BuildOps() => new(
        Prefill: w => session.ForwardPrefill(w),
        Cached: (t, p) => session.ForwardCached(t, p),
        LastLogits: w =>
        {
            var full = session.Forward(w);
            int vocab = config.VocabSize;
            int offset = (w.Length - 1) * vocab;
            var span = offset >= 0 && full.Length >= offset + vocab
                ? full.AsSpan(offset, vocab)
                : full.AsSpan(0, Math.Min(vocab, full.Length));
            return span.ToArray();
        },
        ResetCache: session.ResetCache,
        IsStopToken: id => id == config.EosTokenId,
        VocabSize: config.VocabSize,
        EosTokenId: config.EosTokenId,
        MaxSequenceLength: config.MaxPositionEmbeddings);
}
