using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Shared;
using NivaraChat.SmolLM;
using System.Text;

namespace NivaraChat.SmolLM;

internal sealed class SilkSmolLMChatClient : IChatClient
{
    readonly SilkSmolLMSession session;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly int maxNewTokens;
    readonly float temperature;
    readonly float topP;
    readonly bool useKvCache;
    readonly Random rng = new(0);
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
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        this.session = session;
        this.tokenizer = tokenizer;
        this.config = config;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
        this.temperature = temperature ?? 0f;
        this.topP = Math.Clamp(topP ?? 1f, 0f, 1f);
        this.useKvCache = useKvCache;
        this.rng = new Random(seed ?? 0);
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "SmolLM-135M-Instruct (Silk/OpenCL causal LM + GPT-2 BPE)");

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var text = Generate(messages, cancellationToken);
        return Task.FromResult(new ChatResponse([new ChatMessage(ChatRole.Assistant, text)]));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true);
        var ids = new List<int>(tokenizeCache.Encode(prompt, p => SmolLMChatEncoding.EncodePrompt(tokenizer, p)));

        bool isGreedy = temperature <= 0f;
        lock (rng)
        {
            if (useKvCache)
                session.ResetCache();

            float[]? logits = useKvCache ? session.ForwardPrefill(ids.ToArray()) : null;
            int position = ids.Count;

            for (int t = 0; t < maxNewTokens; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache && logits is not null)
                    next = TokenSampling.Sample(logits.AsSpan(), config.VocabSize, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                else
                {
                    var full = session.Forward(ids.ToArray());
                    int vocab = config.VocabSize;
                    int offset = (ids.Count - 1) * vocab;
                    var span = offset >= 0 && full.Length >= offset + vocab
                        ? full.AsSpan(offset, vocab)
                        : full.AsSpan(0, Math.Min(vocab, full.Length));
                    next = TokenSampling.Sample(span, vocab, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                }

                if (next == config.EosTokenId)
                    yield break;

                ids.Add(next);
                yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode([next]));

                if (useKvCache)
                    logits = session.ForwardCached(next, position++);
            }
        }
    }

    string Generate(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true);
        var ids = new List<int>(tokenizeCache.Encode(prompt, p => SmolLMChatEncoding.EncodePrompt(tokenizer, p)));

        bool isGreedy = temperature <= 0f;
        var sb = new StringBuilder();
        int generated = 0;

        lock (rng)
        {
            if (useKvCache)
                session.ResetCache();

            float[]? logits = useKvCache ? session.ForwardPrefill(ids.ToArray()) : null;
            int position = ids.Count;

            while (generated < maxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache && logits is not null)
                    next = TokenSampling.Sample(logits.AsSpan(), config.VocabSize, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                else
                {
                    var full = session.Forward(ids.ToArray());
                    int vocab = config.VocabSize;
                    int offset = (ids.Count - 1) * vocab;
                    var span = offset >= 0 && full.Length >= offset + vocab
                        ? full.AsSpan(offset, vocab)
                        : full.AsSpan(0, Math.Min(vocab, full.Length));
                    next = TokenSampling.Sample(span, vocab, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                }

                if (next == config.EosTokenId)
                    break;

                ids.Add(next);
                sb.Append(tokenizer.Decode([next]));
                generated++;

                if (useKvCache)
                    logits = session.ForwardCached(next, position++);
            }
        }
        return sb.ToString();
    }

    public void Dispose() => session.Dispose();
}
