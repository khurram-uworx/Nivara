using Microsoft.Extensions.AI;
using Nivara;
using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Shared;
using System.Text;

namespace NivaraChat.Qwen;

internal sealed class SilkQwenChatClient : IChatClient, IDisposable
{
    readonly SilkQwenSession session;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly int maxNewTokens;
    readonly float temperature;
    readonly float topP;
    readonly bool useKvCache;
    readonly Random rng;
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
        this.knownToolNames = knownToolNames;
        this.turnCallback = turnCallback;
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "Qwen2.5-0.5B-Instruct (Silk/OpenCL causal LM + GPT-2 BPE)");

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var ids = Generate(messages, cancellationToken);
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

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var ids = Generate(messages, cancellationToken);
        foreach (var id in ids)
            yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode([id]));
    }

    public void Dispose() => session.Dispose();

    IReadOnlyList<int> Generate(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = QwenChatTemplate.Render(messages, addGenerationPrompt: true);
        var ids = new List<int>(tokenizeCache.Encode(prompt, p => tokenizer.Encode(p)));

        bool greedy = temperature <= 0f;
        var generated = new List<int>();

        lock (rng)
        {
            session.ResetCache();

            float[]? logits = useKvCache ? session.ForwardPrefill(ids.ToArray()) : null;
            int position = ids.Count;

            for (int t = 0; t < maxNewTokens; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache && logits is not null)
                {
                    int vocab = config.VocabSize;
                    int start = logits.Length - vocab;
                    var row = start >= 0 && logits.Length >= start + vocab
                        ? logits.AsSpan(start, vocab)
                        : logits.AsSpan(0, Math.Min(vocab, logits.Length));
                    next = TokenSampling.Sample(row, vocab, temperature, topP, rng);
                }
                else
                {
                    var full = session.Forward(ids.ToArray());
                    int vocab = config.VocabSize;
                    int start = full.Length - vocab;
                    var row = start >= 0 && full.Length >= start + vocab
                        ? full.AsSpan(start, vocab)
                        : full.AsSpan(0, Math.Min(vocab, full.Length));
                    next = TokenSampling.Sample(row, vocab, temperature, topP, rng);
                }

                if (QwenIds.StopIds.Contains(next))
                    break;

                generated.Add(next);

                if (useKvCache)
                    logits = session.ForwardCached(next, position++);
                else
                    ids.Add(next);

                if (generated.Count >= config.MaxPositionEmbeddings - position)
                    break;
            }
        }

        return generated;
    }
}
