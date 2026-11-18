using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.Silk;
using NivaraChat.Shared;
using System.Text;

namespace NivaraChat.Transformer;

/// <summary>
/// An <see cref="IChatClient"/> backed by a <see cref="SilkTinyShakespeareSession"/> (Silk/OpenCL
/// GPU path for the batched TinyShakespeare transformer). Mirrors
/// <c>SilkSmolLMChatClient</c>: prefill + cached decode + overflow + top-p sampling, with a
/// non-cached fallback that re-projects the full window.
/// </summary>
internal sealed class SilkBatchedChatClient : IChatClient
{
    readonly SilkTinyShakespeareSession session;
    readonly TextTokenizer tokenizer;
    readonly int maxSeqLen;
    readonly int maxNewTokens;
    readonly float temperature;
    readonly float topP;
    readonly bool useKvCache;
    readonly Random rng = new(0);

    public SilkBatchedChatClient(
        SilkTinyShakespeareSession session,
        TextTokenizer tokenizer,
        int maxSeqLen,
        int maxNewTokens = 96,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = true)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);

        this.session = session;
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen > 0 ? maxSeqLen : 1;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
        this.temperature = temperature ?? 0f;
        this.topP = Math.Clamp(topP ?? 1f, 0f, 1f);
        this.useKvCache = useKvCache;
        this.rng = new Random(seed ?? 0);
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "TinyShakespeare (Silk/OpenCL batched transformer)");

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();
        foreach (var chunk in Generate(messages, cancellationToken))
            sb.Append(chunk);
        return Task.FromResult(new ChatResponse([new ChatMessage(ChatRole.Assistant, sb.ToString())]));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var chunk in Generate(messages, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return new ChatResponseUpdate(ChatRole.Assistant, chunk);
            await Task.CompletedTask;
        }
    }

    public void Dispose() => session.Dispose();

    IEnumerable<string> Generate(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = FormatConversation(messages);
        var ids = new List<int>(tokenizer.Encode(prompt, addBosEos: true));

        bool isGreedy = temperature <= 0f;
        int generated = 0;

        lock (rng)
        {
            if (useKvCache)
                session.ResetCache();

            float[]? logits = null;
            int position = ids.Count;

            if (useKvCache)
            {
                int seqLen = Math.Min(ids.Count, maxSeqLen);
                var window = ids.GetRange(ids.Count - seqLen, seqLen);
                logits = session.ForwardPrefill(window.ToArray());
                ids = new List<int>(window);
                position = window.Count;

                if (ids.Count > maxSeqLen)
                {
                    int keep = maxSeqLen;
                    var w = ids.GetRange(ids.Count - keep, keep);
                    session.ResetCache();
                    logits = session.ForwardPrefill(w.ToArray());
                    ids = new List<int>(w);
                    position = w.Count;
                }
            }

            while (generated < maxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache && logits is not null)
                {
                    int vocab = tokenizer.VocabSize;
                    next = TokenSampling.Sample(logits.AsSpan(), vocab, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                }
                else
                {
                    int seqLen = Math.Min(ids.Count, maxSeqLen);
                    var window = ids.GetRange(ids.Count - seqLen, seqLen);
                    var full = session.ForwardPrefill(window.ToArray());
                    int vocab = tokenizer.VocabSize;
                    next = TokenSampling.Sample(full.AsSpan(), vocab, isGreedy ? 0f : temperature, isGreedy ? 1f : topP, rng);
                }

                if (next == tokenizer.EosToken)
                    yield break;

                ids.Add(next);
                generated++;
                yield return tokenizer.Decode(new[] { next });

                if (useKvCache)
                {
                    if (position >= maxSeqLen)
                    {
                        int keep = maxSeqLen - 1;
                        var w = ids.GetRange(ids.Count - 1 - keep, keep);
                        session.ResetCache();
                        session.ForwardPrefill(w.ToArray());
                        position = keep;
                    }
                    logits = session.ForwardCached(next, position++);
                }
            }
        }
    }

    static string FormatConversation(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            if (string.IsNullOrEmpty(message.Text))
                continue;

            string role = message.Role == ChatRole.User ? "user" : "assistant";
            sb.Append(role).Append(": ").Append(message.Text).Append('\n');
        }
        sb.Append("assistant:");
        return sb.ToString();
    }
}