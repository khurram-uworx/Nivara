using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace NivaraChat.SmolLM;

/// <summary>
/// An <see cref="IChatClient"/> backed by a pretrained SmolLM/Llama causal LM
/// (<see cref="LlamaForCausalLM{T}"/>). Renders the conversation with
/// <see cref="SmollmChatTemplate"/>, generates autoregressively, and streams the plain-text
/// reply token-by-token. Greedy (argmax) by default; when <paramref name="temperature"/> is
/// greater than 0, tokens are drawn from the temperature-scaled softmax with optional top-p
/// (nucleus) filtering via a seeded RNG for reproducible, varied replies. KV-caching is used
/// when <paramref name="useKvCache"/> is set, avoiding full-prefix re-projection per token.
/// The model runs in eval mode; each call builds its own tensors, so concurrent requests are
/// safe (re-entrant). Plain-chat only in Stage A.
/// </summary>
internal sealed class SmolLMChatClient<T> : IChatClient
    where T : struct, IFloatingPointIeee754<T>
{
    static readonly string[] SpecialTokens =
    [
        SmollmChatTemplate.ImStart,
        SmollmChatTemplate.ImEnd,
        "<|endoftext|>",
    ];

    readonly LlamaForCausalLM<T> model;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly int maxNewTokens;
    readonly float temperature;
    readonly float topP;
    readonly bool useKvCache;
    readonly Random rng;
    readonly int numLayers;
    readonly int kvWidth;

    public SmolLMChatClient(
        LlamaForCausalLM<T> model,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        int maxNewTokens = 64,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.config = config;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
        this.temperature = temperature ?? 0f;
        this.topP = Math.Clamp(topP ?? 1f, 0f, 1f);
        this.useKvCache = useKvCache;
        this.rng = new Random(seed ?? 0);
        this.numLayers = config.NumHiddenLayers;
        this.kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "SmolLM-135M-Instruct (LlamaForCausalLM + GPT-2 BPE)");

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var text = Generate(messages, options, cancellationToken);
        if (SmollmChatTemplate.TryParseToolCall(text, out var calls))
            return Task.FromResult(new ChatResponse(
                [new ChatMessage(ChatRole.Assistant, BuildToolCallContents(calls))]));
        return Task.FromResult(new ChatResponse([new ChatMessage(ChatRole.Assistant, text)]));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        bool hasTools = options?.Tools is { Count: > 0 };

        if (!hasTools)
        {
            await foreach (var update in StreamPlain(messages, cancellationToken))
                yield return update;
            yield break;
        }

        // Tool-calling: buffer the full generated text, then emit the outcome atomically so tool
        // calls never arrive as a stream of partial tokens.
        var text = Generate(messages, options, cancellationToken);
        if (SmollmChatTemplate.TryParseToolCall(text, out var calls))
            yield return new ChatResponseUpdate(ChatRole.Assistant, BuildToolCallContents(calls));
        else
            yield return new ChatResponseUpdate(ChatRole.Assistant, text);
    }

    /// <summary>Builds <see cref="FunctionCallContent"/> items from parsed tool calls, deserializing
    /// each call's JSON arguments into a dictionary for <see cref="FunctionInvokingChatClient"/>.</summary>
    static List<AIContent> BuildToolCallContents(List<(string name, string argsJson)> calls)
    {
        var contents = new List<AIContent>();
        foreach (var (name, argsJson) in calls)
        {
            var arguments = JsonSerializer.Deserialize<Dictionary<string, object?>>(argsJson);
            contents.Add(new FunctionCallContent(
                callId: Guid.NewGuid().ToString("N"),
                name: name,
                arguments: arguments ?? []));
        }
        return contents;
    }

    /// <summary>Streams a plain-chat reply token-by-token (no tools), preserving Stage A behavior.</summary>
    async IAsyncEnumerable<ChatResponseUpdate> StreamPlain(
        IEnumerable<ChatMessage> messages,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true);
        var ids = new List<int>(EncodePrompt(prompt));

        using var cache = useKvCache
            ? new LlamaKVCache<T>(numLayers, kvWidth)
            : null;

        bool isGreedy = temperature <= 0f;
        lock (rng)
        {
            // Seed the cache (or, without KV cache, run a full forward per token below).
            var logits = useKvCache ? SeedCache(ids, cache!) : null;
            int position = ids.Count;

            for (int t = 0; t < maxNewTokens; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache)
                {
                    next = Select(LastRow(logits!), config.VocabSize, isGreedy);
                }
                else
                {
                    var full = model.Forward(ids.ToArray()); // [L, vocab]
                    next = Select(LastRow(full), config.VocabSize, isGreedy);
                }

                if (next == config.EosTokenId)
                    yield break;

                ids.Add(next);
                yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode([next]));

                if (useKvCache)
                    logits = model.ForwardCached(next, position++, cache!);
            }
        }
    }

    public void Dispose() { }

    string Generate(IEnumerable<ChatMessage> messages, ChatOptions? options, CancellationToken cancellationToken)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true, tools: options?.Tools);
        var ids = new List<int>(EncodePrompt(prompt));

        using var cache = useKvCache
            ? new LlamaKVCache<T>(numLayers, kvWidth)
            : null;

        bool isGreedy = temperature <= 0f;
        var sb = new System.Text.StringBuilder();
        int generated = 0;

        lock (rng)
        {
            var logits = useKvCache ? SeedCache(ids, cache!) : null;
            int position = ids.Count;

            while (generated < maxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache)
                {
                    next = Select(LastRow(logits!), config.VocabSize, isGreedy);
                }
                else
                {
                    var full = model.Forward(ids.ToArray());
                    next = Select(LastRow(full), config.VocabSize, isGreedy);
                }

                if (next == config.EosTokenId)
                    break;

                ids.Add(next);
                sb.Append(tokenizer.Decode([next]));
                generated++;

                if (useKvCache)
                    logits = model.ForwardCached(next, position++, cache!);
            }
        }
        return sb.ToString();
    }

    /// <summary>Runs the prompt tokens through the cached forward to seed the per-layer cache
    /// and produce the logits predicting the first generated token.</summary>
    ReverseGradTensor<T> SeedCache(List<int> ids, LlamaKVCache<T> cache)
    {
        ReverseGradTensor<T> logits = null!;
        for (int p = 0; p < ids.Count; p++)
            logits = model.ForwardCached(ids[p], p, cache);
        return logits;
    }

    /// <summary>Selects the next token: argmax when greedy, otherwise temperature softmax with
    /// optional top-p filtering, drawn from the shared seeded RNG.</summary>
    int Select(ReadOnlySpan<T> lastRow, int vocab, bool isGreedy)
    {
        if (lastRow.Length < vocab)
            lastRow = lastRow[..vocab];

        if (isGreedy)
            return ArgMax(lastRow, vocab);

        // Temperature-scaled softmax with numeric stability (subtract max).
        var exp = new double[vocab];
        double max = double.CreateChecked(lastRow[0]);
        for (int i = 1; i < vocab; i++)
        {
            double v = double.CreateChecked(lastRow[i]);
            if (v > max) max = v;
        }

        double sum = 0;
        for (int i = 0; i < vocab; i++)
        {
            double v = double.CreateChecked(lastRow[i]);
            exp[i] = Math.Exp((v - max) / temperature);
            sum += exp[i];
        }

        if (sum <= 0 || !double.IsFinite(sum))
            return ArgMax(lastRow, vocab);

        // Normalize to probabilities.
        for (int i = 0; i < vocab; i++)
            exp[i] /= sum;

        // Optional top-p (nucleus) filtering: keep only the smallest set of tokens whose
        // cumulative probability exceeds topP; renormalize and sample from those.
        if (topP < 1f)
        {
            var order = new int[vocab];
            for (int i = 0; i < vocab; i++) order[i] = i;
            Array.Sort(order, (a, b) => exp[b].CompareTo(exp[a]));

            double cum = 0;
            int cut = vocab;
            for (int i = 0; i < vocab; i++)
            {
                cum += exp[order[i]];
                if (cum >= topP)
                {
                    cut = i + 1;
                    break;
                }
            }

            double keptSum = 0;
            for (int i = 0; i < cut; i++)
                keptSum += exp[order[i]];

            double r = rng.NextDouble() * keptSum;
            double acc = 0;
            for (int i = 0; i < cut; i++)
            {
                acc += exp[order[i]];
                if (r <= acc)
                    return order[i];
            }
            return order[0];
        }

        double rFull = rng.NextDouble() * sum;
        double accFull = 0;
        for (int i = 0; i < vocab; i++)
        {
            accFull += exp[i];
            if (rFull <= accFull)
                return i;
        }
        return ArgMax(lastRow, vocab);
    }

    /// <summary>Returns the model's final-position logits (<c>[vocab]</c>) as a span.</summary>
    static ReadOnlySpan<T> LastRow(ReverseGradTensor<T> logits)
    {
        logits.Data.TryGetSpan(out var span);
        int vocab = logits.Shape[^1];
        int rows = logits.Shape[0];
        int offset = (rows - 1) * vocab;
        return offset >= 0 && span.Length >= offset + vocab ? span.Slice(offset, vocab) : span;
    }

    /// <summary>Returns the argmax vocabulary index of the final-position logits.</summary>
    static int ArgMax(ReadOnlySpan<T> lastRow, int vocab)
    {
        int best = 0;
        double bestValue = double.CreateChecked(lastRow[0]);
        for (int i = 1; i < vocab; i++)
        {
            double v = double.CreateChecked(lastRow[i]);
            if (v > bestValue)
            {
                bestValue = v;
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Encodes the rendered prompt into token ids, keeping special tokens (<c>&lt;|im_start|&gt;</c>,
    /// <c>&lt;|im_end|&gt;</c>, <c>&lt;|endoftext|&gt;</c>) as single ids instead of splitting them
    /// through byte-level BPE.
    /// </summary>
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
