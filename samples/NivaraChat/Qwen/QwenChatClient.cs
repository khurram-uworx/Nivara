using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NivaraChat.Qwen;

/// <summary>
/// An <see cref="IChatClient"/> backed by Qwen2.5-0.5B-Instruct running as a
/// <see cref="LlamaForCausalLM{T}"/> (BF16-on-disk weights upcast to F32), served over the
/// Microsoft.Extensions.AI content model. Inference is the default path: the model is run
/// outside a <c>GradientUtils.Grad()</c> scope, so no graph nodes are built (per ADR-001 /
/// ADR-002 inference-default direction).
///
/// Conversations are rendered byte-identically to HuggingFace's <c>apply_chat_template</c>
/// (<see cref="QwenChatTemplate"/>), encoded by the byte-level BPE tokenizer, and decoded
/// autoregressively with a KV cache, stopping on either <c>&lt;|im_end|&gt;</c> (151645) or
/// <c>&lt;|endoftext|&gt;</c> (151643). Generated assistant turns containing
/// <c>&lt;tool_call&gt;</c> blocks are parsed into <see cref="FunctionCallContent"/> so a
/// <see cref="FunctionInvokingChatClient"/> (or the tool loop in <c>QwenMode</c>) can execute
/// them; plain turns become <see cref="TextContent"/>. <paramref name="turnCallback"/> observes
/// every generated assistant turn (used by the demo to print live progress).
/// </summary>
internal sealed class QwenChatClient<T> : IChatClient
    where T : struct, IFloatingPointIeee754<T>
{
    readonly LlamaForCausalLM<T> model;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LlamaConfig config;
    readonly int maxNewTokens;
    readonly float temperature;
    readonly float topP;
    readonly bool useKvCache;
    readonly Random rng;
    readonly IReadOnlyList<string>? knownToolNames;
    readonly Action<string>? turnCallback;
    readonly int numLayers;
    readonly int kvWidth;

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "Qwen2.5-0.5B-Instruct (LlamaForCausalLM + GPT-2 BPE)");

    public QwenChatClient(
        LlamaForCausalLM<T> model,
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
        this.knownToolNames = knownToolNames;
        this.turnCallback = turnCallback;
        this.numLayers = config.NumHiddenLayers;
        this.kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
    }

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var text = Decode(Generate(messages, cancellationToken));
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
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var generated = Generate(messages, cancellationToken);
        foreach (var id in generated)
            yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode([id]));
    }

    public void Dispose() { }

    /// <summary>Renders, encodes, and autoregressively decodes one assistant turn (ids only).</summary>
    IReadOnlyList<int> Generate(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = QwenChatTemplate.Render(messages, addGenerationPrompt: true);
        var ids = new List<int>(tokenizer.Encode(prompt));

        using var cache = useKvCache ? new LlamaKVCache<T>(numLayers, kvWidth) : null;

        bool greedy = temperature <= 0f;
        var generated = new List<int>();

        lock (rng)
        {
            var logits = useKvCache ? SeedCache(ids, cache!) : null;
            int position = ids.Count;

            for (int t = 0; t < maxNewTokens; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next;
                if (useKvCache)
                {
                    next = Select(LastRow(logits!));
                }
                else
                {
                    var full = model.Forward(ids.ToArray()); // [L, vocab]
                    next = Select(LastRow(full));
                }

                if (QwenIds.StopIds.Contains(next))
                    break;

                generated.Add(next);

                if (useKvCache)
                    logits = model.ForwardCached(next, position++, cache!);
                else
                    ids.Add(next);

                if (generated.Count >= config.MaxPositionEmbeddings - position)
                    break;
            }
        }

        return generated;
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

    /// <summary>Returns the model's final-position logits (<c>[vocab]</c>) as a span.</summary>
    static ReadOnlySpan<T> LastRow(ReverseGradTensor<T> logits)
    {
        logits.Data.TryGetSpan(out var span);
        int vocab = logits.Shape[^1];
        int rows = logits.Shape[0];
        int offset = (rows - 1) * vocab;
        return offset >= 0 && span.Length >= offset + vocab ? span.Slice(offset, vocab) : span;
    }

    /// <summary>Selects the next token: argmax when greedy, otherwise temperature softmax with
    /// optional top-p filtering, drawn from the shared seeded RNG.</summary>
    int Select(ReadOnlySpan<T> lastRow)
    {
        int vocab = config.VocabSize;
        if (lastRow.Length < vocab)
            lastRow = lastRow[..vocab];

        if (temperature <= 0f)
            return ArgMax(lastRow, vocab);

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

        for (int i = 0; i < vocab; i++)
            exp[i] /= sum;

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
                if (cum >= topP) { cut = i + 1; break; }
            }

            double kept = 0;
            for (int i = 0; i < cut; i++) kept += exp[order[i]];

            double r = rng.NextDouble() * kept;
            double acc = 0;
            for (int i = 0; i < cut; i++)
            {
                acc += exp[order[i]];
                if (r <= acc) return order[i];
            }
            return order[0];
        }

        double full = rng.NextDouble() * sum;
        double total = 0;
        for (int i = 0; i < vocab; i++)
        {
            total += exp[i];
            if (full <= total) return i;
        }
        return ArgMax(lastRow, vocab);
    }

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

    string Decode(IReadOnlyList<int> ids) => tokenizer.Decode(ids);
}