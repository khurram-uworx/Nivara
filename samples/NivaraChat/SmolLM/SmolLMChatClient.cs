using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NivaraChat.SmolLM;

/// <summary>
/// An <see cref="IChatClient"/> backed by a pretrained SmolLM/Llama causal LM
/// (<see cref="LlamaForCausalLM{T}"/>). Renders the conversation with
/// <see cref="SmollmChatTemplate"/>, greedy-generates autoregressively, and streams the
/// plain-text reply token-by-token. The model runs in eval mode; each call builds its own
/// tensors, so concurrent requests are safe (re-entrant). Plain-chat only in Stage A.
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

    public SmolLMChatClient(
        LlamaForCausalLM<T> model,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        int maxNewTokens = 64)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.config = config;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "SmolLM-135M-Instruct (LlamaForCausalLM + GPT-2 BPE)");

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
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true);
        var sequence = new List<int>(EncodePrompt(prompt));

        for (int t = 0; t < maxNewTokens; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var logits = model.Forward(sequence.ToArray()); // [L, vocab]
            int next = ArgMaxLastRow(logits, logits.Shape[0], config.VocabSize);
            if (next == config.EosTokenId)
                yield break;

            sequence.Add(next);
            yield return new ChatResponseUpdate(ChatRole.Assistant, tokenizer.Decode([next]));
        }
    }

    public void Dispose() { }

    string Generate(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = SmollmChatTemplate.Render(messages, addGenerationPrompt: true);
        var sequence = new List<int>(EncodePrompt(prompt));

        var sb = new System.Text.StringBuilder();
        int generated = 0;
        while (generated < maxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var logits = model.Forward(sequence.ToArray()); // [L, vocab]
            int next = ArgMaxLastRow(logits, logits.Shape[0], config.VocabSize);
            if (next == config.EosTokenId)
                break;

            sequence.Add(next);
            sb.Append(tokenizer.Decode([next]));
            generated++;
        }
        return sb.ToString();
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

    /// <summary>Returns the argmax vocabulary index of the model's final-position logits.</summary>
    static int ArgMaxLastRow(ReverseGradTensor<T> logits, int rows, int vocab)
    {
        logits.Data.TryGetSpan(out var span);
        int offset = (rows - 1) * vocab;
        int best = 0;
        double bestValue = double.CreateChecked(span[offset]);
        for (int i = 1; i < vocab; i++)
        {
            double v = double.CreateChecked(span[offset + i]);
            if (v > bestValue)
            {
                bestValue = v;
                best = i;
            }
        }
        return best;
    }
}
