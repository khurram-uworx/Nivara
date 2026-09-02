using Nivara.AutoDiff;
using Nivara.Samples;
using System.Numerics;

namespace NivaraChat.SmolLM;

/// <summary>
/// An <see cref="IChatClient"/> backed by a pretrained SmolLM/Llama causal LM
/// (<see cref="LlamaForCausalLM{T}"/>). Renders the conversation with the Hermes ChatML template
/// (<see cref="SmollmChatTemplate"/>), generates autoregressively, and streams the plain-text reply
/// token-by-token. Greedy (argmax) by default; when <paramref name="temperature"/> is greater than
/// 0, tokens are drawn from the temperature-scaled softmax with optional top-p (nucleus) filtering
/// via a seeded RNG for reproducible, varied replies. KV-caching is used when
/// <paramref name="useKvCache"/> is set, avoiding full-prefix re-projection per token. The generation
/// machinery and Hermes tool-call surface are inherited from <see cref="LlamaChatClientBase{T}"/>, so
/// this type stays lean and contributes only its model label. (Tool calling is primarily exercised
/// by the dedicated function-calling client, <see cref="BiggieChatClient{T}"/>.)
/// </summary>
/// <typeparam name="T">The model's floating-point element type.</typeparam>
internal sealed class SmolLMChatClient<T> : LlamaChatClientBase<T>
    where T : struct, IFloatingPointIeee754<T>
{
    public SmolLMChatClient(
        LlamaForCausalLM<T> model,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        string? modelName = null,
        int maxNewTokens = 64,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = false)
        : base(model, tokenizer, config, modelName, maxNewTokens, temperature, topP, seed, useKvCache)
    {
    }

    protected override string ModelLabel => "SmolLM/Llama";
}
