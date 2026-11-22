using System.Numerics;
using Microsoft.Extensions.AI;
using Nivara.AI;
using Nivara.Samples;

namespace NivaraFineTuning;

public static class WorklogSftGeneration
{
    public static WorklogEvalMetrics Evaluate<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model,
        Gpt2BpeTokenizer tokenizer,
        IReadOnlyList<WorklogSftExample> examples,
        string template,
        int maxNewTokens,
        int[]? stopIds = null)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentException.ThrowIfNullOrWhiteSpace(template);
        if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "maxNewTokens must be > 0.");

        stopIds ??= GetStopTokenIds(tokenizer, template);
        return WorklogSftEvaluator.Evaluate(examples, ex =>
        {
            var promptMessages = ex.Messages
                .TakeWhile(m => !string.Equals(m.Role, "assistant", StringComparison.OrdinalIgnoreCase))
                .ToList();
            var promptText = RenderEvalPrompt(promptMessages, template);
            var promptIds = tokenizer.Encode(promptText).ToArray();
            var genIds = GenerateGreedy(model, promptIds, maxNewTokens, stopIds);
            return tokenizer.Decode(genIds);
        });
    }

    public static int[] GenerateGreedy<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model,
        int[] promptIds,
        int maxNewTokens,
        int[] stopIds)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(stopIds);
        if (maxNewTokens <= 0) return [];

        var stop = new HashSet<int>(stopIds);
        var gen = new List<int>(maxNewTokens);
        var logits = model.Forward(promptIds);
        int vocab = logits.Shape[^1];
        for (int t = 0; t < maxNewTokens; t++)
        {
            int next = ArgMaxLastRow(logits.Data.ToArray(), vocab);
            if (stop.Contains(next)) break;
            gen.Add(next);
            var seq = new int[promptIds.Length + gen.Count];
            Array.Copy(promptIds, seq, promptIds.Length);
            for (int i = 0; i < gen.Count; i++) seq[promptIds.Length + i] = gen[i];
            logits = model.Forward(seq);
        }
        return gen.ToArray();
    }

    public static string RenderEvalPrompt(IReadOnlyList<WorklogMessage> messages, string template)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentException.ThrowIfNullOrWhiteSpace(template);

        var chatMessages = messages.Select(m => new ChatMessage(
            m.Role.ToLowerInvariant() switch
            {
                "system" => ChatRole.System,
                "user" => ChatRole.User,
                "assistant" => ChatRole.Assistant,
                "tool" => ChatRole.Tool,
                _ => ChatRole.User
            },
            m.Content ?? string.Empty)).ToList();

        return template.ToLowerInvariant() switch
        {
            "qwen" => QwenChatTemplate.Render(chatMessages, addGenerationPrompt: true),
            "smollm" => SmollmChatTemplate.Render(chatMessages, addGenerationPrompt: true),
            _ => throw new ArgumentException($"Unknown template '{template}'.", nameof(template))
        };
    }

    public static int[] GetStopTokenIds(Gpt2BpeTokenizer tokenizer, string template)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _ = template; // template reserved for future template-specific stops; current set is shared.

        var stops = new List<int>();
        void Add(string token)
        {
            int id = tokenizer.TokenId(token);
            if (id >= 0 && !stops.Contains(id)) stops.Add(id);
        }
        Add("<|im_end|>");
        Add("<|endoftext|>");
        Add("<|eot_id|>");
        return stops.ToArray();
    }

    static int ArgMaxLastRow<TModel>(TModel[] data, int vocab)
        where TModel : struct, IFloatingPointIeee754<TModel>
    {
        int off = Math.Max(0, data.Length - vocab);
        int best = 0;
        TModel bestVal = data[off];
        for (int i = 1; i < vocab; i++)
        {
            TModel v = data[off + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }
}
