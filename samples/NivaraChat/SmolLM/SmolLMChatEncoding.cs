using Nivara.AI;

namespace NivaraChat.SmolLM;

/// <summary>
/// Shared tokenization helpers for SmolLM chat clients (CPU and Silk).
/// </summary>
public static class SmolLMChatEncoding
{
    public static readonly string[] SpecialTokens =
    [
        SmollmChatTemplate.ImStart,
        SmollmChatTemplate.ImEnd,
        "<​|endoftext|>",
    ];

    public static IReadOnlyList<int> EncodePrompt(Gpt2BpeTokenizer tokenizer, string prompt)
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