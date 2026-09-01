using Microsoft.Extensions.AI;
using System.Text;

namespace NivaraChat.SmolLM;

/// <summary>
/// Renders an <c>Microsoft.Extensions.AI</c> conversation into the Hermes ChatML format used by
/// SmolLM instruct models: <c>&lt;|im_start|&gt;role\n...&lt;|im_end|&gt;</c> turns with an
/// optional trailing <c>&lt;|im_start|&gt;assistant\n</c> generation prompt. Plain-chat only in
/// Stage A (no tool-call/result content is emitted).
/// </summary>
internal static class SmollmChatTemplate
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";

    /// <summary>
    /// Renders the conversation, optionally appending the assistant generation prompt so the model
    /// produces the next assistant turn.
    /// </summary>
    public static string Render(IEnumerable<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            if (message is null || string.IsNullOrEmpty(message.Text))
                continue;

            string role = message.Role == ChatRole.System ? "system"
                : message.Role == ChatRole.User ? "user"
                : "assistant";

            sb.Append(ImStart).Append(role).Append('\n');
            sb.Append(message.Text).Append('\n');
            sb.Append(ImEnd).Append('\n');
        }

        if (addGenerationPrompt)
            sb.Append(ImStart).Append("assistant\n");

        return sb.ToString();
    }
}
