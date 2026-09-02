using Microsoft.Extensions.AI;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NivaraChat.SmolLM;

/// <summary>
/// Renders an <c>Microsoft.Extensions.AI</c> conversation into the Hermes ChatML format used by
/// SmolLM instruct models: <c>&lt;|im_start|&gt;role\n...&lt;|im_end|&gt;</c> turns with an
/// optional trailing <c>&lt;|im_start|&gt;assistant\n</c> generation prompt. Plain-chat in
/// Stage A; Stage B adds the Hermes tool-calling surface: a <c>&lt;tools&gt;{json}&lt;/tools&gt;</c>
/// system prompt derived from <c>ChatOptions.Tools</c>, <c>&lt;tool_response&gt;</c> blocks for
/// <see cref="FunctionResultContent"/>, and parsing of generated <c>&lt;tool_call&gt;</c> blocks.
/// </summary>
internal static class SmollmChatTemplate
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";

    const string ToolCallTag = "<tool_call>";
    const string ToolCallCloseTag = "</tool_call>";
    const string ToolResponseTag = "<tool_response>";

    const string ToolSystemPrompt =
        "You are an expert in composing functions. You are given a question and a set of "
        + "possible functions. Based on the question, you will need to make one or more "
        + "function/tool calls to achieve the purpose. If none of the functions can be used, "
        + "point it out and refuse to answer. If the given question lacks the parameters "
        + "required by the function, also point it out.\n\n"
        + "You have access to the following tools:\n<tools>\n{json}\n</tools>\n\n"
        + "The output MUST strictly adhere to the following format, and NO other text MUST be "
        + "included. The example format is as follows. Please make sure the parameter type is "
        + "correct. If no function call is needed, please make the tool calls an empty list '[]'.\n"
        + "<tool_call>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\"}}]</tool_call>";

    /// <summary>
    /// Renders the conversation, optionally appending the assistant generation prompt so the model
    /// produces the next assistant turn. When <paramref name="tools"/> is non-empty, a Hermes
    /// tool-calling system prompt (with the serialized tool schemas) is emitted first.
    /// </summary>
    public static string Render(
        IEnumerable<ChatMessage> messages,
        bool addGenerationPrompt = true,
        IList<AITool>? tools = null)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();

        if (tools is { Count: > 0 })
        {
            sb.Append(ImStart).Append("system\n");
            sb.Append(ToolSystemPrompt.Replace("{json}", RenderTools(tools)));
            sb.Append('\n').Append(ImEnd).Append('\n');
        }

        foreach (var message in messages)
        {
            if (message is null)
                continue;

            string role = message.Role == ChatRole.System ? "system"
                : message.Role == ChatRole.User ? "user"
                : "assistant";

            var toolResults = message.Contents?.OfType<FunctionResultContent>().ToList() ?? [];
            var toolCalls = message.Contents?.OfType<FunctionCallContent>().ToList() ?? [];
            string text = message.Text ?? string.Empty;

            if (string.IsNullOrEmpty(text) && toolResults.Count == 0 && toolCalls.Count == 0)
                continue;

            sb.Append(ImStart).Append(role).Append('\n');

            if (text.Length > 0)
                sb.Append(text).Append('\n');

            foreach (var result in toolResults)
            {
                sb.Append(ToolResponseTag).Append(result.Result ?? string.Empty).Append('\n');
            }

            if (toolCalls.Count > 0)
            {
                sb.Append(ToolCallTag);
                sb.Append(RenderToolCalls(toolCalls));
                sb.Append(ToolCallCloseTag).Append('\n');
            }

            sb.Append(ImEnd).Append('\n');
        }

        if (addGenerationPrompt)
            sb.Append(ImStart).Append("assistant\n");

        return sb.ToString();
    }

    /// <summary>
    /// Parses the first <c>&lt;tool_call&gt;...&lt;/tool_call&gt;</c> block in <paramref name="text"/>,
    /// extracting each function name plus its raw JSON arguments. Returns false when no well-formed
    /// tool call is present so callers can fall back to plain text.
    /// </summary>
    public static bool TryParseToolCall(string text, out List<(string name, string argsJson)> calls)
    {
        calls = [];
        if (string.IsNullOrWhiteSpace(text))
            return false;

        int start = text.IndexOf(ToolCallTag, StringComparison.OrdinalIgnoreCase);
        if (start < 0)
            return false;
        int end = text.IndexOf(ToolCallCloseTag, start + ToolCallTag.Length, StringComparison.OrdinalIgnoreCase);
        if (end < 0)
            return false;

        string inner = text[(start + ToolCallTag.Length)..end].Trim();
        if (inner.Length == 0)
            return false;

        try
        {
            using var doc = JsonDocument.Parse(inner);
            JsonElement root = doc.RootElement;
            List<JsonElement> elements = root.ValueKind == JsonValueKind.Array
                ? [.. root.EnumerateArray()]
                : [root];

            foreach (JsonElement element in elements)
            {
                if (element.ValueKind != JsonValueKind.Object)
                    continue;
                if (!element.TryGetProperty("name", out JsonElement nameEl) || nameEl.ValueKind != JsonValueKind.String)
                    continue;

                string argsJson = element.TryGetProperty("arguments", out JsonElement argsEl)
                    ? argsEl.GetRawText()
                    : "{}";
                calls.Add((nameEl.GetString()!, argsJson));
            }
        }
        catch (JsonException)
        {
            calls = [];
            return false;
        }

        return calls.Count > 0;
    }

    /// <summary>Serializes the Hermes tool definitions from the provided tools' <see cref="AIFunction"/>s
    /// (Name / Description / JsonSchema) into a compact JSON array.</summary>
    internal static string RenderTools(IList<AITool> tools)
    {
        var arr = new JsonArray();
        foreach (var tool in tools)
        {
            if (tool is not AIFunction fn)
                continue;

            var parameters = fn.JsonSchema.ValueKind == JsonValueKind.Undefined
                ? new JsonObject()
                : JsonNode.Parse(fn.JsonSchema.GetRawText()) as JsonObject ?? new JsonObject();

            arr.Add(new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = fn.Name,
                    ["description"] = fn.Description,
                    ["parameters"] = parameters,
                },
            });
        }

        // Emit the tool JSON compactly (single line) so the model sees exactly one contiguous
        // block, matching the SmolLM tool-calling training format.
        return JsonSerializer.Serialize(arr);
    }

    /// <summary>Reconstructs a <c>&lt;tool_call&gt;</c> payload from parsed
    /// <see cref="FunctionCallContent"/> items (assistant history round-trip).</summary>
    static string RenderToolCalls(List<FunctionCallContent> toolCalls)
    {
        var arr = new JsonArray();
        foreach (var call in toolCalls)
        {
            var argsNode = call.Arguments is null
                ? JsonSerializer.SerializeToNode(new Dictionary<string, object?>())
                : JsonSerializer.SerializeToNode(call.Arguments);

            arr.Add(new JsonObject
            {
                ["name"] = call.Name,
                ["arguments"] = argsNode,
            });
        }

        return JsonSerializer.Serialize(arr);
    }
}
