using Microsoft.Extensions.AI;
using System.ComponentModel;
using System.Reflection;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NivaraChat.Qwen;

/// <summary>
/// Special-token ids for Qwen2.5 (verified against the downloaded checkpoint's
/// <c>tokenizer.json</c> / <c>generation_config.json</c>, issue #382 Phase 2):
/// <c>eos_token_id</c> is the array <c>[151645, 151643]</c> — generation must stop on either.
/// </summary>
internal static class QwenIds
{
    public const int EndOfText = 151643;   // <|endoftext|> (also the bos id)
    public const int ImStart = 151644;     // <|im_start|>
    public const int ImEnd = 151645;       // <|im_end|> (primary eos)
    public const int ToolCall = 151657;    // <tool_call>
    public const int ToolCallEnd = 151658; // </tool_call>

    /// <summary>Generation stop ids from <c>generation_config.json</c> <c>eos_token_id</c>.</summary>
    public static readonly int[] StopIds = [ImEnd, EndOfText];
}

/// <summary>
/// Renders a conversation into the Qwen2.5 ChatML format, byte-identical to HuggingFace's
/// <c>apply_chat_template</c> for this checkpoint. Tool-calling mode renders the tools system
/// turn via <see cref="BuildToolsSystemMessage"/> (passed in as the first message, which the
/// non-tools render path turns into the exact header the Torch reference produced). Tool calls
/// are re-emitted from <see cref="FunctionCallContent"/> and tool results land in a
/// <c>user</c> turn wrapped in <c>&lt;tool_response&gt;</c>, matching the real checkpoint
/// template. The source of truth for the whitespace/JSON layout is the ground-truth fixture
/// <c>qwen_tool_prompt.txt</c> / <c>qwen_tool_final_prompt.txt</c> (pinned byte-for-byte by
/// <c>QwenChatTemplateTests</c>).
/// </summary>
internal static class QwenChatTemplate
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";

    public const string DefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    /// <summary>
    /// Bakes the tool-mode system turn (default system text + the tools instructions + the JSON
    /// schemas), byte-identical to the <c>{%- if tools %}</c> branch of the checkpoint template.
    /// </summary>
    public static string BuildToolsSystemMessage(IReadOnlyList<AIFunction> tools, string? systemContent = null)
    {
        ArgumentNullException.ThrowIfNull(tools);

        var sb = new StringBuilder();
        sb.Append(systemContent ?? DefaultSystem);
        sb.Append(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        foreach (var tool in tools)
        {
            sb.Append('\n');
            sb.Append(ToolJson(tool));
        }
        sb.Append(
            "\n</tools>\n\nFor each function call, return a json object with function name and " +
            "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        return sb.ToString();
    }

    /// <summary>
    /// Renders the conversation. When the first message is a system turn it is used verbatim
    /// (for tool mode pass the <see cref="BuildToolsSystemMessage"/> result as the first
    /// system message); otherwise the Qwen default system turn is emitted, as the checkpoint
    /// template's <c>else</c> branch does. Appends the <c>&lt;|im_start|&gt;assistant\n</c>
    /// generation prompt when <paramref name="addGenerationPrompt"/> is set.
    /// </summary>
    public static string Render(IEnumerable<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var list = messages.Where(m => m is not null).ToArray();
        var sb = new StringBuilder();

        if (list.Length == 0 || list[0].Role != ChatRole.System)
        {
            sb.Append(ImStart).Append("system\n").Append(DefaultSystem);
            sb.Append(ImEnd).Append('\n');
        }

        foreach (var message in list)
        {
            if (message.Role == ChatRole.User)
            {
                sb.Append(ImStart).Append("user\n");
                sb.Append(MessageText(message));
                sb.Append(ImEnd).Append('\n');
            }
            else if (message.Role == ChatRole.System)
            {
                sb.Append(ImStart).Append("system\n");
                sb.Append(MessageText(message));
                sb.Append(ImEnd).Append('\n');
            }
            else if (message.Role == ChatRole.Assistant)
            {
                var calls = message.Contents is null
                    ? []
                    : message.Contents.OfType<FunctionCallContent>().ToArray();

                if (calls.Length == 0)
                {
                    sb.Append(ImStart).Append("assistant\n");
                    sb.Append(MessageText(message));
                    sb.Append(ImEnd).Append('\n');
                    continue;
                }

                sb.Append(ImStart).Append("assistant");
                var preface = MessageText(message);
                if (preface.Length > 0)
                    sb.Append('\n').Append(preface);
                foreach (var call in calls)
                {
                    sb.Append("\n<tool_call>\n{\"name\": \"");
                    sb.Append(call.Name);
                    sb.Append("\", \"arguments\": ");
                    sb.Append(ArgumentsJson(call.Arguments));
                    sb.Append("}\n</tool_call>");
                }
                sb.Append(ImEnd).Append('\n');
            }
            else if (message.Role == ChatRole.Tool)
            {
                // Tool results are rendered as a user turn containing <tool_response>.
                sb.Append(ImStart).Append("user\n<tool_response>\n");
                sb.Append(ToolResultText(message));
                sb.Append("\n</tool_response>");
                sb.Append(ImEnd).Append('\n');
            }
        }

        if (addGenerationPrompt)
            sb.Append(ImStart).Append("assistant\n");

        return sb.ToString();
    }

    static string MessageText(ChatMessage message)
        => message.Text ?? "";

    static string ToolResultText(ChatMessage message)
    {
        var result = message.Contents?.OfType<FunctionResultContent>().LastOrDefault()?.Result;
        return result is null ? (message.Text ?? "") : result.ToString() ?? "";
    }

    /// <summary>
    /// Renders a tool's JSON declaration <c>{"type": "function", "function": {...}}</c> with the
    /// Jinja <c>tojson</c> separator/escaping semantics (spaces after <c>:</c> and <c>,</c>,
    /// non-ASCII and <c>'</c> left literal) so it matches the Torch reference byte-for-byte.
    /// </summary>
    static string ToolJson(AIFunction tool)
    {
        var parameters = new JsonObject { ["type"] = "object" };
        var properties = new JsonObject();
        var required = new JsonArray();

        var method = tool.UnderlyingMethod;
        if (method is not null)
        {
            foreach (var p in method.GetParameters())
            {
                var prop = new JsonObject { ["type"] = JsonTypeName(p.ParameterType) };
                var description = p.GetCustomAttribute<DescriptionAttribute>()?.Description;
                if (!string.IsNullOrEmpty(description))
                    prop["description"] = description;
                properties[p.Name!] = prop;
                if (!p.HasDefaultValue)
                    required.Add(p.Name!);
            }
        }

        parameters["properties"] = properties;
        if (required.Count > 0)
            parameters["required"] = required;

        var function = new JsonObject
        {
            ["name"] = tool.Name,
            ["description"] = tool.Description,
            ["parameters"] = parameters,
        };
        var toolRoot = new JsonObject
        {
            ["type"] = "function",
            ["function"] = function,
        };

        return QwenJson.ToSpaced(toolRoot);
    }

    static string JsonTypeName(Type type) => type == typeof(string) ? "string"
        : type == typeof(bool) ? "boolean"
        : type == typeof(short) || type == typeof(ushort) || type == typeof(int) || type == typeof(uint)
            || type == typeof(long) || type == typeof(ulong) || type == typeof(byte) || type == typeof(sbyte)
            ? "integer"
        : "number";

    /// <summary>Renders the parsed tool-call arguments back to compact-with-spaces JSON.</summary>
    static string ArgumentsJson(IDictionary<string, object?>? arguments)
    {
        if (arguments is null || arguments.Count == 0)
            return "{}";

        var obj = new JsonObject();
        foreach (var (key, value) in arguments)
            obj[key] = ToJsonNode(value);
        return QwenJson.ToSpaced(obj);
    }

    static JsonNode? ToJsonNode(object? value) => value switch
    {
        null => null,
        JsonNode node => node,
        JsonElement element => JsonNode.Parse(element.GetRawText()),
        string text => JsonValue.Create(text),
        bool flag => JsonValue.Create(flag),
        int i => JsonValue.Create(i),
        long l => JsonValue.Create(l),
        float f => JsonValue.Create(f),
        double d => JsonValue.Create(d),
        _ => JsonSerializer.SerializeToNode(value),
    };
}

/// <summary>Serializes <see cref="JsonNode"/> trees with Jinja-<c>tojson</c> conventions:
/// spaces after <c>:</c>/<c>,</c>, and relaxed string escaping (literal <c>'</c>, Unicode, no
/// HTML escaping) — mirroring Python's <c>json.dumps(..., ensure_ascii=False)</c>.</summary>
internal static class QwenJson
{
    static readonly JsonSerializerOptions Relaxed = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static string ToSpaced(JsonNode? node) => node switch
    {
        null => "null",
        JsonObject obj => "{" + string.Join(", ", obj.Select(kv => $"{JsonSerializer.Serialize(kv.Key, Relaxed)}: {ToSpaced(kv.Value)}")) + "}",
        JsonArray arr => "[" + string.Join(", ", arr.Select(ToSpaced)) + "]",
        JsonValue value => Scalar(value),
        _ => node.ToJsonString(Relaxed),
    };

    static string Scalar(JsonValue value)
        => value.TryGetValue<string>(out var s) ? JsonSerializer.Serialize(s, Relaxed) : value.ToJsonString();
}