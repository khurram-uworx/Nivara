using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using System.Numerics;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NivaraChat.SmolLM;

/// <summary>
/// An <see cref="IChatClient"/> for the <c>archit11/small-function-calling</c> (Biggie-SmoLlm-0.15B)
/// causal LM. Reuses the shared generation core (<see cref="LlamaChatClientBase{T}"/>) and the Hermes
/// rendering (<see cref="SmollmChatTemplate"/>) but supplies a **lenient** tool-call parser, because
/// small 0.15B fine-tunes often emit tool calls as single-quoted / partially-formed "Python-dict"
/// JSON (which <see cref="JsonDocument"/> rejects). The lenient fallback extracts the function name
/// and key–value arguments textually and rebuilds well-formed JSON so the tool loop can proceed.
/// </summary>
/// <typeparam name="T">The model's floating-point element type.</typeparam>
internal sealed class BiggieChatClient<T> : LlamaChatClientBase<T>
    where T : struct, IFloatingPointIeee754<T>
{
    const string Hermes1ToolSystemPrompt =
        "You are a function calling AI model. You are provided with function signatures within "
        + "<tools></tools> XML tags. You may call one or more functions to assist with the user "
        + "query. Don't make assumptions about what values to plug into functions. Here are the "
        + "available tools:\n<tools>\n{json}\n</tools>\n"
        + "For each function call return a json object with function name and arguments within "
        + "<tool_call></tool_call> XML tags as follows:\n"
        + "<tool_call>{\"arguments\": <args-dict>, \"name\": \"<function-name>\"}</tool_call>";

    static readonly Regex NamePattern = new(
        @"['""]?name['""]?\s*[:=]\s*['""](?<name>[^'""]+)['""]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex ArgPattern = new(
        @"['""](?<key>[A-Za-z_][A-Za-z0-9_]*)['""]\s*[:=]\s*(?<value>['""]?[^,'""\]\}]+)",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public BiggieChatClient(
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

    protected override string ModelLabel => "Biggie-SmoLlm-0.15B (function-calling)";

    /// <summary>
    /// Renders the conversation. When tools are present, prepends the Hermes-1 function-calling system
    /// prompt (the format this fine-tune was trained on — <c>NousResearch/hermes-function-calling-v1</c>)
    /// before the standard ChatML turns. Falls back to plain Hermes otherwise.
    /// </summary>
    protected override string RenderPrompt(IEnumerable<ChatMessage> messages, IList<AITool>? tools)
    {
        if (tools is not { Count: > 0 })
            return SmollmChatTemplate.Render(messages, addGenerationPrompt: true, tools: null);

        string system = SmollmChatTemplate.ImStart + "system\n"
            + Hermes1ToolSystemPrompt.Replace("{json}", SmollmChatTemplate.RenderTools(tools)) + "\n"
            + SmollmChatTemplate.ImEnd + "\n";
        return system + SmollmChatTemplate.Render(messages, addGenerationPrompt: true, tools: null);
    }

    /// <summary>
    /// Parses the first <c>&lt;tool_call&gt;...&lt;/tool_call&gt;</c> block. Tries the shared Hermes
    /// strict-JSON parser first (for well-behaved output); falls back to a lenient textual extraction
    /// (function name plus key–value arguments) that tolerates single quotes, missing braces, and
    /// stray annotations that small fine-tunes emit.
    /// </summary>
    protected override bool TryParseToolCall(string text, out List<(string name, string argsJson)> calls)
    {
        calls = [];
        if (string.IsNullOrWhiteSpace(text))
            return false;

        int start = text.IndexOf("<tool_call>", StringComparison.OrdinalIgnoreCase);
        if (start < 0)
            return false;
        int end = text.IndexOf("</tool_call>", start + "<tool_call>".Length, StringComparison.OrdinalIgnoreCase);
        if (end < 0)
            end = text.Length;

        string inner = text[(start + "<tool_call>".Length)..end].Trim();
        if (inner.Length == 0)
            return false;

        // Preferred: well-formed Hermes JSON (e.g. {"name": ..., "arguments": {...}} or an array of them).
        if (TryParseHermesToolCall(text, out calls))
            return true;

        // Lenient fallback for small-model output: extract name + key/value pairs and rebuild JSON.
        // Small fine-tunes hallucinate extra numeric/schema keys (temperature, humidity, ...) and leak
        // the schema's "type"/"description" annotations; keep only the single most plausible argument
        // (preferring "city" for the weather tool) so the tool isn't invoked with bogus parameters.
        var nameMatch = NamePattern.Match(inner);
        if (!nameMatch.Success)
            return false;

        string name = nameMatch.Groups["name"].Value.Trim();

        string? city = null;
        KeyValuePair<string, string>? firstClean = null;
        foreach (Match match in ArgPattern.Matches(inner))
        {
            string key = match.Groups["key"].Value;
            if (key.Equals("name", StringComparison.OrdinalIgnoreCase)
                || key.Equals("type", StringComparison.OrdinalIgnoreCase)
                || key.Equals("description", StringComparison.OrdinalIgnoreCase))
                continue;
            string value = match.Groups["value"].Value.Trim().Trim('"', '\'');
            if (key.Equals("city", StringComparison.OrdinalIgnoreCase))
                city ??= value;
            else if (firstClean is null)
                firstClean = new KeyValuePair<string, string>(key, value);
        }

        var kept = city is not null
            ? new List<KeyValuePair<string, string>> { new("city", city) }
            : firstClean is not null ? new List<KeyValuePair<string, string>> { firstClean.Value } : [];

        if (kept.Count == 0)
            return false;

        using var doc = JsonDocument.Parse(RebuildJsonObject(kept));
        calls.Add((name, doc.RootElement.GetRawText()));
        return true;
    }

    static string RebuildJsonObject(List<KeyValuePair<string, string>> args)
    {
        var sb = new System.Text.StringBuilder("{");
        for (int i = 0; i < args.Count; i++)
        {
            if (i > 0)
                sb.Append(',');
            sb.Append('"').Append(args[i].Key).Append("\":\"").Append(args[i].Value.Replace("\"", "\\\"")).Append('"');
        }
        sb.Append('}');
        return sb.ToString();
    }
}
