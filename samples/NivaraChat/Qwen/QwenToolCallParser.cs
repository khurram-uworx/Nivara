using Microsoft.Extensions.AI;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NivaraChat.Qwen;

/// <summary>
/// Parses the raw assistant-generated text into <see cref="FunctionCallContent"/> items.
/// Qwen emits structured calls as <c>&lt;tool_call&gt;\n{"name": ..., "arguments": {...}}\n
/// &lt;/tool_call&gt;</c> blocks. The canonical JSON path is a strict <see cref="JsonDocument"/>
/// parse of each block (spacing-agnostic), with a tolerant regex fallback for anything the model
/// wanders from (Phase B lesson: never crash on model output). Argument dictionaries are built
/// from the parsed JSON so <see cref="FunctionCallContent.Arguments"/> is a correct dict the
/// function binder can serialize back.
/// </summary>
internal static class QwenToolCallParser
{
    static readonly Regex ToolCallBlock = new(
        "<tool_call>(.*?)</tool_call>",
        RegexOptions.Compiled | RegexOptions.Singleline | RegexOptions.CultureInvariant);

    static readonly Regex TolerantName = new(
        "\"?name\"?\\s*:\\s*\"([^\"]+)\"",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Extracts tool calls from generated text. <paramref name="knownToolNames"/> (when supplied)
    /// canonicalizes the emitted name against the registered tools so a model that emits
    /// <c>GetWeather</c> still resolves to the <c>getWeather</c> AIFunction.
    /// </summary>
    public static IReadOnlyList<FunctionCallContent> Parse(
        string text,
        IReadOnlyList<string>? knownToolNames = null)
    {
        ArgumentNullException.ThrowIfNull(text);

        var calls = new List<FunctionCallContent>();
        foreach (Match match in ToolCallBlock.Matches(text))
        {
            var inner = match.Groups[1].Value.Trim();
            var parsed = TryParseJson(inner) ?? TryParseTolerant(inner);
            if (parsed is null || parsed.Value.Name is null)
                continue;

            var canonical = CanonicalName(parsed.Value.Name, knownToolNames);
            calls.Add(new FunctionCallContent($"call_{Guid.NewGuid():N}", canonical, parsed.Value.Arguments));
        }
        return calls;
    }

    /// <summary>Strict parse of the JSON between the tool tags.</summary>
    static (string? Name, IDictionary<string, object?>? Arguments)? TryParseJson(string inner)
    {
        try
        {
            using var doc = JsonDocument.Parse(inner);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return null;

            string? name = root.TryGetProperty("name", out var nameEl) && nameEl.ValueKind == JsonValueKind.String
                ? nameEl.GetString()
                : null;
            if (name is null || !root.TryGetProperty("arguments", out var argsEl) || argsEl.ValueKind != JsonValueKind.Object)
                return null;

            var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var prop in argsEl.EnumerateObject())
                arguments[prop.Name] = prop.Value.Clone();
            return (name, arguments);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Last-resort extraction when the strict parse failed (malformed/unexpected text).</summary>
    static (string? Name, IDictionary<string, object?>? Arguments)? TryParseTolerant(string inner)
    {
        var nameMatch = TolerantName.Match(inner);
        if (!nameMatch.Success)
            return null;

        var name = nameMatch.Groups[1].Value;
        // Keep the (unparseable) remainder so the failure is observable, not silently dropped.
        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["__raw"] = inner,
        };
        return (name, arguments);
    }

    static string CanonicalName(string name, IReadOnlyList<string>? knownToolNames)
    {
        if (knownToolNames is null)
            return name;

        foreach (var known in knownToolNames)
        {
            if (string.Equals(known, name, StringComparison.OrdinalIgnoreCase))
                return known;
        }
        return name;
    }
}