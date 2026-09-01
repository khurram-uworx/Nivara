using Microsoft.Agents.AI.Workflows;
using System.Text.Json;

namespace NivaraChat.Executors;

internal sealed class ValidatorExecutor : Executor<string, string>
{
    private readonly List<string> _pending = [];
    private const int ExpectedCount = 2;

    public ValidatorExecutor()
        : base("Validator")
    {
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrEmpty(input))
            _pending.Add(input);

        if (_pending.Count < ExpectedCount)
            return ValueTask.FromResult("");

        string? sentimentJson = null;
        string? entitiesJson = null;
        foreach (var msg in _pending)
        {
            using var doc = JsonDocument.Parse(msg);
            if (doc.RootElement.TryGetProperty("entities", out _))
                entitiesJson = msg;
            else if (doc.RootElement.TryGetProperty("label", out _))
                sentimentJson = msg;
        }
        _pending.Clear();

        sentimentJson ??= "{\"label\":\"unknown\",\"confidence\":0}";
        entitiesJson ??= "{\"entities\":{},\"confidence\":0}";

        using var sentDoc = JsonDocument.Parse(sentimentJson);
        string sentimentLabel = sentDoc.RootElement.GetProperty("label").GetString() ?? "unknown";
        float sentimentConfidence = sentDoc.RootElement.GetProperty("confidence").GetSingle();

        using var entDoc = JsonDocument.Parse(entitiesJson);
        float entityConfidence = entDoc.RootElement.GetProperty("confidence").GetSingle();
        var entitiesRaw = entDoc.RootElement.GetProperty("entities").GetRawText();

        bool hasEntities = entitiesRaw.Contains("\"person\"") || entitiesRaw.Contains("\"org\"")
            || entitiesRaw.Contains("\"date\"") || entitiesRaw.Contains("\"location\"");
        bool hasMeaningfulSentiment = sentimentLabel != "neutral" && sentimentLabel != "unknown";

        float confidence = (hasEntities || hasMeaningfulSentiment) ? 0.9f : 0.3f;
        string status = (hasEntities || hasMeaningfulSentiment) ? "CONSISTENT" : "INCONSISTENT";
        var result = $"{{\"status\":\"{status}\",\"confidence\":{confidence:F1},\"sentiment\":\"{sentimentLabel}\",\"sentimentConfidence\":{sentimentConfidence:F2},\"entities\":{entitiesRaw},\"entityConfidence\":{entityConfidence:F2}}}";
        return ValueTask.FromResult(result);
    }
}
