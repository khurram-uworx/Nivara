using Microsoft.Agents.AI.Workflows;
using System.Text.Json;

namespace NivaraChat;

internal sealed class NivaraResultFormatter : Executor<string, string>
{
    public NivaraResultFormatter()
        : base("NivaraResult")
    {
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        using var doc = JsonDocument.Parse(input);
        var root = doc.RootElement;

        string sentimentLabel = "unknown";
        float sentimentConfidence = 0f;
        if (root.TryGetProperty("sentiment", out var sentObj) && sentObj.TryGetProperty("label", out var labelProp))
        {
            sentimentLabel = labelProp.GetString() ?? "unknown";
            sentimentConfidence = sentObj.GetProperty("confidence").GetSingle();
        }

        string entitiesRaw = "{}";
        if (root.TryGetProperty("entities", out var entObj) && entObj.TryGetProperty("entities", out var innerEnt))
            entitiesRaw = innerEnt.GetRawText();

        float threshold = root.TryGetProperty("threshold", out var thr) ? thr.GetSingle() : 0.8f;

        var result = $"Sentiment: {sentimentLabel} (confidence: {sentimentConfidence:F2})\n"
                   + $"Entities: {entitiesRaw}\n"
                   + $"(Handled by Nivara — no LLM needed, threshold: {threshold:F2})";
        return ValueTask.FromResult(result);
    }
}
