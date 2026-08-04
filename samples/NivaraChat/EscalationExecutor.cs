using Microsoft.Agents.AI.Workflows;
using System.Text.Json;

namespace NivaraChat;

internal sealed class EscalationExecutor : Executor<string, string>
{
    public EscalationExecutor()
        : base("Escalation")
    {
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var originalText = ExtractInput(input);
        var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC");
        var result = $"[ESCALATION] Complaint received at {timestamp}: {originalText}. A human agent will follow up.";
        return ValueTask.FromResult(result);
    }

    private static string ExtractInput(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("text", out var textProp))
                return textProp.GetString() ?? json;
        }
        catch { }
        return json;
    }
}