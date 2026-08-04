using Microsoft.Agents.AI.Workflows;

namespace NivaraChat;

internal sealed class EscalationExecutor : Executor<string, string>
{
    public EscalationExecutor()
        : base("Escalation")
    {
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC");
        var result = $"[ESCALATION] Complaint received at {timestamp}: {input}. A human agent will follow up.";
        return ValueTask.FromResult(result);
    }
}