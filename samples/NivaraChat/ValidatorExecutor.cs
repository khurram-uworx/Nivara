using Microsoft.Agents.AI.Workflows;

namespace NivaraChat;

internal sealed class ValidatorExecutor : Executor<string, string>
{
    public ValidatorExecutor()
        : base("Validator")
    {
    }

    public override ValueTask<string> HandleAsync(string llmResponse, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var hasHallucination = llmResponse.Contains("unknown", StringComparison.OrdinalIgnoreCase)
            || llmResponse.Contains("unverified", StringComparison.OrdinalIgnoreCase)
            || llmResponse.Contains("no record", StringComparison.OrdinalIgnoreCase);

        var confidence = hasHallucination ? 0.3 : 0.9;
        var status = hasHallucination ? "INCONSISTENT" : "CONSISTENT";

        var result = $"{{\"status\":\"{status}\",\"confidence\":{confidence:F1},\"response\":\"{llmResponse.Replace("\"", "\\\"")}\"}}";
        return ValueTask.FromResult(result);
    }
}
