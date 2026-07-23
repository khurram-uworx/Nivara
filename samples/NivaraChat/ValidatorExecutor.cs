using Microsoft.Agents.AI.Workflows;

namespace NivaraChat;

internal sealed partial class ValidatorExecutor : Executor
{
    public ValidatorExecutor()
        : base("Validator")
    {
    }

    [MessageHandler]
    public ValueTask<string> HandleAsync(string llmResponse, IWorkflowContext context)
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
