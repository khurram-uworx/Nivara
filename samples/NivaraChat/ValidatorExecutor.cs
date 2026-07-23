using Microsoft.Agents.AI.Workflows;

namespace NivaraChat;

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

        string? sentiment = null;
        string? entities = null;
        foreach (var msg in _pending)
        {
            if (msg.StartsWith('{'))
                entities = msg;
            else
                sentiment = msg;
        }
        _pending.Clear();

        sentiment ??= "unknown";
        entities ??= "{}";

        bool hasEntities = entities.Contains("\"person\"") || entities.Contains("\"org\"")
            || entities.Contains("\"date\"") || entities.Contains("\"location\"");
        bool hasMeaningfulSentiment = sentiment != "Neutral" && sentiment != "unknown";

        var confidence = (hasEntities || hasMeaningfulSentiment) ? 0.9 : 0.3;
        var status = (hasEntities || hasMeaningfulSentiment) ? "CONSISTENT" : "INCONSISTENT";
        var result = $"{{\"status\":\"{status}\",\"confidence\":{confidence:F1},\"sentiment\":\"{sentiment}\",\"entities\":{entities}}}";
        return ValueTask.FromResult(result);
    }
}
