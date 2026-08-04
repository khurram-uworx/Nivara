using Microsoft.Agents.AI.Workflows;
using System.Text.Json;

namespace NivaraChat;

internal sealed class ConfidenceRouter : Executor<string, string>
{
    private readonly float _threshold;
    private readonly List<string> _pending = [];

    public ConfidenceRouter(float threshold = 0.8f)
        : base("ConfidenceRouter")
    {
        _threshold = threshold;
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrEmpty(input))
            _pending.Add(input);

        if (_pending.Count < 2)
            return ValueTask.FromResult("");

        float sentimentConfidence = 0f;
        float entityConfidence = 0f;
        string sentimentResult = "{}";
        string entitiesResult = "{}";

        foreach (var msg in _pending)
        {
            using var doc = JsonDocument.Parse(msg);
            if (doc.RootElement.TryGetProperty("label", out _))
            {
                sentimentConfidence = doc.RootElement.GetProperty("confidence").GetSingle();
                sentimentResult = msg;
            }
            else if (doc.RootElement.TryGetProperty("entities", out _))
            {
                entityConfidence = doc.RootElement.GetProperty("confidence").GetSingle();
                entitiesResult = msg;
            }
        }
        _pending.Clear();

        bool confident = sentimentConfidence >= _threshold && entityConfidence >= _threshold;

        var result = JsonSerializer.Serialize(new
        {
            confident,
            sentimentConfidence,
            entityConfidence,
            threshold = _threshold,
            sentiment = JsonDocument.Parse(sentimentResult).RootElement,
            entities = JsonDocument.Parse(entitiesResult).RootElement
        });
        return ValueTask.FromResult(result);
    }
}
