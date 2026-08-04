using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Text.Json;

namespace NivaraChat;

internal sealed class ConfidenceRouter : Executor<string, string>
{
    private readonly float _threshold;
    private readonly OllamaApiClient? _chatClient;
    private readonly List<string> _pending = [];

    public ConfidenceRouter(float threshold, OllamaApiClient? chatClient = null)
        : base("ConfidenceRouter")
    {
        _threshold = threshold;
        _chatClient = chatClient;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrEmpty(input))
            _pending.Add(input);

        if (_pending.Count < 2)
            return "";

        float sentimentConfidence = 0f;
        float entityConfidence = 0f;
        string sentimentLabel = "unknown";
        string entitiesRaw = "{}";
        string sentimentJson = "{}";

        foreach (var msg in _pending)
        {
            using var doc = JsonDocument.Parse(msg);
            if (doc.RootElement.TryGetProperty("label", out _))
            {
                sentimentConfidence = doc.RootElement.GetProperty("confidence").GetSingle();
                sentimentLabel = doc.RootElement.GetProperty("label").GetString() ?? "unknown";
                sentimentJson = msg;
            }
            else if (doc.RootElement.TryGetProperty("entities", out _))
            {
                entityConfidence = doc.RootElement.GetProperty("confidence").GetSingle();
                if (doc.RootElement.TryGetProperty("entities", out var innerEnt))
                    entitiesRaw = innerEnt.GetRawText();
            }
        }
        _pending.Clear();

        bool confident = sentimentConfidence >= _threshold && entityConfidence >= _threshold;

        if (confident)
        {
            return $"Sentiment: {sentimentLabel} (confidence: {sentimentConfidence:F2})\n"
                 + $"Entities: {entitiesRaw}\n"
                 + $"(Handled by Nivara — no LLM needed, threshold: {_threshold:F2})";
        }

        if (_chatClient == null)
        {
            return JsonSerializer.Serialize(new
            {
                confident = false,
                sentimentConfidence,
                entityConfidence,
                threshold = _threshold,
                sentiment = JsonDocument.Parse(sentimentJson).RootElement,
                entities = JsonDocument.Parse($"{{\"entities\":{entitiesRaw}}}").RootElement
            });
        }

        var routerData = JsonSerializer.Serialize(new
        {
            confident = false,
            sentimentConfidence,
            entityConfidence,
            threshold = _threshold,
            sentiment = JsonDocument.Parse(sentimentJson).RootElement,
            entities = JsonDocument.Parse($"{{\"entities\":{entitiesRaw}}}").RootElement
        });

        var prompt = "You are a helpful assistant. Analyze the structured data provided and give a clear, concise response.\n\n" + routerData;
        var response = await _chatClient.GetResponseAsync(prompt);
        return response.ToString();
    }
}
