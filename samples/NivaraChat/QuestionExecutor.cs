using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Text.Json;

namespace NivaraChat;

internal sealed class QuestionExecutor : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;

    public QuestionExecutor(OllamaApiClient chatClient)
        : base("Question")
    {
        _chatClient = chatClient;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var originalText = ExtractInput(input);
            var prompt = "You are a helpful assistant. Answer the user's question clearly and concisely.\n\n" + originalText;
            var response = await _chatClient.GetResponseAsync(prompt);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error calling LLM: {ex.Message}";
        }
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