using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Text.Json;

namespace NivaraChat.Executors;

internal sealed class ChitchatExecutor : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;

    public ChitchatExecutor(OllamaApiClient chatClient)
        : base("Chitchat")
    {
        _chatClient = chatClient;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var originalText = ExtractInput(input);
            var prompt = "You are a friendly chatbot. Have a casual, warm conversation with the user. Keep responses short and engaging.\n\n" + originalText;
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