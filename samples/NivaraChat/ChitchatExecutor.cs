using Microsoft.Agents.AI.Workflows;
using OllamaSharp;

namespace NivaraChat;

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
            var prompt = "You are a friendly chatbot. Have a casual, warm conversation with the user. Keep responses short and engaging.\n\n" + input;
            var response = await _chatClient.GetResponseAsync(prompt);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error calling LLM: {ex.Message}";
        }
    }
}