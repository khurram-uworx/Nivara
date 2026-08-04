using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;

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
            var prompt = "You are a helpful assistant. Answer the user's question clearly and concisely.\n\n" + input;
            var response = await _chatClient.GetResponseAsync(prompt);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error calling LLM: {ex.Message}";
        }
    }
}