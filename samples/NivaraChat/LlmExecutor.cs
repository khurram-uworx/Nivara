using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;

namespace NivaraChat;

internal sealed class LlmExecutor : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;

    public LlmExecutor(OllamaApiClient chatClient)
        : base("Ollama LLM")
    {
        _chatClient = chatClient;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var prompt = "You are a helpful assistant. Analyze the structured data provided and give a clear, concise response.\n\n" + input;
            var response = await _chatClient.GetResponseAsync(prompt);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error calling LLM: {ex.Message}";
        }
    }
}
