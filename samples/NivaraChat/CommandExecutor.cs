using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;

namespace NivaraChat;

internal sealed class CommandExecutor : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;
    private readonly AIFunction[] _tools;

    public CommandExecutor(OllamaApiClient chatClient, AIFunction[] tools)
        : base("Command")
    {
        _chatClient = chatClient;
        _tools = tools;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var agent = new ChatClientAgent(_chatClient,
                instructions: "You are an action assistant. Use the provided tools to fulfill the user's request.",
                name: "CommandAgent",
                tools: _tools);

            var response = await agent.RunAsync(input);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error in command execution: {ex.Message}";
        }
    }
}