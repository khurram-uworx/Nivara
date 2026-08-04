using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Text.Json;

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
            var originalText = ExtractInput(input);
            var agent = new ChatClientAgent(_chatClient,
                instructions: "You are an action assistant. Use the provided tools to fulfill the user's request.",
                name: "CommandAgent",
                tools: _tools);

            var response = await agent.RunAsync(originalText);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error in command execution: {ex.Message}";
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