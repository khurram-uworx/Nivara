using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using NivaraChat.Tools;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --tools: an Ollama LLM orchestrator calls the Nivara-trained models as AIFunction tools.
/// </summary>
public static class ToolsMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Nivara as AIFunction Tools ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "sentiment_model.json")))
        {
            Console.WriteLine("Models not found. Run with --train first.");
            return;
        }

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --tools requires --ollama for the LLM orchestrator.");
            return;
        }

        Console.WriteLine("Loading trained models...");
        var (sentimentModel, sentimentTok) = ModeHelpers.LoadSentimentModel(ctx.ModelsDir);
        var (entityModel, entityTok) = ModeHelpers.LoadEntityModel(ctx.ModelsDir);
        var (validatorModel, validatorTok) = ModeHelpers.LoadValidatorModel(ctx.ModelsDir);
        Console.WriteLine("Models loaded.\n");

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        NivaraToolFunctions.Initialize(sentimentModel, sentimentTok, entityModel, entityTok, validatorModel, validatorTok);

        var tools = new AITool[]
        {
            AIFunctionFactory.Create(NivaraToolFunctions.AnalyzeSentiment),
            AIFunctionFactory.Create(NivaraToolFunctions.ExtractEntities),
            AIFunctionFactory.Create(NivaraToolFunctions.ValidateResponse),
        };

        var agent = new ChatClientAgent(chatClient,
            instructions: "You are an analyst. Use the provided Nivara tools to analyze text. Always call tools before generating your response. Present a clear summary of all tool results.",
            name: "NivaraOrchestrator",
            tools: tools);

        Workflow BuildWorkflow() => new WorkflowBuilder(agent)
            .WithOutputFrom(agent)
            .Build();

        string prompt;
        if (ctx.SingleShotText != null)
        {
            prompt = $"Analyze this text using available tools: {ctx.SingleShotText}";
        }
        else
        {
            Console.WriteLine("Type a message to analyze (or 'quit' to exit):\n");
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input == "quit") goto cleanup;
            prompt = $"Analyze this text using available tools: {input}";
        }

        var run = await InProcessExecution.RunAsync(BuildWorkflow(), prompt);
        Console.WriteLine("\n--- Tool Orchestration Results ---");
        ModeHelpers.PrintAgentResults(run);

        if (ctx.SingleShotText == null)
        {
            while (true)
            {
                Console.Write("> ");
                var input = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(input) || input == "quit") break;

                var loopRun = await InProcessExecution.RunAsync(BuildWorkflow(), $"Analyze this text using available tools: {input}");
                Console.WriteLine("\n--- Tool Orchestration Results ---");
                ModeHelpers.PrintAgentResults(loopRun);
                Console.WriteLine();
            }
        }

cleanup:
        sentimentModel.Dispose();
        entityModel.Dispose();
        validatorModel.Dispose();
        Console.WriteLine("\nDone.");
    }
}