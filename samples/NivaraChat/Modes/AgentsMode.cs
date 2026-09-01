using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using NivaraChat.Models;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --agents / --interactive: wires the trained text models into named IAIAgent participants and
/// runs the sentiment → entity → validator (→ LLM) graph interactively or single-shot.
/// </summary>
public static class AgentsMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat Agents ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "sentiment_model.json")))
        {
            Console.WriteLine("Models not found. Run with --train first.");
            return;
        }

        Console.WriteLine("Loading trained models...");
        var (sentimentModel, sentimentTok) = ModeHelpers.LoadSentimentModel(ctx.ModelsDir);
        var (entityModel, entityTok) = ModeHelpers.LoadEntityModel(ctx.ModelsDir);
        var (validatorModel, validatorTok) = ModeHelpers.LoadValidatorModel(ctx.ModelsDir, useAgentsFormat: true);
        Console.WriteLine("Models loaded.\n");

        var sentimentText = new SentimentTextModel(sentimentModel, sentimentTok);
        var entityText = new EntityTextModel(entityModel, entityTok);
        var validatorText = new ValidatorTextModel(validatorModel, validatorTok);

        var sentimentAgent = new NivaraChatClient(sentimentText).AsAIAgent("NivaraSentiment");
        var entityAgent = new NivaraChatClient(entityText).AsAIAgent("NivaraEntity");
        var validatorAgent = new NivaraChatClient(validatorText).AsAIAgent("NivaraValidator");

        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            var ollamaClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            var llmAgent = ollamaClient.AsAIAgent(
                name: "OllamaLLM", instructions:
                    """
                    Trust what NivaraSentiment, NivaraEntity and NivaraValidator agents.
                    Present the gathered fact in user friendly summary,
                    if any fact is missing due to low score, figure out yourself"
                    """);
            Console.WriteLine("Ollama connected.\n");
            Console.WriteLine("Graph: NivaraSentiment -> NivaraEntity -> NivaraValidator -> OllamaLLM\n");

            Workflow BuildWorkflowWithOllama() => new WorkflowBuilder(sentimentAgent)
                .AddEdge(sentimentAgent, entityAgent)
                .AddEdge(entityAgent, validatorAgent)
                .AddEdge(validatorAgent, llmAgent)
                .WithOutputFrom(sentimentAgent, entityAgent, validatorAgent, llmAgent)
                .Build();

            if (ctx.SingleShotText != null)
                await ModeHelpers.RunSingleShot(BuildWorkflowWithOllama(), ctx.SingleShotText);
            else
                await ModeHelpers.RunLoop(BuildWorkflowWithOllama);
        }
        else
        {
            Console.WriteLine("Graph: NivaraSentiment -> NivaraEntity -> NivaraValidator\n");

            Workflow BuildWorkflow() => new WorkflowBuilder(sentimentAgent)
                .AddEdge(sentimentAgent, entityAgent)
                .AddEdge(entityAgent, validatorAgent)
                .WithOutputFrom(sentimentAgent, entityAgent, validatorAgent)
                .Build();

            if (ctx.SingleShotText != null)
                await ModeHelpers.RunSingleShot(BuildWorkflow(), ctx.SingleShotText);
            else
                await ModeHelpers.RunLoop(BuildWorkflow);
        }

        sentimentModel.Dispose();
        entityModel.Dispose();
        validatorModel.Dispose();
        Console.WriteLine("\nDone.");
    }
}