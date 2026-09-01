using Microsoft.Agents.AI.Workflows;
using NivaraChat.Executors;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --handoff: confidence-based handoff — Nivara's sentiment/entity executors answer when
/// confident, otherwise the Ollama LLM fallback path is used.
/// </summary>
public static class HandoffMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Confidence Handoff ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "sentiment_model.json")))
        {
            Console.WriteLine("Models not found. Run with --train first.");
            return;
        }

        Console.WriteLine("Loading trained models...");
        var (sentimentModel, sentimentTok) = ModeHelpers.LoadSentimentModel(ctx.ModelsDir);
        var (entityModel, entityTok) = ModeHelpers.LoadEntityModel(ctx.ModelsDir);
        Console.WriteLine("Models loaded.\n");

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --handoff requires --ollama for the LLM fallback path.");
            sentimentModel.Dispose();
            entityModel.Dispose();
            return;
        }

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        var router = new TextRouter();
        var sentimentExecutor = new SentimentExecutor(sentimentModel, sentimentTok);
        var entityExtractor = new EntityExtractor(entityModel, entityTok);
        var confidenceRouter = new ConfidenceRouter(ctx.ConfidenceThreshold, chatClient);

        Console.WriteLine($"Graph: TextRouter --fan-out--> [Sentiment, Entity] --fan-in--> ConfidenceRouter");
        Console.WriteLine($"  confident (>= {ctx.ConfidenceThreshold:F1}) --> Nivara result (skip LLM)");
        Console.WriteLine($"  uncertain (< {ctx.ConfidenceThreshold:F1}) --> Ollama LLM\n");

        Workflow BuildWorkflow() => new WorkflowBuilder(router)
            .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
            .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, confidenceRouter)
            .WithOutputFrom(confidenceRouter)
            .Build();

        if (ctx.SingleShotText != null)
        {
            var run = await InProcessExecution.RunAsync(BuildWorkflow(), ctx.SingleShotText);
            Console.WriteLine("\n--- Handoff Results ---");
            foreach (var evt in run.NewEvents)
            {
                switch (evt)
                {
                    case ExecutorCompletedEvent executorEvt:
                        if (executorEvt.Data?.ToString() is string data && !string.IsNullOrEmpty(data))
                            Console.WriteLine($"  [{executorEvt.ExecutorId}] {data}");
                        break;
                    case AgentResponseEvent agentEvt:
                        Console.WriteLine($"  [LLM] {agentEvt.Data}");
                        break;
                }
            }
        }
        else
        {
            Console.WriteLine("Type a message to analyze (or 'quit' to exit):\n");

            while (true)
            {
                Console.Write("> ");
                var input = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(input) || input == "quit") break;

                var run = await InProcessExecution.RunAsync(BuildWorkflow(), input);

                Console.WriteLine("\n--- Handoff Results ---");
                foreach (var evt in run.NewEvents)
                {
                    switch (evt)
                    {
                        case ExecutorCompletedEvent executorEvt:
                            if (executorEvt.Data?.ToString() is string data && !string.IsNullOrEmpty(data))
                                Console.WriteLine($"  [{executorEvt.ExecutorId}] {data}");
                            break;
                        case AgentResponseEvent agentEvt:
                            Console.WriteLine($"  [LLM] {agentEvt.Data}");
                            break;
                    }
                }
                Console.WriteLine();
            }
        }

        sentimentModel.Dispose();
        entityModel.Dispose();
        Console.WriteLine("\nDone.");
    }
}