using Microsoft.Agents.AI.Workflows;
using NivaraChat.Executors;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --workflow: Agent Framework fan-out/fan-in pipeline over the trained sentiment/entity/validator
/// models, with an optional Ollama LLM backstop.
/// </summary>
public static class WorkflowMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat Workflow ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "sentiment_model.json")))
        {
            Console.WriteLine("Models not found. Run with --train first.");
            return;
        }

        Console.WriteLine("Loading trained models...");
        var (sentimentModel, sentimentTok) = ModeHelpers.LoadSentimentModel(ctx.ModelsDir);
        var (entityModel, entityTok) = ModeHelpers.LoadEntityModel(ctx.ModelsDir);
        Console.WriteLine("Models loaded.\n");

        var router = new TextRouter();
        var sentimentExecutor = new SentimentExecutor(sentimentModel, sentimentTok);
        var entityExtractor = new EntityExtractor(entityModel, entityTok);
        var validator = new ValidatorExecutor();

        Executor<string, string>? llmExecutor = null;
        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            llmExecutor = new LlmExecutor(chatClient);
            Console.WriteLine("Ollama connected.\n");
        }

        Console.WriteLine(llmExecutor != null
            ? "Graph: TextRouter --fan-out--> [SentimentExecutor, EntityExtractor] --fan-in--> ValidatorExecutor -> Ollama LLM\n"
            : "Graph: TextRouter --fan-out--> [SentimentExecutor, EntityExtractor] --fan-in--> ValidatorExecutor\n");

        Workflow BuildWorkflow() => llmExecutor != null
            ? new WorkflowBuilder(router)
                .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
                .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, validator)
                .AddEdge(validator, llmExecutor)
                .WithOutputFrom(sentimentExecutor, entityExtractor, validator, llmExecutor)
                .Build()
            : new WorkflowBuilder(router)
                .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
                .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, validator)
                .WithOutputFrom(sentimentExecutor, entityExtractor, validator)
                .Build();

        if (ctx.SingleShotText != null)
        {
            var run = await InProcessExecution.RunAsync(BuildWorkflow(), ctx.SingleShotText);
            Console.WriteLine("\n--- Workflow Results ---");
            var events = run.NewEvents.ToList();
            foreach (var evt in events)
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

                Console.WriteLine("\n--- Workflow Results ---");
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