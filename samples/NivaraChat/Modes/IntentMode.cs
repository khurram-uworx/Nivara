using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Nivara.Samples;
using NivaraChat.Executors;
using NivaraChat.Helpers;
using NivaraChat.Tools;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --intent: classifies input into one of five intents and routes to a specialist executor
/// (factual/retrieval, question, command/tools, escalation, chitchat).
/// </summary>
public static class IntentMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat Intent Routing ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "intent_model.json")))
        {
            Console.WriteLine("Intent model not found. Run with --intent-train first.");
            return;
        }

        Console.WriteLine("Loading intent model...");
        var (intentModel, intentTok) = ModeHelpers.LoadIntentModel(ctx.ModelsDir);
        Console.WriteLine("Intent model loaded.\n");

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --intent requires --ollama for specialist executors.");
            intentModel.Dispose();
            return;
        }

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        var minilmDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "minilm");
        CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore = null;
        if (Directory.Exists(minilmDir))
        {
            Console.WriteLine("Loading MiniLM embedding model for factual retrieval...");
            var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
            vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(
                new() { EmbeddingGenerator = generator });
            var collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
            await collection.EnsureCollectionExistsAsync();
            var repoRoot = ModeHelpers.GetRepoRoot();
            var docsDir = Path.Combine(repoRoot, "docs");
            var mdFiles = Directory.Exists(docsDir)
                ? Directory.GetFiles(docsDir, "*.md")
                : [];
            var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
            if (File.Exists(readmePath))
                mdFiles = [.. mdFiles, readmePath];
            if (mdFiles.Length > 0)
            {
                Console.WriteLine($"Indexing {mdFiles.Length} markdown files...");
                await DocumentChunker.IndexMarkdownFiles(collection, mdFiles);
            }
            Console.WriteLine("Factual retrieval ready.\n");
        }
        else
        {
            Console.WriteLine("MiniLM model not found; factual executor will use LLM without retrieval.\n");
        }

        var tools = new AIFunction[]
        {
            AIFunctionFactory.Create(NivaraToolFunctions.AnalyzeSentiment),
            AIFunctionFactory.Create(NivaraToolFunctions.ExtractEntities),
            AIFunctionFactory.Create(NivaraToolFunctions.ValidateResponse),
        };

        var intentClassifier = new IntentClassifier(intentModel, intentTok);
        Executor<string, string> factualExecutor = vectorStore != null
            ? new FactualExecutor(vectorStore, chatClient)
            : new LlmExecutor(chatClient);
        var questionExecutor = new QuestionExecutor(chatClient);
        var commandExecutor = new CommandExecutor(chatClient, tools);
        var escalationExecutor = new EscalationExecutor();
        var chitchatExecutor = new ChitchatExecutor(chatClient);

        Console.WriteLine("Graph: IntentClassifier --> AddSwitch --> [Factual, Question, Command, Escalation, Chitchat]\n");

        Workflow BuildWorkflow() => new WorkflowBuilder(intentClassifier)
            .AddEdge<string>(intentClassifier, factualExecutor,
                condition: msg => ExtractIntent(msg!) == "factual")
            .AddEdge<string>(intentClassifier, questionExecutor,
                condition: msg => ExtractIntent(msg!) == "question")
            .AddEdge<string>(intentClassifier, commandExecutor,
                condition: msg => ExtractIntent(msg!) == "command")
            .AddEdge<string>(intentClassifier, escalationExecutor,
                condition: msg => ExtractIntent(msg!) == "complaint")
            .AddEdge<string>(intentClassifier, chitchatExecutor,
                condition: msg => ExtractIntent(msg!) == "chitchat")
            .WithOutputFrom(factualExecutor, questionExecutor, commandExecutor, escalationExecutor, chitchatExecutor)
            .Build();

        static string ExtractIntent(string json)
        {
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(json);
                return doc.RootElement.GetProperty("intent").GetString() ?? "chitchat";
            }
            catch
            {
                return "chitchat";
            }
        }

        if (ctx.SingleShotText != null)
        {
            var run = await InProcessExecution.RunAsync(BuildWorkflow(), ctx.SingleShotText);
            Console.WriteLine("\n--- Intent Routing Results ---");
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
            Console.WriteLine("Type a message to classify (or 'quit' to exit):\n");

            while (true)
            {
                Console.Write("> ");
                var input = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(input) || input == "quit") break;

                var run = await InProcessExecution.RunAsync(BuildWorkflow(), input);

                Console.WriteLine("\n--- Intent Routing Results ---");
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

        intentModel.Dispose();
    }
}