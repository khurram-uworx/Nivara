using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.Samples;
using NivaraChat;
using NivaraChat.Training;
using OllamaSharp;
using System.Numerics.Tensors;

const string DefaultOllamaUrl = "http://localhost:11434";
const string DefaultModel = "llama3.2";
const string ModelsDir = "samples/data/nivarachat";

var modelsDir = Path.Combine(GetRepoRoot(), ModelsDir);

if (args.Length > 0)
{
    var mode = args[0];
    var ollamaUrl = DefaultOllamaUrl;
    var modelName = DefaultModel;
    string? workflowText = null;
    bool useOllama = false;
    float confidenceThreshold = 0.8f;
    string? docsDir = null;
    int topK = 3;

    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--ollama") { useOllama = true; if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) ollamaUrl = args[++i]; }
        if (args[i] == "--model" && i + 1 < args.Length) modelName = args[++i];
        if (args[i] == "--text" && i + 1 < args.Length) workflowText = args[++i];
        if (args[i] == "--threshold" && i + 1 < args.Length) confidenceThreshold = float.Parse(args[++i]);
        if (args[i] == "--docs-dir" && i + 1 < args.Length) docsDir = args[++i];
        if (args[i] == "--top-k" && i + 1 < args.Length) topK = int.Parse(args[++i]);
    }

    switch (mode)
    {
        case "--train":
            RunTraining(modelsDir);
            break;
        case "--workflow":
            await RunWorkflow(modelsDir, ollamaUrl, modelName, workflowText, useOllama);
            break;
        case "--interactive":
            await RunAgents(modelsDir, ollamaUrl, modelName, null, useOllama);
            break;
        case "--agents":
            await RunAgents(modelsDir, ollamaUrl, modelName, workflowText, useOllama);
            break;
        case "--handoff":
            await RunHandoff(modelsDir, ollamaUrl, modelName, workflowText, useOllama, confidenceThreshold);
            break;
        case "--tools":
            await RunTools(modelsDir, ollamaUrl, modelName, workflowText, useOllama);
            break;
        case "--critic":
            await RunCritic(modelsDir, ollamaUrl, modelName, workflowText, useOllama);
            break;
        case "--embed":
            RunEmbeddingSearch();
            break;
        case "--rag":
            await RunRagPipeline(modelsDir, ollamaUrl, modelName, docsDir, topK, useOllama, workflowText);
            break;
        case "--rag-agent":
            await RunRagAgentPipeline(modelsDir, ollamaUrl, modelName, docsDir, topK, useOllama, workflowText);
            break;
        case "--intent-train":
            RunIntentTraining(modelsDir);
            break;
        case "--intent":
            await RunIntentMode(modelsDir, ollamaUrl, modelName, workflowText, useOllama);
            break;
        default:
            PrintUsage();
            break;
    }
}
else
{
    await RunInteractiveMenu(modelsDir);
}

async Task RunInteractiveMenu(string modelsDir)
{
    while (true)
    {
        var choice = ShowMainMenu();

        switch (choice)
        {
            case "1":
                RunTraining(modelsDir);
                Console.WriteLine();
                break;
            case "2":
                var (useOllama2, url2, model2) = AskOllama();
                await RunWorkflow(modelsDir, url2, model2, null, useOllama2);
                Console.WriteLine();
                break;
            case "3":
                var (useOllama3, url3, model3) = AskOllama();
                await RunAgents(modelsDir, url3, model3, null, useOllama3);
                Console.WriteLine();
                break;
            case "q":
                return;
        }
    }
}

string ShowMainMenu()
{
    Console.WriteLine("=== NivaraChat ===\n");
    Console.WriteLine("Select a mode:");
    Console.WriteLine("  1. Training    - Train sentiment, entity, and validator models");
    Console.WriteLine("  2. Workflow    - Run the Agent Framework workflow pipeline");
    Console.WriteLine("  3. Agents      - Run the agents pipeline with live chat");
    Console.WriteLine("  q. Quit\n");
    Console.Write("> ");
    return Console.ReadLine()?.Trim().ToLower() ?? "";
}

(bool useOllama, string url, string model) AskOllama()
{
    Console.Write("\nUse Ollama for LLM enrichment? (y/n, default: n): ");
    var answer = Console.ReadLine()?.Trim().ToLower();
    if (answer == "y" || answer == "yes")
    {
        Console.WriteLine($"  Using Ollama at {DefaultOllamaUrl} with model {DefaultModel}\n");
        return (true, DefaultOllamaUrl, DefaultModel);
    }
    Console.WriteLine();
    return (false, DefaultOllamaUrl, DefaultModel);
}

string GetRepoRoot()
    => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,
    "..", "..", "..", "..", ".."));


void RunTraining(string modelsDir)
{
    Console.WriteLine("=== NivaraChat Model Training ===\n");
    Directory.CreateDirectory(modelsDir);

    Console.WriteLine("[1/4] Training sentiment classifier...");
    SentimentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: modelsDir);

    Console.WriteLine("\n[2/4] Training entity extractor...");
    EntityTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: modelsDir);

    Console.WriteLine("\n[3/4] Training workflow validator...");
    ValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: modelsDir);

    Console.WriteLine("\n[4/4] Training agents validator...");
    AgentsValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: modelsDir);

    Console.WriteLine("\n=== Training complete! ===");
    if (args.Length > 0)
        Console.WriteLine("Run with --workflow or --agents to test the pipeline, or --interactive for chat.");
    else
        Console.WriteLine("Returning to main menu...");
}

void RunIntentTraining(string modelsDir)
{
    Console.WriteLine("=== NivaraChat Intent Classifier Training ===\n");
    Directory.CreateDirectory(modelsDir);
    IntentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: modelsDir);
    Console.WriteLine("\n=== Intent training complete! ===");
    if (args.Length > 0)
        Console.WriteLine("Run with --intent to test the intent routing.");
    else
        Console.WriteLine("Returning to main menu...");
}

async Task RunIntentMode(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Intent Routing ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "intent_model.json")))
    {
        Console.WriteLine("Intent model not found. Run with --intent-train first.");
        return;
    }

    Console.WriteLine("Loading intent model...");
    var (intentModel, intentTok) = LoadIntentModel(modelsDir);
    Console.WriteLine("Intent model loaded.\n");

    if (!useOllama)
    {
        Console.WriteLine("Error: --intent requires --ollama for specialist executors.");
        intentModel.Dispose();
        return;
    }

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    Console.WriteLine("Ollama connected.\n");

    var minilmDir = Path.Combine(GetRepoRoot(), "samples", "data", "minilm");
    CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore = null;
    if (Directory.Exists(minilmDir))
    {
        Console.WriteLine("Loading MiniLM embedding model for factual retrieval...");
        var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
        vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(
            new() { EmbeddingGenerator = generator });
        var collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
        await collection.EnsureCollectionExistsAsync();
        var repoRoot = GetRepoRoot();
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

    if (singleShotText != null)
    {
        var run = await InProcessExecution.RunAsync(BuildWorkflow(), singleShotText);
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

async Task RunWorkflow(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Workflow ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel(modelsDir);
    var (entityModel, entityTok) = LoadEntityModel(modelsDir);
    Console.WriteLine("Models loaded.\n");

    var router = new TextRouter();
    var sentimentExecutor = new SentimentExecutor(sentimentModel, sentimentTok);
    var entityExtractor = new EntityExtractor(entityModel, entityTok);
    var validator = new ValidatorExecutor();

    Executor<string, string>? llmExecutor = null;
    if (useOllama)
    {
        Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
        var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
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

    if (singleShotText != null)
    {
        var run = await InProcessExecution.RunAsync(BuildWorkflow(), singleShotText);
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

async Task RunAgents(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Agents ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel(modelsDir);
    var (entityModel, entityTok) = LoadEntityModel(modelsDir);
    var (validatorModel, validatorTok) = LoadValidatorModel(modelsDir, useAgentsFormat: true);
    Console.WriteLine("Models loaded.\n");

    var sentimentText = new SentimentTextModel(sentimentModel, sentimentTok);
    var entityText = new EntityTextModel(entityModel, entityTok);
    var validatorText = new ValidatorTextModel(validatorModel, validatorTok);

    var sentimentAgent = new NivaraChatClient(sentimentText).AsAIAgent("NivaraSentiment");
    var entityAgent = new NivaraChatClient(entityText).AsAIAgent("NivaraEntity");
    var validatorAgent = new NivaraChatClient(validatorText).AsAIAgent("NivaraValidator");

    if (useOllama)
    {
        Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
        var ollamaClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
        var llmAgent = ollamaClient.AsAIAgent(
            //new NivaraChatClient(new PassthroughTextModel(ollamaClient)).AsAIAgent("OllamaLLM");
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

        if (singleShotText != null)
            await RunSingleShot(BuildWorkflowWithOllama(), singleShotText);
        else
            await RunLoop(BuildWorkflowWithOllama);
    }
    else
    {
        Console.WriteLine("Graph: NivaraSentiment -> NivaraEntity -> NivaraValidator\n");

        Workflow BuildWorkflow() => new WorkflowBuilder(sentimentAgent)
            .AddEdge(sentimentAgent, entityAgent)
            .AddEdge(entityAgent, validatorAgent)
            .WithOutputFrom(sentimentAgent, entityAgent, validatorAgent)
            .Build();

        if (singleShotText != null)
            await RunSingleShot(BuildWorkflow(), singleShotText);
        else
            await RunLoop(BuildWorkflow);
    }

    sentimentModel.Dispose();
    entityModel.Dispose();
    validatorModel.Dispose();
    Console.WriteLine("\nDone.");
}

async Task RunHandoff(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama, float threshold)
{
    Console.WriteLine("=== NivaraChat — Confidence Handoff ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel(modelsDir);
    var (entityModel, entityTok) = LoadEntityModel(modelsDir);
    Console.WriteLine("Models loaded.\n");

    if (!useOllama)
    {
        Console.WriteLine("Error: --handoff requires --ollama for the LLM fallback path.");
        sentimentModel.Dispose();
        entityModel.Dispose();
        return;
    }

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    Console.WriteLine("Ollama connected.\n");

    var router = new TextRouter();
    var sentimentExecutor = new SentimentExecutor(sentimentModel, sentimentTok);
    var entityExtractor = new EntityExtractor(entityModel, entityTok);
    var confidenceRouter = new ConfidenceRouter(threshold, chatClient);

    Console.WriteLine($"Graph: TextRouter --fan-out--> [Sentiment, Entity] --fan-in--> ConfidenceRouter");
    Console.WriteLine($"  confident (>= {threshold:F1}) --> Nivara result (skip LLM)");
    Console.WriteLine($"  uncertain (< {threshold:F1}) --> Ollama LLM\n");

    Workflow BuildWorkflow() => new WorkflowBuilder(router)
        .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
        .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, confidenceRouter)
        .WithOutputFrom(confidenceRouter)
        .Build();

    if (singleShotText != null)
    {
        var run = await InProcessExecution.RunAsync(BuildWorkflow(), singleShotText);
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

async Task RunTools(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat — Nivara as AIFunction Tools ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    if (!useOllama)
    {
        Console.WriteLine("Error: --tools requires --ollama for the LLM orchestrator.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel(modelsDir);
    var (entityModel, entityTok) = LoadEntityModel(modelsDir);
    var (validatorModel, validatorTok) = LoadValidatorModel(modelsDir);
    Console.WriteLine("Models loaded.\n");

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
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
    if (singleShotText != null)
    {
        prompt = $"Analyze this text using available tools: {singleShotText}";
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
    PrintAgentResults(run);

    if (singleShotText == null)
    {
        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input == "quit") break;

            var loopRun = await InProcessExecution.RunAsync(BuildWorkflow(), $"Analyze this text using available tools: {input}");
            Console.WriteLine("\n--- Tool Orchestration Results ---");
            PrintAgentResults(loopRun);
            Console.WriteLine();
        }
    }

cleanup:
    sentimentModel.Dispose();
    entityModel.Dispose();
    validatorModel.Dispose();
    Console.WriteLine("\nDone.");
}

async Task RunCritic(string modelsDir, string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat — Writer-Critic Loop ===\n");

    if (!File.Exists(Path.Combine(modelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    if (!useOllama)
    {
        Console.WriteLine("Error: --critic requires --ollama for the LLM writer.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (validatorModel, validatorTok) = LoadValidatorModel(modelsDir);
    Console.WriteLine("Models loaded.\n");

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    Console.WriteLine("Ollama connected.\n");

    var critic = new CriticExecutor(validatorModel, validatorTok);
    var loop = new WriterCriticLoop(chatClient, critic);

    Console.WriteLine("Writer (Ollama) → Critic (Nivara validator) → pass/fail → retry if needed\n");

    if (singleShotText != null)
    {
        var result = await loop.RunAsync(singleShotText);
        Console.WriteLine("\n--- Critic Results ---");
        Console.WriteLine(result);
    }
    else
    {
        Console.WriteLine("Type a question (or 'quit' to exit):\n");

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input == "quit") break;

            var result = await loop.RunAsync(input);
            Console.WriteLine("\n--- Critic Results ---");
            Console.WriteLine(result);
            Console.WriteLine();
        }
    }

    validatorModel.Dispose();
    Console.WriteLine("\nDone.");
}

async Task RunSingleShot(Workflow workflow, string text)
{
    var run = await InProcessExecution.RunAsync(workflow, text);
    Console.WriteLine("\n--- Agent Results ---");
    PrintAgentResults(run);
}

async Task RunLoop(Func<Workflow> workflowFactory)
{
    Console.WriteLine("Type a message (or 'quit' to exit):\n");
    while (true)
    {
        Console.Write("> ");
        var input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input) || input == "quit") break;

        var run = await InProcessExecution.RunAsync(workflowFactory(), input);
        Console.WriteLine("\n--- Agent Results ---");
        PrintAgentResults(run);
        Console.WriteLine();
    }
}

void PrintAgentResults(Run run)
{
    var events = run.NewEvents.ToList();
    var streamingBuffers = new Dictionary<string, string>();

    foreach (var evt in events)
    {
        switch (evt)
        {
            case AgentResponseUpdateEvent updateEvt:
                if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text))
                {
                    var id = updateEvt.ExecutorId;
                    streamingBuffers.TryGetValue(id, out var existing);
                    streamingBuffers[id] = (existing ?? "") + text;
                }
                break;
            case ExecutorCompletedEvent executorEvt:
                if (streamingBuffers.TryGetValue(executorEvt.ExecutorId, out var buffered) && buffered.Length > 0)
                {
                    Console.WriteLine($"  [{executorEvt.ExecutorId}] {buffered}");
                    streamingBuffers.Remove(executorEvt.ExecutorId);
                }
                else if (executorEvt.Data?.ToString() is string data && !string.IsNullOrEmpty(data))
                {
                    Console.WriteLine($"  [{executorEvt.ExecutorId}] {data}");
                }
                break;
            case ExecutorFailedEvent failedEvt:
                Console.WriteLine($"  [{failedEvt.ExecutorId}] FAILED: {failedEvt}");
                break;
            case WorkflowErrorEvent errorEvt:
                Console.WriteLine($"  WORKFLOW ERROR: {errorEvt}");
                break;
        }
    }

    foreach (var (id, text) in streamingBuffers)
    {
        if (text.Length > 0)
            Console.WriteLine($"  [{id}] {text}");
    }
}

(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadValidatorModel(string modelsDir, bool useAgentsFormat = false)
{
    var suffix = useAgentsFormat ? "agents_validator" : "validator";
    var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, $"{suffix}_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 2, 40);
    ModelSerializer.Load(model, Path.Combine(modelsDir, $"{suffix}_model.json"));
    model.Eval();
    return (model, tokenizer);
}

(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadSentimentModel(string modelsDir)
{
    var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "sentiment_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 3, 20);
    ModelSerializer.Load(model, Path.Combine(modelsDir, "sentiment_model.json"));
    model.Eval();
    return (model, tokenizer);
}

(TokenClassifierModel<float> model, TextTokenizer tokenizer) LoadEntityModel(string modelsDir)
{
    var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "entity_tokenizer.json"));
    var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
    ModelSerializer.Load(model, Path.Combine(modelsDir, "entity_model.json"));
    model.Eval();
    return (model, tokenizer);
}

(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadIntentModel(string modelsDir)
{
    var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "intent_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
    ModelSerializer.Load(model, Path.Combine(modelsDir, "intent_model.json"));
    model.Eval();
    return (model, tokenizer);
}

void PrintUsage()
{
    Console.WriteLine("Usage: NivaraChat <mode> [options]\n");
    Console.WriteLine("Modes:");
    Console.WriteLine("  --train              Train sentiment, entity, and validator models");
    Console.WriteLine("  --workflow           Run the Agent Framework workflow (Ollama optional)");
    Console.WriteLine("  --interactive        Interactive mode: agents pipeline with live input");
    Console.WriteLine("  --agents             Same as --interactive, with --text for single-shot");
    Console.WriteLine("  --handoff            Confidence-based handoff: Nivara decides if LLM is needed");
    Console.WriteLine("  --tools              LLM orchestrator calls Nivara models as AIFunction tools");
    Console.WriteLine("  --critic             Writer-critic loop: LLM writes, Nivara scores, retry if poor");
    Console.WriteLine("  --embed              Embedding search: index documents, retrieve context via IEmbeddingGenerator");
    Console.WriteLine("  --rag                RAG pipeline: chunk docs, retrieve context, LLM generate answer");
    Console.WriteLine("  --rag-agent          RAG pipeline with TextSearchProvider auto-context injection");
    Console.WriteLine("\nOptions:");
    Console.WriteLine("  --ollama <url>       Ollama endpoint (default: http://localhost:11434)");
    Console.WriteLine("  --model <name>       Model name (default: llama3.2)");
    Console.WriteLine("  --text \"<message>\"   Single-shot: run pipeline on one message and exit");
    Console.WriteLine("  --threshold <float>  Confidence threshold for --handoff (default: 0.8)");
    Console.WriteLine("  --docs-dir <path>    Documents directory for --rag (default: docs/ + README.md)");
    Console.WriteLine("  --top-k <int>        Number of chunks to retrieve for --rag (default: 3)");
}

void RunEmbeddingSearch()
{
    Console.WriteLine("=== NivaraChat — Embedding Search ===\n");

    var minilmDir = Path.Combine(GetRepoRoot(), "samples", "data", "minilm");
    if (!Directory.Exists(minilmDir))
    {
        Console.WriteLine($"MiniLM model not found at: {minilmDir}");
        Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
        return;
    }

    Console.WriteLine("Loading MiniLM embedding model...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
    var metadata = generator.GetService(typeof(EmbeddingGeneratorMetadata), null) as EmbeddingGeneratorMetadata;
    sw.Stop();
    Console.WriteLine($"  Provider:     {metadata?.ProviderName ?? "Nivara-MiniLM"}");
    Console.WriteLine($"  Model:        {metadata?.DefaultModelId ?? "all-minilm-l6-v2"}");
    Console.WriteLine($"  Dimensions:   {generator.EmbeddingDimension}");
    Console.WriteLine($"  Loaded in:    {sw.ElapsedMilliseconds} ms\n");

    var documents = new[]
    {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "The stock market closed at record highs today",
        "Neural networks are inspired by biological brains",
        "The weather forecast predicts rain tomorrow",
        "Deep learning has revolutionized computer vision",
        "Interest rates are expected to rise next quarter",
        "Natural language processing enables text understanding"
    };

    Console.WriteLine($"Indexing {documents.Length} knowledge documents via IEmbeddingGenerator...");
    sw.Restart();
    var docEmbeddings = generator.GenerateAsync(documents).GetAwaiter().GetResult();
    sw.Stop();
    for (int i = 0; i < documents.Length; i++)
        Console.WriteLine($"  [{i}] \"{documents[i]}\"");
    Console.WriteLine($"\nIndexed in {sw.ElapsedMilliseconds} ms — ready for chat\n");

    Console.WriteLine("Type a message and press Enter (or 'quit' to exit):\n");

    while (true)
    {
        Console.Write("> ");
        var query = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(query) || query == "quit") break;

        sw.Restart();
        var queryEmbedding = generator.GenerateVectorAsync(query).GetAwaiter().GetResult();
        sw.Stop();
        var queryVector = queryEmbedding.ToArray();

        var scores = new float[documents.Length];
        for (int i = 0; i < documents.Length; i++)
        {
            var docVector = docEmbeddings[i].Vector;
            scores[i] = TensorPrimitives.CosineSimilarity(queryVector, docVector.Span);
        }

        var ranked = scores
            .Select((score, idx) => (Score: score, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(4)
            .ToList();

        Console.WriteLine($"  Retrieved {ranked.Count} relevant documents ({sw.ElapsedMilliseconds} ms)\n");
        Console.WriteLine("  Context for LLM:");
        for (int rank = 0; rank < ranked.Count; rank++)
            Console.WriteLine($"    #{rank + 1}  {ranked[rank].Score:F4}  \"{documents[ranked[rank].Index]}\"");
        Console.WriteLine("\n  (In a full pipeline, these would be injected into the LLM prompt\n   via TextSearchProvider — see NEXT.md section D)\n");
    }
}

async Task RunRagPipeline(string modelsDir, string ollamaUrl, string modelName, string? docsDir, int topK, bool useOllama, string? singleShotText)
{
    Console.WriteLine("=== NivaraChat — RAG Pipeline ===\n");

    if (!useOllama)
    {
        Console.WriteLine("Error: --rag requires --ollama for LLM generation.");
        return;
    }

    var minilmDir = Path.Combine(GetRepoRoot(), "samples", "data", "minilm");
    if (!Directory.Exists(minilmDir))
    {
        Console.WriteLine($"MiniLM model not found at: {minilmDir}");
        Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
        return;
    }

    Console.WriteLine("Loading MiniLM embedding model...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
    sw.Stop();
    Console.WriteLine($"  Dimensions: {generator.EmbeddingDimension}");
    Console.WriteLine($"  Loaded in:  {sw.ElapsedMilliseconds} ms\n");

    var vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(
        new() { EmbeddingGenerator = generator });
    var collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
    await collection.EnsureCollectionExistsAsync();

    var repoRoot = GetRepoRoot();
    var resolvedDocsDir = docsDir ?? Path.Combine(repoRoot, "docs");
    var mdFiles = Directory.Exists(resolvedDocsDir)
        ? Directory.GetFiles(resolvedDocsDir, "*.md")
        : [];

    var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
    if (File.Exists(readmePath))
        mdFiles = [.. mdFiles, readmePath];

    if (mdFiles.Length == 0)
    {
        Console.WriteLine($"No markdown files found in {resolvedDocsDir}");
        return;
    }

    Console.WriteLine($"Indexing {mdFiles.Length} markdown files...");
    sw.Restart();
    await DocumentChunker.IndexMarkdownFiles(collection, mdFiles);
    sw.Stop();
    Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    Console.WriteLine("Ollama connected.\n");

    Console.WriteLine($"Top-K: {topK} chunks per query. Type a question (or 'quit' to exit):\n");

    async Task RunQuery(string query)
    {
        sw.Restart();
        var searchResults = new List<(DocumentChunk Record, double? Score)>();
        await foreach (var result in collection.SearchAsync(query, topK))
        {
            searchResults.Add((result.Record, result.Score));
        }
        var retrievalMs = sw.ElapsedMilliseconds;

        Console.WriteLine($"\n  Retrieved {searchResults.Count} chunks ({retrievalMs} ms):");
        for (int i = 0; i < searchResults.Count; i++)
            Console.WriteLine($"    #{i + 1}  {searchResults[i].Score:F4}  [{searchResults[i].Record.Source}]  \"{searchResults[i].Record.Text[..Math.Min(100, searchResults[i].Record.Text.Length)]}...\"");

        var context = string.Join("\n\n", searchResults.Select(r => r.Record.Text));
        var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";

        sw.Restart();
        var response = await chatClient.GetResponseAsync(prompt, new ChatOptions { ModelId = modelName });
        var llmMs = sw.ElapsedMilliseconds;

        Console.WriteLine($"\n  LLM response ({llmMs} ms):");
        Console.WriteLine($"  {response.Text}");
        Console.WriteLine();
    }

    if (singleShotText != null)
    {
        await RunQuery(singleShotText);
    }
    else
    {
        while (true)
        {
            Console.Write("> ");
            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query) || query == "quit") break;
            await RunQuery(query);
        }
    }
}

async Task RunRagAgentPipeline(string modelsDir, string ollamaUrl, string modelName, string? docsDir, int topK, bool useOllama, string? singleShotText)
{
    Console.WriteLine("=== NivaraChat — RAG Agent (TextSearchProvider) ===\n");

    if (!useOllama)
    {
        Console.WriteLine("Error: --rag-agent requires --ollama for LLM generation.");
        return;
    }

    var minilmDir = Path.Combine(GetRepoRoot(), "samples", "data", "minilm");
    if (!Directory.Exists(minilmDir))
    {
        Console.WriteLine($"MiniLM model not found at: {minilmDir}");
        Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
        return;
    }

    Console.WriteLine("Loading MiniLM embedding model...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
    sw.Stop();
    Console.WriteLine($"  Dimensions: {generator.EmbeddingDimension}");
    Console.WriteLine($"  Loaded in:  {sw.ElapsedMilliseconds} ms\n");

    var vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(
        new() { EmbeddingGenerator = generator });
    var collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
    await collection.EnsureCollectionExistsAsync();

    var repoRoot = GetRepoRoot();
    var resolvedDocsDir = docsDir ?? Path.Combine(repoRoot, "docs");
    var mdFiles = Directory.Exists(resolvedDocsDir)
        ? Directory.GetFiles(resolvedDocsDir, "*.md")
        : [];

    var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
    if (File.Exists(readmePath))
        mdFiles = [.. mdFiles, readmePath];

    if (mdFiles.Length == 0)
    {
        Console.WriteLine($"No markdown files found in {resolvedDocsDir}");
        return;
    }

    Console.WriteLine($"Indexing {mdFiles.Length} markdown files...");
    sw.Restart();
    await DocumentChunker.IndexMarkdownFiles(collection, mdFiles);
    sw.Stop();
    Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    Console.WriteLine("Ollama connected.\n");

    var textSearch = new TextSearchProvider(
        async (query, ct) =>
        {
            var results = new List<TextSearchProvider.TextSearchResult>();
            await foreach (var result in collection.SearchAsync(query, topK, cancellationToken: ct))
            {
                results.Add(new TextSearchProvider.TextSearchResult
                {
                    Text = result.Record.Text,
                    SourceName = result.Record.Source
                });
            }
            return results;
        },
        new TextSearchProviderOptions
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
            RecentMessageMemoryLimit = 2
        });

    var baseAgent = new ChatClientAgent(chatClient,
        name: "NivaraRagAgent",
        instructions: "You are a helpful assistant that answers questions about the Nivara project using the provided context. Always base your answers on the retrieved context.");

    var agent = new AIAgentBuilder(baseAgent)
        .UseAIContextProviders(textSearch)
        .Build();

    Console.WriteLine($"Top-K: {topK} chunks. TextSearchProvider auto-injects context before each LLM call.");
    Console.WriteLine("Type a question (or 'quit' to exit):\n");

    async Task RunQuery(string query)
    {
        var run = await InProcessExecution.RunAsync(
            new WorkflowBuilder(agent).WithOutputFrom(agent).Build(),
            query);

        Console.WriteLine("\n--- RAG Agent Response ---");
        foreach (var evt in run.NewEvents)
        {
            switch (evt)
            {
                case AgentResponseUpdateEvent updateEvt:
                    if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text))
                        Console.Write(text);
                    break;
                case AgentResponseEvent agentEvt:
                    if (agentEvt.Data is string data && !string.IsNullOrEmpty(data))
                        Console.WriteLine(data);
                    break;
            }
        }
        Console.WriteLine("\n");
    }

    if (singleShotText != null)
    {
        await RunQuery(singleShotText);
    }
    else
    {
        while (true)
        {
            Console.Write("> ");
            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query) || query == "quit") break;
            await RunQuery(query);
        }
    }
}
