using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using NivaraChat;
using NivaraChat.Training;
using OllamaSharp;

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

    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--ollama") { useOllama = true; if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) ollamaUrl = args[++i]; }
        if (args[i] == "--model" && i + 1 < args.Length) modelName = args[++i];
        if (args[i] == "--text" && i + 1 < args.Length) workflowText = args[++i];
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

void PrintUsage()
{
    Console.WriteLine("Usage: NivaraChat <mode> [options]\n");
    Console.WriteLine("Modes:");
    Console.WriteLine("  --train              Train sentiment, entity, and validator models");
    Console.WriteLine("  --workflow           Run the Agent Framework workflow (Ollama optional)");
    Console.WriteLine("  --interactive        Interactive mode: agents pipeline with live input");
    Console.WriteLine("  --agents             Same as --interactive, with --text for single-shot");
    Console.WriteLine("\nOptions:");
    Console.WriteLine("  --ollama <url>       Ollama endpoint (default: http://localhost:11434)");
    Console.WriteLine("  --model <name>       Model name (default: llama3.2)");
    Console.WriteLine("  --text \"<message>\"   Single-shot: run pipeline on one message and exit");
}
