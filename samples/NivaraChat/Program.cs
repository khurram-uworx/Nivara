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

var mode = args.Length > 0 ? args[0] : "--workflow";
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
        RunTraining();
        break;
    case "--workflow":
        await RunWorkflow(ollamaUrl, modelName, workflowText, useOllama);
        break;
    case "--interactive":
        await RunAgents(ollamaUrl, modelName, null, useOllama);
        break;
    case "--agents":
        await RunAgents(ollamaUrl, modelName, workflowText, useOllama);
        break;
    default:
        PrintUsage();
        break;
}

void RunTraining()
{
    Console.WriteLine("=== NivaraChat Model Training ===\n");
    Directory.CreateDirectory(ModelsDir);

    Console.WriteLine("[1/4] Training sentiment classifier...");
    SentimentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n[2/4] Training entity extractor...");
    EntityTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n[3/4] Training workflow validator...");
    ValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n[4/4] Training agents validator...");
    AgentsValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n=== Training complete! ===");
    Console.WriteLine("Run with --workflow or --agents to test the pipeline, or --interactive for chat.");
}

async Task RunWorkflow(string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Workflow ===\n");

    if (!File.Exists(Path.Combine(ModelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel();
    var (entityModel, entityTok) = LoadEntityModel();
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

    Workflow workflow;
    if (llmExecutor != null)
    {
        workflow = new WorkflowBuilder(router)
            .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
            .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, validator)
            .AddEdge(validator, llmExecutor)
            .WithOutputFrom(sentimentExecutor, entityExtractor, validator, llmExecutor)
            .Build();
        Console.WriteLine("Graph: TextRouter --fan-out--> [SentimentExecutor, EntityExtractor] --fan-in--> ValidatorExecutor -> Ollama LLM\n");
    }
    else
    {
        workflow = new WorkflowBuilder(router)
            .AddFanOutEdge(router, new ExecutorBinding[] { sentimentExecutor, entityExtractor })
            .AddFanInBarrierEdge(new ExecutorBinding[] { sentimentExecutor, entityExtractor }, validator)
            .WithOutputFrom(sentimentExecutor, entityExtractor, validator)
            .Build();
        Console.WriteLine("Graph: TextRouter --fan-out--> [SentimentExecutor, EntityExtractor] --fan-in--> ValidatorExecutor\n");
    }
    if (singleShotText != null)
    {
        var run = await InProcessExecution.RunAsync(workflow, singleShotText);
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

            var run = await InProcessExecution.RunAsync(workflow, input);

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

async Task RunAgents(string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Agents ===\n");

    if (!File.Exists(Path.Combine(ModelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel();
    var (entityModel, entityTok) = LoadEntityModel();
    var (validatorModel, validatorTok) = LoadValidatorModel(useAgentsFormat: true);
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
        var llmAgent = new NivaraChatClient(new PassthroughTextModel(ollamaClient)).AsAIAgent("OllamaLLM");
        Console.WriteLine("Ollama connected.\n");

        var workflow = new WorkflowBuilder(sentimentAgent)
            .AddEdge(sentimentAgent, entityAgent)
            .AddEdge(entityAgent, validatorAgent)
            .AddEdge(validatorAgent, llmAgent)
            .WithOutputFrom(sentimentAgent, entityAgent, validatorAgent, llmAgent)
            .Build();
        Console.WriteLine("Graph: NivaraSentiment -> NivaraEntity -> NivaraValidator -> OllamaLLM\n");

        if (singleShotText != null)
            await RunSingleShot(workflow, singleShotText);
        else
            await RunLoop(workflow);
    }
    else
    {
        var workflow = new WorkflowBuilder(sentimentAgent)
            .AddEdge(sentimentAgent, entityAgent)
            .AddEdge(entityAgent, validatorAgent)
            .WithOutputFrom(sentimentAgent, entityAgent, validatorAgent)
            .Build();
        Console.WriteLine("Graph: NivaraSentiment -> NivaraEntity -> NivaraValidator\n");

        if (singleShotText != null)
            await RunSingleShot(workflow, singleShotText);
        else
            await RunLoop(workflow);
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

async Task RunLoop(Workflow workflow)
{
    Console.WriteLine("Type a message (or 'quit' to exit):\n");
    while (true)
    {
        Console.Write("> ");
        var input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input) || input == "quit") break;

        var run = await InProcessExecution.RunAsync(workflow, input);
        Console.WriteLine("\n--- Agent Results ---");
        PrintAgentResults(run);
        Console.WriteLine();
    }
}

void PrintAgentResults(Run run)
{
    var events = run.NewEvents.ToList();
    foreach (var evt in events)
    {
        switch (evt)
        {
            case AgentResponseUpdateEvent updateEvt:
                if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text))
                    Console.WriteLine($"  [{updateEvt.ExecutorId}] {text}");
                break;
            case ExecutorCompletedEvent executorEvt:
                if (executorEvt.Data?.ToString() is string data && !string.IsNullOrEmpty(data))
                    Console.WriteLine($"  [{executorEvt.ExecutorId}] {data}");
                break;
            case ExecutorFailedEvent failedEvt:
                Console.WriteLine($"  [{failedEvt.ExecutorId}] FAILED: {failedEvt}");
                break;
            case WorkflowErrorEvent errorEvt:
                Console.WriteLine($"  WORKFLOW ERROR: {errorEvt}");
                break;
        }
    }
}

(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadValidatorModel(bool useAgentsFormat = false)
{
    var suffix = useAgentsFormat ? "agents_validator" : "validator";
    var tokenizer = TextTokenizer.Load(Path.Combine(ModelsDir, $"{suffix}_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 2, 40);
    ModelSerializer.Load(model, Path.Combine(ModelsDir, $"{suffix}_model.json"));
    model.Eval();
    return (model, tokenizer);
}

(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadSentimentModel()
{
    var tokenizer = TextTokenizer.Load(Path.Combine(ModelsDir, "sentiment_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 3, 20);
    ModelSerializer.Load(model, Path.Combine(ModelsDir, "sentiment_model.json"));
    model.Eval();
    return (model, tokenizer);
}

(TokenClassifierModel<float> model, TextTokenizer tokenizer) LoadEntityModel()
{
    var tokenizer = TextTokenizer.Load(Path.Combine(ModelsDir, "entity_tokenizer.json"));
    var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
    ModelSerializer.Load(model, Path.Combine(ModelsDir, "entity_model.json"));
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
