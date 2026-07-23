using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
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

for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--ollama" && i + 1 < args.Length) ollamaUrl = args[++i];
    if (args[i] == "--model" && i + 1 < args.Length) modelName = args[++i];
}

switch (mode)
{
    case "--train":
        RunTraining();
        break;
    case "--workflow":
        await RunWorkflow(ollamaUrl, modelName);
        break;
    case "--interactive":
        await RunInteractive(ollamaUrl, modelName);
        break;
    default:
        PrintUsage();
        break;
}

void RunTraining()
{
    Console.WriteLine("=== NivaraChat Model Training ===\n");
    Directory.CreateDirectory(ModelsDir);

    if (File.Exists(Path.Combine(ModelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models already exist in samples/data/nivarachat/. Delete to retrain.\n");
        return;
    }

    Console.WriteLine("[1/3] Training sentiment classifier...");
    SentimentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n[2/3] Training entity extractor...");
    EntityTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n[3/3] Training response validator...");
    ValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ModelsDir);

    Console.WriteLine("\n=== Training complete! ===");
    Console.WriteLine("Run with --workflow to test the pipeline, or --interactive for chat.");
}

async Task RunWorkflow(string ollamaUrl, string modelName)
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

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
    var agent = new ChatClientAgent(chatClient, "You are a helpful assistant. Analyze user messages and provide relevant information.");

    var sentimentExecutor = new SentimentExecutor(sentimentModel, sentimentTok);
    var entityExtractor = new EntityExtractor(entityModel, entityTok);
    var validator = new ValidatorExecutor();

    Console.WriteLine("Building workflow graph...\n");
    var workflow = new WorkflowBuilder(sentimentExecutor)
        .AddEdge(sentimentExecutor, entityExtractor)
        .AddEdge(entityExtractor, validator)
        .AddEdge(validator, agent)
        .Build();

    Console.WriteLine("Graph: SentimentExecutor -> EntityExtractor -> ValidatorExecutor -> Ollama LLM\n");
    Console.WriteLine("Type a message to analyze (or 'quit' to exit):\n");

    while (true)
    {
        Console.Write("> ");
        var input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input) || input == "quit") break;

        var run = await InProcessExecution.RunAsync(workflow, input);
        Console.WriteLine($"\n--- Result ---\n{run}\n");
    }

    sentimentModel.Dispose();
    entityModel.Dispose();
    Console.WriteLine("\nDone.");
}

async Task RunInteractive(string ollamaUrl, string modelName)
{
    Console.WriteLine("=== NivaraChat Interactive ===\n");

    if (!File.Exists(Path.Combine(ModelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    Console.WriteLine("Loading trained models...");
    var (sentimentModel, sentimentTok) = LoadSentimentModel();
    var (entityModel, entityTok) = LoadEntityModel();
    Console.WriteLine("Models loaded.\n");

    Console.WriteLine($"Connecting to Ollama at {ollamaUrl} (model: {modelName})...");
    var chatClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);

    Console.WriteLine("\nType a message (or 'quit' to exit):\n");

    while (true)
    {
        Console.Write("> ");
        var input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input) || input == "quit") break;

        var sentiment = AnalyzeSentiment(input, sentimentModel, sentimentTok);
        var entities = AnalyzeEntities(input, entityModel, entityTok);
        Console.WriteLine($"  [Nivara] Sentiment: {sentiment}");

        if (entities.Count > 0)
        {
            Console.WriteLine("  [Nivara] Entities:");
            foreach (var (type, names) in entities)
                Console.WriteLine($"    {type}: {string.Join(", ", names)}");
        }

        var prompt = $"User message: \"{input}\"\n\nClassify the sentiment of this message and identify any named entities (person, organization, date, location). Provide a brief, helpful response.";
        var response = await chatClient.GetResponseAsync(prompt);
        Console.WriteLine($"  [Ollama] {response}\n");
    }

    sentimentModel.Dispose();
    entityModel.Dispose();
    Console.WriteLine("\nDone.");
}

Dictionary<string, List<string>> AnalyzeEntities(string text, TokenClassifierModel<float> model, TextTokenizer tokenizer)
{
    var tokens = tokenizer.Encode(text, fixedLength: 20, addBosEos: false);
    var data = new float[tokens.Length];
    for (int i = 0; i < tokens.Length; i++)
        data[i] = tokens[i];
    var input = ReverseGradTensor<float>.FromMatrix(data, 1, 20, requiresGrad: false);
    var logits = model.Forward(input);

    var wordTokens = TextTokenizer.Tokenize(text);
    var entities = new Dictionary<string, List<string>>();
    var entityClasses = new[] { "O", "B-person", "B-org", "B-date", "B-location" };

    int numClasses = entityClasses.Length;
    for (int i = 0; i < Math.Min(wordTokens.Count, 20); i++)
    {
        int bestClass = 0;
        float bestVal = logits.Data[i * numClasses];
        for (int c = 1; c < numClasses; c++)
        {
            if (logits.Data[i * numClasses + c] > bestVal)
            {
                bestVal = logits.Data[i * numClasses + c];
                bestClass = c;
            }
        }

        string label = entityClasses[bestClass];
        if (label != "O")
        {
            string entityType = label.Replace("B-", "");
            if (!entities.ContainsKey(entityType))
                entities[entityType] = [];
            entities[entityType].Add(wordTokens[i]);
        }
    }
    return entities;
}

string AnalyzeSentiment(string text, TextClassifierModel<float> model, TextTokenizer tokenizer)
{
    var tokens = tokenizer.Encode(text, fixedLength: 20);
    var data = new float[tokens.Length];
    for (int i = 0; i < tokens.Length; i++)
        data[i] = tokens[i];
    var input = ReverseGradTensor<float>.FromMatrix(data, 1, 20, requiresGrad: false);
    var logits = model.Forward(input);

    int bestClass = 0;
    float bestVal = logits.Data[0];
    for (int c = 1; c < 3; c++)
    {
        if (logits.Data[c] > bestVal) { bestVal = logits.Data[c]; bestClass = c; }
    }

    return bestClass switch
    {
        0 => "Negative",
        1 => "Neutral",
        _ => "Positive"
    };
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
    Console.WriteLine("  --train          Train sentiment, entity, and validator models");
    Console.WriteLine("  --workflow       Run the Agent Framework workflow (requires Ollama)");
    Console.WriteLine("  --interactive    Interactive mode: local inference + Ollama chat");
    Console.WriteLine("\nOptions:");
    Console.WriteLine("  --ollama <url>   Ollama endpoint (default: http://localhost:11434)");
    Console.WriteLine("  --model <name>   Model name (default: llama3.2)");
}
