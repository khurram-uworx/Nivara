using NivaraChat.Modes;
using NivaraChat.Qwen;
using NivaraChat.SmolLM;
using NivaraChat.Transformer;

const string DefaultOllamaUrl = "http://localhost:11434";
const string DefaultModel = "llama3.2";
const string ModelsDir = "samples/data/nivarachat";

var modelsDir = Path.Combine(ModeHelpers.GetRepoRoot(), ModelsDir);

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

    var ctx = new ModeContext(modelsDir, ollamaUrl, modelName, useOllama, workflowText, confidenceThreshold, docsDir, topK);

    switch (mode)
    {
        case "--train":
            TrainingMode.RunTrain(ctx, fromCli: true);
            break;
        case "--workflow":
            await WorkflowMode.Run(ctx);
            break;
        case "--interactive":
            await AgentsMode.Run(ctx with { SingleShotText = null });
            break;
        case "--agents":
            await AgentsMode.Run(ctx);
            break;
        case "--handoff":
            await HandoffMode.Run(ctx);
            break;
        case "--tools":
            await ToolsMode.Run(ctx);
            break;
        case "--critic":
            await CriticMode.Run(ctx);
            break;
        case "--embed":
            EmbeddingMode.Run();
            break;
        case "--rag":
            await RagMode.Run(ctx);
            break;
        case "--rag-agent":
            await RagAgentMode.Run(ctx);
            break;
        case "--intent-train":
            TrainingMode.RunIntentTrain(ctx, fromCli: true);
            break;
        case "--intent":
            await IntentMode.Run(ctx);
            break;
        case "--online-learning":
            await OnlineLearningMode.Run(ctx);
            break;
        case "--tinyshakespeare":
            TransformerMode.Run(args.Skip(1).ToArray());
            break;
        case "--smollm":
            await SmollmMode.Run(args.Skip(1).ToArray());
            break;
        case "--qwen":
            await QwenMode.Run(args.Skip(1).ToArray());
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
                TrainingMode.RunTrain(new ModeContext(modelsDir, DefaultOllamaUrl, DefaultModel, false, null, 0.8f, null, 3), fromCli: false);
                Console.WriteLine();
                break;
            case "2":
                var (useOllama2, url2, model2) = AskOllama();
                await WorkflowMode.Run(new ModeContext(modelsDir, url2, model2, useOllama2, null, 0.8f, null, 3));
                Console.WriteLine();
                break;
            case "3":
                var (useOllama3, url3, model3) = AskOllama();
                await AgentsMode.Run(new ModeContext(modelsDir, url3, model3, useOllama3, null, 0.8f, null, 3));
                Console.WriteLine();
                break;
            case "4":
                TransformerMode.RunInteractive();
                Console.WriteLine();
                break;
            case "5":
                await SmollmMode.RunInteractive();
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
    Console.WriteLine("  4. TinyShakespeare - Train/serve a batched transformer as IChatClient");
    Console.WriteLine("  5. SmolLM        - Serve the pretrained SmolLM-135M-Instruct causal LM as IChatClient");
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
    Console.WriteLine("  --intent-train       Train the 5-class intent classifier");
    Console.WriteLine("  --intent             Intent routing: classify input, route to specialist executor");
    Console.WriteLine("  --online-learning    Online learning: classify with LLM fallback, collect feedback, retrain");
    Console.WriteLine("  --tinyshakespeare    TinyShakespeare: train/serve a batched transformer as IChatClient (see --tinyshakespeare --help)");
    Console.WriteLine("  --smollm             SmolLM: serve the pretrained SmolLM-135M-Instruct causal LM as IChatClient (see --smollm --help)");
    Console.WriteLine("  --qwen               Qwen: native function calling with Qwen2.5-0.5B-Instruct (see --qwen --help)");
    Console.WriteLine("\nOptions:");
    Console.WriteLine("  --ollama <url>       Ollama endpoint (default: http://localhost:11434)");
    Console.WriteLine("  --model <name>       Model name (default: llama3.2)");
    Console.WriteLine("  --text \"<message>\"   Single-shot: run pipeline on one message and exit");
    Console.WriteLine("  --threshold <float>  Confidence threshold for --handoff / --online-learning (default: 0.8)");
    Console.WriteLine("  --docs-dir <path>    Documents directory for --rag (default: docs/ + README.md)");
    Console.WriteLine("  --top-k <int>        Number of chunks to retrieve for --rag (default: 3)");
    Console.WriteLine("  --tinyshakespeare options: --n-embd --n-layer --block-size --n-head --epochs --batch-size --lr");
    Console.WriteLine("                        --vocab-size --temperature --max-new-tokens --samples --seed --data");
    Console.WriteLine("                        --prompt --save --load --no-di-demo --help");
    Console.WriteLine("  --smollm options:    --model-dir --precision --text --max-new-tokens --help");
}