using NivaraChat.Helpers;
using NivaraChat.Shared;

namespace NivaraChat.Modes;

/// <summary>
/// --online-learning: classify with the intent model, let the LLM correct low-confidence
/// classifications, collect the corrections into a retrain buffer, and retrain when full.
/// </summary>
public static class OnlineLearningMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Online Learning from LLM Feedback ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "intent_model.json")))
        {
            Console.WriteLine("Intent model not found. Run with --intent-train first.");
            return;
        }

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --online-learning requires --ollama for LLM feedback.");
            return;
        }

        Console.WriteLine("Loading intent model...");
        var (intentModel, intentTok) = ModeHelpers.LoadIntentModel(ctx.ModelsDir);
        SilkTextClassifier? intentSilk = ctx.UseSilk ? SilkTextClassifier.TryCreate(intentModel, 20, 5) : null;
        var silk = ctx.UseSilk ? "Silk (OpenCL)" : "CPU";
        BackendMatrixPrinter.Print(new BackendMatrix(
            new ComponentBackend("embeddings", "not used", "n/a"),
            new ComponentBackend("llm", ctx.UseOllama ? $"Ollama ({ctx.ModelName})" : "not used", "n/a"),
            [new ComponentBackend("intent", intentSilk is not null ? silk : "CPU", "F32")],
            new ComponentBackend("laya-encoder", "not used", "n/a"),
            new ComponentBackend("laya-head-training", "not used", "n/a")));
        Console.WriteLine("Intent model loaded.\n");

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = LlmClientFactory.Create(ctx)
            ?? throw new InvalidOperationException("This mode requires --ollama.");
        Console.WriteLine("Ollama connected.\n");

        var collector = new FeedbackCollector(
            intentModel, intentTok, chatClient, ctx.ModelsDir,
            threshold: ctx.ConfidenceThreshold, retrainThreshold: 10,
            useSilk: ctx.UseSilk, silk: intentSilk);

        Console.WriteLine($"Threshold: {ctx.ConfidenceThreshold:F2} — below this, LLM provides corrected intent.");
        Console.WriteLine($"Retrain buffer: {collector.BufferCount}/10 examples.");
        Console.WriteLine("Type a message to classify (or 'quit' to exit, 'status' to see buffer):\n");

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input == "quit") break;

            if (input == "status")
            {
                Console.WriteLine($"  Buffer: {collector.BufferCount}/10 examples collected, {collector.TotalCollected} total.\n");
                continue;
            }

            var (intent, confidence, collected) = await collector.ClassifyAsync(input);

            Console.WriteLine($"  Intent: {intent} (confidence: {confidence:F3})" +
                (collected ? " [LLM-corrected, added to training buffer]" : ""));

            if (collector.ShouldRetrain())
            {
                Console.WriteLine("\n  Buffer full — triggering incremental retrain...");
                var (newModel, newTok) = collector.Retrain();
                intentModel.Dispose();
                intentSilk?.Dispose();
                intentModel = newModel;
                intentTok = newTok;
                intentSilk = ctx.UseSilk ? SilkTextClassifier.TryCreate(newModel, 20, 5) : null;
                collector = new FeedbackCollector(
                    newModel, newTok, chatClient, ctx.ModelsDir,
                    threshold: ctx.ConfidenceThreshold, retrainThreshold: 10,
                    useSilk: ctx.UseSilk, silk: intentSilk);
                Console.WriteLine("  Retrain complete. Model updated.\n");
            }
            Console.WriteLine();
        }

        intentModel.Dispose();
        intentSilk?.Dispose();
        Console.WriteLine($"\nDone. {collector.TotalCollected} examples collected, {collector.BufferCount} pending.");
    }
}
