using NivaraChat.Helpers;
using NivaraChat.Training;

namespace NivaraChat.Modes;

/// <summary>
/// --train / --intent-train: trains the small Nivara models used by every other mode.
/// </summary>
public static class TrainingMode
{
    public static void RunTrain(ModeContext ctx, bool fromCli)
    {
        Console.WriteLine("=== NivaraChat Model Training ===\n");
        var silk = ctx.UseSilk ? "Silk (OpenCL)" : "CPU";
        Console.WriteLine($"Backend: {(ctx.UseSilk ? "Silk (OpenCL) for sentiment/validator/agents-validator/intent/entity" : "CPU")}\n");
        BackendMatrixPrinter.Print(new BackendMatrix(
            new ComponentBackend("embeddings", "not used", "n/a"),
            new ComponentBackend("llm", "not used", "n/a"),
            [
                new ComponentBackend("sentiment", silk, "F32"),
                new ComponentBackend("entity", silk, "F32"),
                new ComponentBackend("validator", silk, "F32"),
                new ComponentBackend("agents-validator", silk, "F32"),
            ],
            new ComponentBackend("laya-encoder", "not used", "n/a"),
            new ComponentBackend("laya-head-training", "not used", "n/a")));
        Directory.CreateDirectory(ctx.ModelsDir);

        Console.WriteLine("[1/4] Training sentiment classifier...");
        SentimentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir, useSilk: ctx.UseSilk);

        Console.WriteLine("\n[2/4] Training entity extractor...");
        EntityTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir, useSilk: ctx.UseSilk);

        Console.WriteLine("\n[3/4] Training workflow validator...");
        ValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir, useSilk: ctx.UseSilk);

        Console.WriteLine("\n[4/4] Training agents validator...");
        AgentsValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir, useSilk: ctx.UseSilk);

        Console.WriteLine("\n=== Training complete! ===");
        if (fromCli)
            Console.WriteLine("Run with --workflow or --agents to test the pipeline, or --interactive for chat.");
        else
            Console.WriteLine("Returning to main menu...");
    }

    public static void RunIntentTrain(ModeContext ctx, bool fromCli)
    {
        Console.WriteLine("=== NivaraChat Intent Classifier Training ===\n");
        Directory.CreateDirectory(ctx.ModelsDir);
        var silkIntent = ctx.UseSilk ? "Silk (OpenCL)" : "CPU";
        BackendMatrixPrinter.Print(new BackendMatrix(
            new ComponentBackend("embeddings", "not used", "n/a"),
            new ComponentBackend("llm", "not used", "n/a"),
            [new ComponentBackend("intent", silkIntent, "F32")],
            new ComponentBackend("laya-encoder", "not used", "n/a"),
            new ComponentBackend("laya-head-training", "not used", "n/a")));
        IntentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir, useSilk: ctx.UseSilk);
        Console.WriteLine("\n=== Intent training complete! ===");
        if (fromCli)
            Console.WriteLine("Run with --intent to test the intent routing.");
        else
            Console.WriteLine("Returning to main menu...");
    }
}
