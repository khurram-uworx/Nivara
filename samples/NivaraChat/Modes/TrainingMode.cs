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
        Directory.CreateDirectory(ctx.ModelsDir);

        Console.WriteLine("[1/4] Training sentiment classifier...");
        SentimentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir);

        Console.WriteLine("\n[2/4] Training entity extractor...");
        EntityTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir);

        Console.WriteLine("\n[3/4] Training workflow validator...");
        ValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir);

        Console.WriteLine("\n[4/4] Training agents validator...");
        AgentsValidatorTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir);

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
        IntentTrainer.Train(epochs: 20, batchSize: 32, numSamples: 1000, saveDir: ctx.ModelsDir);
        Console.WriteLine("\n=== Intent training complete! ===");
        if (fromCli)
            Console.WriteLine("Run with --intent to test the intent routing.");
        else
            Console.WriteLine("Returning to main menu...");
    }
}