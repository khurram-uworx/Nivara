using Microsoft.Extensions.AI;
using NivaraChat.Executors;
using NivaraChat.Helpers;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --critic: writer-critic loop — the Ollama LLM writes a response, the Nivara validator scores
/// it, and the loop retries until the response passes.
/// </summary>
public static class CriticMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Writer-Critic Loop ===\n");

        if (!File.Exists(Path.Combine(ctx.ModelsDir, "sentiment_model.json")))
        {
            Console.WriteLine("Models not found. Run with --train first.");
            return;
        }

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --critic requires --ollama for the LLM writer.");
            return;
        }

        Console.WriteLine("Loading trained models...");
        var (validatorModel, validatorTok) = ModeHelpers.LoadValidatorModel(ctx.ModelsDir);
        Console.WriteLine("Models loaded.\n");

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        var critic = new CriticExecutor(validatorModel, validatorTok);
        var loop = new WriterCriticLoop(chatClient, critic);

        Console.WriteLine("Writer (Ollama) → Critic (Nivara validator) → pass/fail → retry if needed\n");

        if (ctx.SingleShotText != null)
        {
            var result = await loop.RunAsync(ctx.SingleShotText);
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
}