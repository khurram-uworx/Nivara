using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.Samples;

namespace NivaraChat.Modes;

/// <summary>
/// Shared helpers hoisted out of Program.cs's mode runners: repo-root resolution, the four
/// trained-model loaders, and the interactive single-shot/loop/print helpers used by the
/// agents-style pipelines.
/// </summary>
internal static class ModeHelpers
{
    /// <summary>Resolves the repository root from the app base directory (bin/Release/TFM = 5 levels down).</summary>
    public static string GetRepoRoot()
        => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,
        "..", "..", "..", "..", ".."));

    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) LoadValidatorModel(string modelsDir, bool useAgentsFormat = false)
    {
        var suffix = useAgentsFormat ? "agents_validator" : "validator";
        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, $"{suffix}_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 2, 40);
        ModelSerializer.Load(model, Path.Combine(modelsDir, $"{suffix}_model.json"));
        model.Eval();
        return (model, tokenizer);
    }

    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) LoadSentimentModel(string modelsDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "sentiment_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 3, 20);
        ModelSerializer.Load(model, Path.Combine(modelsDir, "sentiment_model.json"));
        model.Eval();
        return (model, tokenizer);
    }

    public static (TokenClassifierModel<float> model, TextTokenizer tokenizer) LoadEntityModel(string modelsDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "entity_tokenizer.json"));
        var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
        ModelSerializer.Load(model, Path.Combine(modelsDir, "entity_model.json"));
        model.Eval();
        return (model, tokenizer);
    }

    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) LoadIntentModel(string modelsDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "intent_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
        ModelSerializer.Load(model, Path.Combine(modelsDir, "intent_model.json"));
        model.Eval();
        return (model, tokenizer);
    }

    public static async Task RunSingleShot(Workflow workflow, string text)
    {
        var run = await InProcessExecution.RunAsync(workflow, text);
        Console.WriteLine("\n--- Agent Results ---");
        PrintAgentResults(run);
    }

    public static async Task RunLoop(Func<Workflow> workflowFactory)
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

    public static void PrintAgentResults(Run run)
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
}