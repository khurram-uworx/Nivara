using Microsoft.Extensions.AI;
using Nivara.Samples;
using System.Diagnostics;
using System.Numerics;

namespace NivaraChat.SmolLM;

/// <summary>
/// The <c>--smollm</c> mode (Stage A): loads the pretrained SmolLM-135M-Instruct causal LM and
/// serves it as an <see cref="IChatClient"/> via <see cref="SmolLMChatClient{T}"/>. Two sub-modes:
/// <c>chat</c> (interactive REPL or single prompt) and <c>plain</c> (single-shot, no REPL).
/// Plain-chat only in Stage A (no tools).
/// </summary>
public static class SmollmMode
{
    public static async Task Run(string[] args)
    {
        var options = ParseArgs(args);
        if (options.ShowHelp)
        {
            PrintHelp();
            return;
        }

        await Execute(options);
    }

    /// <summary>Interactive-menu entry point: asks for a prompt, defaults every other answer.</summary>
    public static async Task RunInteractive()
    {
        Console.WriteLine("\n=== SmolLM-135M-Instruct — causal LM as IChatClient ===\n");

        Console.Write("Prompt? (blank = \"The capital of France is\"): ");
        var prompt = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(prompt)) prompt = "The capital of France is";

        Console.WriteLine();
        await Execute(new SmollmOptions { Mode = "chat", Text = prompt });
    }

    static SmollmOptions ParseArgs(string[] args)
    {
        var options = new SmollmOptions();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model-dir": options.ModelDir = args[++i]; break;
                case "--precision":
                    options.Precision = args[++i].ToLowerInvariant() switch
                    {
                        "bf16" or "bfloat16" => "bf16",
                        _ => "f32",
                    };
                    break;
                case "--max-new-tokens": options.MaxNewTokens = int.Parse(args[++i]); break;
                case "--text": options.Text = args[++i]; break;
                case "--help": options.ShowHelp = true; break;
                case "-h": options.ShowHelp = true; break;
                default:
                    if (options.Mode.Length == 0) options.Mode = args[i];
                    break;
            }
        }
        return options;
    }

    static async Task Execute(SmollmOptions options)
    {
        if (options.Mode is not ("chat" or "plain"))
        {
            Console.WriteLine("Usage: --smollm {chat|plain} [--text \"...\"] [options]");
            PrintHelp();
            return;
        }

        var modelDir = options.ModelDir ?? Path.Combine(GetRepoRoot(), "samples", "data", "smollm-135m");
        if (!File.Exists(Path.Combine(modelDir, "model.safetensors")))
        {
            Console.WriteLine($"Model files not found in '{modelDir}'. Download SmolLM-135M-Instruct first:");
            Console.WriteLine("  hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m");
            return;
        }

        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"), Path.Combine(modelDir, "merges.txt"));

        Console.WriteLine($"Loading SmolLM-135M-Instruct ({options.Precision})...");
        var loadSw = Stopwatch.StartNew();
        if (options.Precision == "bf16")
            await Run<BFloat16>(config, tokenizer, modelDir, options);
        else
            await Run<float>(config, tokenizer, modelDir, options);
        loadSw.Stop();
        Console.WriteLine($"Model ready in {loadSw.ElapsedMilliseconds} ms.");
    }

    static async Task Run<T>(
        LlamaConfig config,
        Gpt2BpeTokenizer tokenizer,
        string modelDir,
        SmollmOptions options)
        where T : struct, IFloatingPointIeee754<T>
    {
        var tensors = SafeTensorsLoader.Read<T>(Path.Combine(modelDir, "model.safetensors"));
        var model = LlamaLoader.Load<T, T>(config, tensors);

        using var client = new SmolLMChatClient<T>(model, tokenizer, config, options.MaxNewTokens);

        if (options.Mode == "plain" || !string.IsNullOrEmpty(options.Text))
        {
            await RunSingleTurn(client, options.Text ?? "The capital of France is");
            return;
        }

        await RunRepl(client);
    }

    static async Task RunSingleTurn<T>(SmolLMChatClient<T> client, string text)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine($"\nYou: {text}");
        Console.Write("SmolLM: ");
        var sw = Stopwatch.StartNew();
        await foreach (var update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, text)]))
        {
            if (update.Text is not null)
                Console.Write(update.Text);
        }
        sw.Stop();
        Console.WriteLine($"\n\n[streamed in {sw.ElapsedMilliseconds} ms]\n");
    }

    static async Task RunRepl<T>(SmolLMChatClient<T> client)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine("Interactive chat (type 'quit' to exit).\n");
        var history = new List<ChatMessage>();
        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;

            history.Add(new ChatMessage(ChatRole.User, input));
            Console.Write("SmolLM: ");

            var updates = new List<ChatResponseUpdate>();
            await foreach (var update in client.GetStreamingResponseAsync(history))
            {
                if (update.Text is not null)
                {
                    Console.Write(update.Text);
                    updates.Add(update);
                }
            }
            Console.WriteLine("\n");
            history.AddMessages(updates);
        }
    }

    static string GetRepoRoot()
        => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraChat --smollm — serve the pretrained SmolLM-135M-Instruct causal LM as an IChatClient

            Usage:
              dotnet run --project samples/NivaraChat -- --smollm chat    [--text "prompt"]
              dotnet run --project samples/NivaraChat -- --smollm plain   [--text "prompt"]

            Sub-modes (Stage A):
              chat   Interactive REPL (default) or single prompt with --text
              plain  Single-shot plain-text reply, no REPL

            Options:
              --model-dir <path>     Model directory (default samples/data/smollm-135m)
              --precision f32|bf16   Compute precision (default f32)
              --max-new-tokens <n>   Max tokens to generate (default 64)
              --text <string>        Single-shot prompt (skips REPL)
              -h, --help             Show this help
            """);
    }

    sealed class SmollmOptions
    {
        public string Mode { get; set; } = "";
        public string? ModelDir { get; set; }
        public string Precision { get; set; } = "f32";
        public int MaxNewTokens { get; set; } = 64;
        public string? Text { get; set; }
        public bool ShowHelp { get; set; }
    }
}
