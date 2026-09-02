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

    /// <summary>Interactive-menu entry point: prompts for a demo and generation options, loads the
    /// model, and enters the REPL.</summary>
    public static async Task RunInteractive()
    {
        Console.WriteLine("\n=== SmolLM-family causal LM as IChatClient ===\n");

        Console.WriteLine("Choose a demo:");
        Console.WriteLine("  1) Plain chat (sampling options below)");
        Console.WriteLine("  2) Weather tool-calling (GetWeather AIFunction)");
        Console.Write("Choice [1]: ");
        var choice = Console.ReadLine()?.Trim();

        if (choice == "2")
        {
            await Execute(new SmollmOptions { Mode = "tools-weather" });
            return;
        }

        Console.Write("Temperature (0 = greedy, >0 = sampling) [0]: ");
        var tempStr = Console.ReadLine()?.Trim();
        float? temp = float.TryParse(tempStr, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float t) ? t : null;

        Console.Write("Top-p nucleus cutoff (0–1, default 1) [1]: ");
        var topPStr = Console.ReadLine()?.Trim();
        float? topP = float.TryParse(topPStr, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float p) ? p : null;

        Console.Write("Seed (blank = 0): ");
        var seedStr = Console.ReadLine()?.Trim();
        int? seed = int.TryParse(seedStr, out int s) ? s : null;

        Console.Write("KV cache? (y/n, default y) [y]: ");
        var cacheStr = Console.ReadLine()?.Trim();
        bool useKvCache = !string.Equals(cacheStr, "n", StringComparison.OrdinalIgnoreCase);

        Console.Write("Max tokens [64]: ");
        var maxStr = Console.ReadLine()?.Trim();
        bool hasMax = int.TryParse(maxStr, out int m) && m > 0;
        int maxNewTokens = hasMax ? m : 64;

        Console.WriteLine();

        await Execute(new SmollmOptions
        {
            Mode = "chat",
            Temperature = temp,
            TopP = topP ?? 1f,
            HasTopP = topP.HasValue,
            Seed = seed,
            UseKvCache = useKvCache,
            MaxNewTokens = maxNewTokens,
            HasMaxNewTokens = hasMax,
        });
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
                case "--max-new-tokens": options.MaxNewTokens = int.Parse(args[++i]); options.HasMaxNewTokens = true; break;
                case "--temperature": options.Temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                case "--top-p":
                    options.TopP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                    options.HasTopP = true;
                    break;
                case "--seed": options.Seed = int.Parse(args[++i]); break;
                case "--kv-cache": options.UseKvCache = true; break;
                case "--no-kv-cache": options.UseKvCache = false; break;
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
        if (options.Mode is not ("chat" or "plain" or "tools-weather"))
        {
            Console.WriteLine("Usage: --smollm {chat|plain|tools-weather} [--text \"...\"] [options]");
            PrintHelp();
            return;
        }

        // tools-weather uses the dedicated function-calling fine-tune (archit11/small-function-calling,
        // a Biggie-SmoLlm-0.15B SmolLM variant trained on Hermes function-calling data, with the base
        // model's GPT-2 BPE tokenizer). Plain chat/plain use the stock SmolLM-135M-Instruct.
        bool isTools = options.Mode == "tools-weather";
        string modelName = isTools ? "small-function-calling (Biggie-SmoLlm 0.15B)" : "SmolLM-135M-Instruct";
        string defaultDir = isTools ? "smollm-fn-135m" : "smollm-135m";
        var modelDir = options.ModelDir ?? Path.Combine(GetRepoRoot(), "samples", "data", defaultDir);
        if (!File.Exists(Path.Combine(modelDir, "model.safetensors")))
        {
            Console.WriteLine($"Model files not found in '{modelDir}'. Download {modelName} first:");
            if (isTools)
            {
                Console.WriteLine("  hf download archit11/small-function-calling config.json model.safetensors generation_config.json --local-dir samples/data/smollm-fn-135m");
                Console.WriteLine("  hf download nisten/Biggie-SmoLlm-0.15B-Base vocab.json merges.txt tokenizer.json --local-dir samples/data/smollm-fn-135m");
            }
            else
            {
                Console.WriteLine($"  hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m");
            }
            return;
        }

        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"), Path.Combine(modelDir, "merges.txt"));

        Console.WriteLine($"Loading {modelName} ({options.Precision})...");
        var loadSw = Stopwatch.StartNew();
        if (options.Precision == "bf16")
            await Run<BFloat16>(config, tokenizer, modelDir, options, modelName);
        else
            await Run<float>(config, tokenizer, modelDir, options, modelName);
        loadSw.Stop();
        Console.WriteLine($"Model ready in {loadSw.ElapsedMilliseconds} ms.");
    }

    static async Task Run<T>(
        LlamaConfig config,
        Gpt2BpeTokenizer tokenizer,
        string modelDir,
        SmollmOptions options,
        string modelName)
        where T : struct, IFloatingPointIeee754<T>
    {
        var tensors = SafeTensorsLoader.Read<T>(Path.Combine(modelDir, "model.safetensors"));
        var model = LlamaLoader.Load<T, T>(config, tensors);

        int maxNewTokens = options.Mode == "tools-weather" && !options.HasMaxNewTokens
            ? 256
            : options.MaxNewTokens;

        // The Biggie-SmoLlm function-calling fine-tune (archit11/small-function-calling) needs the
        // lenient parser in BiggieChatClient; everything else (SmolLM/SmolLM2 Hermes) uses SmolLMChatClient.
        bool useBiggie = IsBiggieFunctionModel(modelDir);
        using LlamaChatClientBase<T> client = useBiggie
            ? new BiggieChatClient<T>(
                model, tokenizer, config, modelName,
                maxNewTokens: maxNewTokens,
                temperature: options.Temperature,
                topP: options.HasTopP ? options.TopP : null,
                seed: options.Seed,
                useKvCache: options.UseKvCache)
            : new SmolLMChatClient<T>(
                model, tokenizer, config, modelName,
                maxNewTokens: maxNewTokens,
                temperature: options.Temperature,
                topP: options.HasTopP ? options.TopP : null,
                seed: options.Seed,
                useKvCache: options.UseKvCache);

        if (options.Mode == "tools-weather")
        {
            await RunToolsWeather(client, options);
            return;
        }

        if (options.Mode == "plain" || !string.IsNullOrEmpty(options.Text))
        {
            await RunSingleTurn(client, options.Text ?? "The capital of France is");
            return;
        }

        await RunRepl(client);
    }

    /// <summary>True when the model directory holds the Biggie-SmoLlm function-calling fine-tune
    /// that ships without its own tokenizer and benefits from the lenient tool-call parser.</summary>
    static bool IsBiggieFunctionModel(string modelDir)
        => string.Equals(Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir)), "smollm-fn-135m",
            StringComparison.OrdinalIgnoreCase);

    static async Task RunSingleTurn<T>(LlamaChatClientBase<T> client, string text)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine($"\nYou: {text}");
        Console.Write("SmolLM: ");
        var sw = Stopwatch.StartNew();
        int tokens = 0;
        await foreach (var update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, text)]))
        {
            if (update.Text is not null)
            {
                Console.Write(update.Text);
                tokens++;
            }
        }
        sw.Stop();
        Console.WriteLine($"\n\n[streamed {tokens} tokens in {sw.ElapsedMilliseconds} ms — {TokensPerSecond(tokens, sw.Elapsed)}]\n");
    }

    static string TokensPerSecond(int tokens, TimeSpan elapsed)
    {
        if (tokens <= 0 || elapsed.TotalSeconds <= 0) return "0 tok/s";
        return $"{tokens / elapsed.TotalSeconds:F1} tok/s";
    }

    static async Task RunRepl<T>(LlamaChatClientBase<T> client)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine("Interactive chat (type 'quit' or 'exit' to leave).\n");
        var history = new List<ChatMessage>();
        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine()?.Trim();
            if (input is null || input.Equals("quit", StringComparison.OrdinalIgnoreCase)
                || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            if (string.IsNullOrEmpty(input))
            {
                Console.WriteLine("(empty prompt, try a question)\n");
                continue;
            }

            history.Add(new ChatMessage(ChatRole.User, input));
            Console.Write("SmolLM: ");

            var updates = new List<ChatResponseUpdate>();
            var sw = Stopwatch.StartNew();
            int tokens = 0;
            await foreach (var update in client.GetStreamingResponseAsync(history))
            {
                if (update.Text is not null)
                {
                    Console.Write(update.Text);
                    updates.Add(update);
                    tokens++;
                }
            }
            sw.Stop();
            Console.WriteLine($"\n[{tokens} tokens in {sw.ElapsedMilliseconds} ms — {TokensPerSecond(tokens, sw.Elapsed)}]\n");
            history.AddMessages(updates);
        }
    }

    /// <summary>Runs the Stage B <c>tools-weather</c> demo: wraps the client with
    /// <c>FunctionInvokingChatClient</c> and the single <c>GetWeather</c> tool, then either answers
    /// one prompt or enters a REPL. The framework drives the tool loop internally, returning only
    /// the final natural-language answer (raw <c>&lt;tool_call&gt;</c> markup is not shown).</summary>
    static async Task RunToolsWeather<T>(LlamaChatClientBase<T> client, SmollmOptions options)
        where T : struct, IFloatingPointIeee754<T>
    {
        var tools = SmollmTools.GetWeatherTools();
        using var funcClient = new FunctionInvokingChatClient(client)
        {
            // The small function-calling fine-tune tends to keep requesting the tool rather than
            // closing with a final answer; cap iterations so a runaway loop terminates quickly.
            MaximumIterationsPerRequest = 3,
        };

        if (!string.IsNullOrEmpty(options.Text))
        {
            await RunToolsWeatherSingleTurn(funcClient, tools, options.Text);
            return;
        }

        await RunToolsWeatherRepl(funcClient, tools);
    }

    static async Task RunToolsWeatherSingleTurn(FunctionInvokingChatClient funcClient, AITool[] tools, string text)
    {
        Console.WriteLine($"\nYou: {text}");
        Console.Write("SmolLM: ");
        var sw = Stopwatch.StartNew();
        var response = await funcClient.GetResponseAsync(
            [new ChatMessage(ChatRole.User, text)],
            new ChatOptions { Tools = tools });
        sw.Stop();
        Console.WriteLine(response.Text);
        Console.WriteLine($"\n[tool-call loop in {sw.ElapsedMilliseconds} ms]\n");
    }

    static async Task RunToolsWeatherRepl(FunctionInvokingChatClient funcClient, AITool[] tools)
    {
        Console.WriteLine("Ask about the weather (type 'quit' or 'exit' to leave).\n");
        var history = new List<ChatMessage>();
        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine()?.Trim();
            if (input is null || input.Equals("quit", StringComparison.OrdinalIgnoreCase)
                || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            if (string.IsNullOrEmpty(input))
            {
                Console.WriteLine("(empty prompt, try a question)\n");
                continue;
            }

            history.Add(new ChatMessage(ChatRole.User, input));
            Console.Write("SmolLM: ");
            var sw = Stopwatch.StartNew();
            var response = await funcClient.GetResponseAsync(
                history, new ChatOptions { Tools = tools });
            sw.Stop();
            Console.WriteLine(response.Text);

            if (response.Text is not null)
                history.Add(new ChatMessage(ChatRole.Assistant, response.Text));

            Console.WriteLine($"\n[tool-call loop in {sw.ElapsedMilliseconds} ms]\n");
        }
    }

    static string GetRepoRoot()
        => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraChat --smollm — serve SmolLM-family causal LMs as IChatClient

            Models:
              chat / plain     SmolLM-135M-Instruct      samples/data/smollm-135m
              tools-weather    small-function-calling     samples/data/smollm-fn-135m
                               (archit11 fine-tune of Biggie-SmoLlm-0.15B, trained on Hermes
                               function-calling data; GPT-2 BPE tokenizer from the base model)

            Usage:
              dotnet run --project samples/NivaraChat -- --smollm chat           [--text "prompt"]
              dotnet run --project samples/NivaraChat -- --smollm plain          [--text "prompt"]
              dotnet run --project samples/NivaraChat -- --smollm tools-weather  [--text "prompt"]

            Sub-modes:
              chat            Interactive REPL (default) or single prompt with --text
              plain           Single-shot plain-text reply, no REPL
              tools-weather   Native tool-calling: model emits <tool_call> → FunctionInvokingChatClient
                              invokes the deterministic GetWeather AIFunction → <tool_response> → final answer.
                              Default max tokens 256; only the final answer is shown.

            Options:
              --model-dir <path>     Model directory (default depends on sub-mode, above)
              --precision f32|bf16   Compute precision (default f32)
              --max-new-tokens <n>   Max tokens to generate (default 64; 256 for tools-weather)
              --temperature <t>      Sampling temperature; >0 enables sampling (default 0 = greedy)
              --top-p <p>            Nucleus (top-p) cutoff for sampling, 0-1 (default 1 = off)
              --seed <n>             RNG seed for reproducible sampling (default 0)
              --kv-cache             Use the KV cache for faster generation (default)
              --no-kv-cache          Re-run the full forward per token (no cache)
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
        public bool HasMaxNewTokens { get; set; }
        public float? Temperature { get; set; }
        public float TopP { get; set; } = 1f;
        public bool HasTopP { get; set; }
        public int? Seed { get; set; }
        public bool UseKvCache { get; set; } = true;
        public string? Text { get; set; }
        public bool ShowHelp { get; set; }
    }
}
