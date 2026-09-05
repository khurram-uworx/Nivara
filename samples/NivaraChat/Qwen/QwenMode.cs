using Microsoft.Extensions.AI;
using Nivara.Samples;
using System.Diagnostics;
using System.Numerics;

namespace NivaraChat.Qwen;

/// <summary>
/// The <c>--qwen</c> mode (issue #382 Phase 3): serves Qwen2.5-0.5B-Instruct as an
/// <see cref="IChatClient"/> via <see cref="QwenChatClient{T}"/>. Sub-modes:
/// <c>tools-weather</c> runs the native function-calling loop (<c>&lt;tool_call&gt;</c> emitted by
/// the model, executed by <c>FunctionInvokingChatClient</c>, result fed back as
/// <c>&lt;tool_response&gt;</c>, capped at <see cref="MaxToolLoopIterations"/>), while
/// <c>chat</c> / <c>plain</c> are plain text (streaming) with no tools. The <c>--smollm</c> mode
/// is untouched.
/// </summary>
public static class QwenMode
{
    const int MaxToolLoopIterations = 3;

    public static async Task Run(string[] args)
    {
        var options = ParseArgs(args);
        if (options.ShowHelp)
        {
            PrintHelp();
            return;
        }

        if (options.Mode is not ("tools-weather" or "chat" or "plain"))
        {
            Console.WriteLine("Usage: --qwen {tools-weather|chat|plain} [--text \"...\"] [options]");
            PrintHelp();
            return;
        }

        var modelDir = options.ModelDir ?? Path.Combine(GetRepoRoot(), "samples", "data", "qwen2.5-0.5b-instruct");
        if (!File.Exists(Path.Combine(modelDir, "model.safetensors"))
            || !File.Exists(Path.Combine(modelDir, "tokenizer.json")))
        {
            Console.WriteLine($"Qwen model files not found in '{modelDir}'. Download Qwen2.5-0.5B-Instruct first:");
            Console.WriteLine("  hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/qwen2.5-0.5b-instruct");
            return;
        }

        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"),
            Path.Combine(modelDir, "merges.txt"),
            tokenizerJsonPath: Path.Combine(modelDir, "tokenizer.json"));

        Console.WriteLine($"Loading Qwen2.5-0.5B-Instruct ({options.Precision})...");
        var loadSw = Stopwatch.StartNew();
        if (options.Precision == "bf16")
            await Execute<BFloat16>(config, tokenizer, modelDir, options);
        else
            await Execute<float>(config, tokenizer, modelDir, options);
        loadSw.Stop();
        Console.WriteLine($"Model ready in {loadSw.ElapsedMilliseconds} ms.");
    }

    static async Task Execute<T>(
        LlamaConfig config,
        Gpt2BpeTokenizer tokenizer,
        string modelDir,
        QwenOptions options)
        where T : struct, IFloatingPointIeee754<T>
    {
        var tensors = SafeTensorsLoader.Read<T>(Path.Combine(modelDir, "model.safetensors"));
        var model = LlamaLoader.Load<T, T>(config, tensors);

        using var client = new QwenChatClient<T>(
            model, tokenizer, config,
            maxNewTokens: options.MaxNewTokens,
            temperature: options.Temperature,
            topP: options.HasTopP ? options.TopP : null,
            seed: options.Seed,
            useKvCache: options.UseKvCache,
            knownToolNames: options.Mode == "tools-weather" ? [QwenSampleTools.WeatherToolName] : null);

        if (options.Mode == "tools-weather")
        {
            await RunToolsWeather(client, options.Text);
            return;
        }

        if (options.Mode == "plain" || !string.IsNullOrEmpty(options.Text))
        {
            await RunSingleTurn(client, options.Text ?? "The capital of France is");
            return;
        }

        await RunRepl(client);
    }

    /// <summary>
    /// Native function-calling demo: the model generates a <c>&lt;tool_call&gt;</c>, the framework
    /// executes <c>GetWeather</c>, the result is fed back as a <c>&lt;tool_response&gt;</c> user
    /// turn, and the model produces the final answer. The loop is capped at
    /// <see cref="MaxToolLoopIterations"/> so a model that never answers still exits cleanly.
    /// </summary>
    static async Task RunToolsWeather<T>(QwenChatClient<T> client, string? initialText)
        where T : struct, IFloatingPointIeee754<T>
    {
        var weather = QwenSampleTools.CreateWeatherTool();
        var history = new List<ChatMessage>
        {
            new(ChatRole.System, QwenChatTemplate.BuildToolsSystemMessage([weather])),
        };
        using var loop = new FunctionInvokingChatClient(client)
        {
            MaximumIterationsPerRequest = MaxToolLoopIterations,
        };

        Console.WriteLine($"Native Qwen tool calling (GetWeather; loop cap {MaxToolLoopIterations}).\n");

        string? prompt = initialText;
        while (true)
        {
            if (prompt is null)
            {
                Console.Write("You: ");
                var input = Console.ReadLine()?.Trim();
                if (input is null || input is "quit" or "exit")
                    break;
                if (input.Length == 0)
                    continue;
                prompt = input;
            }

            history.Add(new ChatMessage(ChatRole.User, prompt));
            Console.WriteLine($"You: {prompt}");

            var sw = Stopwatch.StartNew();
            var response = await loop.GetResponseAsync(history, new ChatOptions { Tools = [weather] });
            sw.Stop();

            foreach (var message in response.Messages)
            {
                history.Add(message);
                PrintTurn(message);
            }
            Console.WriteLine($"[{response.Messages.Count} turn(s) in {sw.ElapsedMilliseconds} ms]\n");

            var final = response.Messages.LastOrDefault(m => m.Role == ChatRole.Assistant);
            if (final is null || string.IsNullOrWhiteSpace(final.Text))
                Console.WriteLine("(no final text answer produced — tool loop hit the iteration cap)");

            if (initialText is not null)
                break;
            prompt = null;
        }
    }

    static void PrintTurn(ChatMessage message)
    {
        var calls = message.Contents?.OfType<FunctionCallContent>().ToArray() ?? [];
        if (message.Role == ChatRole.Tool)
        {
            var result = message.Contents?.OfType<FunctionResultContent>().LastOrDefault()?.Result?.ToString() ?? "";
            Console.WriteLine($"[tool] {result}");
        }
        else if (calls.Length > 0)
        {
            foreach (var call in calls)
            {
                var argsText = call.Arguments is null
                    ? "{}"
                    : string.Join(", ", call.Arguments.Select(kv => $"{kv.Key}: {kv.Value}"));
                Console.WriteLine($"[assistant → {call.Name}({argsText})]");
            }
        }
        else if (message.Role == ChatRole.Assistant && !string.IsNullOrWhiteSpace(message.Text))
        {
            Console.WriteLine($"Qwen: {message.Text}");
        }
    }

    static async Task RunSingleTurn<T>(QwenChatClient<T> client, string text)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine($"\nYou: {text}");
        Console.Write("Qwen: ");
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

    static async Task RunRepl<T>(QwenChatClient<T> client)
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
            Console.Write("Qwen: ");

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

    static string TokensPerSecond(int tokens, TimeSpan elapsed)
    {
        if (tokens <= 0 || elapsed.TotalSeconds <= 0) return "0 tok/s";
        return $"{tokens / elapsed.TotalSeconds:F1} tok/s";
    }

    static string GetRepoRoot()
        => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    static QwenOptions ParseArgs(string[] args)
    {
        var options = new QwenOptions();
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--model-dir" && i + 1 < args.Length) options.ModelDir = args[++i];
            else if (args[i] == "--precision" && i + 1 < args.Length) options.Precision = args[++i];
            else if (args[i] == "--max-new-tokens" && i + 1 < args.Length) options.MaxNewTokens = int.Parse(args[++i]);
            else if (args[i] == "--temperature" && i + 1 < args.Length) options.Temperature = float.Parse(args[++i]);
            else if (args[i] == "--top-p" && i + 1 < args.Length) { options.TopP = float.Parse(args[++i]); options.HasTopP = true; }
            else if (args[i] == "--seed" && i + 1 < args.Length) options.Seed = int.Parse(args[++i]);
            else if (args[i] == "--kv-cache") options.UseKvCache = true;
            else if (args[i] == "--no-kv-cache") options.UseKvCache = false;
            else if (args[i] == "--text" && i + 1 < args.Length) options.Text = args[++i];
            else if (args[i] is "-h" or "--help") options.ShowHelp = true;
        }
        options.Mode = args.Length > 0 ? args[0] : "";
        return options;
    }

    static void PrintHelp()
    {
        Console.WriteLine("""
            NivaraChat --qwen — native function calling with Qwen2.5-0.5B-Instruct (issue #382)

            Usage:
              dotnet run --project samples/NivaraChat -- --qwen tools-weather [--text "prompt"]
              dotnet run --project samples/NivaraChat -- --qwen chat         [--text "prompt"]
              dotnet run --project samples/NivaraChat -- --qwen plain        [--text "prompt"]

            Sub-modes:
              tools-weather  Native <tool_call> → GetWeather → <tool_response> → final answer,
                             loop capped at 3 iterations
              chat           Interactive REPL (or single prompt with --text)
              plain          Single-shot plain-text reply, no REPL

            Options:
              --model-dir <path>     Model directory (default samples/data/qwen2.5-0.5b-instruct)
              --precision f32|bf16   Compute precision (default f32; BF16 weights upcast to F32)
              --max-new-tokens <n>   Max tokens to generate per turn (default 128)
              --temperature <t>      Sampling temperature; >0 enables sampling (default 0 = greedy)
              --top-p <p>            Nucleus (top-p) cutoff for sampling, 0-1 (default 1 = off)
              --seed <n>             RNG seed for reproducible sampling (default 0)
              --kv-cache             Use the KV cache for faster generation (default)
              --no-kv-cache          Re-run the full forward per token (no cache)
              --text <string>        Single-shot prompt (skips REPL)
              -h, --help             Show this help
            """);
    }

    sealed class QwenOptions
    {
        public string Mode { get; set; } = "";
        public string? ModelDir { get; set; }
        public string Precision { get; set; } = "f32";
        public int MaxNewTokens { get; set; } = 128;
        public float? Temperature { get; set; }
        public float TopP { get; set; } = 1f;
        public bool HasTopP { get; set; }
        public int? Seed { get; set; }
        public bool UseKvCache { get; set; } = true;
        public string? Text { get; set; }
        public bool ShowHelp { get; set; }
    }
}