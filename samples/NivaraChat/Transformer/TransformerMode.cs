using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;

namespace NivaraChat.Transformer;

/// <summary>
/// The <c>--tinyshakespeare</c> mode: trains a word-level batched causal
/// transformer on TinyShakespeare using Nivara AutoDiff, then serves it as an
/// <see cref="IChatClient"/> (Microsoft.Extensions.AI) wired through DI.
/// </summary>
public static class TransformerMode
{
    public static void Run(string[] args)
    {
        var options = ParseArgs(args);
        if (options.ShowHelp)
        {
            PrintHelp();
            return;
        }

        Execute(options);
    }

    /// <summary>
    /// Interactive-menu entry point: asks for the key options, defaulting every
    /// answer so an unfamiliar user can just press Enter.
    /// </summary>
    public static void RunInteractive()
    {
        Console.WriteLine("\n=== TinyShakespeare — batched transformer as IChatClient ===\n");

        Console.Write("Load a saved model? (path, blank = train a new one): ");
        var loadInput = Console.ReadLine()?.Trim();
        string? loadPath = string.IsNullOrEmpty(loadInput) ? null : loadInput;

        Console.Write("Vocab size? (blank = 8000; 1200 = ~3x faster smoke run): ");
        var vocabInput = Console.ReadLine()?.Trim();
        int vocabSize = string.IsNullOrEmpty(vocabInput) ? 8000 : int.Parse(vocabInput);

        Console.Write("Prompt? (blank = \"ROMEO:\"): ");
        var prompt = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(prompt)) prompt = "ROMEO:";

        Console.WriteLine();
        Execute(new TransformerOptions
        {
            LoadPath = loadPath,
            MaxVocabSize = vocabSize,
            Prompt = prompt,
        });
    }

    static TransformerOptions ParseArgs(string[] args)
    {
        var options = new TransformerOptions();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--n-embd": options.NEmbd = int.Parse(args[++i]); break;
                case "--n-layer": options.NLayer = int.Parse(args[++i]); break;
                case "--block-size": options.BlockSize = int.Parse(args[++i]); break;
                case "--n-head": options.NHead = int.Parse(args[++i]); break;
                case "--dropout": options.Dropout = double.Parse(args[++i]); break;
                case "--epochs": options.Epochs = int.Parse(args[++i]); break;
                case "--batch-size": options.BatchSize = int.Parse(args[++i]); break;
                case "--lr": options.LearningRate = double.Parse(args[++i]); break;
                case "--beta1": options.Beta1 = double.Parse(args[++i]); break;
                case "--beta2": options.Beta2 = double.Parse(args[++i]); break;
                case "--vocab-size": options.MaxVocabSize = int.Parse(args[++i]); break;
                case "--seed": options.RngSeed = int.Parse(args[++i]); break;
                case "--temperature": options.Temperature = float.Parse(args[++i]); break;
                case "--max-new-tokens": options.MaxNewTokens = int.Parse(args[++i]); break;
                case "--samples": options.SampleCount = int.Parse(args[++i]); break;
                case "--data": options.DataPath = args[++i]; break;
                case "--prompt": options.Prompt = args[++i]; break;
                case "--save": options.SavePath = args[++i]; break;
                case "--load": options.LoadPath = args[++i]; break;
                case "--no-di-demo": options.ShowDiDemo = false; break;
                case "--help": options.ShowHelp = true; break;
                case "-h": options.ShowHelp = true; break;
            }
        }

        return options;
    }

    static void Execute(TransformerOptions options)
    {
        TextTokenizer? tokenizer = TryLoadTokenizer(options.LoadPath);
        List<string>? docs = null;

        if (tokenizer == null)
        {
            Console.Write("Loading TinyShakespeare corpus... ");
            string corpusPath = TinyShakespeare.Load(options.DataPath);
            Console.WriteLine(corpusPath);
            docs = TinyShakespeare.ReadDocuments(corpusPath);
            Console.WriteLine($"documents: {docs.Count}");

            tokenizer = TextTokenizer.FromDocuments(docs, options.MaxVocabSize);
            Console.WriteLine($"tokenizer: vocab {tokenizer.VocabSize} (pad {tokenizer.PadToken}, unk {tokenizer.UnkToken}, bos {tokenizer.BosToken}, eos {tokenizer.EosToken})");
        }

        using var model = new BatchedTransformer<float>(
            tokenizer.VocabSize, options.NEmbd, options.NLayer, options.NHead, options.BlockSize, dropout: options.Dropout);

        if (!string.IsNullOrWhiteSpace(options.LoadPath))
        {
            ModelSerializer.Load(model, options.LoadPath);
            Console.WriteLine($"Loaded model: {options.LoadPath}");
        }
        else
        {
            int totalParams = 0;
            foreach (var p in model.GetParameters().Values)
                totalParams += p.Length;
            Console.WriteLine($"model: {options.NLayer}L x {options.NEmbd}D, {options.NHead} heads, block={options.BlockSize}, dropout={options.Dropout}");
            Console.WriteLine($"params: {totalParams}");

            Train(model, tokenizer, docs!, options);
        }

        if (!string.IsNullOrWhiteSpace(options.SavePath))
        {
            ModelSerializer.Save(model, options.SavePath);
            Console.WriteLine($"Saved model: {options.SavePath}");
            string tokenizerPath = Path.ChangeExtension(options.SavePath, ".tokenizer.json");
            tokenizer.Save(tokenizerPath);
            Console.WriteLine($"Saved tokenizer: {tokenizerPath}");
        }

        RunSamples(model, tokenizer, options.Prompt ?? "ROMEO:", options);

        if (options.ShowDiDemo)
        {
            RunDiDemo(model, tokenizer, options.Prompt ?? "ROMEO:", options);
        }
    }

    static TextTokenizer? TryLoadTokenizer(string? loadPath)
    {
        if (string.IsNullOrWhiteSpace(loadPath))
            return null;

        string tokenizerPath = Path.ChangeExtension(loadPath, ".tokenizer.json");
        if (!File.Exists(tokenizerPath))
            return null;

        var loaded = TextTokenizer.Load(tokenizerPath);
        Console.WriteLine($"Loaded tokenizer: {tokenizerPath} (vocab {loaded.VocabSize})");
        return loaded;
    }

    static void Train(BatchedTransformer<float> model, TextTokenizer tokenizer, List<string> docs, TransformerOptions options)
    {
        Console.WriteLine("tokenizing...");
        var allTokens = new List<int>();
        foreach (var doc in docs)
        {
            var tokens = tokenizer.Encode(doc, addBosEos: false);
            allTokens.AddRange(tokens);
            allTokens.Add(tokenizer.EosToken);
        }
        int nTokens = allTokens.Count;
        Console.WriteLine($"tokens: {nTokens}");

        int nBatches = Math.Max(1, (nTokens - 1) / (options.BatchSize * options.BlockSize));
        Console.WriteLine($"batches/epoch: {nBatches}");

        var lossFn = new CrossEntropyLoss<float>();
        var optimizer = new Adam<float>((float)options.LearningRate, options.Beta1, options.Beta2);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var rng = new Random(options.RngSeed);
        var sw = Stopwatch.StartNew();

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            double epochLoss = 0;
            int batchCount = 0;
            var epochSw = Stopwatch.StartNew();

            for (int batchIdx = 0; batchIdx < nBatches; batchIdx++)
            {
                using var gradScope = GradientUtils.Grad();

                var inputFloats = new float[options.BatchSize * options.BlockSize];
                var targetFloats = new float[options.BatchSize * options.BlockSize];

                for (int b = 0; b < options.BatchSize; b++)
                {
                    int start = rng.Next(nTokens - options.BlockSize - 1);
                    for (int t = 0; t < options.BlockSize; t++)
                    {
                        inputFloats[b * options.BlockSize + t] = allTokens[start + t];
                        targetFloats[b * options.BlockSize + t] = allTokens[start + t + 1];
                    }
                }

                var inputCol = NivaraColumn<float>.Create(inputFloats);
                var inputTensor = new ReverseGradTensor<float>(inputCol, requiresGrad: false);
                inputTensor.Reshape(options.BatchSize, options.BlockSize);

                var logits = model.Forward(inputTensor);

                var targets = new int[options.BatchSize * options.BlockSize];
                for (int i = 0; i < targets.Length; i++)
                    targets[i] = (int)targetFloats[i];

                var loss = lossFn.Forward(logits, targets);

                float lossVal = float.CreateChecked(loss[0]);
                if (float.IsNaN(lossVal) || float.IsInfinity(lossVal))
                {
                    Console.WriteLine($"  NaN at epoch {epoch} batch {batchIdx}! loss={lossVal}");
                    break;
                }

                loss.Backward();
                optimizer.Step();
                optimizer.ZeroGrad();

                epochLoss += lossVal;
                batchCount++;
            }

            epochSw.Stop();

            double avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
            double tokPerSec = (double)batchCount * options.BatchSize * options.BlockSize / epochSw.Elapsed.TotalSeconds;
            Console.WriteLine($"epoch {epoch}/{options.Epochs} | loss {avgLoss:F4} | {epochSw.Elapsed.TotalSeconds:F1}s | {tokPerSec:F0} tok/s");
        }

        sw.Stop();
        Console.WriteLine($"\ntime: {sw.Elapsed.TotalSeconds:F2}s");
    }

    static void RunSamples(BatchedTransformer<float> model, TextTokenizer tokenizer,
        string prompt, TransformerOptions options)
    {
        Console.WriteLine($"\n--- samples (prompt: {prompt}) ---");
        using var client = new BatchedChatClient(model, tokenizer, options.Temperature, options.MaxNewTokens);

        for (int i = 0; i < options.SampleCount; i++)
        {
            var response = client.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)])
                .GetAwaiter().GetResult();
            Console.WriteLine($"\nsample {i}: {response.Text?.Trim()}");
        }
    }

    static void RunDiDemo(BatchedTransformer<float> model, TextTokenizer tokenizer,
        string prompt, TransformerOptions options)
    {
        Console.WriteLine("\n--- DI demo (IChatClient via Microsoft.Extensions.AI) ---");
        var services = new ServiceCollection();
        services.AddSingleton(model);
        services.AddSingleton(tokenizer);
        services.AddChatClient(sp => new BatchedChatClient(
            sp.GetRequiredService<BatchedTransformer<float>>(),
            sp.GetRequiredService<TextTokenizer>(),
            options.Temperature,
            options.MaxNewTokens));

        using var provider = services.BuildServiceProvider();
        using var client = provider.GetRequiredService<IChatClient>();

        var response = client.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)])
            .GetAwaiter().GetResult();
        Console.WriteLine($"reply: {response.Text?.Trim()}");
    }

    static void PrintHelp()
    {
        Console.WriteLine("""
NivaraChat --tinyshakespeare — Word-level batched Transformer chat mode

Trains a causal transformer with batched multi-head attention on TinyShakespeare,
then exposes it as an IChatClient (Microsoft.Extensions.AI) wired through DI.

Options:
  --n-embd <int>          Embedding dimension (default: 96)
  --n-layer <int>         Number of transformer layers (default: 2)
  --block-size <int>      Context window / max sequence length (default: 64)
  --n-head <int>          Number of attention heads (default: 4)
  --dropout <float>       Dropout probability (default: 0.1)
  --epochs <int>          Training epochs (default: 20)
  --batch-size <int>      Batch size (default: 32)
  --lr <float>            Learning rate (default: 3e-3)
  --beta1 <float>         Adam beta1 (default: 0.9)
  --beta2 <float>         Adam beta2 (default: 0.95)
  --vocab-size <int>      Max word-vocab size (default: 8000)
  --temperature <float>   Sampling temperature (default: 0.8)
  --max-new-tokens <int>  Max tokens per generated reply (default: 96)
  --samples <int>         Number of generated samples (default: 5)
  --seed <int>            RNG seed (default: 42)
  --data <path>           Corpus path (default: samples/data/tinyshakespeare.txt, downloaded on first use)
  --prompt <text>         Chat with the model using this user prompt
  --save <path>           Save trained model to JSON
  --load <path>           Load model from JSON (pass the same --n-embd/--n-layer/
                          --block-size/--n-head/--vocab-size used at save time;
                          the matching <path>.tokenizer.json is loaded too)
  --no-di-demo            Skip the DI + IChatClient demo at the end
  --help, -h              Show this help
""");
    }

    sealed class TransformerOptions
    {
        public int NEmbd = 96;
        public int NLayer = 2;
        public int BlockSize = 64;
        public int NHead = 4;
        public int Epochs = 20;
        public int BatchSize = 32;
        public double LearningRate = 3e-3;
        public double Beta1 = 0.9;
        public double Beta2 = 0.95;
        public double Dropout = 0.1;
        public int RngSeed = 42;
        public int MaxVocabSize = 8000;
        public int MaxNewTokens = 96;
        public float Temperature = 0.8f;
        public int SampleCount = 5;
        public string? SavePath = null;
        public string? LoadPath = null;
        public string? DataPath = null;
        public string? Prompt = null;
        public bool ShowDiDemo = true;
        public bool ShowHelp = false;
    }
}
