using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NivaraChatClient;
using System.Diagnostics;

int nEmbd = 96;
int nLayer = 2;
int blockSize = 64;
int nHead = 4;
int epochs = 20;
int batchSize = 32;
double learningRate = 3e-3;
double beta1 = 0.9, beta2 = 0.95;
double dropout = 0.1;
int rngSeed = 42;
int maxVocabSize = 8000;
int maxNewTokens = 96;
float temperature = 0.8f;
int sampleCount = 5;
string? savePath = null;
string? loadPath = null;
string? dataPath = null;
string? prompt = null;
bool showDiDemo = true;
bool help = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--n-embd": nEmbd = int.Parse(args[++i]); break;
        case "--n-layer": nLayer = int.Parse(args[++i]); break;
        case "--block-size": blockSize = int.Parse(args[++i]); break;
        case "--n-head": nHead = int.Parse(args[++i]); break;
        case "--dropout": dropout = double.Parse(args[++i]); break;
        case "--epochs": epochs = int.Parse(args[++i]); break;
        case "--batch-size": batchSize = int.Parse(args[++i]); break;
        case "--lr": learningRate = double.Parse(args[++i]); break;
        case "--beta1": beta1 = double.Parse(args[++i]); break;
        case "--beta2": beta2 = double.Parse(args[++i]); break;
        case "--vocab-size": maxVocabSize = int.Parse(args[++i]); break;
        case "--seed": rngSeed = int.Parse(args[++i]); break;
        case "--temperature": temperature = float.Parse(args[++i]); break;
        case "--max-new-tokens": maxNewTokens = int.Parse(args[++i]); break;
        case "--samples": sampleCount = int.Parse(args[++i]); break;
        case "--data": dataPath = args[++i]; break;
        case "--prompt": prompt = args[++i]; break;
        case "--save": savePath = args[++i]; break;
        case "--load": loadPath = args[++i]; break;
        case "--no-di-demo": showDiDemo = false; break;
        case "--help": help = true; break;
        case "-h": help = true; break;
    }
}

if (help)
{
    Console.WriteLine("""
NivaraChatClient — Word-level batched Transformer chat sample on Nivara AutoDiff

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
    return;
}

TextTokenizer? tokenizer = TryLoadTokenizer(loadPath);
List<string>? docs = null;

if (tokenizer == null)
{
    Console.Write("Loading TinyShakespeare corpus... ");
    string corpusPath = TinyShakespeare.Load(dataPath);
    Console.WriteLine(corpusPath);
    docs = TinyShakespeare.ReadDocuments(corpusPath);
    Console.WriteLine($"documents: {docs.Count}");

    tokenizer = TextTokenizer.FromDocuments(docs, maxVocabSize);
    Console.WriteLine($"tokenizer: vocab {tokenizer.VocabSize} (pad {tokenizer.PadToken}, unk {tokenizer.UnkToken}, bos {tokenizer.BosToken}, eos {tokenizer.EosToken})");
}

using var model = new BatchedTransformer<float>(
    tokenizer.VocabSize, nEmbd, nLayer, nHead, blockSize, dropout: dropout);

if (!string.IsNullOrWhiteSpace(loadPath))
{
    ModelSerializer.Load(model, loadPath);
    Console.WriteLine($"Loaded model: {loadPath}");
}
else
{
    int totalParams = 0;
    foreach (var p in model.GetParameters().Values)
        totalParams += p.Length;
    Console.WriteLine($"model: {nLayer}L x {nEmbd}D, {nHead} heads, block={blockSize}, dropout={dropout}");
    Console.WriteLine($"params: {totalParams}");

    Train(model, tokenizer, docs!, epochs, batchSize, learningRate, beta1, beta2, rngSeed, blockSize);
}

if (!string.IsNullOrWhiteSpace(savePath))
{
    ModelSerializer.Save(model, savePath);
    Console.WriteLine($"Saved model: {savePath}");
    string tokenizerPath = Path.ChangeExtension(savePath, ".tokenizer.json");
    tokenizer.Save(tokenizerPath);
    Console.WriteLine($"Saved tokenizer: {tokenizerPath}");
}

RunSamples(model, tokenizer, prompt ?? "ROMEO:", sampleCount, temperature, maxNewTokens);

if (showDiDemo)
{
    RunDiDemo(model, tokenizer, prompt ?? "ROMEO:", temperature, maxNewTokens);
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

static void Train(BatchedTransformer<float> model, TextTokenizer tokenizer, List<string> docs,
    int epochs, int batchSize, double learningRate, double beta1, double beta2,
    int rngSeed, int blockSize)
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

    int nBatches = Math.Max(1, (nTokens - 1) / (batchSize * blockSize));
    Console.WriteLine($"batches/epoch: {nBatches}");

    var lossFn = new CrossEntropyLoss<float>();
    var optimizer = new Adam<float>((float)learningRate, beta1, beta2);
    optimizer.AddParameterGroup(model.GetParameters().Values);

    var rng = new Random(rngSeed);
    var sw = Stopwatch.StartNew();

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        double epochLoss = 0;
        int batchCount = 0;
        var epochSw = Stopwatch.StartNew();

        for (int batchIdx = 0; batchIdx < nBatches; batchIdx++)
        {
            using var gradScope = GradientUtils.Grad();

            var inputFloats = new float[batchSize * blockSize];
            var targetFloats = new float[batchSize * blockSize];

            for (int b = 0; b < batchSize; b++)
            {
                int start = rng.Next(nTokens - blockSize - 1);
                for (int t = 0; t < blockSize; t++)
                {
                    inputFloats[b * blockSize + t] = allTokens[start + t];
                    targetFloats[b * blockSize + t] = allTokens[start + t + 1];
                }
            }

            var inputCol = NivaraColumn<float>.Create(inputFloats);
            var inputTensor = new ReverseGradTensor<float>(inputCol, requiresGrad: false);
            inputTensor.Reshape(batchSize, blockSize);

            var logits = model.Forward(inputTensor);

            var targets = new int[batchSize * blockSize];
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
        double tokPerSec = (double)batchCount * batchSize * blockSize / epochSw.Elapsed.TotalSeconds;
        Console.WriteLine($"epoch {epoch}/{epochs} | loss {avgLoss:F4} | {epochSw.Elapsed.TotalSeconds:F1}s | {tokPerSec:F0} tok/s");
    }

    sw.Stop();
    Console.WriteLine($"\ntime: {sw.Elapsed.TotalSeconds:F2}s");
}

static void RunSamples(BatchedTransformer<float> model, TextTokenizer tokenizer,
    string prompt, int sampleCount, float temperature, int maxNewTokens)
{
    Console.WriteLine($"\n--- samples (prompt: {prompt}) ---");
    using var client = new BatchedChatClient(model, tokenizer, temperature, maxNewTokens);

    for (int i = 0; i < sampleCount; i++)
    {
        var response = client.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)])
            .GetAwaiter().GetResult();
        Console.WriteLine($"\nsample {i}: {response.Text?.Trim()}");
    }
}

static void RunDiDemo(BatchedTransformer<float> model, TextTokenizer tokenizer,
    string prompt, float temperature, int maxNewTokens)
{
    Console.WriteLine("\n--- DI demo (IChatClient via Microsoft.Extensions.AI) ---");
    var services = new ServiceCollection();
    services.AddSingleton(model);
    services.AddSingleton(tokenizer);
    services.AddChatClient(sp => new BatchedChatClient(
        sp.GetRequiredService<BatchedTransformer<float>>(),
        sp.GetRequiredService<TextTokenizer>(),
        temperature,
        maxNewTokens));

    using var provider = services.BuildServiceProvider();
    using var client = provider.GetRequiredService<IChatClient>();

    var response = client.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)])
        .GetAwaiter().GetResult();
    Console.WriteLine($"reply: {response.Text?.Trim()}");
}
