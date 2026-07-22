using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NivaraGpt;
using System.Diagnostics;

int nEmbd = 64;
int nLayer = 2;
int blockSize = 32;
int nHead = 4;
int epochs = 20;
int batchSize = 64;
double learningRate = 3e-3;
double beta1 = 0.9, beta2 = 0.95;
double initStd = 0.02;
bool weightTying = true;
bool lrDecay = false;
double temperature = 0.8;
int topK = 0;
double dropout = 0.1;
int rngSeed = 42;
int numSamples = 10;
string? savePath = null;
string? loadPath = null;
string? dumpWeightsPath = null;
bool help = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--n-embd": nEmbd = int.Parse(args[++i]); break;
        case "--n-layer": nLayer = int.Parse(args[++i]); break;
        case "--block-size": blockSize = int.Parse(args[++i]); break;
        case "--n-head": nHead = int.Parse(args[++i]); break;
        case "--epochs": epochs = int.Parse(args[++i]); break;
        case "--batch-size": batchSize = int.Parse(args[++i]); break;
        case "--lr": learningRate = double.Parse(args[++i]); break;
        case "--beta1": beta1 = double.Parse(args[++i]); break;
        case "--beta2": beta2 = double.Parse(args[++i]); break;
        case "--init-std": initStd = double.Parse(args[++i]); break;
        case "--dropout": dropout = double.Parse(args[++i]); break;
        case "--no-weight-tying": weightTying = false; break;
        case "--lr-decay": lrDecay = true; break;
        case "--temperature": temperature = double.Parse(args[++i]); break;
        case "--top-k": topK = int.Parse(args[++i]); break;
        case "--no-dropout": dropout = 0.0; break;
        case "--seed": rngSeed = int.Parse(args[++i]); break;
        case "--samples": numSamples = int.Parse(args[++i]); break;
        case "--save": savePath = args[++i]; break;
        case "--load": loadPath = args[++i]; break;
        case "--dump-weights": dumpWeightsPath = args[++i]; break;
        case "--help": help = true; break;
        case "-h": help = true; break;
    }
}

if (help)
{
    Console.WriteLine("""
NivaraGpt — Character-level Transformer on Nivara AutoDiff

Options:
  --n-embd <int>          Embedding dimension (default: 64)
  --n-layer <int>         Number of transformer layers (default: 2)
  --block-size <int>      Context window / max sequence length (default: 32)
  --n-head <int>          Number of attention heads (default: 4)
  --dropout <float>       Dropout probability (default: 0.1)
  --epochs <int>          Training epochs (default: 20)
  --batch-size <int>      Batch size (default: 64)
  --lr <float>            Learning rate (default: 3e-3)
  --beta1 <float>         Adam beta1 (default: 0.9)
  --beta2 <float>         Adam beta2 (default: 0.95)
  --init-std <float>      Weight init std dev (default: 0.02)
  --no-weight-tying       Use separate lm_head instead of weight tying
  --lr-decay              Linear LR decay to zero over epochs
  --temperature <float>   Sampling temperature (default: 0.8)
  --top-k <int>           Top-k sampling, 0 = disabled (default: 0)
  --no-dropout            Disable all dropout
  --save <path>           Save trained model to JSON
  --load <path>           Load model from JSON
  --samples <int>         Number of generated samples (default: 10)
  --seed <int>            RNG seed (default: 42)
  --help, -h              Show this help
""");
    return;
}

Console.Write("Loading names dataset... ");
var namesPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "samples", "data", "names.txt");
if (!File.Exists(namesPath))
    namesPath = Path.Combine("samples", "data", "names.txt");
if (!File.Exists(namesPath))
    namesPath = Path.Combine("..", "data", "names.txt");
if (!File.Exists(namesPath))
    throw new FileNotFoundException($"Could not find names.txt. Place it in samples/data/names.txt relative to the repo root.");

var namesText = await File.ReadAllTextAsync(namesPath);

var docs = namesText.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Where(l => l.Length > 0).ToList();

var tokenizer = new Tokenizer(docs);
Console.WriteLine($"vocab: {tokenizer.VocabSize}, docs: {docs.Count}");

using var model = new NivaraGptModel<float>(
    tokenizer.VocabSize, nEmbd, nLayer, blockSize, nHead,
    dropout: dropout, weightTying: weightTying, initStd: initStd);

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

    Train(model, tokenizer, docs, epochs, batchSize, learningRate, beta1, beta2, lrDecay, rngSeed, blockSize);
}

if (!string.IsNullOrWhiteSpace(savePath))
{
    ModelSerializer.Save(model, savePath);
    Console.WriteLine($"Saved model: {savePath}");
}

if (!string.IsNullOrWhiteSpace(dumpWeightsPath))
{
    Console.Write($"Dumping weights to {dumpWeightsPath}... ");
    var parameters = model.GetParameters();
    using (var fs = File.CreateText(dumpWeightsPath))
    {
        fs.WriteLine("{");
        int idx = 0;
        foreach (var kvp in parameters)
        {
            int len = kvp.Value.Length;
            var data = new float[len];
            kvp.Value.Tensor.Data.CopyTo(data, 0f);
            float mean = data.Average();
            float variance = data.Sum(x => (x - mean) * (x - mean)) / data.Length;
            float std = MathF.Sqrt(variance);
            float min = data.Min();
            float max = data.Max();
            float l2 = MathF.Sqrt(data.Sum(x => x * x));
            idx++;
            string comma = idx < parameters.Count ? "," : "";
            fs.WriteLine($"  \"{kvp.Key}\": {{\"len\":{data.Length},\"mean\":{mean:F6},\"std\":{std:F6},\"min\":{min:F6},\"max\":{max:F6},\"l2\":{l2:F4}}}{comma}");
        }
        fs.WriteLine("}");
    }
    Console.WriteLine("done");
}

Console.WriteLine("\n--- generation ---");
var rng = new Random(rngSeed);
for (int i = 0; i < numSamples; i++)
{
    var name = Generate(model, tokenizer, blockSize, rng, temperature, topK);
    Console.WriteLine($"sample {i}: {name}");
}

static void Train(NivaraGptModel<float> model, Tokenizer tokenizer, List<string> docs,
    int epochs, int batchSize, double learningRate, double beta1, double beta2,
    bool lrDecay, int rngSeed, int blockSize)
{
    Console.WriteLine("tokenizing...");
    var allTokens = new List<int>();
    foreach (var doc in docs)
    {
        var tokens = tokenizer.Encode(doc);
        allTokens.AddRange(tokens);
        allTokens.Add(tokenizer.EOS);
    }
    int nTokens = allTokens.Count;
    Console.WriteLine($"tokens: {nTokens}");

    int nBatches = (nTokens - 1) / (batchSize * blockSize);
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

        if (lrDecay)
        {
            float decayed = (float)(learningRate * (1.0 - (double)epoch / epochs));
            optimizer.SetGroupLearningRate(0, decayed);
        }

        double avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
        double tokPerSec = (double)batchCount * batchSize * blockSize / epochSw.Elapsed.TotalSeconds;
        Console.WriteLine($"epoch {epoch}/{epochs} | loss {avgLoss:F4} | {epochSw.Elapsed.TotalSeconds:F1}s | {tokPerSec:F0} tok/s");
    }

    sw.Stop();
    Console.WriteLine($"\ntime: {sw.Elapsed.TotalSeconds:F2}s");
}

static string Generate(NivaraGptModel<float> model, Tokenizer tokenizer,
    int blockSize, Random rng, double temperature, int topK)
{
    model.Eval();
    var sampler = new Sampler<float>();

    var context = new List<int> { tokenizer.BOS };
    for (int pos = 0; pos < blockSize; pos++)
    {
        var data = new float[blockSize];
        int start = Math.Max(0, context.Count - blockSize);
        int count = Math.Min(context.Count, blockSize);
        for (int i = 0; i < count; i++)
            data[i] = context[start + i];

        var col = NivaraColumn<float>.Create(data);
        var input = new ReverseGradTensor<float>(col, requiresGrad: false);

        var logits = model.Forward(input);

        int lastIdx = count - 1;
        var lastPosLogits = new float[tokenizer.VocabSize];
        int logitOffset = lastIdx * tokenizer.VocabSize;
        for (int v = 0; v < tokenizer.VocabSize; v++)
            lastPosLogits[v] = float.CreateChecked(logits[logitOffset + v]);

        var logitsTensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(lastPosLogits), requiresGrad: false);
        logitsTensor.Reshape(tokenizer.VocabSize);

        int next = sampler.Sample(logitsTensor, temperature, topK);
        context.Add(next);
        if (next == tokenizer.EOS) break;
    }

    model.Train();

    var sb = new System.Text.StringBuilder();
    foreach (var t in context.Skip(1))
    {
        if (t == tokenizer.EOS) break;
        sb.Append(tokenizer.Decode(t));
    }
    return sb.ToString();
}
