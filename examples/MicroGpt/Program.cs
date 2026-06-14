using MicroGpt;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Net.Http;

int nEmbd = 16;
int nLayer = 1;
int blockSize = 8;
int nHead = 4;
int numSteps = 1000;
double learningRate = 1e-2;
double beta1 = 0.9, beta2 = 0.95, epsAdam = 1e-8;
double initStd = 0.02;
bool weightTying = true;
bool lrDecay = false;
double temperature = 1.0;
int rngSeed = 42;
int numSamples = 5;
bool help = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--n-embd": nEmbd = int.Parse(args[++i]); break;
        case "--n-layer": nLayer = int.Parse(args[++i]); break;
        case "--block-size": blockSize = int.Parse(args[++i]); break;
        case "--n-head": nHead = int.Parse(args[++i]); break;
        case "--steps": numSteps = int.Parse(args[++i]); break;
        case "--lr": learningRate = double.Parse(args[++i]); break;
        case "--beta1": beta1 = double.Parse(args[++i]); break;
        case "--beta2": beta2 = double.Parse(args[++i]); break;
        case "--init-std": initStd = double.Parse(args[++i]); break;
        case "--no-weight-tying": weightTying = false; break;
        case "--lr-decay": lrDecay = true; break;
        case "--temperature": temperature = double.Parse(args[++i]); break;
        case "--seed": rngSeed = int.Parse(args[++i]); break;
        case "--samples": numSamples = int.Parse(args[++i]); break;
        case "--help": help = true; break;
        case "-h": help = true; break;
    }
}

if (help)
{
    Console.WriteLine("""
MicroGpt on Nivara AutoDiff

Options:
  --n-embd <int>          Embedding dimension (default: 16)
  --n-layer <int>         Number of transformer layers (default: 1)
  --block-size <int>      Context window / block size (default: 8)
  --n-head <int>          Number of attention heads (default: 4)
  --steps <int>           Training steps (default: 1000)
  --lr <float>            Learning rate (default: 0.01)
  --beta1 <float>         Adam beta1 (default: 0.9)
  --beta2 <float>         Adam beta2 (default: 0.95)
  --init-std <float>      Weight init std dev (default: 0.02)
  --no-weight-tying       Use separate lm_head instead of weight tying
  --lr-decay              Linear LR decay to zero over steps
  --temperature <float>   Sampling temperature (default: 1.0)
  --seed <int>            RNG seed (default: 42)
  --samples <int>         Number of generated samples (default: 5)
  --help, -h              Show this help

Karpathy defaults for A vs C comparison:
  --block-size 16 --beta1 0.85 --beta2 0.99 --init-std 0.08
  --no-weight-tying --lr-decay --temperature 0.5 --samples 20
""");
    return;
}

Console.Write("Downloading names dataset... ");
using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
var namesText = await client.GetStringAsync(
    "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");

var docs = namesText.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Where(l => l.Length > 0).ToList();

var tokenizer = new Tokenizer(docs);
Console.WriteLine($"vocab: {tokenizer.VocabSize}, docs: {docs.Count}");

using var model = new MicroGptModel<float>(
    tokenizer.VocabSize, nEmbd, nLayer, blockSize, nHead, initStd, weightTying);
Console.WriteLine($"model: {nLayer}L x {nEmbd}D, {nHead} heads, block={blockSize}");

int totalParams = (tokenizer.VocabSize * nEmbd) + (blockSize * nEmbd)
    + nLayer * (4 * nEmbd * nEmbd + nEmbd * (4 * nEmbd) + (4 * nEmbd) * nEmbd);
if (!weightTying)
    totalParams += tokenizer.VocabSize * nEmbd;
Console.WriteLine($"params: {totalParams}");

var optimizer = new Adam<float>((float)learningRate, beta1, beta2, epsAdam);
optimizer.AddParameterGroup(model.Parameters, (float)learningRate);

var rng = new Random(rngSeed);
var sw = Stopwatch.StartNew();

for (int step = 0; step < numSteps; step++)
{
    var doc = docs[step % docs.Count];
    var tokens = tokenizer.Encode(doc);
    int seqLen = Math.Min(tokens.Count - 1, blockSize - 1);
    if (seqLen <= 0) continue;

    float lossVal = 0f;

    using (GradientUtils.Grad())
    {
        var keys = new List<ReverseGradTensor<float>>[nLayer];
        var values = new List<ReverseGradTensor<float>>[nLayer];
        for (int i = 0; i < nLayer; i++) { keys[i] = []; values[i] = []; }

        for (int t = 0; t < seqLen; t++)
        {
            var logits = model.Forward(tokens[t], t, keys, values);

            var logSm = ReverseGradOperations.LogSoftmax(logits);
            float[] sel = new float[tokenizer.VocabSize];
            sel[tokens[t + 1]] = 1f;
            var col = NivaraColumn<float>.Create(sel);
            var selector = new ReverseGradTensor<float>(col, requiresGrad: false);
            selector.Reshape(tokenizer.VocabSize, 1);
            var picked = ReverseGradOperations.MatMul(logSm, selector);
            var posLoss = ReverseGradOperations.Negate(picked);
            var scaleCol = NivaraColumn<float>.Create([1f / seqLen]);
            var scaleTensor = new ReverseGradTensor<float>(scaleCol, requiresGrad: false);
            var scaled = ReverseGradOperations.Multiply(posLoss, scaleTensor);

            scaled.Backward();
            lossVal += scaled.Data[0];
        }

        if (lrDecay)
        {
            float decayed = (float)(learningRate * (1 - (double)step / numSteps));
            optimizer.SetGroupLearningRate(0, decayed);
        }
        optimizer.Step();
        optimizer.ZeroGrad();
    }

    if ((step + 1) % Math.Max(1, numSteps / 20) == 0 || step == 0)
        Console.WriteLine($"step {step + 1} / {numSteps} | loss {lossVal:F4}");
}

sw.Stop();
Console.WriteLine($"\ntime: {sw.Elapsed.TotalSeconds:F2}s");

Console.WriteLine("\n--- generation ---");
for (int i = 0; i < numSamples; i++)
{
    var name = Generate(model, tokenizer, blockSize, nLayer, nHead, nEmbd, rng, temperature);
    Console.WriteLine($"sample {i}: {name}");
}

static string Generate(
    MicroGptModel<float> model, Tokenizer tokenizer,
    int blockSize, int nLayer, int nHead, int nEmbd, Random rng, double temperature)
{
    var keys = new List<ReverseGradTensor<float>>[nLayer];
    var values = new List<ReverseGradTensor<float>>[nLayer];
    for (int i = 0; i < nLayer; i++) { keys[i] = []; values[i] = []; }

    var tokens = new List<int> { tokenizer.BOS };
    for (int pos = 0; pos < blockSize; pos++)
    {
        var logits = model.Forward(tokens[pos], pos, keys, values);
        var buf = new float[tokenizer.VocabSize];
        logits.Data.CopyTo(buf.AsSpan(), 0f);

        // Softmax with temperature
        float max = buf[0];
        for (int j = 1; j < buf.Length; j++) if (buf[j] > max) max = buf[j];
        var exps = new double[buf.Length];
        double sum = 0;
        for (int j = 0; j < buf.Length; j++)
        {
            exps[j] = Math.Exp((buf[j] - max) / temperature);
            sum += exps[j];
        }
        var probs = new float[buf.Length];
        for (int j = 0; j < buf.Length; j++) probs[j] = (float)(exps[j] / sum);

        double d = rng.NextDouble(), cum = 0;
        int next = probs.Length - 1;
        for (int j = 0; j < probs.Length; j++) { cum += probs[j]; if (d < cum) { next = j; break; } }

        tokens.Add(next);
        if (next == tokenizer.EOS) break;
    }

    var res = "";
    foreach (var t in tokens.Skip(1))
    {
        if (t == tokenizer.EOS) break;
        res += tokenizer.Decode(t);
    }
    return res;
}
