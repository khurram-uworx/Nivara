using MicroGpt;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Net.Http;

// MicroGPT on Nivara AutoDiff — exact neural network parameters as AutoGrad-Engine.
// A vs B comparison: both use the same architecture, optimizer, and data.

Console.Write("Downloading names dataset... ");
using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
var namesText = await client.GetStringAsync(
    "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");

var docs = namesText.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Where(l => l.Length > 0).ToList();

var tokenizer = new Tokenizer(docs);
Console.WriteLine($"vocab: {tokenizer.VocabSize}, docs: {docs.Count}");

// Neural network parameters — identical to AutoGrad-Engine defaults
int nEmbd = 16;
int nLayer = 1;
int blockSize = 8;
int nHead = 4;
int numSteps = 1000;
double learningRate = 1e-2;
double beta1 = 0.9, beta2 = 0.95, epsAdam = 1e-8;

using var model = new MicroGptModel<float>(tokenizer.VocabSize, nEmbd, nLayer, blockSize, nHead);
Console.WriteLine($"model: {nLayer}L x {nEmbd}D, {nHead} heads, block={blockSize}");

// Parameter count
int totalParams = (tokenizer.VocabSize * nEmbd) + (blockSize * nEmbd)
    + nLayer * (4 * nEmbd * nEmbd + nEmbd * (4 * nEmbd) + (4 * nEmbd) * nEmbd);
Console.WriteLine($"params: {totalParams}");

var optimizer = new Adam<float>((float)learningRate, beta1, beta2, epsAdam);
optimizer.AddParameterGroup(model.Parameters, (float)learningRate);

var rng = new Random(42);
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

            // Per-position cross-entropy loss averaged over sequence length
            // Matches AutoGrad-Engine: loss = -(log(probs[target]) / seqLen
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

        optimizer.Step();
        optimizer.ZeroGrad();
    }

    Console.WriteLine($"step {step + 1} / {numSteps} | loss {lossVal:F4}");
}

sw.Stop();
Console.WriteLine($"\ntime: {sw.Elapsed.TotalSeconds:F2}s");

Console.WriteLine("\n--- generation ---");
for (int i = 0; i < 5; i++)
{
    var name = Generate(model, tokenizer, blockSize, nLayer, nHead, nEmbd, rng);
    Console.WriteLine($"sample {i}: {name}");
}

static string Generate(
    MicroGptModel<float> model, Tokenizer tokenizer,
    int blockSize, int nLayer, int nHead, int nEmbd, Random rng)
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

        // Softmax
        float max = buf[0];
        for (int j = 1; j < buf.Length; j++) if (buf[j] > max) max = buf[j];
        var exps = new double[buf.Length];
        double sum = 0;
        for (int j = 0; j < buf.Length; j++) { exps[j] = Math.Exp(buf[j] - max); sum += exps[j]; }
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
