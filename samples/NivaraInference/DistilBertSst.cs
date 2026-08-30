using Microsoft.ML.Tokenizers;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace NivaraInference;

static class DistilBertSst
{
    public static readonly string[] CompareSentences =
    [
        "This movie was an absolute joy from start to finish.",
        "A complete waste of time, boring and predictable.",
        "The acting was brilliant and the plot kept me on the edge of my seat.",
        "Terrible script, awful performances, I want my money back.",
        "An emotional masterpiece that will stay with you long after the credits.",
        "Not funny at all, the jokes fall completely flat.",
        "Visually stunning with a captivating story to match.",
        "Poorly paced and overlong, nothing happens for the first hour.",
    ];

    public const string LabelsPath = "samples/data/compare_distilbert_sst_cs.bin";

    public static DistilBertForSequenceClassification<float> Load(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = new DistilBertForSequenceClassification<float>(config.ToBertConfig(), numClasses: 2);
        model.LoadWeights(tensors);
        return model;
    }

    public static BertTokenizer LoadTokenizer(string modelDir)
        => MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));

    public static ReverseGradTensor<float> PredictLogits(
        DistilBertForSequenceClassification<float> model,
        BertTokenizer tokenizer,
        string text,
        int maxLen)
    {
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen);
        var inputIds = GradientUtils.Constant(tokenIds);
        var mask = GradientUtils.Constant(attnMask);
        return model.Forward(inputIds, mask, 1, tokenIds.Length);
    }

    public static DistilBertForSequenceClassification<BFloat16> LoadBFloat16(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = new DistilBertForSequenceClassification<BFloat16>(config.ToBertConfig(), numClasses: 2);
        model.LoadWeights<BFloat16>(tensors);
        return model;
    }

    public static ReverseGradTensor<BFloat16> PredictLogitsBFloat16(
        DistilBertForSequenceClassification<BFloat16> model,
        BertTokenizer tokenizer,
        string text,
        int maxLen)
    {
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (BFloat16)x));
        return model.Forward(intIds, mask, 1, intIds.Length);
    }

    public static DistilBertForSequenceClassification<Half> LoadHalf(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = new DistilBertForSequenceClassification<Half>(config.ToBertConfig(), numClasses: 2);
        model.LoadWeights<Half>(tensors);
        return model;
    }

    public static ReverseGradTensor<Half> PredictLogitsHalf(
        DistilBertForSequenceClassification<Half> model,
        BertTokenizer tokenizer,
        string text,
        int maxLen)
    {
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (Half)x));
        return model.Forward(intIds, mask, 1, intIds.Length);
    }

    public static (int ArgMax, float[] Probs) Softmax(ReverseGradTensor<float> logits)
    {
        int n = logits.Shape[^1];
        var logitData = new float[n];
        logits.Data.TryGetSpan(out var span);
        if (!span.IsEmpty)
            span.Slice(0, n).CopyTo(logitData);
        return Softmax(logitData.AsSpan());
    }

    public static (int ArgMax, float[] Probs) Softmax(ReadOnlySpan<float> logits)
    {
        float max = TensorPrimitives.Max(logits);

        var shifted = new float[logits.Length];
        TensorPrimitives.Add(logits, -max, shifted);
        var exps = new float[logits.Length];
        TensorPrimitives.Exp(shifted, exps);
        float sum = TensorPrimitives.Sum(exps);
        TensorPrimitives.Divide(exps, sum, exps);

        int argMax = 0;
        for (int i = 1; i < logits.Length; i++)
            if (exps[i] > exps[argMax]) argMax = i;

        return (argMax, exps);
    }

    public static string Label(int argMax) => argMax == 0 ? "NEGATIVE" : "POSITIVE";

    public static int CountParameters(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => tensors.Values.Sum(t => t.Data.Length);

    public static double WeightMb(Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => tensors.Values.Sum(t => t.Data.Length * 4.0) / (1024.0 * 1024.0);

    public static void SaveCompareOutput(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string modelDir,
        string path)
    {
        var model = Load(tensors, modelDir);
        var tokenizer = LoadTokenizer(modelDir);
        model.Eval();

        int n = CompareSentences.Length;
        var logits = new float[n * 2];
        var probs = new float[n * 2];

        for (int s = 0; s < n; s++)
        {
            var output = PredictLogits(model, tokenizer, CompareSentences[s], maxLen: 128);
            var (argMax, p) = Softmax(output);
            logits[s * 2] = output.Data[0];
            logits[s * 2 + 1] = output.Data[1];
            probs[s * 2] = p[0];
            probs[s * 2 + 1] = p[1];
            Console.WriteLine($"  [{s}] {Label(argMax),8} ({p[argMax] * 100:F1}%)  \"{CompareSentences[s]}\"");
        }

        using (var fs = File.Create(path))
        {
            var header = BitConverter.GetBytes(n);
            fs.Write(header, 0, 4);
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
            fs.Write(MemoryMarshal.AsBytes(probs.AsSpan()));
        }
        Console.WriteLine($"Saved logits + softmax probs to {path}");
    }

    public static void SaveBFloat16CompareOutput(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        string modelDir,
        string path)
    {
        var model = LoadBFloat16(tensors, modelDir);
        var tokenizer = LoadTokenizer(modelDir);
        model.Eval();

        int n = CompareSentences.Length;
        var logits = new float[n * 2];
        var probs = new float[n * 2];

        for (int s = 0; s < n; s++)
        {
            var output = PredictLogitsBFloat16(model, tokenizer, CompareSentences[s], maxLen: 128);
            int len = output.Shape[^1];
            var bf16Logits = new float[len];
            output.Data.TryGetSpan(out var span);
            if (!span.IsEmpty)
            {
                int take = Math.Min(len, span.Length);
                for (int i = 0; i < take; i++)
                    bf16Logits[i] = (float)span[i];
            }

            var (argMax, p) = Softmax(bf16Logits.AsSpan());
            logits[s * 2] = bf16Logits[0];
            logits[s * 2 + 1] = bf16Logits[1];
            probs[s * 2] = p[0];
            probs[s * 2 + 1] = p[1];
            Console.WriteLine($"  [{s}] {Label(argMax),8} ({p[argMax] * 100:F1}%)  \"{CompareSentences[s]}\"");
        }

        using (var fs = File.Create(path))
        {
            var header = BitConverter.GetBytes(n);
            fs.Write(header, 0, 4);
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
            fs.Write(MemoryMarshal.AsBytes(probs.AsSpan()));
        }
        Console.WriteLine($"Saved BFloat16 logits + softmax probs to {path}");
    }

    public static void SaveHalfCompareOutput(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir,
        string path)
    {
        var model = LoadHalf(tensors, modelDir);
        var tokenizer = LoadTokenizer(modelDir);
        model.Eval();

        int n = CompareSentences.Length;
        var logits = new float[n * 2];
        var probs = new float[n * 2];

        for (int s = 0; s < n; s++)
        {
            var output = PredictLogitsHalf(model, tokenizer, CompareSentences[s], maxLen: 128);
            int len = output.Shape[^1];
            var halfLogits = new float[len];
            output.Data.TryGetSpan(out var span);
            if (!span.IsEmpty)
            {
                int take = Math.Min(len, span.Length);
                for (int i = 0; i < take; i++)
                    halfLogits[i] = (float)span[i];
            }

            var (argMax, p) = Softmax(halfLogits.AsSpan());
            logits[s * 2] = halfLogits[0];
            logits[s * 2 + 1] = halfLogits[1];
            probs[s * 2] = p[0];
            probs[s * 2 + 1] = p[1];
            Console.WriteLine($"  [{s}] {Label(argMax),8} ({p[argMax] * 100:F1}%)  \"{CompareSentences[s]}\"");
        }

        using (var fs = File.Create(path))
        {
            var header = BitConverter.GetBytes(n);
            fs.Write(header, 0, 4);
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
            fs.Write(MemoryMarshal.AsBytes(probs.AsSpan()));
        }
        Console.WriteLine($"Saved Half logits + softmax probs to {path}");
    }

    public static void PrintCompareDiff(
        string pyPath,
        string csPath,
        int sentenceCount)
    {
        if (!File.Exists(pyPath) || !File.Exists(csPath))
        {
            Console.WriteLine($"Skipping diff: missing {pyPath} or {csPath}.");
            return;
        }

        var py = ReadCompareOutput(pyPath, sentenceCount);
        var cs = ReadCompareOutput(csPath, sentenceCount);
        if (py == null || cs == null)
        {
            Console.WriteLine("Skipping diff: reference files malformed or length mismatch.");
            return;
        }

        var pyValue = py.Value;
        var csValue = cs.Value;

        float maxAbs = 0f, sumAbs = 0f;
        int argmaxAgree = 0;
        for (int s = 0; s < sentenceCount; s++)
        {
            for (int c = 0; c < 2; c++)
            {
                float diff = MathF.Abs(csValue.Logits[s * 2 + c] - pyValue.Logits[s * 2 + c]);
                if (diff > maxAbs) maxAbs = diff;
                sumAbs += diff;
            }
            if (argmaxOf(csValue.Logits, s) == argmaxOf(pyValue.Logits, s)) argmaxAgree++;
        }

        Console.WriteLine();
        Console.WriteLine($"max abs logit diff: {maxAbs:F8}");
        Console.WriteLine($"mean abs logit diff: {sumAbs / (sentenceCount * 2):F8}");
        Console.WriteLine($"argmax agreement: {argmaxAgree}/{sentenceCount}");
        Console.WriteLine();

        for (int s = 0; s < sentenceCount; s++)
        {
            string csLabel = Label(argmaxOf(csValue.Logits, s));
            string pyLabel = Label(argmaxOf(pyValue.Logits, s));
            Console.WriteLine($"  [{s}] C# {csLabel,8}  Py {pyLabel,8}  {(csLabel == pyLabel ? "" : "<- MISMATCH")}");
        }

        static int argmaxOf(float[] logits, int s) => logits[s * 2 + 1] > logits[s * 2] ? 1 : 0;
    }

    static (float[] Logits, float[] Probs)? ReadCompareOutput(string path, int sentenceCount)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 4) return null;
        int n = BitConverter.ToInt32(bytes, 0);
        if (n != sentenceCount) return null;
        int floatCount = n * 2;
        int expected = 4 + floatCount * 8;
        if (bytes.Length < expected) return null;

        var logits = new float[floatCount];
        var probs = new float[floatCount];
        Buffer.BlockCopy(bytes, 4, logits, 0, floatCount * 4);
        Buffer.BlockCopy(bytes, 4 + floatCount * 4, probs, 0, floatCount * 4);
        return (logits, probs);
    }
}
