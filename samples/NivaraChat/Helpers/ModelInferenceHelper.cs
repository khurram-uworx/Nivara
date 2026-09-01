using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using System.Numerics.Tensors;

namespace NivaraChat.Helpers;

internal static class ModelInferenceHelper
{
    public static ReverseGradTensor<float> ToTensor(TextTokenizer tokenizer, string input, int maxSeqLen, bool addBosEos = true)
    {
        var tokens = tokenizer.Encode(input, fixedLength: maxSeqLen, addBosEos: addBosEos);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        return ReverseGradTensor<float>.FromMatrix(data, 1, maxSeqLen, requiresGrad: false);
    }

    public static int ArgMax(NivaraColumn<float> logits, int offset, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (logits[offset + i] > logits[offset + best]) best = i;
        return best;
    }

    public static float SoftmaxConfidence(NivaraColumn<float> logits, int offset, int count, int predClass)
    {
        var slice = new float[count];
        for (int i = 0; i < count; i++)
            slice[i] = logits[offset + i];

        float maxVal = TensorPrimitives.Max(slice.AsSpan());
        TensorPrimitives.Add(slice.AsSpan(), -maxVal, slice);
        TensorPrimitives.Exp(slice.AsSpan(), slice);
        float sumExp = TensorPrimitives.Sum(slice.AsSpan());
        return slice[predClass] / sumExp;
    }

    public static int RunClassifier(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, int numClasses, bool addBosEos = true)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos);
        var logits = model.Forward(tensorInput);
        return ArgMax(logits.Data, 0, numClasses);
    }

    public static (int bestClass, float confidence) RunClassifierWithConfidence(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, int numClasses, bool addBosEos = true)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos);
        var logits = model.Forward(tensorInput);
        int bestClass = ArgMax(logits.Data, 0, numClasses);
        float confidence = SoftmaxConfidence(logits.Data, 0, numClasses, bestClass);
        return (bestClass, confidence);
    }

    public static float RunTokenClassifierWithConfidence(
        TokenClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, ReadOnlySpan<string> entityClasses)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos: false);
        var logits = model.Forward(tensorInput);

        int numClasses = entityClasses.Length;
        int numTokens = Math.Min(TextTokenizer.Tokenize(input).Count, maxSeqLen);
        if (numTokens == 0) return 0f;

        float totalConfidence = 0f;
        for (int i = 0; i < numTokens; i++)
        {
            int bestClass = ArgMax(logits.Data, i * numClasses, numClasses);
            totalConfidence += SoftmaxConfidence(logits.Data, i * numClasses, numClasses, bestClass);
        }
        return totalConfidence / numTokens;
    }

    public static Dictionary<string, List<string>> RunTokenClassifier(
        TokenClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, ReadOnlySpan<string> entityClasses)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos: false);
        var logits = model.Forward(tensorInput);

        var wordTokens = TextTokenizer.Tokenize(input);
        var entities = new Dictionary<string, List<string>>();
        foreach (var cls in entityClasses)
        {
            if (cls == "O") continue;
            var entityType = cls.Replace("B-", "");
            entities[entityType] = [];
        }

        int numClasses = entityClasses.Length;
        for (int i = 0; i < Math.Min(wordTokens.Count, maxSeqLen); i++)
        {
            int bestClass = ArgMax(logits.Data, i * numClasses, numClasses);
            string label = entityClasses[bestClass];
            if (label != "O")
            {
                var entityType = label.Replace("B-", "");
                if (entities.ContainsKey(entityType))
                    entities[entityType].Add(wordTokens[i]);
            }
        }

        return entities;
    }
}
