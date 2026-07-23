using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;

namespace NivaraChat;

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
        float maxVal = float.MinValue;
        for (int i = 0; i < count; i++)
            if (logits[offset + i] > maxVal) maxVal = logits[offset + i];

        float sumExp = 0f;
        for (int i = 0; i < count; i++)
            sumExp += MathF.Exp(logits[offset + i] - maxVal);
        return MathF.Exp(logits[offset + predClass] - maxVal) / sumExp;
    }

    public static int RunClassifier(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, int numClasses, bool addBosEos = true)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos);
        var logits = model.Forward(tensorInput);
        return ArgMax(logits.Data, 0, numClasses);
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
