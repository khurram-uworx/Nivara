using Microsoft.Extensions.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using System.ComponentModel;
using System.Text.Json;

namespace Nivara.Samples;

/// <summary>
/// Factory methods that wrap Nivara's trained text models as
/// <see cref="AIFunction"/> tools for use with any <see cref="IChatClient"/>
/// function-calling loop (Microsoft.Extensions.AI). Captures the model and
/// tokenizer as trusted state.
/// </summary>
public static class NivaraAIFunctions
{
    public const string SentimentToolName = "analyzeSentiment";
    public const string EntitiesToolName = "extractEntities";
    public const string ValidatorToolName = "validateResponse";

    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];
    private static readonly string[] SentimentLabels = ["negative", "neutral", "positive"];

    public static AIFunction CreateSentimentTool(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        int maxSeqLen = 20, int numClasses = 3,
        string? name = null, string? description = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        name ??= SentimentToolName;
        description ??= "Analyze sentiment of text. Returns positive/negative/neutral with confidence score.";

        return AIFunctionFactory.Create(
            ([Description("The text to analyze")] string text) =>
            {
                var (bestClass, confidence) = RunClassifierWithConfidence(model, tokenizer, text, maxSeqLen, numClasses);
                var label = (bestClass >= 0 && bestClass < SentimentLabels.Length) ? SentimentLabels[bestClass] : "unknown";
                return JsonSerializer.Serialize(new { label, confidence });
            },
            name: name,
            description: description);
    }

    public static AIFunction CreateEntityTool(
        TokenClassifierModel<float> model, TextTokenizer tokenizer,
        int maxSeqLen = 20,
        string? name = null, string? description = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        name ??= EntitiesToolName;
        description ??= "Extract named entities (person, organization, date, location) from text.";

        return AIFunctionFactory.Create(
            ([Description("The text to extract entities from")] string text) =>
            {
                var entities = RunTokenClassifier(model, tokenizer, text, maxSeqLen, EntityClasses);
                float confidence = RunTokenClassifierWithConfidence(model, tokenizer, text, maxSeqLen, EntityClasses);
                return JsonSerializer.Serialize(new { entities, confidence });
            },
            name: name,
            description: description);
    }

    public static AIFunction CreateValidatorTool(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        int maxSeqLen = 40,
        string? name = null, string? description = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        name ??= ValidatorToolName;
        description ??= "Validate whether a response is consistent with the original text.";

        return AIFunctionFactory.Create(
            ([Description("The original text")] string original,
             [Description("The response to validate")] string response) =>
            {
                var input = $"{original} || {response}";
                var (bestClass, confidence) = RunClassifierWithConfidence(model, tokenizer, input, maxSeqLen, 2);
                return JsonSerializer.Serialize(new { consistent = bestClass == 1, confidence });
            },
            name: name,
            description: description);
    }

    public static IReadOnlyList<AIFunction> CreateRegistered(
        TextClassifierModel<float>? sentimentModel, TextTokenizer? sentimentTok,
        TokenClassifierModel<float>? entityModel, TextTokenizer? entityTok,
        TextClassifierModel<float>? validatorModel, TextTokenizer? validatorTok,
        string toolsCsv = "all")
    {
        var selection = ParseSelection(toolsCsv);
        var result = new List<AIFunction>();

        if (selection.Contains("all") || selection.Contains("sentiment"))
        {
            if (sentimentModel is not null && sentimentTok is not null)
                result.Add(CreateSentimentTool(sentimentModel, sentimentTok));
        }
        if (selection.Contains("all") || selection.Contains("entities"))
        {
            if (entityModel is not null && entityTok is not null)
                result.Add(CreateEntityTool(entityModel, entityTok));
        }
        if (selection.Contains("all") || selection.Contains("validate"))
        {
            if (validatorModel is not null && validatorTok is not null)
                result.Add(CreateValidatorTool(validatorModel, validatorTok));
        }

        return result;
    }

    private static HashSet<string> ParseSelection(string toolsCsv)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var part in (toolsCsv ?? "all").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var v = part.ToLowerInvariant();
            if (v is "all" or "sentiment" or "entities" or "validate")
                set.Add(v);
        }
        if (set.Count == 0) set.Add("all");
        return set;
    }

    // ---- float inference helpers ----

    private static (int bestClass, float confidence) RunClassifierWithConfidence(
        TextClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, int numClasses, bool addBosEos = true)
    {
        var tensorInput = ToTensor(tokenizer, input, maxSeqLen, addBosEos);
        var logits = model.Forward(tensorInput);
        int bestClass = ArgMax(logits.Data, 0, numClasses);
        float confidence = SoftmaxConfidence(logits.Data, 0, numClasses, bestClass);
        return (bestClass, confidence);
    }

    private static Dictionary<string, List<string>> RunTokenClassifier(
        TokenClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, string[] entityClasses)
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

    private static float RunTokenClassifierWithConfidence(
        TokenClassifierModel<float> model, TextTokenizer tokenizer,
        string input, int maxSeqLen, string[] entityClasses)
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

    private static ReverseGradTensor<float> ToTensor(TextTokenizer tokenizer, string input, int maxSeqLen, bool addBosEos = true)
    {
        var tokens = tokenizer.Encode(input, fixedLength: maxSeqLen, addBosEos: addBosEos);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        return ReverseGradTensor<float>.FromMatrix(data, 1, maxSeqLen, requiresGrad: false);
    }

    private static int ArgMax(NivaraColumn<float> logits, int offset, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (logits[offset + i] > logits[offset + best]) best = i;
        return best;
    }

    private static float SoftmaxConfidence(NivaraColumn<float> logits, int offset, int count, int predClass)
    {
        var slice = new float[count];
        for (int i = 0; i < count; i++)
            slice[i] = logits[offset + i];

        float maxVal = System.Numerics.Tensors.TensorPrimitives.Max(slice.AsSpan());
        System.Numerics.Tensors.TensorPrimitives.Add(slice.AsSpan(), -maxVal, slice);
        System.Numerics.Tensors.TensorPrimitives.Exp(slice.AsSpan(), slice);
        float sumExp = System.Numerics.Tensors.TensorPrimitives.Sum(slice.AsSpan());
        return slice[predClass] / sumExp;
    }
}
