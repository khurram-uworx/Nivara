using Nivara.AutoDiff.Nn;
using NivaraChat.Helpers;
using System.ComponentModel;
using System.Text.Json;

namespace NivaraChat.Tools;

public static class NivaraToolFunctions
{
    private static TextClassifierModel<float>? _sentimentModel;
    private static TextTokenizer? _sentimentTokenizer;
    private static TokenClassifierModel<float>? _entityModel;
    private static TextTokenizer? _entityTokenizer;
    private static TextClassifierModel<float>? _validatorModel;
    private static TextTokenizer? _validatorTokenizer;
    private static SilkTextClassifier? _sentimentSilk;
    private static SilkTextClassifier? _validatorSilk;

    private static readonly string[] SentimentClasses = ["negative", "neutral", "positive"];
    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public static void Initialize(
        TextClassifierModel<float> sentimentModel, TextTokenizer sentimentTok,
        TokenClassifierModel<float> entityModel, TextTokenizer entityTok,
        TextClassifierModel<float> validatorModel, TextTokenizer validatorTok,
        SilkTextClassifier? sentimentSilk = null, SilkTextClassifier? validatorSilk = null)
    {
        _sentimentModel = sentimentModel;
        _sentimentTokenizer = sentimentTok;
        _entityModel = entityModel;
        _entityTokenizer = entityTok;
        _validatorModel = validatorModel;
        _validatorTokenizer = validatorTok;
        _sentimentSilk = sentimentSilk;
        _validatorSilk = validatorSilk;
    }

    [Description("Analyze sentiment of text. Returns positive/negative/neutral with confidence score.")]
    public static string AnalyzeSentiment(
        [Description("The text to analyze")] string text)
    {
        EnsureSentimentModel();
        (int bestClass, float confidence) r;
        if (_sentimentSilk is not null)
            r = ModelInferenceHelper.RunClassifierWithConfidence(_sentimentSilk, _sentimentTokenizer!, text, maxSeqLen: 20, numClasses: 3);
        else
            r = ModelInferenceHelper.RunClassifierWithConfidence(_sentimentModel!, _sentimentTokenizer!, text, maxSeqLen: 20, numClasses: 3);
        var bestClass = r.bestClass;
        var confidence = r.confidence;
        return JsonSerializer.Serialize(new { label = SentimentClasses[bestClass], confidence });
    }

    [Description("Extract named entities (person, organization, date, location) from text.")]
    public static string ExtractEntities(
        [Description("The text to extract entities from")] string text)
    {
        EnsureEntityModel();
        var entities = ModelInferenceHelper.RunTokenClassifier(
            _entityModel!, _entityTokenizer!, text, maxSeqLen: 20, EntityClasses);
        float confidence = ModelInferenceHelper.RunTokenClassifierWithConfidence(
            _entityModel!, _entityTokenizer!, text, maxSeqLen: 20, EntityClasses);
        return JsonSerializer.Serialize(new { entities, confidence });
    }

    [Description("Validate whether a response is consistent with the original text.")]
    public static string ValidateResponse(
        [Description("The original text")] string original,
        [Description("The response to validate")] string response)
    {
        EnsureValidatorModel();
        var input = $"{original} || {response}";
        (int bestClass, float confidence) r;
        if (_validatorSilk is not null)
            r = ModelInferenceHelper.RunClassifierWithConfidence(_validatorSilk, _validatorTokenizer!, input, maxSeqLen: 40, numClasses: 2);
        else
            r = ModelInferenceHelper.RunClassifierWithConfidence(_validatorModel!, _validatorTokenizer!, input, maxSeqLen: 40, numClasses: 2);
        return JsonSerializer.Serialize(new { consistent = r.bestClass == 1, confidence = r.confidence });
    }

    private static void EnsureSentimentModel()
    {
        if (_sentimentModel == null || _sentimentTokenizer == null)
            throw new InvalidOperationException("NivaraToolFunctions.Initialize() must be called before using tools.");
    }

    private static void EnsureEntityModel()
    {
        if (_entityModel == null || _entityTokenizer == null)
            throw new InvalidOperationException("NivaraToolFunctions.Initialize() must be called before using tools.");
    }

    private static void EnsureValidatorModel()
    {
        if (_validatorModel == null || _validatorTokenizer == null)
            throw new InvalidOperationException("NivaraToolFunctions.Initialize() must be called before using tools.");
    }
}
