using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using NivaraChat.Helpers;

namespace NivaraChat.Models;

internal sealed class SentimentTextModel : ITextModel
{
    private readonly TextClassifierModel<float> _model;
    private readonly SilkTextClassifier? _silk;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraSentiment";

    private static readonly string[] Labels = ["Negative", "Neutral", "Positive"];
    private const int NumClasses = 3;

    public SentimentTextModel(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20, SilkTextClassifier? silk = null)
    {
        _model = model;
        _model.Eval();
        _silk = silk;
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public static (SentimentTextModel model, TextTokenizer tokenizer) Load(string saveDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(saveDir, "sentiment_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 3, 20);
        ModelSerializer.Load(model, Path.Combine(saveDir, "sentiment_model.json"));
        model.Eval();
        return (new SentimentTextModel(model, tokenizer), tokenizer);
    }

    public string Process(string input)
    {
        (int bestClass, float confidence) r;
        if (_silk is not null)
            r = ModelInferenceHelper.RunClassifierWithConfidence(_silk, _tokenizer, input, _maxSeqLen, NumClasses);
        else
        {
            var tensorInput = ModelInferenceHelper.ToTensor(_tokenizer, input, _maxSeqLen);
            var logits = _model.Forward(tensorInput);
            int bc = ModelInferenceHelper.ArgMax(logits.Data, 0, NumClasses);
            float cf = ModelInferenceHelper.SoftmaxConfidence(logits.Data, 0, NumClasses, bc);
            r = (bc, cf);
        }
        int bestClass = r.bestClass;
        float confidence = r.confidence;
        var label = Labels[bestClass];
        if (confidence < 0.4f)
            return $"Unable to determine sentiment (confidence: {confidence:F2})";
        return $"{label} (confidence: {confidence:F2})";
    }
}
