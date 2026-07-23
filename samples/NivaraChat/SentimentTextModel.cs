using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;

namespace NivaraChat;

internal sealed class SentimentTextModel : ITextModel
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraSentiment";

    private static readonly string[] Labels = ["Negative", "Neutral", "Positive"];
    private const int NumClasses = 3;

    public SentimentTextModel(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
    {
        _model = model;
        _model.Eval();
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
        var tensorInput = ModelInferenceHelper.ToTensor(_tokenizer, input, _maxSeqLen);
        var logits = _model.Forward(tensorInput);
        int bestClass = ModelInferenceHelper.ArgMax(logits.Data, 0, NumClasses);
        float confidence = ModelInferenceHelper.SoftmaxConfidence(logits.Data, 0, NumClasses, bestClass);
        var label = Labels[bestClass];
        if (confidence < 0.4f)
            return $"Unable to determine sentiment (confidence: {confidence:F2})";
        return $"{label} (confidence: {confidence:F2})";
    }
}
