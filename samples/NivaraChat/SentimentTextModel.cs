using Nivara.AutoDiff;
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
        var tokens = _tokenizer.Encode(input, fixedLength: _maxSeqLen);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        var tensorInput = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(tensorInput);

        int bestClass = 0;
        float bestVal = logits.Data[0];
        for (int c = 1; c < 3; c++)
        {
            if (logits.Data[c] > bestVal) { bestVal = logits.Data[c]; bestClass = c; }
        }

        float confidence = SoftmaxConfidence(logits, bestClass);
        var label = Labels[bestClass];
        if (confidence < 0.4f)
            return $"Unable to determine sentiment (confidence: {confidence:F2})";
        return $"{label} (confidence: {confidence:F2})";
    }

    private static float SoftmaxConfidence(ReverseGradTensor<float> logits, int predClass)
    {
        float maxVal = float.MinValue;
        for (int i = 0; i < logits.Data.Length; i++)
            if (logits.Data[i] > maxVal) maxVal = logits.Data[i];

        float sumExp = 0f;
        for (int i = 0; i < logits.Data.Length; i++)
            sumExp += MathF.Exp(logits.Data[i] - maxVal);
        return MathF.Exp(logits.Data[predClass] - maxVal) / sumExp;
    }
}
