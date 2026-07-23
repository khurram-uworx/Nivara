using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;

namespace NivaraChat;

internal sealed class ValidatorTextModel : ITextModel
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraValidator";

    public ValidatorTextModel(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 40)
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public static ValidatorTextModel Load(string saveDir, bool useAgentsFormat = false)
    {
        var suffix = useAgentsFormat ? "agents_validator" : "validator";
        var tokenizer = TextTokenizer.Load(Path.Combine(saveDir, $"{suffix}_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 2, 40);
        ModelSerializer.Load(model, Path.Combine(saveDir, $"{suffix}_model.json"));
        model.Eval();
        return new ValidatorTextModel(model, tokenizer);
    }

    public string Process(string input)
    {
        var tokens = _tokenizer.Encode(input, fixedLength: _maxSeqLen);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        var tensorInput = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(tensorInput);

        float validScore = logits.Data[0];
        float invalidScore = logits.Data[1];
        float confidence = MathF.Exp(validScore) / (MathF.Exp(validScore) + MathF.Exp(invalidScore));
        bool isValid = validScore > invalidScore;
        var label = isValid ? "VALID" : "INVALID";
        return $"{{\"validation\":\"{label}\",\"confidence\":{confidence:F2}}}";
    }
}
