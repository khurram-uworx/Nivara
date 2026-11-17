using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using NivaraChat.Helpers;

namespace NivaraChat.Models;

internal sealed class ValidatorTextModel : ITextModel
{
    private readonly TextClassifierModel<float> _model;
    private readonly SilkTextClassifier? _silk;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraValidator";

    public ValidatorTextModel(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 40, SilkTextClassifier? silk = null)
    {
        _model = model;
        _model.Eval();
        _silk = silk;
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
        float[] logitsArr;
        if (_silk is not null)
        {
            var toks = _tokenizer.Encode(input, fixedLength: _maxSeqLen);
            logitsArr = _silk.ForwardLogits(toks);
        }
        else
        {
            var tensorInput = ModelInferenceHelper.ToTensor(_tokenizer, input, _maxSeqLen);
            var logits = _model.Forward(tensorInput);
            logitsArr = logits.Data.ToArray();
        }

        float validScore = logitsArr[0];
        float invalidScore = logitsArr[1];
        float confidence = MathF.Exp(validScore) / (MathF.Exp(validScore) + MathF.Exp(invalidScore));
        bool isValid = validScore > invalidScore;
        var label = isValid ? "VALID" : "INVALID";
        return $"{{\"validation\":\"{label}\",\"confidence\":{confidence:F2}}}";
    }
}
