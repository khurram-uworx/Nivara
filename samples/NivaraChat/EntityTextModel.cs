using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using System.Text.Json;

namespace NivaraChat;

internal sealed class EntityTextModel : ITextModel
{
    private readonly TokenClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraEntity";

    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public EntityTextModel(TokenClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public static EntityTextModel Load(string saveDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(saveDir, "entity_tokenizer.json"));
        var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, 20);
        ModelSerializer.Load(model, Path.Combine(saveDir, "entity_model.json"));
        model.Eval();
        return new EntityTextModel(model, tokenizer);
    }

    public string Process(string input)
    {
        var entities = ModelInferenceHelper.RunTokenClassifier(
            _model, _tokenizer, input, _maxSeqLen, EntityClasses);

        int entityCount = entities.Values.Sum(e => e.Count);
        var confidence = entityCount > 0 ? 0.85f : 0.2f;
        var result = JsonSerializer.Serialize(entities);
        if (confidence < 0.4f)
            return $"Unable to extract entities (confidence: {confidence:F2})\n{result}";
        return $"{result}\n(confidence: {confidence:F2})";
    }
}
