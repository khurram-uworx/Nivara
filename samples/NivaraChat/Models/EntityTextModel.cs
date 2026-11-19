using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using NivaraChat.Helpers;
using System.Text.Json;

namespace NivaraChat.Models;

internal sealed class EntityTextModel : ITextModel
{
    private readonly TokenClassifierModel<float> _model;
    private readonly SilkTokenClassifier? _silk;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public string Name => "NivaraEntity";

    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public EntityTextModel(TokenClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20, SilkTokenClassifier? silk = null)
    {
        _model = model;
        _model.Eval();
        _silk = silk;
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
        float[] logits;
        if (_silk is not null)
        {
            var tokens = _tokenizer.Encode(input, fixedLength: _maxSeqLen, addBosEos: false);
            logits = _silk.ForwardLogits(tokens);
        }
        else
        {
            var tensorInput = ModelInferenceHelper.ToTensor(_tokenizer, input, _maxSeqLen, addBosEos: false);
            logits = _model.Forward(tensorInput).Data.ToArray();
        }

        var wordTokens = TextTokenizer.Tokenize(input);
        var entities = new Dictionary<string, List<string>>();
        foreach (var cls in EntityClasses)
        {
            if (cls == "O") continue;
            var entityType = cls.Replace("B-", "");
            entities[entityType] = [];
        }

        int numClasses = EntityClasses.Length;
        for (int i = 0; i < Math.Min(wordTokens.Count, _maxSeqLen); i++)
        {
            int bestClass = ModelInferenceHelper.ArgMax(logits, i * numClasses, numClasses);
            string label = EntityClasses[bestClass];
            if (label != "O")
            {
                var entityType = label.Replace("B-", "");
                if (entities.ContainsKey(entityType))
                    entities[entityType].Add(wordTokens[i]);
            }
        }

        int entityCount = entities.Values.Sum(e => e.Count);
        var confidence = entityCount > 0 ? 0.85f : 0.2f;
        var result = JsonSerializer.Serialize(entities);
        if (confidence < 0.4f)
            return $"Unable to extract entities (confidence: {confidence:F2})\n{result}";
        return $"{result}\n(confidence: {confidence:F2})";
    }
}
