using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using System.Text.Json;

namespace NivaraChat;

internal sealed class EntityTextModel : ITextModel
{
    private readonly TokenClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private const int DefaultMaxSeqLen = 20;
    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public string Name => "NivaraEntity";

    public EntityTextModel(TokenClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = DefaultMaxSeqLen)
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public static EntityTextModel Load(string saveDir)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(saveDir, "entity_tokenizer.json"));
        var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, 5, DefaultMaxSeqLen);
        ModelSerializer.Load(model, Path.Combine(saveDir, "entity_model.json"));
        model.Eval();
        return new EntityTextModel(model, tokenizer);
    }

    public string Process(string input)
    {
        var tokens = _tokenizer.Encode(input, fixedLength: _maxSeqLen, addBosEos: false);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        var tensorInput = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(tensorInput);

        var wordTokens = TextTokenizer.Tokenize(input);
        var entities = new Dictionary<string, List<string>>
        {
            ["person"] = [],
            ["org"] = [],
            ["date"] = [],
            ["location"] = []
        };

        int numClasses = EntityClasses.Length;
        int entityCount = 0;
        for (int i = 0; i < Math.Min(wordTokens.Count, _maxSeqLen); i++)
        {
            int bestClass = 0;
            float bestVal = logits.Data[i * numClasses];
            for (int c = 1; c < numClasses; c++)
            {
                if (logits.Data[i * numClasses + c] > bestVal)
                {
                    bestVal = logits.Data[i * numClasses + c];
                    bestClass = c;
                }
            }

            string label = EntityClasses[bestClass];
            if (label != "O")
            {
                string entityType = label.Replace("B-", "");
                if (entities.ContainsKey(entityType))
                    entities[entityType].Add(wordTokens[i]);
                entityCount++;
            }
        }

        var confidence = entityCount > 0 ? 0.85f : 0.2f;
        var result = JsonSerializer.Serialize(entities);
        if (confidence < 0.4f)
            return $"Unable to extract entities (confidence: {confidence:F2})\n{result}";
        return $"{result}\n(confidence: {confidence:F2})";
    }
}
