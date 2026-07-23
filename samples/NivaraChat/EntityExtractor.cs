using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using System.Text.Json;

namespace NivaraChat;

internal sealed class EntityExtractor : Executor<string, string>
{
    private readonly TokenClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public EntityExtractor(TokenClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
        : base("EntityExtractor")
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var tokens = _tokenizer.Encode(text, fixedLength: _maxSeqLen, addBosEos: false);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(input);

        var wordTokens = TextTokenizer.Tokenize(text);
        var entities = new Dictionary<string, List<string>>
        {
            ["person"] = [],
            ["org"] = [],
            ["date"] = [],
            ["location"] = []
        };

        int numClasses = EntityClasses.Length;
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
            }
        }

        return ValueTask.FromResult(JsonSerializer.Serialize(entities));
    }
}
