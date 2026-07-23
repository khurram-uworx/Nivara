using Microsoft.Agents.AI.Workflows;
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
        var entities = ModelInferenceHelper.RunTokenClassifier(
            _model, _tokenizer, text, _maxSeqLen, EntityClasses);

        return ValueTask.FromResult(JsonSerializer.Serialize(entities));
    }
}
