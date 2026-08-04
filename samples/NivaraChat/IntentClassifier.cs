using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using System.Text.Json;

namespace NivaraChat;

internal sealed class IntentClassifier : Executor<string, string>
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private static readonly string[] Intents = ["factual", "question", "command", "complaint", "chitchat"];

    public IntentClassifier(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
        : base("IntentClassifier")
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _model, _tokenizer, text, _maxSeqLen, numClasses: 5);
        var result = JsonSerializer.Serialize(new { intent = Intents[bestClass], confidence, text });
        return ValueTask.FromResult(result);
    }
}