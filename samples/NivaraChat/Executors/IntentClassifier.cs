using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using NivaraChat.Helpers;
using System.Text.Json;

namespace NivaraChat.Executors;

internal sealed class IntentClassifier : Executor<string, string>
{
    private readonly TextClassifierModel<float>? _model;
    private readonly SilkTextClassifier? _silk;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private static readonly string[] Intents = ["factual", "question", "command", "complaint", "chitchat"];

    public IntentClassifier(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20, SilkTextClassifier? silk = null)
        : base("IntentClassifier")
    {
        _model = model;
        _model.Eval();
        _silk = silk;
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        (int bestClass, float confidence) r;
        if (_silk is not null)
            r = ModelInferenceHelper.RunClassifierWithConfidence(_silk, _tokenizer, text, _maxSeqLen, numClasses: 5);
        else
            r = ModelInferenceHelper.RunClassifierWithConfidence(_model!, _tokenizer, text, _maxSeqLen, numClasses: 5);
        var result = JsonSerializer.Serialize(new { intent = Intents[r.bestClass], confidence = r.confidence, text });
        return ValueTask.FromResult(result);
    }
}
