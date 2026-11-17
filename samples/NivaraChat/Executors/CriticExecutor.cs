using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using NivaraChat.Helpers;
using System.Text.Json;

namespace NivaraChat.Executors;

internal sealed class CriticExecutor : Executor<string, string>
{
    private readonly TextClassifierModel<float>? _model;
    private readonly SilkTextClassifier? _silk;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public CriticExecutor(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 40, SilkTextClassifier? silk = null)
        : base("Critic")
    {
        _model = model;
        _model.Eval();
        _silk = silk;
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        return ScoreAsync(input, cancellationToken);
    }

    public ValueTask<string> ScoreAsync(string input, CancellationToken cancellationToken = default)
    {
        (int bestClass, float confidence) r;
        if (_silk is not null)
            r = ModelInferenceHelper.RunClassifierWithConfidence(_silk, _tokenizer, input, _maxSeqLen, numClasses: 2);
        else
            r = ModelInferenceHelper.RunClassifierWithConfidence(_model!, _tokenizer, input, _maxSeqLen, numClasses: 2);

        var result = JsonSerializer.Serialize(new
        {
            score = r.confidence,
            verdict = r.bestClass == 1 ? "GOOD" : "POOR",
            acceptable = r.bestClass == 1 && r.confidence >= 0.8f
        });
        return ValueTask.FromResult(result);
    }
}
