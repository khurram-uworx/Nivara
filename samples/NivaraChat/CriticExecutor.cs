using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using System.Text.Json;

namespace NivaraChat;

internal sealed class CriticExecutor : Executor<string, string>
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public CriticExecutor(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 40)
        : base("Critic")
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        return ScoreAsync(input, cancellationToken);
    }

    public ValueTask<string> ScoreAsync(string input, CancellationToken cancellationToken = default)
    {
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _model, _tokenizer, input, _maxSeqLen, numClasses: 2);

        var result = JsonSerializer.Serialize(new
        {
            score = confidence,
            verdict = bestClass == 1 ? "GOOD" : "POOR",
            acceptable = bestClass == 1 && confidence >= 0.8f
        });
        return ValueTask.FromResult(result);
    }
}
