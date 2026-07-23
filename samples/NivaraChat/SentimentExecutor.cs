using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;

namespace NivaraChat;

internal sealed class SentimentExecutor : Executor<string, string>
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private static readonly string[] Classes = ["negative", "neutral", "positive"];

    public SentimentExecutor(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
        : base("Sentiment")
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        int bestClass = ModelInferenceHelper.RunClassifier(_model, _tokenizer, text, _maxSeqLen, numClasses: 3);
        return ValueTask.FromResult(Classes[bestClass]);
    }
}
