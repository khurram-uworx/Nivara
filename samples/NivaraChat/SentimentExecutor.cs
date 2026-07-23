using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff;
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
        var tokens = _tokenizer.Encode(text, fixedLength: _maxSeqLen);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];
        var input = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(input);

        int bestClass = 0;
        float bestVal = logits.Data[0];
        for (int c = 1; c < 3; c++)
        {
            if (logits.Data[c] > bestVal) { bestVal = logits.Data[c]; bestClass = c; }
        }

        return ValueTask.FromResult(Classes[bestClass]);
    }
}
