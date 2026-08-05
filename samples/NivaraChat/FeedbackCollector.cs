using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using NivaraChat.Training;

namespace NivaraChat;

internal sealed class FeedbackCollector
{
    static readonly string[] ValidIntents = ["factual", "question", "command", "complaint", "chitchat"];

    readonly TextClassifierModel<float> _model;
    readonly TextTokenizer _tokenizer;
    readonly IChatClient _chatClient;
    readonly int _maxSeqLen;
    readonly float _threshold;
    readonly string _modelsDir;

    readonly List<(string text, int label)> _buffer = [];
    readonly int _retrainThreshold;

    public int BufferCount => _buffer.Count;
    public int TotalCollected { get; private set; }

    public FeedbackCollector(
        TextClassifierModel<float> model,
        TextTokenizer tokenizer,
        IChatClient chatClient,
        string modelsDir,
        float threshold = 0.6f,
        int maxSeqLen = 20,
        int retrainThreshold = 50)
    {
        _model = model;
        _tokenizer = tokenizer;
        _chatClient = chatClient;
        _modelsDir = modelsDir;
        _threshold = threshold;
        _maxSeqLen = maxSeqLen;
        _retrainThreshold = retrainThreshold;
    }

    public async Task<(string intent, float confidence, bool collected)> ClassifyAsync(string input)
    {
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _model, _tokenizer, input, _maxSeqLen, numClasses: 5);

        string nivaraIntent = ValidIntents[bestClass];

        if (confidence >= _threshold)
            return (nivaraIntent, confidence, false);

        var llmIntent = await AskLlmForIntent(input);
        if (llmIntent != null)
        {
            _buffer.Add((input, Array.IndexOf(ValidIntents, llmIntent)));
            TotalCollected++;
            return (llmIntent, confidence, true);
        }

        return (nivaraIntent, confidence, false);
    }

    async Task<string?> AskLlmForIntent(string input)
    {
        var prompt =
            $"Classify the intent of the following text as exactly one of: factual, question, command, complaint, chitchat.\n" +
            $"Return ONLY the single word label, nothing else.\n\n" +
            $"Text: {input}\n\nIntent:";

        try
        {
            var response = await _chatClient.GetResponseAsync(prompt);
            var raw = response.Text?.Trim().ToLower() ?? "";
            var cleaned = raw.Trim('.', ',', '!', '?', '"', '\'');

            if (Array.Exists(ValidIntents, intent => intent == cleaned))
                return cleaned;

            foreach (var intent in ValidIntents)
                if (cleaned.Contains(intent))
                    return intent;
        }
        catch
        {
        }

        return null;
    }

    public bool ShouldRetrain() => _buffer.Count >= _retrainThreshold;

    public (TextClassifierModel<float> model, TextTokenizer tokenizer) Retrain()
    {
        Console.WriteLine($"  Retraining with {_buffer.Count} LLM-validated examples...");

        var feedbackTexts = _buffer.Select(p => p.text).ToArray();
        var feedbackLabels = _buffer.Select(p => p.label).ToArray();

        var (model, tokenizer) = IntentTrainer.TrainIncremental(
            feedbackTexts, feedbackLabels,
            modelsDir: _modelsDir,
            additionalEpochs: 5,
            batchSize: 16);

        _buffer.Clear();
        return (model, tokenizer);
    }
}
