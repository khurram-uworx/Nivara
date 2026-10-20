using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using NivaraChat.Modes;
using Nivara.AI;
using NivaraChat.Qwen;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class QwenToolsNivaraLoopTests
{
    static string QwenModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static string NivaraModelsDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "nivarachat");

    static readonly Gpt2BpeTokenizer? CachedTokenizer;
    static readonly (LlamaForCausalLM<float> Model, LlamaConfig Config)? CachedModel;
    static readonly (TextClassifierModel<float> Sentiment, TextTokenizer SentTok)? CachedSentiment;

    static QwenToolsNivaraLoopTests()
    {
        var vocab = Path.Combine(QwenModelDir, "vocab.json");
        var merges = Path.Combine(QwenModelDir, "merges.txt");
        var tokenizerJson = Path.Combine(QwenModelDir, "tokenizer.json");
        var safetensors = Path.Combine(QwenModelDir, "model.safetensors");
        var configJson = Path.Combine(QwenModelDir, "config.json");

        if (File.Exists(vocab) && File.Exists(merges) && File.Exists(tokenizerJson)
            && File.Exists(safetensors) && File.Exists(configJson))
        {
            CachedTokenizer = new Gpt2BpeTokenizer(vocab, merges, tokenizerJsonPath: tokenizerJson);
            var tensors = SafeTensorsLoader.Read<float>(safetensors);
            var config = LlamaConfig.FromJson(File.ReadAllText(configJson));
            var model = LlamaLoader.Load<float, float>(config, tensors);
            CachedModel = (model, config);
        }

        var sentimentModelPath = Path.Combine(NivaraModelsDir, "sentiment_model.json");
        if (File.Exists(sentimentModelPath))
        {
            var (m, t) = ModeHelpers.LoadSentimentModel(NivaraModelsDir);
            CachedSentiment = (m, t);
        }
    }

    [Test]
    public async Task NivaraSentimentTool_FiresAndClosesWithCleanFinalAnswer()
    {
        if (CachedModel is null || CachedTokenizer is null)
            Assert.Ignore("Qwen model files absent; skipping.");
        if (CachedSentiment is null)
            Assert.Ignore("Nivara sentiment model absent; skipping.");

        var (model, config) = CachedModel.Value;
        var (sentimentModel, sentimentTok) = CachedSentiment.Value;
        var tool = NivaraAIFunctions.CreateSentimentTool(sentimentModel, sentimentTok);

        using var inner = new QwenChatClient<float>(
            model, CachedTokenizer, config,
            maxNewTokens: 128,
            knownToolNames: [tool.Name]);
        using var loop = new FunctionInvokingChatClient(inner)
        {
            MaximumIterationsPerRequest = 3,
        };

        var history = new List<ChatMessage>
        {
            new(ChatRole.System, QwenChatTemplate.BuildToolsSystemMessage([tool])),
            new(ChatRole.User, "Analyze the sentiment of: 'I absolutely love this product, it works great!'"),
        };

        var response = await loop.GetResponseAsync(history, new ChatOptions { Tools = [tool] });
        var messages = response.Messages;

        Assert.That(messages, Is.Not.Empty);
        Assert.That(messages.Any(m => m.Contents?.OfType<FunctionCallContent>().Any() == true), Is.True, "expected an assistant tool call");
        Assert.That(messages.Any(m => m.Role == ChatRole.Tool), Is.True, "expected a tool result turn");

        var finalAssistant = messages.LastOrDefault(m => m.Role == ChatRole.Assistant);
        Assert.That(finalAssistant, Is.Not.Null);
        Assert.That(finalAssistant!.Text, Is.Not.Null.And.Not.Empty);
        Assert.That(finalAssistant.Text!.ToLowerInvariant(), Does.Contain("positive"));
    }
}
