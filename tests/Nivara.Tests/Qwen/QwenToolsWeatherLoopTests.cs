using Microsoft.Extensions.AI;
using Nivara.Samples;
using NivaraChat.Qwen;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

/// <summary>
/// End-to-end native function calling against the real Qwen2.5-0.5B-Instruct checkpoint
/// (issue #382 acceptance): the model emits <c>&lt;tool_call&gt;</c>, the framework executes
/// <c>GetWeather</c>, the result feeds back as <c>&lt;tool_response&gt;</c>, and the loop closes
/// with a clean natural-language final answer within the cap. Skipped when the model files are
/// absent (CI/clean).
/// </summary>
[TestFixture]
public class QwenToolsWeatherLoopTests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static readonly Gpt2BpeTokenizer? CachedTokenizer;
    static readonly (LlamaForCausalLM<float> Model, LlamaConfig Config)? CachedModel;

    static QwenToolsWeatherLoopTests()
    {
        var vocab = Path.Combine(ModelDir, "vocab.json");
        var merges = Path.Combine(ModelDir, "merges.txt");
        var tokenizerJson = Path.Combine(ModelDir, "tokenizer.json");
        var safetensors = Path.Combine(ModelDir, "model.safetensors");
        var configJson = Path.Combine(ModelDir, "config.json");

        if (!File.Exists(vocab) || !File.Exists(merges) || !File.Exists(tokenizerJson)
            || !File.Exists(safetensors) || !File.Exists(configJson))
            return;

        CachedTokenizer = new Gpt2BpeTokenizer(vocab, merges, tokenizerJsonPath: tokenizerJson);

        // BF16 on disk -> F32 compute (SafeTensorsLoader upcasts); qkvBias auto-detected
        // (Qwen2.5-0.5B is the bias variant). Same load path as QwenInstructParityTests.
        var tensors = SafeTensorsLoader.Read<float>(safetensors);
        var config = LlamaConfig.FromJson(File.ReadAllText(configJson));
        var model = LlamaLoader.Load<float, float>(config, tensors);
        CachedModel = (model, config);
    }

    [Test]
    public async Task ToolsWeather_Loop_ClosesWithCleanFinalAnswer()
    {
        if (CachedModel is null || CachedTokenizer is null)
            Assert.Ignore("Qwen model files absent; skipping tool-loop acceptance verification.");

        var (model, config) = CachedModel.Value;
        var weather = QwenSampleTools.CreateWeatherTool();

        using var inner = new QwenChatClient<float>(
            model, CachedTokenizer, config,
            maxNewTokens: 128,
            knownToolNames: [QwenSampleTools.WeatherToolName]);
        using var loop = new FunctionInvokingChatClient(inner)
        {
            MaximumIterationsPerRequest = 3,
        };

        var history = new List<ChatMessage>
        {
            new(ChatRole.System, QwenChatTemplate.BuildToolsSystemMessage([weather])),
            new(ChatRole.User, "What's the weather in Paris?"),
        };

        var response = await loop.GetResponseAsync(history, new ChatOptions { Tools = [weather] });
        var messages = response.Messages;

        Assert.That(messages, Is.Not.Empty);

        // Native function calling must actually happen: an assistant turn with a tool call,
        // then a tool result turn.
        Assert.That(
            messages.Any(m => m.Contents?.OfType<FunctionCallContent>().Any() == true),
            Is.True,
            "expected an assistant <tool_call> turn");
        Assert.That(messages.Any(m => m.Role == ChatRole.Tool), Is.True, "expected a tool result turn");

        // The loop must close with a concrete answer (not loop until the cap, not blank).
        var finalAssistant = messages.LastOrDefault(m => m.Role == ChatRole.Assistant);
        Assert.That(finalAssistant, Is.Not.Null, "loop ended without a final assistant answer");
        Assert.That(finalAssistant!.Text, Is.Not.Null.And.Not.Empty, "final answer must not be blank");

        // Semantic check: the answer must fire the weather conclusion back, i.e. the tool result
        // was consumed rather than ignored.
        Assert.That(finalAssistant!.Text!.ToLowerInvariant(), Does.Contain("partly cloudy"));
    }
}