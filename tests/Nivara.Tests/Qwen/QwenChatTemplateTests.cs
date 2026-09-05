using Microsoft.Extensions.AI;
using NivaraChat.Qwen;
using NUnit.Framework;
using System.Text;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Pins <see cref="QwenChatTemplate"/> (and the tool-call parse→render round trip that the live
/// tool loop exercises) byte-for-byte against the Torch ground-truth prompts produced by
/// <c>qwen_tool_reference.py</c> (issue #382 Phase 3). Skipped when the fixtures aren't
/// downloaded (CI/clean).
/// </summary>
[TestFixture]
public class QwenChatTemplateTests
{
    static readonly string ToolsSystem;
    static readonly string UserPrompt = "What's the weather in Paris?";

    static QwenChatTemplateTests()
    {
        var weather = QwenSampleTools.CreateWeatherTool();
        ToolsSystem = QwenChatTemplate.BuildToolsSystemMessage([weather]);
    }

    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static string? FixtureContent(string name)
    {
        var path = Path.Combine(ModelDir, name);
        return File.Exists(path) ? File.ReadAllText(path, Encoding.UTF8) : null;
    }

    [Test]
    public void Render_ToolPrompt_MatchesTorchFixtureBytes()
    {
        var expected = FixtureContent("qwen_tool_prompt.txt");
        if (expected is null)
            Assert.Ignore("Qwen fixtures absent; skipping byte-exact template verification.");

        var rendered = QwenChatTemplate.Render(
            [new ChatMessage(ChatRole.System, ToolsSystem), new ChatMessage(ChatRole.User, UserPrompt)],
            addGenerationPrompt: true);

        Assert.That(rendered, Is.EqualTo(expected));
    }

    [Test]
    public void Render_AssistantToolCallAndToolResponse_MatchesTorchFixtureBytes_RoundTrip()
    {
        var expected = FixtureContent("qwen_tool_final_prompt.txt");
        if (expected is null)
            Assert.Ignore("Qwen fixtures absent; skipping byte-exact round-trip verification.");

        // The raw turn the model generates (ground-truth tool turn, 19 ids, byte-exact text).
        var rawToolCall = "<tool_call>\n{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

        // Parse, then re-render — exactly what the live loop does between iterations.
        var calls = QwenToolCallParser.Parse(rawToolCall, [QwenSampleTools.WeatherToolName]);
        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo(QwenSampleTools.WeatherToolName));

        var assistant = new ChatMessage(ChatRole.Assistant, (string?)null);
        assistant.Contents.Add(calls[0]);
        var tool = new ChatMessage(ChatRole.Tool, (string?)null);
        tool.Contents.Add(new FunctionResultContent(
            calls[0].CallId, "Partly cloudy, 18°C. Light breeze from the northwest."));

        var rendered = QwenChatTemplate.Render(
            [
                new ChatMessage(ChatRole.System, ToolsSystem),
                new ChatMessage(ChatRole.User, UserPrompt),
                assistant,
                tool,
            ],
            addGenerationPrompt: true);

        Assert.That(rendered, Is.EqualTo(expected));
    }

    [Test]
    public void Render_PlainChat_EmitsQwenDefaultSystemTurn()
    {
        var rendered = QwenChatTemplate.Render(
            [new ChatMessage(ChatRole.User, UserPrompt)],
            addGenerationPrompt: true);

        Assert.That(rendered, Is.EqualTo(
            "<|im_start|>system\n" + QwenChatTemplate.DefaultSystem + "<|im_end|>\n" +
            "<|im_start|>user\n" + UserPrompt + "<|im_end|>\n" +
            "<|im_start|>assistant\n"));
    }

    [Test]
    public void BuildToolsSystemMessage_ContainsToolsInstructionsAndToolSchema()
    {
        Assert.That(ToolsSystem, Does.StartWith(QwenChatTemplate.DefaultSystem));
        Assert.That(ToolsSystem, Does.Contain("# Tools"));
        Assert.That(ToolsSystem, Does.Contain("<tools>"));
        Assert.That(ToolsSystem, Does.Contain(QwenSampleTools.WeatherToolName));
        Assert.That(ToolsSystem, Does.Contain("\"description\": \"The city name, e.g. 'Paris' or 'New York'\""));
        Assert.That(ToolsSystem, Does.Contain("{\"name\": <function-name>, \"arguments\": <args-json-object>}"));
    }
}