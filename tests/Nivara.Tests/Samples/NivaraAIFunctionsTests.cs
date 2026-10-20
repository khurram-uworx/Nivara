using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Serialization;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public class NivaraAIFunctionsTests
{
    static string ModelsDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "nivarachat");

    static (TextClassifierModel<float> model, TextTokenizer tok)? Cached;

    [OneTimeSetUp]
    public void SetUp()
    {
        var path = Path.Combine(ModelsDir, "sentiment_model.json");
        if (!File.Exists(path)) return;
        var tok = TextTokenizer.Load(Path.Combine(ModelsDir, "sentiment_tokenizer.json"));
        var model = new TextClassifierModel<float>(tok.VocabSize, 32, 64, 3, 20);
        ModelSerializer.Load(model, path);
        model.Eval();
        Cached = (model, tok);
    }

    [OneTimeTearDown]
    public void TearDown() => Cached?.model.Dispose();

    [Test]
    public async Task SentimentTool_InvokesAndReturnsPositiveLabel()
    {
        if (Cached is null) Assert.Ignore("Nivara sentiment model absent; skipping.");
        var (model, tok) = Cached.Value;
        var tool = NivaraAIFunctions.CreateSentimentTool(model, tok);

        var args = new AIFunctionArguments { ["text"] = "I absolutely love this product, it works great!" };
        var result = await tool.InvokeAsync(args);
        var text = result?.ToString() ?? "";
        Assert.That(text, Does.Contain("positive"));
    }
}
