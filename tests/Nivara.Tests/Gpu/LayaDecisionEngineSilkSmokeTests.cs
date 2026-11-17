using Nivara.Silk;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public sealed class LayaDecisionEngineSilkSmokeTests
{
    static string ModelDir
    {
        get
        {
            var root = Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", ".."));
            return Path.Combine(root, "samples", "data", "laya");
        }
    }

    [Test]
    public void ConstructSilk_DecideAndExtractFeatures_Run()
    {
        if (!File.Exists(Path.Combine(ModelDir, "model.safetensors")))
            Assert.Ignore("Laya checkpoint absent.");
        SilkRuntime? rt = null;
        try { rt = new SilkRuntime(); }
        catch (Exception ex) { Assert.Ignore($"OpenCL unavailable: {ex.Message}"); }
        finally { rt?.Dispose(); }

        using var engine = new LayaRetrievalDecisionHead.LayaDecisionEngine(ModelDir, useSilk: true);
        var chunk = new MultiViewChunk("c0", "src", "", "title", "", "public", "hello world", null);
        var features = engine.ExtractFeatures("query", chunk);
        Assert.That(features.Values.Length, Is.GreaterThan(0));

        var retrieved = new RetrievedChunk("c0", "src", "title", "hello world", 0d);
        var decision = engine.Decide("query", retrieved);
        Assert.That(decision, Is.Not.Null);
    }
}
