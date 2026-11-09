using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
[Category("Gpu")]
public class SilkQwenKvCacheTests
{
    [Test]
    public void GenerateSilk_UseKvCacheOnOff_ReturnsIdenticalTokens_Synthetic()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk KV-cache verification."); }

        var (tensors, config) = QwenTestWeights.Synthesize();
        int[] prompt = [1, 2, 3];

        using var session = SilkQwenSession.Open(QwenTestWeights.ToSilkConfig(config, 16), tensors);
        var on = NivaraInference.Qwen.GenerateSilk(session, config, prompt, 4, true);
        var off = NivaraInference.Qwen.GenerateSilk(session, config, prompt, 4, false);

        Assert.That(off, Is.EqualTo(on));
    }
}
