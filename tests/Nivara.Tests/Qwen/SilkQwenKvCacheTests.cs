using Nivara.Silk;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class SilkQwenKvCacheTests
{
    [Test]
    public void GenerateSilk_UseKvCacheOnOff_ReturnsIdenticalTokens_Synthetic()
    {
        var (tensors, config) = SilkQwenTestWeights.Synthesize();
        int[] prompt = [1, 2, 3];

        using var session = SilkQwenSession.Open(SilkQwenTestWeights.ToSilkConfig(config, 16), tensors);
        var on = NivaraInference.Qwen.GenerateSilk(session, config, prompt, 4, true);
        var off = NivaraInference.Qwen.GenerateSilk(session, config, prompt, 4, false);

        Assert.That(off, Is.EqualTo(on));
    }
}
