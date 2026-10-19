using NivaraChat.Shared;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class TokenizePrefixCacheTests
{
    [Test]
    public void UnchangedPrompt_HitsCache_NoReencode()
    {
        var cache = new TokenizePrefixCache();
        int encodes = 0;
        Func<string, System.Collections.Generic.IReadOnlyList<int>> enc = s => { encodes++; return new[] { 1, 2, 3 }; };

        var first = cache.Encode("hello", enc);
        var second = cache.Encode("hello", enc);

        Assert.That(cache.Hits, Is.EqualTo(1));
        Assert.That(encodes, Is.EqualTo(1));
        Assert.That(second, Is.EqualTo(first));
    }

    [Test]
    public void ExtendedPrompt_ReusesPrefix()
    {
        var cache = new TokenizePrefixCache();
        int encodes = 0;
        Func<string, System.Collections.Generic.IReadOnlyList<int>> enc = s =>
        {
            encodes++;
            return s == "ab" ? new[] { 10, 20 } : new[] { 30 };
        };

        cache.Encode("ab", enc);
        var res = cache.Encode("abcd", enc);

        Assert.That(res, Is.EqualTo(new[] { 10, 20, 30 }));
    }
}