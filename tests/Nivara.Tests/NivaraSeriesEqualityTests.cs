using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraSeriesEqualityTests
{
    [Test]
    public void Equals_SameValuesNoIndex_ReturnsTrueAndHashMatches()
    {
        var a = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraSeries<int>.Create(new[] { 1, 2, 3 });

        Assert.That(a.Equals(b), Is.True);
        Assert.That(a == b, Is.True);
        Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
    }

    [Test]
    public void Equals_DifferentValues_ReturnsFalse()
    {
        var a = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraSeries<int>.Create(new[] { 1, 2, 4 });

        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Equals_DefaultIndexEqualsExplicitPositionalIndex()
    {
        var a = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraSeries<int>.Create(new[] { 1, 2, 3 }, new object[] { 0, 1, 2 });

        Assert.That(a.Equals(b), Is.True);
        Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
    }

    [Test]
    public void Equals_DifferentLabels_ReturnsFalse()
    {
        var a = NivaraSeries<int>.Create(new[] { 1, 2, 3 }, new[] { "a", "b", "c" });
        var b = NivaraSeries<int>.Create(new[] { 1, 2, 3 }, new[] { "a", "b", "d" });

        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Equals_SameLabels_ReturnsTrue()
    {
        var a = NivaraSeries<int>.Create(new[] { 1, 2, 3 }, new[] { "a", "b", "c" });
        var b = NivaraSeries<int>.Create(new[] { 1, 2, 3 }, new[] { "a", "b", "c" });

        Assert.That(a.Equals(b), Is.True);
        Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
    }
}
