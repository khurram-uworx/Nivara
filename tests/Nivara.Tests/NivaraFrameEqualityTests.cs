using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraFrameEqualityTests
{
    [Test]
    public void Equals_TwoIdenticallyBuiltFrames_ReturnsTrueAndHashMatches()
    {
        var frame1 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })), ("B", (IColumn)NivaraColumn<string>.Create(new[] { "x", "y", "z" })) });
        var frame2 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })), ("B", (IColumn)NivaraColumn<string>.Create(new[] { "x", "y", "z" })) });

        Assert.That(frame1, Is.EqualTo(frame2));
        Assert.That(frame1.Equals(frame2), Is.True);
        Assert.That(frame1 == frame2, Is.True);
        Assert.That(frame1.GetHashCode(), Is.EqualTo(frame2.GetHashCode()));
    }

    [Test]
    public void Equals_DifferentColumnOrder_ReturnsFalse()
    {
        var frame1 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })), ("B", (IColumn)NivaraColumn<int>.Create(new[] { 4, 5, 6 })) });
        var frame2 = new NivaraFrame(new[] { ("B", (IColumn)NivaraColumn<int>.Create(new[] { 4, 5, 6 })), ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })) });

        Assert.That(frame1.Equals(frame2), Is.False);
    }

    [Test]
    public void Equals_DifferentValues_ReturnsFalse()
    {
        var frame1 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })) });
        var frame2 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 4 })) });

        Assert.That(frame1.Equals(frame2), Is.False);
    }

    [Test]
    public void Equals_AgreesWithSchema_OnMetadata()
    {
        var s1 = new Nivara.Schema(new[] { ("A", typeof(int)) });
        var s2 = new Nivara.Schema(new[] { ("A", typeof(int)) });

        Assert.That(s1.Equals(s2), Is.True);
        // Frames built from identical columns share the same schema equality outcome.
        var f1 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1 })) });
        var f2 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1 })) });
        Assert.That(f1.Schema.Equals(f2.Schema), Is.EqualTo(f1.Equals(f2)));
    }

    [Test]
    public void Equals_Disposed_Throws()
    {
        var frame1 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })) });
        var frame2 = new NivaraFrame(new[] { ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 2, 3 })) });
        frame1.Dispose();

        Assert.Throws<ObjectDisposedException>(() => frame1.Equals(frame2));
    }
}
