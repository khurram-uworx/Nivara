using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveSilkParityTests
{
    static NivaraFrame Frame(float[][] content) =>
        MultiViewScoring.BuildFrame(
            [new MultiViewChunk("a#0", "a.md", "a.md", "A", "A", "public", "first", null)],
            [content[0]], [content[0]], [content[0]], [content[0]]);

    [Test]
    public void Compare_IdenticalFrames_IsZero()
    {
        var v = new[] { new[] { 1f, 2f, 3f } };
        var a = Frame(v);
        var b = Frame(v);
        var r = AdaptiveSilkParity.Compare(a, b, ["VectorContent"]);
        Assert.That(r.MaxAbsoluteDifference, Is.EqualTo(0.0));
        Assert.That(r.MismatchedRows, Is.EqualTo(0));
        Assert.That(r.WithinTolerance, Is.True);
    }

    [Test]
    public void Compare_DifferentVectors_ReportsMaxAbsAndMismatches()
    {
        var a = Frame(new[] { new[] { 1f, 2f, 3f } });
        var b = Frame(new[] { new[] { 1f, 2.5f, 3f } });
        var r = AdaptiveSilkParity.Compare(a, b, ["VectorContent"]);
        Assert.That(r.MaxAbsoluteDifference, Is.EqualTo(0.5).Within(1e-6));
        Assert.That(r.MismatchedRows, Is.EqualTo(1));
        Assert.That(r.WithinTolerance, Is.False);
    }

    [Test]
    public void Compare_RowCountMismatch_Throws()
    {
        var chunks = new[]
        {
            new MultiViewChunk("a#0", "a.md", "a.md", "A", "A", "public", "first", null),
            new MultiViewChunk("b#0", "b.md", "b.md", "B", "B", "public", "second", null),
        };
        var a = MultiViewScoring.BuildFrame(chunks,
            [new[] { 1f }, new[] { 2f }], [new[] { 1f }, new[] { 2f }],
            [new[] { 1f }, new[] { 2f }], [new[] { 1f }, new[] { 2f }]);
        var b = MultiViewScoring.BuildFrame([chunks[0]],
            [new[] { 1f }], [new[] { 1f }], [new[] { 1f }], [new[] { 1f }]);
        Assert.Throws<ArgumentException>(() => AdaptiveSilkParity.Compare(a, b, ["VectorContent"]));
    }
}
