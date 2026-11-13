using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraColumnEqualityTests
{
    [Test]
    public void Equals_Structural_SameContentDifferentInstances_ReturnsTrue()
    {
        var a = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        Assert.That(a, Is.EqualTo(b));
        Assert.That(a.Equals(b), Is.True);
        Assert.That(a == b, Is.True);
        Assert.That(a != b, Is.False);
        Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
    }

    [Test]
    public void Equals_Structural_DifferentValues_ReturnsFalse()
    {
        var a = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraColumn<int>.Create(new[] { 1, 2, 4 });

        Assert.That(a.Equals(b), Is.False);
        Assert.That(a == b, Is.False);
        Assert.That(a != b, Is.True);
    }

    [Test]
    public void Equals_Structural_DifferentLength_ReturnsFalse()
    {
        var a = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraColumn<int>.Create(new[] { 1, 2 });

        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Equals_Structural_NullMaskDiffers_ReturnsFalse()
    {
        var a = NivaraColumn<object>.Create((object[])new object?[] { "a", null, "c" });
        var b = NivaraColumn<object>.Create(new object[] { "a", "b", "c" });

        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Equals_Structural_NullsAtSamePositions_ReturnsTrue()
    {
        var a = NivaraColumn<object>.Create((object[])new object?[] { "a", null, "c" });
        var b = NivaraColumn<object>.Create((object[])new object?[] { "a", null, "c" });

        Assert.That(a.Equals(b), Is.True);
        Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
    }

    [Test]
    public void Equals_Structural_DifferentGenericTypes_ReturnsFalseViaObject()
    {
        var a = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        object b = NivaraColumn<long>.Create(new[] { 1L, 2L, 3L });

        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Equals_Structural_Disposed_Throws()
    {
        var a = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var b = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        a.Dispose();

        Assert.Throws<ObjectDisposedException>(() => a.Equals(b));
    }

    [Test]
    public void ElementWise_EqualTo_ScalarAndColumn()
    {
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var scalarMask = col.EqualTo(2);
        var colMask = col.EqualTo(NivaraColumn<int>.Create(new[] { 1, 0, 3 }));

        Assert.That(scalarMask.ToArray(), Is.EqualTo(new[] { false, true, false }));
        Assert.That(colMask.ToArray(), Is.EqualTo(new[] { true, false, true }));
    }

    [Test]
    public void ElementWise_NotEqualTo_GreaterThanOrEqual_LessThanOrEqual()
    {
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        Assert.That(col.NotEqualTo(2).ToArray(), Is.EqualTo(new[] { true, false, true }));
        Assert.That(col.GreaterThanOrEqual(2).ToArray(), Is.EqualTo(new[] { false, true, true }));
        Assert.That(col.LessThanOrEqual(2).ToArray(), Is.EqualTo(new[] { true, true, false }));
    }

    [Test]
    public void ElementWise_NullMask_Propagates()
    {
        var col = NivaraColumn<object>.Create((object[])new object?[] { "1", null, "3" });
        var mask = col.EqualTo("1");

        Assert.That(mask.Length, Is.EqualTo(3));
        Assert.That(mask[0], Is.True);
        Assert.That(mask[1], Is.False);
        Assert.That(mask.IsNull(1), Is.True);
    }
}
