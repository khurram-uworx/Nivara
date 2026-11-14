using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Ownership contract for NivaraFrame disposal (issue #1159).
/// A frame owns only the columns it created; borrowed columns are shared and
/// are not disposed by the borrowing frame.
/// </summary>
public class NivaraFrameOwnershipTests
{
    static NivaraFrame MakeSource()
        => NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));

    // ---------- WithColumn ----------

    [Test]
    public void WithColumn_DisposeDerived_SourceRemainsUsable()
    {
        using var source = MakeSource();
        var derived = source.WithColumn("B", NivaraColumn<int>.Create(new[] { 4, 5, 6 }));

        derived.Dispose();

        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
        Assert.That(source.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void WithColumn_DisposeSource_DerivedBecomesInvalid()
    {
        var source = MakeSource();
        var derived = source.WithColumn("B", NivaraColumn<int>.Create(new[] { 4, 5, 6 }));

        source.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = derived.GetColumn<int>("A").Length);

        derived.Dispose();
    }

    [Test]
    public void WithColumn_DisposeDerived_DisposesOnlyAddedColumn()
    {
        var sourceColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var addedColumn = NivaraColumn<int>.Create(new[] { 4, 5, 6 });
        using var source = NivaraFrame.Create(("A", sourceColumn));
        var derived = source.WithColumn("B", addedColumn);

        derived.Dispose();

        Assert.That(addedColumn.IsDisposed, Is.True);
        Assert.That(sourceColumn.IsDisposed, Is.False);
    }

    // ---------- SelectColumns ----------

    [Test]
    public void SelectColumns_DisposeDerived_SourceRemainsUsable()
    {
        using var source = MakeSource();
        var derived = source.SelectColumns("A");

        derived.Dispose();

        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
    }

    [Test]
    public void SelectColumns_DisposeSource_DerivedBecomesInvalid()
    {
        var source = MakeSource();
        var derived = source.SelectColumns("A");

        source.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = derived.GetColumn<int>("A").Length);

        derived.Dispose();
    }

    [Test]
    public void SelectColumns_DisposeDerived_DoesNotDisposeSourceColumns()
    {
        using var source = MakeSource();
        var column = source.GetColumn<int>("A");
        var derived = source.SelectColumns("A");

        derived.Dispose();

        Assert.That(column.IsDisposed, Is.False);
        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
    }

    // ---------- WithoutColumn ----------

    [Test]
    public void WithoutColumn_DisposeDerived_SourceRemainsUsable()
    {
        using var source = NivaraFrame.Create(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("B", NivaraColumn<int>.Create(new[] { 4, 5, 6 })));
        var derived = source.WithoutColumn("B");

        derived.Dispose();

        Assert.That(source.ColumnCount, Is.EqualTo(2));
        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
    }

    [Test]
    public void WithoutColumn_DisposeSource_DerivedBecomesInvalid()
    {
        var source = NivaraFrame.Create(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("B", NivaraColumn<int>.Create(new[] { 4, 5, 6 })));
        var derived = source.WithoutColumn("B");

        source.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = derived.GetColumn<int>("A").Length);

        derived.Dispose();
    }

    // ---------- RenameColumn ----------

    static NivaraFrame MakeTwoColumnSource()
        => NivaraFrame.Create(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("B", NivaraColumn<int>.Create(new[] { 4, 5, 6 })));

    [Test]
    public void RenameColumn_DisposeDerived_SourceRemainsUsable()
    {
        using var source = MakeTwoColumnSource();
        var derived = source.RenameColumn("A", "A2");

        derived.Dispose();

        Assert.That(source.HasColumn("A"), Is.True);
        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
    }

    [Test]
    public void RenameColumn_DisposeSource_DerivedBecomesInvalid()
    {
        var source = MakeTwoColumnSource();
        var derived = source.RenameColumn("A", "A2");

        source.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = derived.GetColumn<int>("A2").Length);

        derived.Dispose();
    }

    // ---------- Where (FilterByMask) ----------

    [Test]
    public void Where_DisposeDerived_SourceRemainsUsable()
    {
        using var source = MakeSource();
        var derived = source.Where(r => (int)r["A"]! > 1);

        derived.Dispose();

        Assert.That(source.RowCount, Is.EqualTo(3));
        Assert.That(source.GetColumn<int>("A")[0], Is.EqualTo(1));
    }

    [Test]
    public void Where_DisposeSource_DerivedRemainsUsable()
    {
        // Where builds fresh columns, so the derived frame does not share storage
        // with the source; disposing the source leaves the derived usable.
        var source = MakeSource();
        var derived = source.Where(r => (int)r["A"]! > 1);

        source.Dispose();

        Assert.That(derived.RowCount, Is.EqualTo(2));

        derived.Dispose();
    }

    // ---------- Fresh-column operations ----------

    [Test]
    public void Take_DisposeDerived_SourceRemainsUsable()
    {
        using var source = MakeSource();
        var derived = source.Take(2);

        derived.Dispose();

        Assert.That(source.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Take_DisposeSource_DerivedRemainsUsable()
    {
        var source = MakeSource();
        var derived = source.Take(2);

        source.Dispose();

        Assert.That(derived.RowCount, Is.EqualTo(2));

        derived.Dispose();
    }

    // ---------- Source ownership release ----------

    [Test]
    public void DisposeSource_ReleasesOwnedColumns()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("A", column));

        frame.Dispose();

        Assert.That(column.IsDisposed, Is.True);
    }

    [Test]
    public void DisposeDerived_DoesNotDisposeSourceColumns()
    {
        using var source = MakeSource();
        var column = source.GetColumn<int>("A");
        var derived = source.SelectColumns("A");

        derived.Dispose();

        Assert.That(column.IsDisposed, Is.False);
    }

    // ---------- Idempotency ----------

    [Test]
    public void Dispose_IsIdempotent()
    {
        var frame = MakeSource();
        frame.Dispose();
        Assert.DoesNotThrow(() => frame.Dispose());
    }

    // ---------- Mixed ownership (Normalize) ----------

    [Test]
    public void Normalize_DisposeDerived_DisposesOnlyNormalizedColumns()
    {
        var sourceColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        using var source = NivaraFrame.Create(("A", sourceColumn));
        var derived = source.Normalize();

        derived.Dispose();

        Assert.That(sourceColumn.IsDisposed, Is.False);
    }
}
