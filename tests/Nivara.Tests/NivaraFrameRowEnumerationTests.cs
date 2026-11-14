using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public sealed class NivaraFrameRowEnumerationTests
{
    static NivaraFrame CreateFrame()
    {
        return NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35 })));
    }

    [Test]
    public void Enumerate_ZeroRows_YieldsNoRows()
    {
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(Array.Empty<string>())),
            ("Age", NivaraColumn<int>.Create(Array.Empty<int>())));

        var count = 0;
        foreach (var row in frame)
        {
            count++;
            _ = row.RowIndex;
        }

        Assert.That(count, Is.EqualTo(0));
        Assert.That(frame.RowCount, Is.EqualTo(0));
    }

    [Test]
    public void Enumerate_OneRow_YieldsSingleRowInOrder()
    {
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(new[] { "Solo" })),
            ("Age", NivaraColumn<int>.Create(new[] { 42 })));

        var rows = new List<(int Index, string Name, int Age)>();
        foreach (var row in frame)
            rows.Add((row.RowIndex, row.GetValue<string>("Name"), row.GetValue<int>("Age")));

        Assert.That(rows, Has.Count.EqualTo(1));
        Assert.That(rows[0], Is.EqualTo((0, "Solo", 42)));
    }

    [Test]
    public void Enumerate_MultipleRows_YieldsRowsInOrderWithCorrectRowIndex()
    {
        using var frame = CreateFrame();

        var indices = new List<int>();
        var names = new List<string>();
        var ages = new List<int>();

        foreach (var row in frame)
        {
            indices.Add(row.RowIndex);
            names.Add(row.GetValue<string>("Name"));
            ages.Add(row.GetValue<int>("Age"));
        }

        Assert.That(indices, Is.EqualTo(new[] { 0, 1, 2 }));
        Assert.That(names, Is.EqualTo(new[] { "Alice", "Bob", "Charlie" }));
        Assert.That(ages, Is.EqualTo(new[] { 25, 30, 35 }));
    }

    [Test]
    public void GetRow_ReturnsRowUsableOutsideLoop_ForFrameLifetime()
    {
        using var frame = CreateFrame();

        var row = frame.GetRow(1);

        Assert.That(row.RowIndex, Is.EqualTo(1));
        Assert.That(row.GetValue<string>("Name"), Is.EqualTo("Bob"));
        Assert.That(row.GetValue<int>("Age"), Is.EqualTo(30));
        Assert.That(row["Name"], Is.EqualTo("Bob"));
    }

    [Test]
    public void GetRow_OutOfRange_ThrowsArgumentOutOfRangeException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ArgumentOutOfRangeException>(() => frame.GetRow(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.GetRow(frame.RowCount));
    }

    [Test]
    public void Enumerate_DisposedFrame_ThrowsObjectDisposedException()
    {
        var frame = CreateFrame();
        frame.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            foreach (var _ in frame) { }
        });
    }

    [Test]
    public void GetRow_DisposedFrame_ThrowsObjectDisposedException()
    {
        var frame = CreateFrame();
        frame.Dispose();

        Assert.Throws<ObjectDisposedException>(() => frame.GetRow(0));
    }

    [Test]
    public void Enumerator_MoveNext_OnDisposedFrame_ThrowsObjectDisposedException()
    {
        var frame = CreateFrame();
        using var enumerator = frame.GetEnumerator();
        frame.Dispose();

        Assert.Throws<ObjectDisposedException>(() => enumerator.MoveNext());
    }

    [Test]
    public void NivaraRow_Create_AllowsPredicateUnitTestingWithoutFrame()
    {
        var nameColumn = NivaraColumn<string>.Create(new[] { "Alice", "Bob" });
        var ageColumn = NivaraColumn<int>.Create(new[] { 25, 40 });
        IColumn[] columns = [nameColumn, ageColumn];
        var map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            ["Name"] = 0,
            ["Age"] = 1
        };

        Func<NivaraRow, bool> predicate = row => row.GetValue<int>("Age") >= 30;

        var row0 = NivaraRow.Create(columns, map, 0);
        var row1 = NivaraRow.Create(columns, map, 1);

        Assert.That(predicate(row0), Is.False);
        Assert.That(predicate(row1), Is.True);
        Assert.That(row1.GetValue<string>("Name"), Is.EqualTo("Bob"));
    }

    [Test]
    public void FromRows_AndRowEnumeration_RoundTrip_LabeledVectors()
    {
        var source = new (string Label, int[] Vector)[]
        {
            ("r1", new[] { 1, 2, 3 }),
            ("r2", new[] { 4, 5, 6 })
        };

        using var frame = NivaraFrame.FromRows(source);

        var roundTripped = new List<(string Label, int[] Vector)>();
        foreach (var row in frame)
        {
            var label = row.GetValue<string>("Label");
            var vector = new[]
            {
                row.GetValue<int>("Col_0"),
                row.GetValue<int>("Col_1"),
                row.GetValue<int>("Col_2")
            };
            roundTripped.Add((label, vector));
        }

        Assert.That(roundTripped.Count, Is.EqualTo(source.Length));
        Assert.That(roundTripped[0].Label, Is.EqualTo("r1"));
        Assert.That(roundTripped[0].Vector, Is.EqualTo(new[] { 1, 2, 3 }));
        Assert.That(roundTripped[1].Label, Is.EqualTo("r2"));
        Assert.That(roundTripped[1].Vector, Is.EqualTo(new[] { 4, 5, 6 }));
    }

    [Test]
    [Category("Performance")]
    public void Enumerate_DoesNotAllocatePerRow()
    {
        const int rowCount = 10_000;
        var names = Enumerable.Repeat("x", rowCount).ToArray();
        var ages = Enumerable.Range(0, rowCount).ToArray();
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(names)),
            ("Age", NivaraColumn<int>.Create(ages)));

        // Warm up and stabilize
        for (int i = 0; i < 5; i++)
            Consume(frame);

        long best = long.MaxValue;
        for (int sample = 0; sample < 7; sample++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long before = GC.GetAllocatedBytesForCurrentThread();
            Consume(frame);
            long after = GC.GetAllocatedBytesForCurrentThread();
            best = Math.Min(best, after - before);
        }

        // Enumerating 10k rows should not allocate per row (allow small fixed overhead for iterator/setup).
        Assert.That(best, Is.LessThanOrEqualTo(4096),
            $"Enumeration allocated {best} bytes for {rowCount} rows; expected near-zero per-row allocations.");
    }

    static int Consume(NivaraFrame frame)
    {
        int sum = 0;
        foreach (var row in frame)
            sum += row.RowIndex;
        return sum;
    }
}