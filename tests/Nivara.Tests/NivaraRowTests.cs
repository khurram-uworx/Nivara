using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraRowTests
{
    static NivaraFrame CreateFrame()
    {
        return NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35 })),
            ("Score", NivaraColumn<double>.Create(new[] { 85.5, 92.0, 78.5 })));
    }

    [Test]
    public void Where_TypedGetValue_FiltersCorrectly()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => row.GetValue<int>("Age") > 30);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("Charlie"));
    }

    [Test]
    public void Where_IndexerCast_FiltersCorrectly()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => (int)row["Age"]! >= 30);

        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.GetColumn<string>("Name").ToArray(), Is.EqualTo(new[] { "Bob", "Charlie" }));
    }

    [Test]
    public void Where_ColumnNames_AreCaseInsensitive()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => row.GetValue<int>("age") > 25);

        Assert.That(result.RowCount, Is.EqualTo(2));
    }

    [Test]
    public void Where_IsNull_SelectsOnlyNullRows()
    {
        var ages = NivaraColumn.CreateFromNullable(new int?[] { 30, null, 35 });
        using var frame = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })), ("Age", ages));

        var result = frame.Where(row => row.IsNull("Age"));

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("b"));
    }

    [Test]
    public void Where_IsNullAndTypedValue_ComposeCorrectly()
    {
        var ages = NivaraColumn.CreateFromNullable(new int?[] { 30, null, 35 });
        using var frame = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })), ("Age", ages));

        var result = frame.Where(row => !row.IsNull("Age") && row.GetValue<int>("Age") > 30);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("c"));
    }

    [Test]
    public void Indexer_NullCell_ReturnsNull()
    {
        var ages = NivaraColumn.CreateFromNullable(new int?[] { 30, null, 35 });
        using var frame = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })), ("Age", ages));

        var result = frame.Where(row => row["Age"] == null);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("b"));
    }

    [Test]
    public void Where_RowIndex_SelectsSpecificPosition()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => row.RowIndex == 1);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("Bob"));
    }

    [Test]
    public void GetValue_MissingColumn_ThrowsColumnNotFoundException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ColumnNotFoundException>(() => frame.Where(row => row.GetValue<int>("Nope") > 0));
    }

    [Test]
    public void GetValue_TypeMismatch_ThrowsColumnTypeMismatchException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ColumnTypeMismatchException>(() => frame.Where(row => row.GetValue<string>("Age") == "x"));
    }

    [Test]
    public void GetValue_NullableElementColumn_TypeMismatch_ThrowsColumnTypeMismatchException()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        Assert.Throws<ColumnTypeMismatchException>(() => frame.Where(row => row.GetValue<long>("Age") > 0));
    }

    [Test]
    public void GetValue_BlankColumnName_ThrowsArgumentException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ArgumentException>(() => frame.Where(row => row.GetValue<int>("  ") > 0));
    }

    [Test]
    public void GetValue_NullableElementColumn_ReadsUnderlyingValue()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        var result = frame.Where(row => !row.IsNull("Age") && row.GetValue<int>("Age") > 15);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("c"));
    }

    [Test]
    public void GetValue_NullableElementColumn_NullCell_ReturnsDefault()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        var result = frame.Where(row =>
            row.RowIndex == 1
            ? row.IsNull("Age") && row.GetValue<int>("Age") == default
            : false);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("b"));
    }

    [Test]
    public void GetValue_NullableElementType_ReadsNullableValue()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        var result = frame.Where(row => row.GetValue<int?>("Age") is null);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("b"));
    }

    [Test]
    public void TryGetValue_NullableElementColumn_ReturnsTrueWithValue()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        var result = frame.Where(row => row.TryGetValue<int>("Age", out var age) && !row.IsNull("Age") && age > 15);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("c"));
    }

    [Test]
    public void TryGetValue_NullableElementColumn_NullCell_ReturnsDefault()
    {
        var ages = NivaraColumn<int?>.Create(new int?[] { 10, null, 30 });
        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })),
            ("Age", ages));

        var result = frame.Where(row =>
            row.RowIndex == 1
            ? row.TryGetValue<int>("Age", out var age) && row.IsNull("Age") && age == default
            : false);

        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("Name")[0], Is.EqualTo("b"));
    }

    [Test]
    public void Where_PredicateException_PropagatesUnwrapped()
    {
        using var frame = CreateFrame();

        Assert.Throws<ApplicationException>(() => frame.Where(row => throw new ApplicationException("boom")));
    }

    [Test]
    public void TryGetValue_ExistingColumn_ReturnsTrueWithValue()
    {
        using var frame = CreateFrame();

        var seen = frame.Where(row => row.TryGetValue<int>("Age", out var age) && age > 25);

        Assert.That(seen.RowCount, Is.EqualTo(2));
    }

    [Test]
    public void TryGetValue_MissingColumn_ReturnsFalse()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => row.TryGetValue<int>("Nope", out var value) && value > 0);

        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    [Test]
    public void TryGetValue_TypeMismatch_ReturnsFalse()
    {
        using var frame = CreateFrame();

        var result = frame.Where(row => row.TryGetValue<string>("Age", out var value) && value == "x");

        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    [Test]
    public void DefaultRow_Access_ThrowsInvalidOperationException()
    {
        NivaraRow row = default;

        Assert.Throws<InvalidOperationException>(() => _ = row["Age"]);
        Assert.Throws<InvalidOperationException>(() => row.GetValue<int>("Age"));
        Assert.Throws<InvalidOperationException>(() => row.IsNull("Age"));
    }

    [Test]
    public void Where_NullFrame_ThrowsArgumentNullException()
    {
        NivaraFrame frame = null!;

        Assert.Throws<ArgumentNullException>(() => frame.Where(row => true));
    }

    [Test]
    public void Where_NullPredicate_ThrowsArgumentNullException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ArgumentNullException>(() => frame.Where((Func<NivaraRow, bool>)null!));
    }

    [Test]
    public void Where_NullableElementColumn_GetValue_AllocatesLikeFilterOnly()
    {
        var values = new int?[10_000];
        for (int i = 0; i < values.Length; i++)
            values[i] = i % 100 == 0 ? null : i;

        using var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(Enumerable.Repeat("x", values.Length).ToArray())),
            ("Age", NivaraColumn<int?>.Create(values)));

        Func<NivaraRow, bool> readPredicate = static row => row.GetValue<int>("Age") > 15;
        Func<NivaraRow, bool> baselinePredicate = static row => row.RowIndex >= 0;

        var (readAlloc, baselineAlloc) = MeasureReadAndBaseline(frame, readPredicate, baselinePredicate);

        // The nullable-element GetValue<T> read must be allocation-free per row: a Where that
        // reads the column should cost the same as one that only inspects RowIndex on the same
        // source frame. Windows measures delta ≈ 0; the Linux CI runner records ~26.7 B/row of
        // platform/JIT variance (measured 762 KB vs 495 KB baseline), so the margin is 350 KB
        // (35 B/row @ 10 000 rows) to absorb that. The old cached MethodInfo.Invoke reader added
        // an object[] and two boxings per row (~88 B × 10 000 ≈ 880 KB), still caught by this
        // margin with ample headroom.
        Assert.That(readAlloc, Is.LessThanOrEqualTo(baselineAlloc + 350_000),
            $"GetValue read path allocated {readAlloc} B vs filter-only baseline {baselineAlloc} B");
    }

    static int RunWhere(NivaraFrame frame, Func<NivaraRow, bool> predicate)
    {
        using var result = frame.Where(predicate);
        return result.RowCount;
    }

    /// <summary>
    /// Best-of-N steady-state allocation of <see cref="RunWhere"/> for each predicate, measured
    /// with both alternated per sample so GC/finalizer drift hits both sides equally (per the
    /// Tensor allocation-guard pattern; cleans the flakiness of measuring two blocks back-to-back).
    /// </summary>
    static (long Read, long Baseline) MeasureReadAndBaseline(
        NivaraFrame frame, Func<NivaraRow, bool> readPredicate, Func<NivaraRow, bool> baselinePredicate)
    {
        long readBest = long.MaxValue;
        long baselineBest = long.MaxValue;

        for (int sample = 0; sample < 7; sample++)
        {
            for (int i = 0; i < 3; i++)
            {
                RunWhere(frame, readPredicate);
                RunWhere(frame, baselinePredicate);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            readBest = Math.Min(readBest, MeasureWhere(frame, readPredicate));
            baselineBest = Math.Min(baselineBest, MeasureWhere(frame, baselinePredicate));
        }

        return (readBest, baselineBest);
    }

    static long MeasureWhere(NivaraFrame frame, Func<NivaraRow, bool> predicate)
    {
        long pre = GC.GetAllocatedBytesForCurrentThread();
        RunWhere(frame, predicate);
        return GC.GetAllocatedBytesForCurrentThread() - pre;
    }
}
