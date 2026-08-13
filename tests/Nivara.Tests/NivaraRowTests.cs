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
    public void GetValue_BlankColumnName_ThrowsArgumentException()
    {
        using var frame = CreateFrame();

        Assert.Throws<ArgumentException>(() => frame.Where(row => row.GetValue<int>("  ") > 0));
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
}
