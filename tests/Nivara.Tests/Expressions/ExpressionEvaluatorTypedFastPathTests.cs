using Nivara.Expressions;
using NUnit.Framework;

namespace Nivara.Tests.Expressions;

[TestFixture]
public class ExpressionEvaluatorTypedFastPathTests
{
    static NivaraFrame CreateFrame()
    {
        var ids = NivaraColumn<int>.Create(new[] { 100, 200, 300, 400, 500 });
        var names = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie", "David", "Eve" });
        var scores = NivaraColumn<double>.Create(new[] { 85.5, 92.0, 78.5, 95.0, 88.0 });
        var active = NivaraColumn<bool>.Create(new[] { true, false, true, true, false });
        var bonus = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        return NivaraFrame.Create(
            ("ID", ids),
            ("Name", names),
            ("Score", scores),
            ("Active", active),
            ("Bonus", bonus)
        );
    }

    [Test]
    public void NumericComparisons_TypedPath_FiltersCorrectly()
    {
        using var frame = CreateFrame();

        using var geFiltered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("ID") >= 300)
            .Collect();
        Assert.That(geFiltered.RowCount, Is.EqualTo(3), ">= filter should keep 300, 400, 500");

        using var leFiltered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("ID") <= 200)
            .Collect();
        Assert.That(leFiltered.RowCount, Is.EqualTo(2), "<= filter should keep 100, 200");

        using var doubleFiltered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Score") < 90.0)
            .Collect();
        Assert.That(doubleFiltered.RowCount, Is.EqualTo(3), "double < filter should keep 85.5, 78.5, 88.0");
    }

    [Test]
    public void StringEquality_TypedPath_FiltersCorrectly()
    {
        using var frame = CreateFrame();

        using var filtered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Name") == "Alice")
            .Collect();

        Assert.That(filtered.RowCount, Is.EqualTo(1));
        Assert.That(filtered.GetColumn<string>("Name")[0], Is.EqualTo("Alice"));
    }

    [Test]
    public void BooleanAndNotEqual_TypedPath_FiltersCorrectly()
    {
        using var frame = CreateFrame();

        using var boolFiltered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Active") == true)
            .Collect();
        Assert.That(boolFiltered.RowCount, Is.EqualTo(3), "boolean filter should keep active rows");

        using var notEqualFiltered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Name") != "Bob")
            .Collect();
        Assert.That(notEqualFiltered.RowCount, Is.EqualTo(4), "!= filter should keep everyone but Bob");
    }

    [Test]
    public void NotEqual_WithNulls_ExcludesNullRowsUsingSqlSemantics()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", null!, "Charlie", "David", "Eve" });
        using var frame = NivaraFrame.Create(("Name", names));

        using var filtered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Name") != "Alice")
            .Collect();

        Assert.That(filtered.RowCount, Is.EqualTo(3), "null name row must be excluded by != (SQL semantics)");
    }

    [Test]
    public void ComparisonSelect_WithNulls_CarriesNullMask()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", null!, "Charlie", "David", "Eve" });
        using var frame = NivaraFrame.Create(("Name", names));

        using var selected = frame.AsQueryFrame()
            .Select(ColumnExpressions.Col("Name") == "Alice")
            .Collect();

        var comparisonColumn = selected.GetColumn<bool>("(Name == Alice)");

        Assert.That(comparisonColumn.HasNulls, Is.True, "typed comparison result should carry a null mask");
        Assert.That(comparisonColumn.IsNull(1), Is.True, "null input row should produce a null comparison");
        Assert.That(comparisonColumn[0], Is.True);
        Assert.That(comparisonColumn[2], Is.False);
        Assert.That(comparisonColumn[4], Is.False);
    }

    [Test]
    public void ScalarMultiply_TypedPath_ProducesTypedElementType()
    {
        using var frame = CreateFrame();

        using var selected = frame.AsQueryFrame()
            .Select(ColumnExpressions.Col("ID") * 2)
            .Collect();

        var doubled = selected.GetColumn<int>("(ID * 2)");
        Assert.That(doubled.Length, Is.EqualTo(5));
        Assert.That(doubled.ToArray(), Is.EqualTo(new[] { 200, 400, 600, 800, 1000 }));
    }

    [Test]
    public void ScalarDivide_TypedPath_ProducesIntegerDivision()
    {
        using var frame = CreateFrame();

        using var selected = frame.AsQueryFrame()
            .Select(ColumnExpressions.Col("ID") / 2)
            .Collect();

        var halved = selected.GetColumn<int>("(ID / 2)");
        Assert.That(halved.ToArray(), Is.EqualTo(new[] { 50, 100, 150, 200, 250 }));
    }

    [Test]
    public void ColumnAddition_TypedPath_ProducesCorrectValues()
    {
        using var frame = CreateFrame();

        using var selected = frame.AsQueryFrame()
            .Select(ColumnExpressions.Col("ID") + ColumnExpressions.Col("Bonus"))
            .Collect();

        var sum = selected.GetColumn<int>("(ID + Bonus)");
        Assert.That(sum.ToArray(), Is.EqualTo(new[] { 101, 202, 303, 404, 505 }));
    }

    [Test]
    public void DoubleScalarArithmetic_TypedPath_ProducesCorrectValues()
    {
        using var frame = CreateFrame();

        using var selected = frame.AsQueryFrame()
            .Select(ColumnExpressions.Col("Score") * 2.0)
            .Collect();

        var doubled = selected.GetColumn<double>("(Score * 2)");
        Assert.That(doubled.ToArray(), Is.EqualTo(new[] { 171.0, 184.0, 157.0, 190.0, 176.0 }));
    }

    [Test]
    public void BoxedFallback_GuidComparison_StillWorks()
    {
        var g1 = Guid.NewGuid();
        var g2 = Guid.NewGuid();
        var guids = NivaraColumn<Guid>.Create(new[] { g1, g2, g1, g2, g1 });
        using var frame = NivaraFrame.Create(("G", guids));

        using var filtered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("G") == g1)
            .Collect();

        Assert.That(filtered.RowCount, Is.EqualTo(3), "Guid equality should fall back to boxed path and still filter");
    }

    [Test]
    public void MemoryStorageColumn_Comparison_FallsBackToBoxed_FiltersCorrectly()
    {
        var ids = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3, 4, 5 });
        using var frame = NivaraFrame.Create(("ID", ids));

        using var filtered = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("ID") > 2)
            .Collect();

        Assert.That(filtered.RowCount, Is.EqualTo(3), "null row must be excluded and 3, 4, 5 kept");
    }
}
