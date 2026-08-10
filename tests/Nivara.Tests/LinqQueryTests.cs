using Nivara.Execution;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class LinqQueryTests
{
    [Test]
    public void Where_WithLambda_FiltersCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Where(x => x["Numbers"] > 3)
            .ToNivaraFrame();

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(4));
        Assert.That(numbers[1], Is.EqualTo(5));
    }

    [Test]
    public void Select_WithLambda_ProjectsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Select(x => x["Letters"])
            .ToNivaraFrame();

        // Assert
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Letters" }));
        Assert.That(result.RowCount, Is.EqualTo(5));
    }

    [Test]
    public void Select_WithMultipleLambdas_ProjectsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Select(x => x["Letters"], x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Letters", "Numbers" }));
    }

    [Test]
    public void OrderBy_WithLambda_SortsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(1));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(3));
    }

    [Test]
    public void OrderByDescending_WithLambda_SortsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 3, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderByDescending(x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(3));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(1));
    }

    [Test]
    public void ChainedLinqOperations_RunCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 5, 1, 4, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "e", "a", "d", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Where(x => x["Numbers"] > 2)
            .OrderBy(x => x["Numbers"])
            .Select(x => x["Letters"])
            .ToNivaraFrame();

        // Assert
        // Expected: 5, 4, 3 -> Sorted: 3, 4, 5 -> Letters: c, d, e
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnNames.Count, Is.EqualTo(1));

        var letters = result.GetColumn<string>("Letters");
        Assert.That(letters[0], Is.EqualTo("c")); // 3
        Assert.That(letters[1], Is.EqualTo("d")); // 4
        Assert.That(letters[2], Is.EqualTo("e")); // 5
    }

    [Test]
    public void ToRowList_WithFilter_MaterializesRowViews()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var scores = NivaraColumn<int>.CreateFromNullable(new int?[] { 10, null, 30 });
        var frame = NivaraFrame.Create(("Numbers", numbers), ("Scores", scores));

        // Act
        var rows = frame.AsQueryFrame()
            .Where(x => x["Numbers"] > 1)
            .ToRowList();

        // Assert
        Assert.That(rows, Has.Count.EqualTo(2));
        Assert.That(rows[0].GetValue<int>("Numbers"), Is.EqualTo(2));
        Assert.That(rows[0].IsNull("Scores"), Is.True);
        Assert.That(rows[1].GetValue<int>("Numbers"), Is.EqualTo(3));
        Assert.That(rows[1].GetValue<int>("Scores"), Is.EqualTo(30));
        Assert.That(rows[1].IsNull("Scores"), Is.False);
    }

    [Test]
    public void OrderBy_WithComputedKey_SortsByMaterializedColumn()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
        var col2 = NivaraColumn<int>.Create(new[] { 30, 10, 20 });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Twice", col2));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Numbers"] * 2)
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(1));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(3));
    }

    [Test]
    public void OrderBy_WithColumnSumExpression_SortsByResult()
    {
        // Arrange
        var left = NivaraColumn<int>.Create(new[] { 1, 5, 3 });
        var right = NivaraColumn<int>.Create(new[] { 10, 2, 3 });
        var frame = NivaraFrame.Create(("A", left), ("B", right));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["A"] + x["B"])
            .ToNivaraFrame();

        // Assert
        // Sums: 11, 7, 6 -> sorted order: (3,3), (5,2), (1,10)
        Assert.That(result.GetColumn<int>("A")[0], Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("A")[1], Is.EqualTo(5));
        Assert.That(result.GetColumn<int>("A")[2], Is.EqualTo(1));
    }

    [Test]
    public void OrderBy_WithMixedTypeComputedKey_UsesFallbackPath()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var scores = NivaraColumn<double>.Create(new[] { 85.5, 92.0, 78.5 });
        var frame = NivaraFrame.Create(("ID", ids), ("Score", scores));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Score"] + x["ID"])
            .ToNivaraFrame();

        // Assert
        // Keys: 86.5, 94.0, 81.5 -> sorted: 3, 1, 2
        Assert.That(result.GetColumn<int>("ID")[0], Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("ID")[1], Is.EqualTo(1));
        Assert.That(result.GetColumn<int>("ID")[2], Is.EqualTo(2));
    }

    [Test]
    public void OrderByDescending_WithComputedKey_SortsDescending()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 3, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderByDescending(x => x["Numbers"] * 2)
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(3));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(1));
    }

    [Test]
    public void OrderBy_WithComputedKey_NullsPlacement_MatchSortSemantics()
    {
        // Arrange
        var ages = NivaraColumn<int>.CreateFromNullable(new int?[] { 30, null, 20, 40, null });
        var names = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Age", ages), ("Name", names));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Age"] * 2)
            .ToNivaraFrame();

        // Assert
        // Keys: 60, null, 40, 80, null -> ascending with NullsLast: 20, 30, 40, null, null
        var sortedAges = result.GetColumn<int>("Age");
        Assert.That(sortedAges[0], Is.EqualTo(20));
        Assert.That(sortedAges[1], Is.EqualTo(30));
        Assert.That(sortedAges[2], Is.EqualTo(40));
        Assert.That(sortedAges.IsNull(3), Is.True);
        Assert.That(sortedAges.IsNull(4), Is.True);
    }

    [Test]
    public void OrderBy_WithComputedKey_AfterFilter_SortsFilteredRows()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var vals = NivaraColumn<int>.Create(new[] { 10, 50, 20, 40, 30 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act
        var result = frame.AsQueryFrame()
            .Where(x => x["ID"] > 1)
            .OrderBy(x => x["Val"] / 10)
            .ToNivaraFrame();

        // Assert
        // IDs 2..5, keys 5,2,4,3 -> sorted: 3, 5, 4, 2
        Assert.That(result.GetColumn<int>("ID")[0], Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("ID")[1], Is.EqualTo(5));
        Assert.That(result.GetColumn<int>("ID")[2], Is.EqualTo(4));
        Assert.That(result.GetColumn<int>("ID")[3], Is.EqualTo(2));
    }

    [Test]
    public void OrderBy_WithLiteralKey_DoesNotThrow()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x.Lit(5))
            .ToNivaraFrame();

        // Assert
        // A constant key leaves rows in their original relative order (stable sort)
        Assert.That(result.GetColumn<int>("Numbers")[0], Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("Numbers")[1], Is.EqualTo(1));
        Assert.That(result.GetColumn<int>("Numbers")[2], Is.EqualTo(2));
    }

    [Test]
    public void SortByExpression_OnQueryFrame_SortsByComputedColumn()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var vals = NivaraColumn<int>.Create(new[] { 100, 300, 200 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act
        var result = frame.AsQueryFrame()
            .SortByExpression(Nivara.Expressions.ColumnExpressions.Col("Val") * 3)
            .Collect();

        // Assert
        var sortedIds = result.GetColumn<int>("ID");
        Assert.That(sortedIds[0], Is.EqualTo(1));
        Assert.That(sortedIds[1], Is.EqualTo(3));
        Assert.That(sortedIds[2], Is.EqualTo(2));
    }

    [Test]
    public void OrderBy_WithComputedKey_UnderParallelStrategy_MatchesLazy()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 5, 3, 1, 4, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 50, 30, 10, 40, 20 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        var queryFrame = frame.AsQueryFrame().OrderBy(x => x["Val"] * 2);
        var plan = queryFrame.ToQueryPlan();

        // Act
        var engine = new ExecutionEngine();
        var parallelResult = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Parallel));

        // Assert
        var sortedIds = parallelResult.GetColumn<int>("ID");
        Assert.That(sortedIds[0], Is.EqualTo(1));
        Assert.That(sortedIds[1], Is.EqualTo(2));
        Assert.That(sortedIds[2], Is.EqualTo(3));
        Assert.That(sortedIds[3], Is.EqualTo(4));
        Assert.That(sortedIds[4], Is.EqualTo(5));
    }

    [Test]
    public void OrderBy_WithComputedKey_UnderStreamingStrategy_FallsBackToLazy()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 5, 3, 1, 4, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 50, 30, 10, 40, 20 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        var queryFrame = frame.AsQueryFrame().OrderBy(x => x["Val"] + x["ID"]);
        var plan = queryFrame.ToQueryPlan();

        // Act
        var engine = new ExecutionEngine();
        var streamingResult = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Streaming));

        // Assert
        // Keys (Val+ID): 55, 33, 11, 44, 22 -> sorted: 1, 2, 3, 4, 5
        var sortedIds = streamingResult.GetColumn<int>("ID");
        Assert.That(sortedIds[0], Is.EqualTo(1));
        Assert.That(sortedIds[1], Is.EqualTo(2));
        Assert.That(sortedIds[2], Is.EqualTo(3));
        Assert.That(sortedIds[3], Is.EqualTo(4));
        Assert.That(sortedIds[4], Is.EqualTo(5));
    }

    [Test]
    public void ThenBy_WithColumnReference_PreservesPrimaryOrderWithinTies()
    {
        // Arrange
        var names = NivaraColumn<string>.Create(new[] { "b", "a", "b", "a", "b" });
        var ages = NivaraColumn<int>.Create(new[] { 30, 20, 10, 40, 25 });
        var frame = NivaraFrame.Create(("Name", names), ("Age", ages));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Name"])
            .ThenBy(x => x["Age"])
            .ToNivaraFrame();

        // Assert
        // Name asc, then Age asc within each name: a20, a40, b10, b25, b30
        Assert.That(result.GetColumn<string>("Name").ToArray(), Is.EqualTo(new[] { "a", "a", "b", "b", "b" }));
        Assert.That(result.GetColumn<int>("Age").ToArray(), Is.EqualTo(new[] { 20, 40, 10, 25, 30 }));
    }

    [Test]
    public void ThenBy_WithComputedKey_SortsByMaterializedColumn()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 1, 2, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 10, 30, 20, 40 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["ID"])
            .ThenBy(x => x["Val"] * 2)
            .ToNivaraFrame();

        // Assert
        // ID asc, Val*2 asc within each ID: (1,10), (1,30), (2,20), (2,40)
        Assert.That(result.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 1, 2, 2 }));
        Assert.That(result.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 10, 30, 20, 40 }));
    }

    [Test]
    public void OrderBy_WithComputedKey_ThenBy_WithColumnReference_SortsLexicographically()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 3, 1, 3, 1 });
        var vals = NivaraColumn<int>.Create(new[] { 2, 4, 1, 3 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["ID"] * 2)
            .ThenBy(x => x["Val"])
            .ToNivaraFrame();

        // Assert
        // ID*2 asc, then Val asc within ties: (2,3),(2,4),(6,1),(6,2) -> ID:1,1,3,3; Val:3,4,1,2
        Assert.That(result.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 1, 3, 3 }));
        Assert.That(result.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 3, 4, 1, 2 }));
    }

    [Test]
    public void ThenByDescending_WithComputedKey_SortsDescendingWithinPrimaryGroups()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 1, 2, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 10, 30, 20, 40 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["ID"])
            .ThenByDescending(x => x["Val"])
            .ToNivaraFrame();

        // Assert
        // ID asc, Val desc within each ID: (1,30),(1,10),(2,40),(2,20)
        Assert.That(result.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 1, 2, 2 }));
        Assert.That(result.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 30, 10, 40, 20 }));
    }

    [Test]
    public void ThenBy_WithColumnReferences_DoesNotThrow()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var vals = NivaraColumn<int>.Create(new[] { 30, 10, 20 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act & Assert
        var result = frame.AsQueryFrame().OrderBy(x => x["ID"]).ThenBy(x => x["Val"]).ToNivaraFrame();
        Assert.That(result.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 2, 3 }));
        Assert.That(result.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 30, 10, 20 }));
    }

    [Test]
    public void ThenBy_WithComputedKey_UnderParallelStrategy_MatchesLazy()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 1, 2, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 10, 30, 20, 40 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        var queryFrame = frame.AsQueryFrame().OrderBy(x => x["ID"]).ThenBy(x => x["Val"] * 2);
        var plan = queryFrame.ToQueryPlan();

        // Act
        var engine = new ExecutionEngine();
        var parallelResult = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Parallel));

        // Assert
        Assert.That(parallelResult.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 1, 2, 2 }));
        Assert.That(parallelResult.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 10, 30, 20, 40 }));
    }

    [Test]
    public void ThenBy_WithComputedKey_UnderStreamingStrategy_FallsBackToLazy()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 1, 1, 2, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 10, 30, 20, 40 });
        var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        var queryFrame = frame.AsQueryFrame().OrderBy(x => x["ID"]).ThenBy(x => x["Val"] + x["ID"]);
        var plan = queryFrame.ToQueryPlan();

        // Act
        var engine = new ExecutionEngine();
        var streamingResult = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Streaming));

        // Assert
        // ID asc, (Val+ID) asc within each ID: keys 11, 31, 22, 42 -> (1,10),(1,30),(2,20),(2,40)
        Assert.That(streamingResult.GetColumn<int>("ID").ToArray(), Is.EqualTo(new[] { 1, 1, 2, 2 }));
        Assert.That(streamingResult.GetColumn<int>("Val").ToArray(), Is.EqualTo(new[] { 10, 30, 20, 40 }));
    }
}
