using NUnit.Framework;
using Nivara.Expressions;

namespace Nivara.Tests;

[TestFixture]
public class QueryFrameTests
{
    [Test]
    public void QueryFrame_Filter_BuildsQueryPlan()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Numbers") > 3);
        
        // Assert
        Assert.That(filteredQuery, Is.Not.Null);
        Assert.That(filteredQuery.IsLazy, Is.False); // Memory source is not lazy
        Assert.That(filteredQuery.Schema.ColumnNames, Is.EqualTo(new[] { "Numbers", "Letters" }));
    }

    [Test]
    public void QueryFrame_Select_BuildsQueryPlan()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var selectedQuery = queryFrame.Select("Numbers");
        
        // Assert
        Assert.That(selectedQuery, Is.Not.Null);
        Assert.That(selectedQuery.Schema.ColumnNames, Is.EqualTo(new[] { "Numbers" }));
        Assert.That(selectedQuery.Schema.ColumnNames.Count, Is.EqualTo(1));
    }

    [Test]
    public void QueryFrame_GroupBy_BuildsQueryPlan()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var groupedQuery = queryFrame.GroupBy("Numbers");
        
        // Assert
        Assert.That(groupedQuery, Is.Not.Null);
        Assert.That(groupedQuery.Schema.ColumnNames, Is.EqualTo(new[] { "Numbers" }));
    }

    [Test]
    public void QueryFrame_ChainedOperations_BuildsQueryPlan()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var chainedQuery = queryFrame
            .Filter(ColumnExpressions.Col("Numbers") > 2)
            .Select("Numbers", "Letters")
            .GroupBy("Numbers");
        
        // Assert
        Assert.That(chainedQuery, Is.Not.Null);
        Assert.That(chainedQuery.Schema.ColumnNames, Is.EqualTo(new[] { "Numbers" }));
    }

    [Test]
    public void QueryFrame_Collect_ExecutesQuery()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Numbers") > 3);
        var result = filteredQuery.Collect();
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(2)); // Values 4 and 5
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        
        var numbersColumn = result.GetColumn<int>("Numbers");
        Assert.That(numbersColumn[0], Is.EqualTo(4));
        Assert.That(numbersColumn[1], Is.EqualTo(5));
    }

    [Test]
    public void QueryFrame_SelectColumns_ExecutesQuery()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var selectedQuery = queryFrame.Select("Letters");
        var result = selectedQuery.Collect();
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(5));
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Letters" }));
        
        var lettersColumn = result.GetColumn<string>("Letters");
        Assert.That(lettersColumn[0], Is.EqualTo("a"));
        Assert.That(lettersColumn[4], Is.EqualTo("e"));
    }

    [Test]
    public void QueryFrame_GroupByDistinct_ExecutesQuery()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var groupedQuery = queryFrame.GroupBy("Numbers");
        var result = groupedQuery.Collect();
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3)); // Distinct values: 1, 2, 3
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        
        var numbersColumn = result.GetColumn<int>("Numbers");
        var distinctValues = new HashSet<int>();
        for (int i = 0; i < numbersColumn.Length; i++)
        {
            distinctValues.Add(numbersColumn[i]);
        }
        
        Assert.That(distinctValues, Is.EqualTo(new HashSet<int> { 1, 2, 3 }));
    }

    [Test]
    public void QueryFrame_ExplainPlan_ReturnsDescription()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Numbers") > 3);
        var explanation = filteredQuery.ExplainPlan();
        
        // Assert
        Assert.That(explanation, Is.Not.Null);
        Assert.That(explanation, Does.Contain("Query Execution Plan"));
        Assert.That(explanation, Does.Contain("Filter"));
    }

    [Test]
    public void QueryFrame_AnalyzeOptimizations_ReturnsOptimizations()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var filteredQuery = queryFrame
            .Filter(ColumnExpressions.Col("Numbers") > 2)
            .Filter(ColumnExpressions.Col("Numbers") < 5);
        var optimizations = filteredQuery.AnalyzeOptimizations();
        
        // Assert
        Assert.That(optimizations, Is.Not.Null);
        Assert.That(optimizations.Count, Is.GreaterThan(0));
    }

    [Test]
    public void QueryFrame_NullCondition_ThrowsArgumentNullException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var queryFrame = frame.AsQueryFrame();
        
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => queryFrame.Filter(null!));
    }

    [Test]
    public void QueryFrame_EmptySelect_ThrowsArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var queryFrame = frame.AsQueryFrame();
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => queryFrame.Select(new string[0]));
    }

    [Test]
    public void QueryFrame_EmptyGroupBy_ThrowsArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var queryFrame = frame.AsQueryFrame();
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => queryFrame.GroupBy(new string[0]));
    }

    [Test]
    public void QueryFrame_ToString_ReturnsDescription()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        
        // Act
        var queryFrame = frame.AsQueryFrame();
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Numbers") > 2);
        var description = filteredQuery.ToString();
        
        // Assert
        Assert.That(description, Is.Not.Null);
        Assert.That(description, Does.Contain("QueryFrame"));
        Assert.That(description, Does.Contain("Filter"));
    }
}