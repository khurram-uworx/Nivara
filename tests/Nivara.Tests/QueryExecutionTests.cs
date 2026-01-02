using Nivara.Expressions;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class QueryExecutionTests
{
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void MultipleCollectCalls_ShouldReExecuteQuery()
    {
        // Arrange
        var testData = "Name,Age\nAlice,30\nBob,25\nCharlie,35";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            var query = Csv.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25);

            // Act - First Collect() call
            var result1 = query.Collect();

            // Act - Second Collect() call (should re-execute)
            var result2 = query.Collect();

            // Assert
            Assert.That(result1.RowCount, Is.EqualTo(2), "First collect should return 2 rows (Alice and Charlie)");
            Assert.That(result2.RowCount, Is.EqualTo(2), "Second collect should return 2 rows (Alice and Charlie)");

            // Verify the data is the same
            var name1_1 = result1.GetColumn<string>("Name")[0];
            var name1_2 = result1.GetColumn<string>("Name")[1];
            var name2_1 = result2.GetColumn<string>("Name")[0];
            var name2_2 = result2.GetColumn<string>("Name")[1];

            Assert.That(name1_1, Is.EqualTo(name2_1), "First row name should be the same");
            Assert.That(name1_2, Is.EqualTo(name2_2), "Second row name should be the same");

            // Clean up
            result1.Dispose();
            result2.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void CollectAfterDispose_ShouldThrowObjectDisposedException()
    {
        // Arrange
        var testData = "Name,Age\nAlice,30";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            var query = Csv.ScanCsvAsQueryFrame(tempFile);

            // Act - Dispose the query
            query.Dispose();

            // Assert - Collect should throw ObjectDisposedException
            Assert.Throws<ObjectDisposedException>(() => query.Collect());
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 9: Query optimization during execution")]
    public void QueryOptimization_ShouldBeAppliedDuringExecution()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query with multiple filters (optimization opportunity)
            var query = Csv.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Filter(ColumnExpressions.Col("Salary") > 45000)
                .Select("Name", "Age");

            // Act
            var result = query.Collect();

            // Assert - Should return Alice and Charlie (both age > 25 and salary > 45000)
            Assert.That(result.RowCount, Is.EqualTo(2), "Should return 2 rows after filtering");
            Assert.That(result.ColumnCount, Is.EqualTo(2), "Should return 2 columns after selection");

            var names = result.GetColumn<string>("Name");
            Assert.That(names[0], Is.EqualTo("Alice"));
            Assert.That(names[1], Is.EqualTo("Charlie"));

            result.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 9: Query optimization during execution")]
    public void QueryOptimizer_AnalyzeOptimizations_ShouldProvideUsefulSuggestions()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query with optimization opportunities
            var query = Csv.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Filter(ColumnExpressions.Col("Salary") > 45000)  // Multiple filters
                .Select("Name", "Age");  // Select both columns at once

            // Act
            var suggestions = query.AnalyzeOptimizations();

            // Assert
            Assert.That(suggestions, Is.Not.Empty, "Should provide optimization suggestions");

            var suggestionsText = string.Join(" ", suggestions);
            Assert.That(suggestionsText.ToLower(), Does.Contain("filter").Or.Contain("select"),
                "Should suggest optimizations for multiple filters or selects");

            // Verify the query still works
            var result = query.Collect();
            Assert.That(result.RowCount, Is.EqualTo(1));
            result.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
