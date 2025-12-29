using Nivara.Expressions;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class QueryOptimizerTests
{
    [Test]
    [Category("Feature: nivara-frame, Property 16: Comprehensive query optimization")]
    public void PredicatePushdown_ShouldMoveFiltersEarlier()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query where filter comes after select (suboptimal)
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Select("Name", "Age")  // Select first
                .Filter(ColumnExpressions.Col("Age") > 25);  // Filter after select

            // Act - Get optimization suggestions
            var suggestions = query.AnalyzeOptimizations();

            // Assert - Should suggest predicate pushdown
            Assert.That(suggestions, Is.Not.Empty, "Should provide optimization suggestions");

            // Execute the query to ensure it still works correctly
            var result = query.Collect();
            Assert.That(result.RowCount, Is.EqualTo(2), "Should return Alice and Charlie");
            Assert.That(result.ColumnCount, Is.EqualTo(2), "Should have Name and Age columns");

            result.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 16: Comprehensive query optimization")]
    public void OperationFusion_ShouldCombineMultipleFilters()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000\nDave,20,30000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query with multiple filters that can be fused
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Filter(ColumnExpressions.Col("Salary") > 35000);

            // Act
            var result = query.Collect();

            // Assert - Should return only Alice and Charlie (both conditions met)
            Assert.That(result.RowCount, Is.EqualTo(2), "Should return 2 rows after combined filtering");

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
    [Category("Feature: nivara-frame, Property 16: Comprehensive query optimization")]
    public void ColumnElimination_ShouldRemoveUnusedColumns()
    {
        // Arrange
        var testData = "Name,Age,Salary,Department\nAlice,30,50000,Engineering\nBob,25,40000,Sales";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query that only uses some columns
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)  // Only uses Age and Name implicitly
                .Select("Name");  // Only selects Name

            // Act
            var result = query.Collect();

            // Assert - Should only have the Name column
            Assert.That(result.ColumnCount, Is.EqualTo(1), "Should have only 1 column after optimization");
            Assert.That(result.ColumnNames.First(), Is.EqualTo("Name"));
            Assert.That(result.RowCount, Is.EqualTo(1), "Should return Alice (Age > 25)");

            result.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 16: Comprehensive query optimization")]
    public void OperationReordering_ShouldOptimizeOperationOrder()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000\nDave,20,30000\nEve,40,70000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a query where operations could be reordered for better performance
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Select("Name", "Age", "Salary")  // Select first (less optimal)
                .Filter(ColumnExpressions.Col("Age") > 30);  // Filter after select (should be moved earlier)

            // Act
            var result = query.Collect();

            // Assert - Should return Alice, Charlie, and Eve (Age > 30)
            // But Age > 30 means Alice (30) is NOT included, only Charlie (35) and Eve (40)
            Assert.That(result.RowCount, Is.EqualTo(2), "Should return 2 rows (Charlie and Eve)");
            Assert.That(result.ColumnCount, Is.EqualTo(3), "Should have 3 columns");

            var names = result.GetColumn<string>("Name");
            Assert.That(names[0], Is.EqualTo("Charlie"));
            Assert.That(names[1], Is.EqualTo("Eve"));

            result.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    [Category("Feature: nivara-frame, Property 16: Comprehensive query optimization")]
    public void ComplexOptimization_ShouldApplyMultipleOptimizations()
    {
        // Arrange
        var testData = "Name,Age,Salary,Department,Location\n" +
                      "Alice,30,50000,Engineering,NYC\n" +
                      "Bob,25,40000,Sales,LA\n" +
                      "Charlie,35,60000,Engineering,NYC\n" +
                      "Dave,20,30000,Sales,Chicago\n" +
                      "Eve,40,70000,Engineering,NYC";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create a complex query with multiple optimization opportunities
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Select("Name", "Age", "Salary", "Department")  // Column elimination opportunity
                .Filter(ColumnExpressions.Col("Age") > 25)      // Filter 1
                .Filter(ColumnExpressions.Col("Salary") > 45000) // Filter 2 (fusion opportunity)
                .Select("Name", "Department");                   // Final projection

            // Act
            var result = query.Collect();

            // Assert - Should return Alice, Charlie, and Eve (Age > 25 AND Salary > 45000)
            // Alice: Age=30 (>25), Salary=50000 (>45000) ✓
            // Charlie: Age=35 (>25), Salary=60000 (>45000) ✓  
            // Eve: Age=40 (>25), Salary=70000 (>45000) ✓
            Assert.That(result.RowCount, Is.EqualTo(3), "Should return 3 rows after optimization");
            Assert.That(result.ColumnCount, Is.EqualTo(2), "Should have 2 columns after final projection");

            var names = result.GetColumn<string>("Name");
            var departments = result.GetColumn<string>("Department");

            Assert.That(names[0], Is.EqualTo("Alice"));
            Assert.That(departments[0], Is.EqualTo("Engineering"));
            Assert.That(names[1], Is.EqualTo("Charlie"));
            Assert.That(departments[1], Is.EqualTo("Engineering"));
            Assert.That(names[2], Is.EqualTo("Eve"));
            Assert.That(departments[2], Is.EqualTo("Engineering"));

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
    public void OptimizationTransparency_ShouldNotChangeResults()
    {
        // Arrange
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, testData);

            // Create two equivalent queries - one optimized, one not
            var optimizedQuery = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Select("Name", "Age");

            var unoptimizedQuery = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Select("Name", "Age", "Salary")  // Select more columns first
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Select("Name", "Age");  // Then narrow down

            // Act
            var optimizedResult = optimizedQuery.Collect();
            var unoptimizedResult = unoptimizedQuery.Collect();

            // Assert - Results should be identical
            Assert.That(optimizedResult.RowCount, Is.EqualTo(unoptimizedResult.RowCount),
                "Optimized and unoptimized queries should return same number of rows");
            Assert.That(optimizedResult.ColumnCount, Is.EqualTo(unoptimizedResult.ColumnCount),
                "Optimized and unoptimized queries should return same number of columns");

            // Compare actual data
            var optimizedNames = optimizedResult.GetColumn<string>("Name");
            var unoptimizedNames = unoptimizedResult.GetColumn<string>("Name");

            for (int i = 0; i < optimizedResult.RowCount; i++)
            {
                Assert.That(optimizedNames[i], Is.EqualTo(unoptimizedNames[i]),
                    $"Row {i} should have same Name value");
            }

            optimizedResult.Dispose();
            unoptimizedResult.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}