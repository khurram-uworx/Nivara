using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Nivara;
using Nivara.IO;
using Nivara.Expressions;

namespace Nivara.Tests;

/// <summary>
/// Property-based tests for query execution engine
/// Tests universal properties that should hold for all valid queries
/// </summary>
[TestFixture]
public class QueryExecutionPropertyTests
{
    /// <summary>
    /// Property 8: Collect execution barrier
    /// Tests that Collect() acts as an execution barrier and multiple calls produce consistent results
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void Property_CollectExecutionBarrier_MultipleCallsProduceSameResults()
    {
        // Test with various data sizes and query types
        var testCases = new[]
        {
            new { Data = "Name,Age\nAlice,30", Description = "Single row" },
            new { Data = "Name,Age\nAlice,30\nBob,25", Description = "Two rows" },
            new { Data = "Name,Age\nAlice,30\nBob,25\nCharlie,35\nDave,20", Description = "Multiple rows" },
            new { Data = "Name,Age", Description = "Header only (empty data)" }
        };

        foreach (var testCase in testCases)
        {
            var tempFile = Path.GetTempFileName();
            
            try
            {
                File.WriteAllText(tempFile, testCase.Data);
                
                // Test simple query
                var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile);
                
                // Property: Multiple Collect() calls should produce identical results
                var result1 = query.Collect();
                var result2 = query.Collect();
                var result3 = query.Collect();
                
                // Assert structural equality
                Assert.That(result2.RowCount, Is.EqualTo(result1.RowCount), 
                    $"Second collect should have same row count as first for {testCase.Description}");
                Assert.That(result3.RowCount, Is.EqualTo(result1.RowCount), 
                    $"Third collect should have same row count as first for {testCase.Description}");
                
                Assert.That(result2.ColumnCount, Is.EqualTo(result1.ColumnCount), 
                    $"Second collect should have same column count as first for {testCase.Description}");
                Assert.That(result3.ColumnCount, Is.EqualTo(result1.ColumnCount), 
                    $"Third collect should have same column count as first for {testCase.Description}");
                
                // Assert schema equality
                Assert.That(result2.ColumnNames.SequenceEqual(result1.ColumnNames), Is.True, 
                    $"Second collect should have same column names as first for {testCase.Description}");
                Assert.That(result3.ColumnNames.SequenceEqual(result1.ColumnNames), Is.True, 
                    $"Third collect should have same column names as first for {testCase.Description}");
                
                // Assert data equality (if there's data)
                if (result1.RowCount > 0)
                {
                    foreach (var columnName in result1.ColumnNames)
                    {
                        // Handle different column types appropriately
                        if (columnName.Equals("Age", StringComparison.OrdinalIgnoreCase))
                        {
                            var col1 = result1.GetColumn<int>(columnName);
                            var col2 = result2.GetColumn<int>(columnName);
                            var col3 = result3.GetColumn<int>(columnName);
                            
                            for (int i = 0; i < result1.RowCount; i++)
                            {
                                Assert.That(col2[i], Is.EqualTo(col1[i]), 
                                    $"Second collect row {i} column {columnName} should equal first for {testCase.Description}");
                                Assert.That(col3[i], Is.EqualTo(col1[i]), 
                                    $"Third collect row {i} column {columnName} should equal first for {testCase.Description}");
                            }
                        }
                        else
                        {
                            var col1 = result1.GetColumn<string>(columnName);
                            var col2 = result2.GetColumn<string>(columnName);
                            var col3 = result3.GetColumn<string>(columnName);
                            
                            for (int i = 0; i < result1.RowCount; i++)
                            {
                                Assert.That(col2[i], Is.EqualTo(col1[i]), 
                                    $"Second collect row {i} column {columnName} should equal first for {testCase.Description}");
                                Assert.That(col3[i], Is.EqualTo(col1[i]), 
                                    $"Third collect row {i} column {columnName} should equal first for {testCase.Description}");
                            }
                        }
                    }
                }
                
                result1.Dispose();
                result2.Dispose();
                result3.Dispose();
            }
            finally
            {
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            }
        }
    }

    /// <summary>
    /// Property 8: Collect execution barrier
    /// Tests that Collect() works correctly with filtered queries
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void Property_CollectExecutionBarrier_FilteredQueriesConsistent()
    {
        var testData = "Name,Age,Salary\nAlice,30,50000\nBob,25,40000\nCharlie,35,60000\nDave,20,30000\nEve,40,70000";
        var tempFile = Path.GetTempFileName();
        
        try
        {
            File.WriteAllText(tempFile, testData);
            
            // Test various filter conditions
            var filterConditions = new[]
            {
                ColumnExpressions.Col("Age") > 25,
                ColumnExpressions.Col("Age") >= 30,
                ColumnExpressions.Col("Salary") > 45000,
                ColumnExpressions.Col("Age") > 20  // Should match all
            };
            
            foreach (var condition in filterConditions)
            {
                var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                    .Filter(condition);
                
                // Property: Multiple executions of the same filtered query should be identical
                var result1 = query.Collect();
                var result2 = query.Collect();
                
                Assert.That(result2.RowCount, Is.EqualTo(result1.RowCount), 
                    "Filtered query should produce same row count on re-execution");
                
                if (result1.RowCount > 0)
                {
                    var names1 = result1.GetColumn<string>("Name");
                    var names2 = result2.GetColumn<string>("Name");
                    
                    for (int i = 0; i < result1.RowCount; i++)
                    {
                        Assert.That(names2[i], Is.EqualTo(names1[i]), 
                            $"Filtered query row {i} should be identical on re-execution");
                    }
                }
                
                result1.Dispose();
                result2.Dispose();
            }
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    /// <summary>
    /// Property 8: Collect execution barrier
    /// Tests that Collect() properly handles resource management
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void Property_CollectExecutionBarrier_ResourceManagement()
    {
        var testData = "Name,Age\nAlice,30\nBob,25";
        var tempFile = Path.GetTempFileName();
        
        try
        {
            File.WriteAllText(tempFile, testData);
            
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile);
            
            // Property: Each Collect() call should return a new, independent frame
            var result1 = query.Collect();
            var result2 = query.Collect();
            
            // Dispose first result
            result1.Dispose();
            
            // Property: Second result should still be usable after first is disposed
            Assert.That(result2.RowCount, Is.EqualTo(2), 
                "Second result should remain usable after first result is disposed");
            Assert.That(result2.GetColumn<string>("Name")[0], Is.EqualTo("Alice"), 
                "Second result data should be accessible after first result is disposed");
            
            result2.Dispose();
            
            // Property: Query should still be usable for new Collect() calls
            var result3 = query.Collect();
            Assert.That(result3.RowCount, Is.EqualTo(2), 
                "Query should be reusable after previous results are disposed");
            
            result3.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    /// <summary>
    /// Property 8: Collect execution barrier
    /// Tests that disposed queries properly throw ObjectDisposedException
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void Property_CollectExecutionBarrier_DisposedQueryThrows()
    {
        var testData = "Name,Age\nAlice,30";
        var tempFile = Path.GetTempFileName();
        
        try
        {
            File.WriteAllText(tempFile, testData);
            
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile);
            
            // Property: Query should work before disposal
            var result = query.Collect();
            Assert.That(result.RowCount, Is.EqualTo(1));
            result.Dispose();
            
            // Dispose the query
            query.Dispose();
            
            // Property: Collect() on disposed query should throw ObjectDisposedException
            Assert.Throws<ObjectDisposedException>(() => query.Collect(), 
                "Collect() on disposed query should throw ObjectDisposedException");
            
            // Property: Multiple calls to Collect() on disposed query should all throw
            Assert.Throws<ObjectDisposedException>(() => query.Collect(), 
                "Second Collect() call on disposed query should also throw");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    /// <summary>
    /// Property 8: Collect execution barrier
    /// Tests that complex queries maintain consistency across multiple executions
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 8: Collect execution barrier")]
    public void Property_CollectExecutionBarrier_ComplexQueriesConsistent()
    {
        var testData = "Name,Age,Salary,Department\n" +
                      "Alice,30,50000,Engineering\n" +
                      "Bob,25,40000,Sales\n" +
                      "Charlie,35,60000,Engineering\n" +
                      "Dave,20,30000,Sales\n" +
                      "Eve,40,70000,Engineering";
        var tempFile = Path.GetTempFileName();
        
        try
        {
            File.WriteAllText(tempFile, testData);
            
            // Create a complex query with multiple operations
            var query = CsvExtensions.ScanCsvAsQueryFrame(tempFile)
                .Filter(ColumnExpressions.Col("Age") > 25)
                .Filter(ColumnExpressions.Col("Salary") > 45000)
                .Select("Name", "Department");
            
            // Property: Complex queries should produce identical results on multiple executions
            var executions = new IFrame[5];
            
            for (int i = 0; i < executions.Length; i++)
            {
                executions[i] = query.Collect();
            }
            
            // All executions should have the same structure
            for (int i = 1; i < executions.Length; i++)
            {
                Assert.That(executions[i].RowCount, Is.EqualTo(executions[0].RowCount), 
                    $"Execution {i} should have same row count as first execution");
                Assert.That(executions[i].ColumnCount, Is.EqualTo(executions[0].ColumnCount), 
                    $"Execution {i} should have same column count as first execution");
                Assert.That(executions[i].ColumnNames.SequenceEqual(executions[0].ColumnNames), Is.True, 
                    $"Execution {i} should have same column names as first execution");
            }
            
            // All executions should have the same data
            if (executions[0].RowCount > 0)
            {
                for (int i = 1; i < executions.Length; i++)
                {
                    var names0 = executions[0].GetColumn<string>("Name");
                    var namesI = executions[i].GetColumn<string>("Name");
                    var dept0 = executions[0].GetColumn<string>("Department");
                    var deptI = executions[i].GetColumn<string>("Department");
                    
                    for (int row = 0; row < executions[0].RowCount; row++)
                    {
                        Assert.That(namesI[row], Is.EqualTo(names0[row]), 
                            $"Execution {i} row {row} Name should match first execution");
                        Assert.That(deptI[row], Is.EqualTo(dept0[row]), 
                            $"Execution {i} row {row} Department should match first execution");
                    }
                }
            }
            
            // Clean up
            foreach (var execution in executions)
            {
                execution.Dispose();
            }
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}