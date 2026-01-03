using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Expressions;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Integration tests for complex scenarios combining multiple operations and testing edge cases.
/// These tests validate the robustness and correctness of component interactions under various conditions.
/// </summary>
[TestFixture]
public class ComplexScenarioIntegrationTests
{
    [Test]
    [Category("Integration")]
    public void ComplexQueryChaining_WithMultipleFiltersAndSelections_WorksCorrectly()
    {
        // Arrange - Create a realistic dataset
        var employeeIds = NivaraColumn<int>.Create(Enumerable.Range(1, 500).ToArray());
        var names = NivaraColumn<string>.Create(Enumerable.Range(1, 500).Select(i => $"Employee{i:D3}").ToArray());
        var departments = NivaraColumn<string>.Create(Enumerable.Range(1, 500).Select(i => $"Dept{(i % 5) + 1}").ToArray());
        var salaries = NivaraColumn<double>.Create(Enumerable.Range(1, 500).Select(i => (double)(30000 + (i * 100) + (i % 10) * 1000)).ToArray());
        var isActive = NivaraColumn<bool>.Create(Enumerable.Range(1, 500).Select(i => i % 7 != 0).ToArray());
        var hireDate = NivaraColumn<DateTime>.Create(Enumerable.Range(1, 500).Select(i => DateTime.Now.AddDays(-i * 30)).ToArray());

        var frame = NivaraFrame.Create(
            ("EmployeeID", employeeIds),
            ("Name", names),
            ("Department", departments),
            ("Salary", salaries),
            ("IsActive", isActive),
            ("HireDate", hireDate)
        );

        // Act - Build complex query with multiple filters and transformations
        var result = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Salary") > 50000.0)
            .Filter(ColumnExpressions.Col("IsActive") == true)
            .Filter(ColumnExpressions.Col("Department") == "Dept1")
            .Select("EmployeeID", "Name", "Salary", "Department")
            .Collect();

        // Assert - Verify complex query results
        Assert.That(result.RowCount, Is.GreaterThan(0), "Complex query should return results");
        Assert.That(result.ColumnCount, Is.EqualTo(4), "Query should return selected columns");

        // Verify all returned rows meet all criteria
        var resultSalaries = result.GetColumn<double>("Salary");
        var resultDepartments = result.GetColumn<string>("Department");
        var resultIds = result.GetColumn<int>("EmployeeID");

        for (int i = 0; i < result.RowCount; i++)
        {
            Assert.That(resultSalaries[i], Is.GreaterThan(50000.0), $"Row {i} should have salary > 50000");
            Assert.That(resultDepartments[i], Is.EqualTo("Dept1"), $"Row {i} should be in Dept1");

            // Verify the original employee was active
            var originalIndex = resultIds[i] - 1; // IDs are 1-based
            Assert.That(isActive[originalIndex], Is.True, $"Row {i} should have been active");
        }

        // Clean up
        frame.Dispose();
        result.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void SeriesOperations_WithComplexAlignment_HandlesEdgeCases()
    {
        // Arrange - Create series with various alignment scenarios

        // Series 1: Sequential labels
        var series1Values = NivaraColumn<double>.Create(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var series1Index = NivaraColumn<object>.Create(new object[] { "A", "B", "C", "D", "E" });
        var series1 = new NivaraSeries<double>(series1Values, series1Index);

        // Series 2: Overlapping labels with gaps
        var series2Values = NivaraColumn<double>.Create(new[] { 100.0, 200.0, 300.0 });
        var series2Index = NivaraColumn<object>.Create(new object[] { "B", "D", "F" });
        var series2 = new NivaraSeries<double>(series2Values, series2Index);

        // Series 3: No overlapping labels
        var series3Values = NivaraColumn<double>.Create(new[] { 1000.0, 2000.0 });
        var series3Index = NivaraColumn<object>.Create(new object[] { "X", "Y" });
        var series3 = new NivaraSeries<double>(series3Values, series3Index);

        // Act & Assert - Test various alignment scenarios

        // Scenario 1: Partial overlap
        var aligned12 = series1.AlignBoth(series2);
        Assert.That(aligned12.Left.Length, Is.EqualTo(2), "Should find 2 matching indices (B, D)");
        Assert.That(aligned12.Right.Length, Is.EqualTo(2), "Should find 2 matching indices (B, D)");
        Assert.That(aligned12.Left.GetByLabel("B"), Is.EqualTo(20.0), "Aligned value should be correct");
        Assert.That(aligned12.Right.GetByLabel("B"), Is.EqualTo(100.0), "Aligned value should be correct");

        // Scenario 2: No overlap
        var aligned13 = series1.AlignBoth(series3);
        Assert.That(aligned13.Left.Length, Is.EqualTo(0), "Should find no matching indices");
        Assert.That(aligned13.Right.Length, Is.EqualTo(0), "Should find no matching indices");

        // Scenario 3: Series operations with alignment
        var addResult = series1.Add(series2); // Should only include aligned values
        Assert.That(addResult.Length, Is.EqualTo(2), "Addition should only include aligned values");
        Assert.That(addResult.GetByLabel("B"), Is.EqualTo(120.0), "Addition should work correctly (20 + 100)");
        Assert.That(addResult.GetByLabel("D"), Is.EqualTo(240.0), "Addition should work correctly (40 + 200)");

        // Clean up
        series1.Dispose();
        series2.Dispose();
        series3.Dispose();
        aligned12.Left.Dispose();
        aligned12.Right.Dispose();
        aligned13.Left.Dispose();
        aligned13.Right.Dispose();
        addResult.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void MixedTypeArithmetic_WithVectorizedAndScalar_WorksCorrectly()
    {
        // Arrange - Create columns with different vectorization characteristics
        var vectorizedInts = NivaraColumn<int>.Create(Enumerable.Range(1, 1000).ToArray());
        var vectorizedDoubles = NivaraColumn<double>.Create(Enumerable.Range(1, 1000).Select(i => i * 1.5).ToArray());
        var scalarStrings = NivaraColumn<string>.Create(Enumerable.Range(1, 1000).Select(i => $"Item{i}").ToArray());

        // Verify storage types
        Assert.That(vectorizedInts.Diagnostics.StorageType, Is.EqualTo(StorageType.Tensor), "Ints should use tensor storage");
        Assert.That(vectorizedDoubles.Diagnostics.StorageType, Is.EqualTo(StorageType.Tensor), "Doubles should use tensor storage");
        Assert.That(scalarStrings.Diagnostics.StorageType, Is.EqualTo(StorageType.Memory), "Strings should use memory storage");

        // Act - Perform operations that mix vectorized and scalar operations
        var doubledInts = vectorizedInts * 2; // Vectorized operation
        var scaledDoubles = vectorizedDoubles * 0.5; // Vectorized operation
        var intComparison = vectorizedInts.GreaterThan(500); // Vectorized comparison
        var stringComparison = scalarStrings.Equals("Item500"); // Scalar comparison

        // Assert - Verify mixed operations work correctly
        Assert.That(doubledInts.Length, Is.EqualTo(1000), "Vectorized operations should preserve length");
        Assert.That(scaledDoubles.Length, Is.EqualTo(1000), "Vectorized operations should preserve length");
        Assert.That(intComparison.Length, Is.EqualTo(1000), "Vectorized comparisons should preserve length");
        Assert.That(stringComparison.Length, Is.EqualTo(1000), "Scalar comparisons should preserve length");

        // Verify some results
        Assert.That(doubledInts[0], Is.EqualTo(2), "Vectorized multiplication should work correctly");
        Assert.That(scaledDoubles[0], Is.EqualTo(0.75), "Vectorized scaling should work correctly (1.5 * 0.5)");
        Assert.That(intComparison[600], Is.EqualTo(true), "Vectorized comparison should work correctly (601 > 500)");
        Assert.That(stringComparison[499], Is.EqualTo(true), "Scalar comparison should work correctly");

        // Create frame with mixed types and perform query
        var mixedFrame = NivaraFrame.Create(
            ("ID", vectorizedInts),
            ("Value", vectorizedDoubles),
            ("Name", scalarStrings)
        );

        var queryResult = mixedFrame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("ID") > 750)
            .Filter(ColumnExpressions.Col("Value") < 1200.0)
            .Select("ID", "Name")
            .Collect();

        Assert.That(queryResult.RowCount, Is.GreaterThan(0), "Mixed-type query should return results");

        // Clean up
        doubledInts.Dispose();
        scaledDoubles.Dispose();
        intComparison.Dispose();
        stringComparison.Dispose();
        mixedFrame.Dispose();
        queryResult.Dispose();
        vectorizedInts.Dispose();
        vectorizedDoubles.Dispose();
        scalarStrings.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void ErrorRecovery_AcrossMultipleOperations_ProvidesUsefulContext()
    {
        // Arrange - Create frame for testing error scenarios
        var ids = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var names = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie", "David", "Eve" });
        var frame = NivaraFrame.Create(("ID", ids), ("Name", names));

        // Test 1: Column not found in query
        var ex1 = Assert.Throws<QueryExecutionException>(() =>
        {
            var query = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("NonExistentColumn") > 0)
                .Collect();
        });
        Assert.That(ex1.Message, Contains.Substring("NonExistentColumn"), "Error should mention missing column");
        Assert.That(ex1.Message, Contains.Substring("Available columns"), "Error should list available columns");

        // Test 2: Type mismatch in frame access
        var ex2 = Assert.Throws<ColumnTypeMismatchException>(() =>
        {
            var wrongTypeColumn = frame.GetColumn<double>("Name"); // Name is string, not double
        });
        Assert.That(ex2.Message, Contains.Substring("Name"), "Error should mention column name");
        Assert.That(ex2.Message, Contains.Substring("Double"), "Error should mention expected type");
        Assert.That(ex2.Message, Contains.Substring("String"), "Error should mention actual type");

        // Test 3: Index out of bounds
        var ex3 = Assert.Throws<IndexOutOfRangeException>(() =>
        {
            var value = ids[10]; // Only 5 elements
        });
        Assert.That(ex3.Message, Contains.Substring("10"), "Error should mention invalid index");
        Assert.That(ex3.Message, Contains.Substring("5"), "Error should mention actual length");

        // Test 4: Series label not found
        var series = new NivaraSeries<int>(ids, NivaraColumn<object>.Create(new object[] { "a", "b", "c", "d", "e" }));
        var ex4 = Assert.Throws<KeyNotFoundException>(() =>
        {
            var value = series.GetByLabel("z"); // Label doesn't exist
        });
        Assert.That(ex4.Message, Contains.Substring("z"), "Error should mention missing label");

        // Test 5: Arithmetic operation on incompatible types
        try
        {
            var result = names.Multiply("invalid"); // String multiplication should fail
            Assert.Fail("String multiplication should throw exception");
        }
        catch (InvalidOperationException ex5)
        {
            Assert.That(ex5.Message, Contains.Substring("String"), "Error should mention type");
        }

        // Clean up
        frame.Dispose();
        series.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void PerformanceCharacteristics_UnderDifferentWorkloads_BehaveConsistently()
    {
        // Arrange - Create columns of different sizes to test performance scaling
        var smallColumn = NivaraColumn<int>.Create(Enumerable.Range(1, 100).ToArray());
        var mediumColumn = NivaraColumn<int>.Create(Enumerable.Range(1, 1000).ToArray());
        var largeColumn = NivaraColumn<int>.Create(Enumerable.Range(1, 10000).ToArray());

        // Act - Measure performance characteristics
        var smallDiag = smallColumn.Diagnostics;
        var mediumDiag = mediumColumn.Diagnostics;
        var largeDiag = largeColumn.Diagnostics;

        // Assert - Verify performance characteristics scale appropriately

        // All should be vectorizable types
        Assert.That(smallDiag.IsVectorizable, Is.True, "All int columns should be vectorizable");
        Assert.That(mediumDiag.IsVectorizable, Is.True, "All int columns should be vectorizable");
        Assert.That(largeDiag.IsVectorizable, Is.True, "All int columns should be vectorizable");

        // Memory usage should scale with size
        Assert.That(mediumDiag.EstimatedMemoryUsage, Is.GreaterThan(smallDiag.EstimatedMemoryUsage),
            "Medium column should use more memory than small column");
        Assert.That(largeDiag.EstimatedMemoryUsage, Is.GreaterThan(mediumDiag.EstimatedMemoryUsage),
            "Large column should use more memory than medium column");

        // Kernel recommendations should depend on size and hardware
        if (Vector.IsHardwareAccelerated)
        {
            // Small column should use scalar due to overhead
            Assert.That(smallDiag.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Small columns should use scalar kernel due to overhead");

            // Large column should use vectorized
            Assert.That(largeDiag.RecommendedKernel, Is.EqualTo(KernelType.Vectorized),
                "Large columns should use vectorized kernel");
        }

        // Performance characteristics should reflect vectorization capability
        var smallPerf = smallDiag.Performance;
        var largePerf = largeDiag.Performance;

        Assert.That(smallPerf.SupportsVectorization, Is.True, "All vectorizable types support vectorization");
        Assert.That(largePerf.SupportsVectorization, Is.True, "All vectorizable types support vectorization");

        // Throughput multiplier should be higher for vectorized operations
        if (largeDiag.RecommendedKernel == KernelType.Vectorized)
        {
            Assert.That(largePerf.ThroughputMultiplier, Is.GreaterThan(1.0),
                "Vectorized operations should have throughput advantage");
        }

        // Test actual performance with operations
        var startTime = DateTime.Now;
        var largeResult = largeColumn * 2;
        var endTime = DateTime.Now;
        var duration = endTime - startTime;

        Assert.That(duration.TotalSeconds, Is.LessThan(5.0), "Large column operations should complete reasonably quickly");
        Assert.That(largeResult.Length, Is.EqualTo(10000), "Operations should preserve length");

        // Clean up
        smallColumn.Dispose();
        mediumColumn.Dispose();
        largeColumn.Dispose();
        largeResult.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void ResourceManagement_UnderMemoryPressure_HandlesGracefully()
    {
        // Arrange - Create multiple large columns to test memory management
        var columns = new List<NivaraColumn<double>>();
        var series = new List<NivaraSeries<double>>();
        var frames = new List<NivaraFrame>();

        try
        {
            // Create multiple large data structures
            for (int i = 0; i < 10; i++)
            {
                var data = Enumerable.Range(1, 10000).Select(x => (double)(x + i * 10000)).ToArray();
                var column = NivaraColumn<double>.Create(data);
                columns.Add(column);

                var seriesData = NivaraColumn<double>.Create(data);
                var seriesInstance = new NivaraSeries<double>(seriesData);
                series.Add(seriesInstance);

                var frameColumn = NivaraColumn<double>.Create(data);
                var frame = NivaraFrame.Create(($"Data{i}", frameColumn));
                frames.Add(frame);
            }

            // Act - Perform operations on all structures
            var totalMemoryUsage = columns.Sum(c => c.Diagnostics.EstimatedMemoryUsage);
            var operationResults = new List<NivaraColumn<double>>();

            foreach (var column in columns)
            {
                var result = column * 2.0;
                operationResults.Add(result);
            }

            // Assert - Verify all operations completed successfully
            Assert.That(columns.Count, Is.EqualTo(10), "All columns should be created");
            Assert.That(series.Count, Is.EqualTo(10), "All series should be created");
            Assert.That(frames.Count, Is.EqualTo(10), "All frames should be created");
            Assert.That(operationResults.Count, Is.EqualTo(10), "All operations should complete");

            Assert.That(totalMemoryUsage, Is.GreaterThan(0), "Memory usage should be tracked");
            Assert.That(totalMemoryUsage, Is.LessThan(100_000_000), "Memory usage should be reasonable"); // Less than 100MB

            // Verify operations produced correct results
            for (int i = 0; i < operationResults.Count; i++)
            {
                Assert.That(operationResults[i].Length, Is.EqualTo(10000), $"Result {i} should have correct length");
                Assert.That(operationResults[i][0], Is.EqualTo(columns[i][0] * 2.0), $"Result {i} should have correct values");
            }

            // Clean up operation results
            foreach (var result in operationResults)
            {
                result.Dispose();
            }
        }
        finally
        {
            // Clean up all resources
            foreach (var frame in frames)
            {
                frame.Dispose();
            }

            foreach (var seriesInstance in series)
            {
                seriesInstance.Dispose();
            }

            foreach (var column in columns)
            {
                column.Dispose();
            }
        }

        // Verify cleanup completed without exceptions
        Assert.Pass("Resource management under memory pressure completed successfully");
    }

    [Test]
    [Category("Integration")]
    public void ConcurrentOperations_OnSharedData_MaintainDataIntegrity()
    {
        // Arrange - Create shared data structures
        var sharedColumn = NivaraColumn<int>.Create(Enumerable.Range(1, 1000).ToArray());
        var results = new List<NivaraColumn<int>>();
        var exceptions = new List<Exception>();

        // Act - Perform concurrent operations (simulated with sequential operations for deterministic testing)
        var tasks = new List<Task>();

        for (int i = 0; i < 5; i++)
        {
            int multiplier = i + 1;
            var task = Task.Run(() =>
            {
                try
                {
                    // Each task performs different operations on the shared column
                    var result = sharedColumn * multiplier;
                    lock (results)
                    {
                        results.Add(result);
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions)
                    {
                        exceptions.Add(ex);
                    }
                }
            });
            tasks.Add(task);
        }

        // Wait for all tasks to complete
        Task.WaitAll(tasks.ToArray());

        // Assert - Verify concurrent operations completed successfully
        Assert.That(exceptions.Count, Is.EqualTo(0), "No exceptions should occur during concurrent operations");
        Assert.That(results.Count, Is.EqualTo(5), "All concurrent operations should complete");

        // Verify each result is correct
        for (int i = 0; i < results.Count; i++)
        {
            Assert.That(results[i].Length, Is.EqualTo(1000), $"Result {i} should have correct length");
            // Note: We can't guarantee order of results due to concurrency, so we just verify they're all valid
            Assert.That(results[i][0], Is.GreaterThan(0), $"Result {i} should have positive values");
        }

        // Verify original column is unchanged
        Assert.That(sharedColumn[0], Is.EqualTo(1), "Original column should be unchanged");
        Assert.That(sharedColumn.Length, Is.EqualTo(1000), "Original column length should be unchanged");

        // Clean up
        foreach (var result in results)
        {
            result.Dispose();
        }
        sharedColumn.Dispose();
    }
}