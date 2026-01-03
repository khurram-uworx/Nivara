using Nivara.Diagnostics;
using Nivara.Expressions;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Comprehensive integration test that demonstrates all Nivara components working together seamlessly.
/// This test validates the complete end-to-end functionality including columns, series, frames, 
/// queries, I/O operations, vectorization, and diagnostics.
/// </summary>
[TestFixture]
public class ComprehensiveIntegrationTest
{
    [Test]
    [Category("Integration")]
    public void CompleteWorkflow_AllComponents_WorkSeamlessly()
    {
        // === PHASE 1: Column Creation and Storage Selection ===

        // Create vectorizable columns (should use TensorStorage)
        var ids = NivaraColumn<int>.Create(Enumerable.Range(1, 1000).ToArray());
        var scores = NivaraColumn<double>.Create(Enumerable.Range(1, 1000).Select(i => i * 0.75 + 25).ToArray());
        var active = NivaraColumn<bool>.Create(Enumerable.Range(1, 1000).Select(i => i % 3 != 0).ToArray());

        // Create non-vectorizable columns (should use MemoryStorage)
        var names = NivaraColumn<string>.Create(Enumerable.Range(1, 1000).Select(i => $"User{i:D4}").ToArray());
        var timestamps = NivaraColumn<DateTime>.Create(Enumerable.Range(1, 1000).Select(i => DateTime.Now.AddDays(-i)).ToArray());

        // Verify automatic storage selection
        Assert.That(ids.Diagnostics.StorageType, Is.EqualTo(StorageType.Tensor), "Integer columns should use tensor storage");
        Assert.That(scores.Diagnostics.StorageType, Is.EqualTo(StorageType.Tensor), "Double columns should use tensor storage");
        Assert.That(active.Diagnostics.StorageType, Is.EqualTo(StorageType.Tensor), "Boolean columns should use tensor storage");
        Assert.That(names.Diagnostics.StorageType, Is.EqualTo(StorageType.Memory), "String columns should use memory storage");
        Assert.That(timestamps.Diagnostics.StorageType, Is.EqualTo(StorageType.Memory), "DateTime columns should use memory storage");

        // === PHASE 2: Vectorization and Performance Characteristics ===

        // Verify vectorization recommendations
        Assert.That(ids.Diagnostics.IsVectorizable, Is.True, "Integer columns should be vectorizable");
        Assert.That(scores.Diagnostics.IsVectorizable, Is.True, "Double columns should be vectorizable");
        Assert.That(names.Diagnostics.IsVectorizable, Is.False, "String columns should not be vectorizable");

        // Check performance characteristics
        var idsPerf = ids.Diagnostics.Performance;
        var namesPerf = names.Diagnostics.Performance;

        Assert.That(idsPerf.SupportsVectorization, Is.True, "Vectorizable columns should support vectorization");
        Assert.That(namesPerf.SupportsVectorization, Is.False, "Non-vectorizable columns should not support vectorization");
        Assert.That(idsPerf.ThroughputMultiplier, Is.GreaterThan(1.0), "Vectorizable columns should have throughput advantage");
        Assert.That(namesPerf.ThroughputMultiplier, Is.EqualTo(1.0), "Non-vectorizable columns should have baseline throughput");

        // === PHASE 3: Arithmetic Operations with Vectorization ===

        // Perform vectorized arithmetic operations
        var doubledScores = scores * 2.0;
        var scaledIds = ids * 2; // Scale integers by integer scalar
        var highScoreMask = scores.GreaterThan(75.0);

        // Verify operations completed successfully
        Assert.That(doubledScores.Length, Is.EqualTo(1000), "Arithmetic operations should preserve length");
        Assert.That(scaledIds.Length, Is.EqualTo(1000), "Scalar operations should preserve length");
        Assert.That(highScoreMask.Length, Is.EqualTo(1000), "Comparison operations should preserve length");

        // Verify some results
        Assert.That(doubledScores[0], Is.EqualTo(scores[0] * 2.0), "Scalar multiplication should work correctly");
        Assert.That(scaledIds[0], Is.EqualTo(ids[0] * 2), "Integer scalar multiplication should work correctly");
        Assert.That(highScoreMask[999], Is.EqualTo(scores[999] > 75.0), "Comparison operations should work correctly");

        // === PHASE 4: Series Creation and Label-based Access ===

        // Create series with custom labels
        var userLabels = NivaraColumn<object>.Create(names.ToArray().Cast<object>().ToArray());
        var scoreSeries = new NivaraSeries<double>(scores, userLabels);
        var idSeries = new NivaraSeries<int>(ids); // Default integer indexing

        // Verify series functionality
        Assert.That(scoreSeries.Length, Is.EqualTo(1000), "Series should preserve length");
        Assert.That(idSeries.Length, Is.EqualTo(1000), "Series should preserve length");

        // Test label-based access
        var firstUserScore = scoreSeries.GetByLabel("User0001");
        Assert.That(firstUserScore, Is.EqualTo(scores[0]), "Label-based access should work correctly");

        // Test position-based access
        Assert.That(idSeries[0], Is.EqualTo(1), "Position-based access should work correctly");

        // === PHASE 5: Frame Creation and Schema Management ===

        // Create comprehensive frame
        var frame = NivaraFrame.Create(
            ("ID", ids),
            ("Name", names),
            ("Score", scores),
            ("Active", active),
            ("Timestamp", timestamps)
        );

        // Verify frame properties
        Assert.That(frame.RowCount, Is.EqualTo(1000), "Frame should have correct row count");
        Assert.That(frame.ColumnCount, Is.EqualTo(5), "Frame should have correct column count");
        Assert.That(frame.ColumnNames, Contains.Item("ID"), "Frame should contain ID column");
        Assert.That(frame.ColumnNames, Contains.Item("Score"), "Frame should contain Score column");

        // Verify schema
        var schema = frame.Schema;
        Assert.That(schema.GetColumnType("ID"), Is.EqualTo(typeof(int)), "Schema should track correct types");
        Assert.That(schema.GetColumnType("Name"), Is.EqualTo(typeof(string)), "Schema should track correct types");
        Assert.That(schema.GetColumnType("Timestamp"), Is.EqualTo(typeof(DateTime)), "Schema should track correct types");

        // === PHASE 6: Query Engine Integration ===

        // Build complex query
        var queryResult = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Score") > 50.0)
            .Filter(ColumnExpressions.Col("Active") == true)
            .Select("ID", "Name", "Score")
            .Collect();

        // Verify query results
        Assert.That(queryResult.RowCount, Is.GreaterThan(0), "Query should return results");
        Assert.That(queryResult.ColumnCount, Is.EqualTo(3), "Query should return selected columns");

        // Verify all returned rows meet criteria
        var resultScores = queryResult.GetColumn<double>("Score");
        var resultIds = queryResult.GetColumn<int>("ID");

        for (int i = 0; i < queryResult.RowCount; i++)
        {
            Assert.That(resultScores[i], Is.GreaterThan(50.0), $"Row {i} should have score > 50");
            // Verify the corresponding active flag was true (by checking the original data)
            var originalIndex = resultIds[i] - 1; // IDs are 1-based
            Assert.That(active[originalIndex], Is.True, $"Row {i} should have been active");
        }

        // === PHASE 7: Series Alignment and Operations ===

        // Create two series with overlapping indices for alignment testing
        var series1Values = NivaraColumn<int>.Create(new[] { 10, 20, 30, 40 });
        var series1Index = NivaraColumn<object>.Create(new object[] { "A", "B", "C", "D" });
        var series1 = new NivaraSeries<int>(series1Values, series1Index);

        var series2Values = NivaraColumn<int>.Create(new[] { 100, 200, 300 });
        var series2Index = NivaraColumn<object>.Create(new object[] { "B", "C", "E" });
        var series2 = new NivaraSeries<int>(series2Values, series2Index);

        // Perform alignment
        var aligned = series1.AlignBoth(series2);

        // Verify alignment results
        Assert.That(aligned.Left.Length, Is.EqualTo(2), "Alignment should find 2 matching indices (B, C)");
        Assert.That(aligned.Right.Length, Is.EqualTo(2), "Alignment should find 2 matching indices (B, C)");

        // Verify aligned values
        Assert.That(aligned.Left.GetByLabel("B"), Is.EqualTo(20), "Aligned left series should have correct value for B");
        Assert.That(aligned.Right.GetByLabel("B"), Is.EqualTo(100), "Aligned right series should have correct value for B");
        Assert.That(aligned.Left.GetByLabel("C"), Is.EqualTo(30), "Aligned left series should have correct value for C");
        Assert.That(aligned.Right.GetByLabel("C"), Is.EqualTo(200), "Aligned right series should have correct value for C");

        // === PHASE 8: Memory Management and Resource Cleanup ===

        // Verify memory usage is reasonable
        var totalMemoryUsage = ids.Diagnostics.EstimatedMemoryUsage +
                              scores.Diagnostics.EstimatedMemoryUsage +
                              names.Diagnostics.EstimatedMemoryUsage +
                              active.Diagnostics.EstimatedMemoryUsage +
                              timestamps.Diagnostics.EstimatedMemoryUsage;

        Assert.That(totalMemoryUsage, Is.GreaterThan(0), "Memory usage should be tracked");
        Assert.That(totalMemoryUsage, Is.LessThan(10_000_000), "Memory usage should be reasonable for 1000 rows"); // Less than 10MB

        // === PHASE 9: Comprehensive Cleanup ===

        // Dispose all resources in proper order
        queryResult.Dispose();
        frame.Dispose();

        // Dispose series and their columns
        series1.Dispose();
        series2.Dispose();
        aligned.Left.Dispose();
        aligned.Right.Dispose();
        scoreSeries.Dispose();
        idSeries.Dispose();

        // Dispose arithmetic operation results
        doubledScores.Dispose();
        scaledIds.Dispose();
        highScoreMask.Dispose();

        // Dispose original columns
        ids.Dispose();
        scores.Dispose();
        active.Dispose();
        names.Dispose();
        timestamps.Dispose();

        // === PHASE 10: Verification of Complete Integration ===

        // This test successfully demonstrates:
        // ✓ Automatic storage selection (Tensor vs Memory)
        // ✓ Vectorization detection and performance characteristics
        // ✓ Arithmetic operations with proper type handling
        // ✓ Series creation with label-based and position-based access
        // ✓ Frame creation with schema management
        // ✓ Complex query execution with filtering and selection
        // ✓ Series alignment operations
        // ✓ Memory usage tracking and diagnostics
        // ✓ Proper resource management and disposal

        Assert.Pass("All Nivara components integrated successfully and work seamlessly together");
    }

    [Test]
    [Category("Integration")]
    public void VectorizationThreshold_BehavesCorrectly()
    {
        // Test vectorization threshold behavior with different sizes
        var vectorSize = Vector<int>.Count;

        // Small column (below threshold)
        var smallColumn = NivaraColumn<int>.Create(Enumerable.Range(1, vectorSize).ToArray());

        // Large column (above threshold)
        var largeColumn = NivaraColumn<int>.Create(Enumerable.Range(1, vectorSize * 20).ToArray());

        // Very large column (well above threshold)
        var veryLargeColumn = NivaraColumn<int>.Create(Enumerable.Range(1, vectorSize * 100).ToArray());

        // Verify threshold behavior
        var smallDiag = smallColumn.Diagnostics;
        var largeDiag = largeColumn.Diagnostics;
        var veryLargeDiag = veryLargeColumn.Diagnostics;

        // All should be vectorizable types
        Assert.That(smallDiag.IsVectorizable, Is.True, "All int columns should be vectorizable types");
        Assert.That(largeDiag.IsVectorizable, Is.True, "All int columns should be vectorizable types");
        Assert.That(veryLargeDiag.IsVectorizable, Is.True, "All int columns should be vectorizable types");

        // Kernel recommendations should depend on size and hardware
        if (Vector.IsHardwareAccelerated)
        {
            // Small column should use scalar due to overhead
            Assert.That(smallDiag.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Small columns should use scalar kernel due to vectorization overhead");

            // Large columns should use vectorized if above threshold
            if (largeDiag.Length >= vectorSize * 4)
            {
                Assert.That(largeDiag.RecommendedKernel, Is.EqualTo(KernelType.Vectorized),
                    "Large columns above threshold should use vectorized kernel");
            }

            // Very large columns should definitely use vectorized
            Assert.That(veryLargeDiag.RecommendedKernel, Is.EqualTo(KernelType.Vectorized),
                "Very large columns should use vectorized kernel");
        }
        else
        {
            // Without hardware acceleration, all should use scalar
            Assert.That(smallDiag.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Without hardware acceleration, should use scalar kernel");
            Assert.That(largeDiag.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Without hardware acceleration, should use scalar kernel");
            Assert.That(veryLargeDiag.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Without hardware acceleration, should use scalar kernel");
        }

        // Clean up
        smallColumn.Dispose();
        largeColumn.Dispose();
        veryLargeColumn.Dispose();
    }
}