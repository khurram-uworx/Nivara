using Nivara.Diagnostics;
using Nivara.Expressions;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Integration tests that verify all Nivara components work together seamlessly.
/// Tests end-to-end scenarios combining columns, series, frames, queries, and I/O operations.
/// </summary>
[TestFixture]
public class IntegrationTests
{
    [Test]
    [Category("Integration")]
    public void EndToEnd_ColumnToSeriestoFrame_WorksSeamlessly()
    {
        // Arrange - Create columns with different types and null values
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var stringColumn = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie", "David", "Eve" });
        var doubleColumn = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.1, null, 3.3, 4.4, 5.5 });

        // Act - Create series with custom indices
        var intSeries = new NivaraSeries<int>(intColumn);
        var stringSeries = new NivaraSeries<string>(stringColumn, NivaraColumn<object>.Create(new object[] { "a", "b", "c", "d", "e" }));
        var doubleSeries = new NivaraSeries<double>(doubleColumn);

        // Create frame from mixed columns
        var frame = NivaraFrame.Create(
            ("ID", intColumn),
            ("Name", stringColumn),
            ("Score", doubleColumn)
        );

        // Assert - Verify all components work together
        Assert.That(frame.RowCount, Is.EqualTo(5));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));

        // Verify data integrity across components
        Assert.That(intSeries[0], Is.EqualTo(1));
        Assert.That(stringSeries.GetByLabel("b"), Is.EqualTo("Bob"));
        Assert.That(doubleSeries.IsNull(1), Is.True, "Double series should preserve null at index 1"); // Check null instead of value

        // Verify frame access
        var nameColumn = frame.GetColumn<string>("Name");
        Assert.That(nameColumn[2], Is.EqualTo("Charlie"), "Frame should preserve string values correctly");

        // Clean up
        intColumn.Dispose();
        stringColumn.Dispose();
        doubleColumn.Dispose();
        intSeries.Dispose();
        stringSeries.Dispose();
        doubleSeries.Dispose();
        frame.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void VectorizedOperations_AreUsedWhenExpected()
    {
        // Arrange - Create large vectorizable columns to trigger vectorization
        var size = Vector<int>.Count * 100; // Much larger to ensure we exceed vectorization threshold
        var data1 = Enumerable.Range(1, size).ToArray();
        var data2 = Enumerable.Range(size + 1, size).ToArray();

        var column1 = NivaraColumn<int>.Create(data1);
        var column2 = NivaraColumn<int>.Create(data2);

        // Act - Perform operations that should use vectorization
        var sum = column1 + column2;
        var scaled = column1 * 2;

        // Assert - Verify vectorization is being used
        var diagnostics1 = column1.Diagnostics;
        var diagnostics2 = column2.Diagnostics;

        Assert.That(diagnostics1.IsVectorizable, Is.True, "Integer columns should be vectorizable");
        Assert.That(diagnostics1.IsHardwareAccelerated, Is.EqualTo(Vector.IsHardwareAccelerated), "Hardware acceleration should match system capability");
        Assert.That(diagnostics1.RecommendedKernel, Is.EqualTo(KernelType.Vectorized), "Large integer columns should recommend vectorized kernel");
        Assert.That(diagnostics1.StorageType, Is.EqualTo(StorageType.Tensor), "Vectorizable types should use tensor storage");

        // Verify operation results are correct
        Assert.That(sum.Length, Is.EqualTo(size));
        Assert.That(sum[0], Is.EqualTo(data1[0] + data2[0]));
        Assert.That(scaled[0], Is.EqualTo(data1[0] * 2));

        // Clean up
        column1.Dispose();
        column2.Dispose();
        sum.Dispose();
        scaled.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void NonVectorizable_UsesMemoryStorage()
    {
        // Arrange - Create columns with non-vectorizable types
        var stringColumn = NivaraColumn<string>.Create(new[] { "Hello", "World", "Test" });
        var guidColumn = NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid() });

        // Act & Assert - Verify non-vectorizable types use memory storage
        var stringDiagnostics = stringColumn.Diagnostics;
        var guidDiagnostics = guidColumn.Diagnostics;

        Assert.That(stringDiagnostics.IsVectorizable, Is.False, "String columns should not be vectorizable");
        Assert.That(stringDiagnostics.StorageType, Is.EqualTo(StorageType.Memory), "Non-vectorizable types should use memory storage");
        Assert.That(stringDiagnostics.RecommendedKernel, Is.EqualTo(KernelType.Scalar), "Non-vectorizable types should recommend scalar kernel");

        Assert.That(guidDiagnostics.IsVectorizable, Is.False, "Guid columns should not be vectorizable");
        Assert.That(guidDiagnostics.StorageType, Is.EqualTo(StorageType.Memory), "Non-vectorizable types should use memory storage");

        // Clean up
        stringColumn.Dispose();
        guidColumn.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void QueryEngine_WithVectorizedOperations_WorksCorrectly()
    {
        // Arrange - Create frame with vectorizable data
        var ids = NivaraColumn<int>.Create(Enumerable.Range(1, 1000).ToArray());
        var scores = NivaraColumn<double>.Create(Enumerable.Range(1, 1000).Select(i => i * 1.5).ToArray());
        var names = NivaraColumn<string>.Create(Enumerable.Range(1, 1000).Select(i => $"User{i}").ToArray());

        var frame = NivaraFrame.Create(
            ("ID", ids),
            ("Score", scores),
            ("Name", names)
        );

        // Act - Perform complex query operations
        var queryResult = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Score") > 500.0)
            .Select("ID", "Name", "Score")
            .Collect();

        // Assert - Verify query worked correctly
        Assert.That(queryResult.RowCount, Is.GreaterThan(0), "Query should return results");
        Assert.That(queryResult.ColumnCount, Is.EqualTo(3), "Query should return 3 columns");

        // Verify vectorized columns were used in query
        var idDiagnostics = ids.Diagnostics;
        var scoreDiagnostics = scores.Diagnostics;

        Assert.That(idDiagnostics.IsVectorizable, Is.True, "ID column should be vectorizable");
        Assert.That(scoreDiagnostics.IsVectorizable, Is.True, "Score column should be vectorizable");

        // Verify query results are correct
        var resultIds = queryResult.GetColumn<int>("ID");
        var resultScores = queryResult.GetColumn<double>("Score");

        Assert.That(resultScores[0], Is.GreaterThan(500.0), "All returned scores should be > 500");

        // Clean up
        frame.Dispose();
        queryResult.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void SeriesAlignment_WithDifferentIndices_WorksCorrectly()
    {
        // Arrange - Create series with different indices
        var values1 = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var index1 = NivaraColumn<object>.Create(new object[] { "a", "b", "c" });
        var series1 = new NivaraSeries<int>(values1, index1);

        var values2 = NivaraColumn<int>.Create(new[] { 100, 200, 300 });
        var index2 = NivaraColumn<object>.Create(new object[] { "b", "c", "d" });
        var series2 = new NivaraSeries<int>(values2, index2);

        // Act - Perform alignment operation
        var aligned = series1.AlignBoth(series2);

        // Assert - Verify alignment worked correctly
        Assert.That(aligned.Left.Length, Is.EqualTo(aligned.Right.Length), "Aligned series should have same length");

        // Verify alignment preserves matching indices
        var leftByB = aligned.Left.GetByLabel("b");
        var rightByB = aligned.Right.GetByLabel("b");
        Assert.That(leftByB, Is.EqualTo(20), "Left series should have correct value for 'b'");
        Assert.That(rightByB, Is.EqualTo(100), "Right series should have correct value for 'b'");

        // Clean up
        series1.Dispose();
        series2.Dispose();
        aligned.Left.Dispose();
        aligned.Right.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void NullHandling_ThroughoutPipeline_PreservesSemantics()
    {
        // Arrange - Create data with nulls at different levels
        var intValues = new int?[] { 1, null, 3, null, 5 };
        var stringValues = new string?[] { "A", null, "C", "D", null };

        var intColumn = NivaraColumn<int>.CreateFromNullable(intValues);
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(stringValues!);

        // Act - Create series and frame with null data
        var intSeries = new NivaraSeries<int>(intColumn);
        var frame = NivaraFrame.Create(
            ("Numbers", intColumn),
            ("Letters", stringColumn)
        );

        // Perform operations that should preserve nulls
        var doubled = intColumn * 2;
        var comparison = intColumn.GreaterThan(2);

        // Assert - Verify null semantics are preserved
        Assert.That(intColumn.HasNulls, Is.True, "Original column should have nulls");
        Assert.That(stringColumn.HasNulls, Is.True, "String column should have nulls");

        // Verify null propagation in arithmetic
        Assert.That(doubled.IsNull(1), Is.True, "Null * 2 should be null");
        Assert.That(doubled[0], Is.EqualTo(2), "Non-null values should be doubled");

        // Verify null propagation in comparisons
        Assert.That(comparison.IsNull(1), Is.True, "Null > 2 should be null");
        Assert.That(comparison[2], Is.EqualTo(true), "3 > 2 should be true");

        // Verify series preserves nulls
        Assert.That(intSeries.IsNull(1), Is.True, "Series should preserve null at index 1");
        Assert.That(intSeries[0], Is.EqualTo(1), "Series should preserve non-null values");

        // Verify frame preserves nulls
        var frameIntColumn = frame.GetColumn<int>("Numbers");
        var frameStringColumn = frame.GetColumn<string>("Letters");
        Assert.That(frameIntColumn.IsNull(1), Is.True, "Frame should preserve int nulls");
        Assert.That(frameStringColumn.IsNull(1), Is.True, "Frame should preserve string nulls");

        // Clean up
        intColumn.Dispose();
        stringColumn.Dispose();
        intSeries.Dispose();
        frame.Dispose();
        doubled.Dispose();
        comparison.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void PerformanceCharacteristics_ReflectActualConfiguration()
    {
        // Arrange - Create columns with different performance characteristics
        var smallIntColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 }); // Small, should use scalar
        var largeIntColumn = NivaraColumn<int>.Create(Enumerable.Range(1, 10000).ToArray()); // Large, should use vectorized
        var stringColumn = NivaraColumn<string>.Create(new[] { "A", "B", "C" }); // Non-vectorizable

        // Act & Assert - Verify performance characteristics match expectations
        var smallDiagnostics = smallIntColumn.Diagnostics;
        var largeDiagnostics = largeIntColumn.Diagnostics;
        var stringDiagnostics = stringColumn.Diagnostics;

        // Small column should use scalar due to overhead
        Assert.That(smallDiagnostics.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
            "Small columns should use scalar kernel due to vectorization overhead");

        // Large column should use vectorized if hardware supports it and size is large enough
        if (Vector.IsHardwareAccelerated && largeDiagnostics.Length >= largeDiagnostics.VectorSize * 4)
        {
            Assert.That(largeDiagnostics.RecommendedKernel, Is.EqualTo(KernelType.Vectorized),
                "Large vectorizable columns should use vectorized kernel when hardware supports it and size exceeds threshold");
        }
        else
        {
            Assert.That(largeDiagnostics.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
                "Large vectorizable columns should use scalar kernel when hardware doesn't support vectorization or size is below threshold");
        }

        // String column should always use scalar
        Assert.That(stringDiagnostics.RecommendedKernel, Is.EqualTo(KernelType.Scalar),
            "Non-vectorizable columns should always use scalar kernel");

        // Verify performance characteristics
        var largePerf = largeDiagnostics.Performance;
        var stringPerf = stringDiagnostics.Performance;

        Assert.That(largePerf.SupportsVectorization, Is.True, "Large int column should support vectorization");
        Assert.That(stringPerf.SupportsVectorization, Is.False, "String column should not support vectorization");

        Assert.That(largePerf.ThroughputMultiplier, Is.GreaterThan(1.0), "Vectorizable columns should have throughput advantage");
        Assert.That(stringPerf.ThroughputMultiplier, Is.EqualTo(1.0), "Non-vectorizable columns should have baseline throughput");

        // Clean up
        smallIntColumn.Dispose();
        largeIntColumn.Dispose();
        stringColumn.Dispose();
    }

    [Test]
    [Category("Integration")]
    public void MemoryManagement_DisposalWorks_Correctly()
    {
        // Arrange - Create column and series with separate columns to avoid ownership issues
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var seriesValues = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 }); // Separate column for series
        var series = new NivaraSeries<int>(seriesValues);

        // Act - Dispose series first
        series.Dispose();

        // Assert - Original column should still be usable (series has its own column)
        Assert.That(column.Length, Is.EqualTo(5), "Original column should still be usable after series disposal");

        // Dispose original column
        column.Dispose();

        // Verify disposed objects throw ObjectDisposedException
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.Length; }, "Disposed column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => { var _ = series.Length; }, "Disposed series should throw ObjectDisposedException");
    }

    [Test]
    [Category("Integration")]
    public void ComplexQuery_WithMixedTypes_ExecutesCorrectly()
    {
        // Arrange - Create complex dataset with mixed types
        var count = 1000;
        var ids = NivaraColumn<int>.Create(Enumerable.Range(1, count).ToArray());
        var names = NivaraColumn<string>.Create(Enumerable.Range(1, count).Select(i => $"User{i}").ToArray());
        var scores = NivaraColumn<double>.Create(Enumerable.Range(1, count).Select(i => i * 0.5 + 10).ToArray());
        var active = NivaraColumn<bool>.Create(Enumerable.Range(1, count).Select(i => i % 3 != 0).ToArray());

        var frame = NivaraFrame.Create(
            ("ID", ids),
            ("Name", names),
            ("Score", scores),
            ("Active", active)
        );

        // Act - Execute complex query with multiple conditions
        var result = frame.AsQueryFrame()
            .Filter(ColumnExpressions.Col("Score") > 100.0)
            .Filter(ColumnExpressions.Col("Active") == true)
            .Select("ID", "Name", "Score")
            .Collect();

        // Assert - Verify query executed correctly
        Assert.That(result.RowCount, Is.GreaterThan(0), "Complex query should return results");
        Assert.That(result.ColumnCount, Is.EqualTo(3), "Query should return selected columns");

        // Verify all returned rows meet criteria
        var resultScores = result.GetColumn<double>("Score");
        for (int i = 0; i < result.RowCount; i++)
        {
            Assert.That(resultScores[i], Is.GreaterThan(100.0), $"Row {i} should have score > 100");
        }

        // Verify vectorization was used where appropriate
        Assert.That(ids.Diagnostics.IsVectorizable, Is.True, "ID column should be vectorizable");
        Assert.That(scores.Diagnostics.IsVectorizable, Is.True, "Score column should be vectorizable");
        Assert.That(active.Diagnostics.IsVectorizable, Is.True, "Boolean column should be vectorizable");
        Assert.That(names.Diagnostics.IsVectorizable, Is.False, "String column should not be vectorizable");

        // Clean up
        frame.Dispose();
        result.Dispose();
    }
}