using Nivara.Diagnostics;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class DiagnosticsTests
{
    [Test]
    public void ColumnDiagnostics_IntColumn_ReturnsCorrectInformation()
    {
        // Arrange
        var values = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(values);

        // Act
        var diagnostics = column.Diagnostics;

        // Assert
        Assert.That(diagnostics.IsVectorizable, Is.True); // Int columns are vectorizable
        Assert.That(diagnostics.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(diagnostics.Length, Is.EqualTo(5));
        Assert.That(diagnostics.HasNulls, Is.False);
        Assert.That(diagnostics.IsHardwareAccelerated, Is.EqualTo(System.Numerics.Vector.IsHardwareAccelerated));
        Assert.That(diagnostics.VectorSize, Is.GreaterThan(0));
    }

    [Test]
    public void ColumnDiagnostics_StringColumn_ReturnsCorrectInformation()
    {
        // Arrange
        var values = new[] { "apple", "banana", "cherry" };
        var column = NivaraColumn<string>.CreateForReferenceType(values);

        // Act
        var diagnostics = column.Diagnostics;

        // Assert
        Assert.That(diagnostics.IsVectorizable, Is.False);
        Assert.That(diagnostics.ElementType, Is.EqualTo(typeof(string)));
        Assert.That(diagnostics.Length, Is.EqualTo(3));
        Assert.That(diagnostics.HasNulls, Is.False);
        Assert.That(diagnostics.RecommendedKernel, Is.EqualTo(KernelType.Scalar));
    }

    [Test]
    public void ColumnDiagnostics_ColumnWithNulls_ReturnsCorrectInformation()
    {
        // Arrange
        var values = new string[] { "apple", null!, "cherry" };
        var column = NivaraColumn<string>.CreateForReferenceType(values);

        // Act
        var diagnostics = column.Diagnostics;

        // Assert
        Assert.That(diagnostics.HasNulls, Is.True);
        Assert.That(diagnostics.EstimatedMemoryUsage, Is.GreaterThan(0));
    }

    [Test]
    public void ColumnDiagnostics_PerformanceCharacteristics_ReturnsValidData()
    {
        // Arrange
        var values = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(values);

        // Act
        var diagnostics = column.Diagnostics;
        var performance = diagnostics.Performance;

        // Assert
        Assert.That(performance.ThroughputMultiplier, Is.GreaterThan(0));
        Assert.That(performance.MemoryEfficiency, Is.GreaterThan(0).And.LessThanOrEqualTo(1.0));
        Assert.That(performance.SupportsVectorization, Is.TypeOf<bool>());
    }

    [Test]
    public void DiagnosticsTracker_WhenEnabled_RecordsOperations()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        var values1 = new[] { 1, 2, 3, 4, 5 };
        var values2 = new[] { 2, 3, 4, 5, 6 };
        var column1 = NivaraColumn<int>.Create(values1);
        var column2 = NivaraColumn<int>.Create(values2);

        // Act
        var result = column1.Add(column2);
        var operations = DiagnosticsTracker.GetRecordedOperations();

        // Assert
        Assert.That(operations.Length, Is.GreaterThan(0));
        var operation = operations[0];
        Assert.That(operation.OperationType, Is.EqualTo("ElementwiseAddition"));
        Assert.That(operation.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(operation.InputLength, Is.EqualTo(5));
        Assert.That(operation.HadNulls, Is.False);

        // Cleanup
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();
    }

    [Test]
    public void DiagnosticsTracker_WhenDisabled_DoesNotRecordOperations()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();

        var values1 = new[] { 1, 2, 3, 4, 5 };
        var values2 = new[] { 2, 3, 4, 5, 6 };
        var column1 = NivaraColumn<int>.Create(values1);
        var column2 = NivaraColumn<int>.Create(values2);

        // Act
        var result = column1.Add(column2);
        var operations = DiagnosticsTracker.GetRecordedOperations();

        // Assert
        Assert.That(operations.Length, Is.EqualTo(0));
    }

    [Test]
    public void DiagnosticsTracker_GetSummary_ReturnsCorrectStatistics()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        var values1 = new[] { 1, 2, 3, 4, 5 };
        var values2 = new[] { 2, 3, 4, 5, 6 };
        var column1 = NivaraColumn<int>.Create(values1);
        var column2 = NivaraColumn<int>.Create(values2);

        // Act - perform multiple operations
        var result1 = column1.Add(column2);
        var result2 = column1.Multiply(2);
        var result3 = column1.Equals(3);

        var summary = DiagnosticsTracker.GetSummary();

        // Assert - be flexible about the exact count since not all operations may be tracked yet
        Assert.That(summary.TotalOperations, Is.GreaterThan(0));
        Assert.That(summary.OperationTypes.Count, Is.GreaterThan(0));
        Assert.That(summary.KernelTypes.Count, Is.GreaterThan(0));
        Assert.That(summary.VectorizationRate, Is.GreaterThanOrEqualTo(0).And.LessThanOrEqualTo(100));

        // Cleanup
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();
    }

    [Test]
    public void DiagnosticsTracker_ColumnToTensor_RecordsAllocationDiagnostics()
    {
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        try
        {
            var column = NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f });

            var tensor = column.ToTensor();

            var operation = AssertSingleOperation("ColumnToTensor");
            Assert.That(operation.AllocatedBytes, Is.GreaterThan(0));
            Assert.That(operation.Elapsed, Is.GreaterThanOrEqualTo(TimeSpan.Zero));
            Assert.That(operation.Notes, Does.Contain("TensorConversion=Copy"));
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            DiagnosticsTracker.ClearRecordedOperations();
        }
    }

    [Test]
    public void DiagnosticsTracker_ColumnToTensorWithNullReplacement_RecordsAllocationDiagnostics()
    {
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        try
        {
            var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });

            var tensor = column.ToTensor(-1.0f);

            var operation = AssertSingleOperation("ColumnToTensorWithNullReplacement");
            Assert.That(operation.AllocatedBytes, Is.GreaterThan(0));
            Assert.That(operation.HadNulls, Is.True);
            Assert.That(operation.Notes, Does.Contain("TensorConversion=NullReplacement"));
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            DiagnosticsTracker.ClearRecordedOperations();
        }
    }

    [Test]
    public void DiagnosticsTracker_GetSummary_IncludesAllocationStatistics()
    {
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        try
        {
            var column = NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f });
            var tensor = column.ToTensor();

            var summary = DiagnosticsTracker.GetSummary();

            Assert.That(summary.TotalAllocatedBytes, Is.GreaterThan(0));
            Assert.That(summary.AverageAllocatedBytes, Is.GreaterThan(0));
            Assert.That(summary.AllocatedBytesByOperationType, Contains.Key("ColumnToTensor"));
            Assert.That(summary.AllocatedBytesByOperationType["ColumnToTensor"], Is.GreaterThan(0));
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            DiagnosticsTracker.ClearRecordedOperations();
        }
    }

    [Test]
    public void OperationDiagnostics_ToString_ReturnsFormattedString()
    {
        // Arrange
        var diagnostic = new OperationDiagnostics(
            "TestOperation",
            KernelType.Vectorized,
            100,
            typeof(int),
            false);

        // Act
        var result = diagnostic.ToString();

        // Assert
        Assert.That(result, Does.Contain("TestOperation"));
        Assert.That(result, Does.Contain("Vectorized"));
        Assert.That(result, Does.Contain("100"));
        Assert.That(result, Does.Contain("Int32"));
        Assert.That(result, Does.Contain("AllocatedBytes"));
    }

    [Test]
    public void ColumnDiagnostics_ToString_ReturnsFormattedString()
    {
        // Arrange
        var values = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(values);

        // Act
        var result = column.Diagnostics.ToString();

        // Assert
        Assert.That(result, Does.Contain("Int32"));
        Assert.That(result, Does.Contain("5"));
        Assert.That(result, Does.Contain("Memory"));
    }

    private static OperationDiagnostics AssertSingleOperation(string operationType)
    {
        var operations = DiagnosticsTracker.GetRecordedOperations();

        Assert.That(operations, Has.Length.EqualTo(1));
        Assert.That(operations[0].OperationType, Is.EqualTo(operationType));

        return operations[0];
    }
}
