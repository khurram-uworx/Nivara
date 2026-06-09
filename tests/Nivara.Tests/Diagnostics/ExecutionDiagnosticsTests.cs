using Nivara.Diagnostics;
using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Diagnostics;

[TestFixture]
public class ExecutionDiagnosticsTests
{
    [Test]
    public void ExecutionDiagnostics_RecordOperationTiming_TracksCorrectly()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        var duration = TimeSpan.FromMilliseconds(100);
        const long rowsProcessed = 1000;
        const long memoryUsed = 1024 * 1024; // 1MB

        // Act
        diagnostics.RecordOperationTiming("Filter", duration, rowsProcessed, memoryUsed);

        // Assert
        var timings = diagnostics.OperationTimings;
        Assert.That(timings.Count, Is.EqualTo(1));

        var timing = timings[0];
        Assert.That(timing.OperationType, Is.EqualTo("Filter"));
        Assert.That(timing.Duration, Is.EqualTo(duration));
        Assert.That(timing.RowsProcessed, Is.EqualTo(rowsProcessed));
        Assert.That(timing.MemoryUsed, Is.EqualTo(memoryUsed));
        Assert.That(timing.Throughput, Is.EqualTo(10000).Within(0.1)); // 1000 rows / 0.1 seconds
    }

    [Test]
    public void ExecutionDiagnostics_RecordWarning_StoresWarningCorrectly()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        var warning = new PerformanceWarning(
            PerformanceWarningSeverity.Warning,
            "Slow operation detected",
            "Consider optimization");

        // Act
        diagnostics.RecordWarning(warning);

        // Assert
        var warnings = diagnostics.Warnings;
        Assert.That(warnings.Count, Is.EqualTo(1));
        Assert.That(warnings[0], Is.EqualTo(warning));
    }

    [Test]
    public void ExecutionDiagnostics_RecordOptimization_StoresOptimizationCorrectly()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        var optimization = new OptimizationApplied(
            "Predicate Pushdown",
            "Moved filter operation before sort",
            25.5);

        // Act
        diagnostics.RecordOptimization(optimization);

        // Assert
        var optimizations = diagnostics.OptimizationsApplied;
        Assert.That(optimizations.Count, Is.EqualTo(1));
        Assert.That(optimizations[0], Is.EqualTo(optimization));
    }

    [Test]
    public void ExecutionDiagnostics_GenerateReport_IncludesAllSections()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        diagnostics.StartExecution();

        // Add some sample data
        diagnostics.RecordOperationTiming("Filter", TimeSpan.FromMilliseconds(50), 1000, 1024);
        diagnostics.RecordOperationTiming("Sort", TimeSpan.FromMilliseconds(150), 1000, 2048);

        diagnostics.RecordWarning(new PerformanceWarning(
            PerformanceWarningSeverity.Info,
            "Test warning",
            "Test suggestion"));

        diagnostics.RecordOptimization(new OptimizationApplied(
            "Test Optimization",
            "Test description",
            10.0));

        Thread.Sleep(10); // Small delay to ensure measurable execution time
        diagnostics.EndExecution();

        // Act
        var report = diagnostics.GenerateReport();

        // Assert
        Assert.That(report, Does.Contain("Execution Diagnostics Report"));
        Assert.That(report, Does.Contain("Operation Breakdown:"));
        Assert.That(report, Does.Contain("Optimizations Applied:"));
        Assert.That(report, Does.Contain("Performance Warnings:"));
        Assert.That(report, Does.Contain("Performance Analysis:"));
        Assert.That(report, Does.Contain("Filter"));
        Assert.That(report, Does.Contain("Sort"));
    }

    [Test]
    public void ExecutionDiagnostics_GetSummary_ProvidesAccurateMetrics()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        diagnostics.StartExecution();
        diagnostics.ParallelismDegree = 4;
        diagnostics.ExecutionStrategy = ExecutionStrategy.Parallel;

        diagnostics.RecordOperationTiming("Filter", TimeSpan.FromMilliseconds(100), 500, 1024);
        diagnostics.RecordOperationTiming("Sort", TimeSpan.FromMilliseconds(200), 500, 2048);
        diagnostics.RecordWarning(new PerformanceWarning(PerformanceWarningSeverity.Info, "Test", null));
        diagnostics.RecordOptimization(new OptimizationApplied("Test", "Test", null));

        Thread.Sleep(10);
        diagnostics.EndExecution();

        // Act
        var summary = diagnostics.GetSummary();

        // Assert
        Assert.That(summary.TotalRowsProcessed, Is.EqualTo(1000));
        Assert.That(summary.OperationCount, Is.EqualTo(2));
        Assert.That(summary.WarningCount, Is.EqualTo(1));
        Assert.That(summary.OptimizationCount, Is.EqualTo(1));
        Assert.That(summary.ExecutionStrategy, Is.EqualTo(ExecutionStrategy.Parallel));
        Assert.That(summary.ParallelismDegree, Is.EqualTo(4));
        Assert.That(summary.AverageThroughput, Is.GreaterThan(0));
    }

    [Test]
    public void DiagnosticHelper_ExecuteWithDiagnostics_RecordsSuccessfulOperation()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        var operationExecuted = false;

        // Act
        var result = DiagnosticHelper.ExecuteWithDiagnostics(
            diagnostics,
            "TestOperation",
            () =>
            {
                operationExecuted = true;
                Thread.Sleep(10); // Simulate some work
                return "success";
            },
            100);

        // Assert
        Assert.That(result, Is.EqualTo("success"));
        Assert.That(operationExecuted, Is.True);
        Assert.That(diagnostics.OperationTimings.Count, Is.EqualTo(1));

        var timing = diagnostics.OperationTimings[0];
        Assert.That(timing.OperationType, Is.EqualTo("TestOperation"));
        Assert.That(timing.RowsProcessed, Is.EqualTo(100));
        Assert.That(timing.Duration, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void DiagnosticHelper_ExecuteWithDiagnostics_RecordsFailedOperation()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();
        var testException = new InvalidOperationException("Test error");

        // Act & Assert
        var exception = Assert.Throws<InvalidOperationException>(() =>
            DiagnosticHelper.ExecuteWithDiagnostics<object>(
                diagnostics,
                "FailingOperation",
                () => throw testException,
                50));

        Assert.That(exception, Is.EqualTo(testException));
        Assert.That(diagnostics.OperationTimings.Count, Is.EqualTo(1));
        Assert.That(diagnostics.Warnings.Count, Is.EqualTo(1));

        var timing = diagnostics.OperationTimings[0];
        Assert.That(timing.OperationType, Is.EqualTo("FailingOperation (Failed)"));

        var warning = diagnostics.Warnings[0];
        Assert.That(warning.Severity, Is.EqualTo(PerformanceWarningSeverity.Critical));
        Assert.That(warning.Message, Does.Contain("FailingOperation failed"));
    }

    [Test]
    public void DiagnosticScope_DisposalRecordsTiming()
    {
        // Arrange
        var diagnostics = new ExecutionDiagnostics();

        // Act
        using (var scope = DiagnosticHelper.CreateScope(diagnostics, "ScopedOperation"))
        {
            scope.SetRowCount(200);
            Thread.Sleep(10); // Simulate work
        } // Disposal should record timing

        // Assert
        Assert.That(diagnostics.OperationTimings.Count, Is.EqualTo(1));

        var timing = diagnostics.OperationTimings[0];
        Assert.That(timing.OperationType, Is.EqualTo("ScopedOperation"));
        Assert.That(timing.RowsProcessed, Is.EqualTo(200));
        Assert.That(timing.Duration, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void PerformanceWarning_CreatesWithCorrectProperties()
    {
        // Arrange
        var severity = PerformanceWarningSeverity.Warning;
        var message = "Test warning message";
        var suggestion = "Test suggestion";

        // Act
        var warning = new PerformanceWarning(severity, message, suggestion);

        // Assert
        Assert.That(warning.Severity, Is.EqualTo(severity));
        Assert.That(warning.Message, Is.EqualTo(message));
        Assert.That(warning.Suggestion, Is.EqualTo(suggestion));
        Assert.That(warning.Timestamp, Is.LessThanOrEqualTo(DateTime.UtcNow));
    }

    [Test]
    public void OptimizationApplied_CreatesWithCorrectProperties()
    {
        // Arrange
        var name = "Test Optimization";
        var description = "Test description";
        var improvement = 15.5;

        // Act
        var optimization = new OptimizationApplied(name, description, improvement);

        // Assert
        Assert.That(optimization.OptimizationName, Is.EqualTo(name));
        Assert.That(optimization.Description, Is.EqualTo(description));
        Assert.That(optimization.EstimatedImprovement, Is.EqualTo(improvement));
        Assert.That(optimization.Timestamp, Is.LessThanOrEqualTo(DateTime.UtcNow));
    }

    [Test]
    public void ImportFromDiagnosticsTracker_ImportsAndClears()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        try
        {
            var values1 = new[] { 1, 2, 3 };
            var values2 = new[] { 2, 3, 4 };
            var column1 = NivaraColumn<int>.Create(values1);
            var column2 = NivaraColumn<int>.Create(values2);
            column1.Add(column2); // This triggers DiagnosticsTracker recording

            var diagnostics = new ExecutionDiagnostics();

            // Act
            diagnostics.ImportFromDiagnosticsTracker();

            // Assert
            Assert.That(diagnostics.KernelOperations.Count, Is.GreaterThan(0));
            Assert.That(DiagnosticsTracker.GetRecordedOperations().Length, Is.EqualTo(0));
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            DiagnosticsTracker.ClearRecordedOperations();
        }
    }

    [Test]
    public void ImportFromDiagnosticsTracker_WhenDisabled_ImportsNothing()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();

        var diagnostics = new ExecutionDiagnostics();

        // Act
        diagnostics.ImportFromDiagnosticsTracker();

        // Assert
        Assert.That(diagnostics.KernelOperations.Count, Is.EqualTo(0));
    }

    [Test]
    public void GenerateReport_IncludesKernelOperations_WhenPresent()
    {
        // Arrange
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        try
        {
            var values1 = new[] { 1, 2, 3 };
            var values2 = new[] { 2, 3, 4 };
            var column1 = NivaraColumn<int>.Create(values1);
            var column2 = NivaraColumn<int>.Create(values2);
            column1.Add(column2);

            var diagnostics = new ExecutionDiagnostics();
            diagnostics.ImportFromDiagnosticsTracker();

            // Act
            var report = diagnostics.GenerateReport();

            // Assert
            Assert.That(report, Does.Contain("Kernel Operations:"));
            Assert.That(report, Does.Contain("ElementwiseAddition"));
        }
        finally
        {
            DiagnosticsTracker.IsEnabled = false;
            DiagnosticsTracker.ClearRecordedOperations();
        }
    }
}
