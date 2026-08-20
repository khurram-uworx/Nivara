using Nivara.Diagnostics;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingBudgetDiagnosticTests
{
    [Test]
    public async Task StreamChunksAsync_MemoryBudgetExceeded_RecordsWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 5000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1;
        context.ExecutionDiagnostics = diagnostics;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(diagnostics.Warnings, Is.Not.Empty,
                "A budget exceeded warning should be recorded when accumulated frames exceed MemoryBudget");

            var budgetWarning = diagnostics.Warnings.FirstOrDefault(w =>
                w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase));
            Assert.That(budgetWarning, Is.Not.Null,
                "Should have a warning mentioning 'budget'");
            Assert.That(budgetWarning!.Severity, Is.EqualTo(PerformanceWarningSeverity.Warning));
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_BudgetNotExceeded_NoWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            var budgetWarnings = diagnostics.Warnings.Where(w =>
                w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase)).ToList();
            Assert.That(budgetWarnings, Is.Empty,
                "No budget warning should be recorded when budget is not exceeded");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public void Execute_SyncBudgetExceeded_RecordsWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 5000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.Warnings, Is.Not.Empty,
            "A budget exceeded warning should be recorded in sync path");
        var budgetWarning = diagnostics.Warnings.FirstOrDefault(w =>
            w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase));
        Assert.That(budgetWarning, Is.Not.Null);
        Assert.That(budgetWarning!.Severity, Is.EqualTo(PerformanceWarningSeverity.Warning));
    }

    [Test]
    public void Execute_SyncBudgetNotExceeded_NoWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        var budgetWarnings = diagnostics.Warnings.Where(w =>
            w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase)).ToList();
        Assert.That(budgetWarnings, Is.Empty,
            "No budget warning in sync path when budget is not exceeded");
    }

    [Test]
    public async Task ExecuteAsync_BudgetExceeded_RecordsWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 5000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1;
        context.ExecutionDiagnostics = diagnostics;

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(diagnostics.Warnings, Is.Not.Empty,
            "A budget exceeded warning should be recorded in async path");
        var budgetWarning = diagnostics.Warnings.FirstOrDefault(w =>
            w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase));
        Assert.That(budgetWarning, Is.Not.Null);
        Assert.That(budgetWarning!.Severity, Is.EqualTo(PerformanceWarningSeverity.Warning));
    }

    [Test]
    public async Task ExecuteAsync_BudgetNotExceeded_NoWarning()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var diagnostics = new ExecutionDiagnostics();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = await strategy.ExecuteAsync(plan, context);

        var budgetWarnings = diagnostics.Warnings.Where(w =>
            w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase)).ToList();
        Assert.That(budgetWarnings, Is.Empty,
            "No budget warning in async path when budget is not exceeded");
    }
}
