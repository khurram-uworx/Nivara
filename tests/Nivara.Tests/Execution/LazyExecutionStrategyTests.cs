using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class LazyExecutionStrategyTests
{
    [Test]
    public void Execute_WithValidPlanAndContext_ReturnsFrame()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_DelegatesToQueryExecutor()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Execute_ProgressReporting_FiresCorrectly()
    {
        var strategy = new LazyExecutionStrategy();
        var tracker = new ProgressTracker();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        context.Progress = tracker;

        using var result = strategy.Execute(plan, context);

        Assert.That(tracker.Reports.Count, Is.GreaterThanOrEqualTo(1));
        Assert.That(tracker.Reports[0].OperationName, Does.Contain("Starting"));
    }

    [Test]
    public void Execute_Cancellation_ThrowsOperationCanceledException()
    {
        var strategy = new LazyExecutionStrategy();
        using var cts = new CancellationTokenSource();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        context.CancellationToken = cts.Token;

        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => strategy.Execute(plan, context));
    }

    [Test]
    public void Execute_DiagnosticsRecordsTiming()
    {
        var strategy = new LazyExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.OperationTimings.Count, Is.EqualTo(1));
        Assert.That(diagnostics.OperationTimings[0].OperationType, Is.EqualTo("LazyExecution"));
    }

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var strategy = new LazyExecutionStrategy();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(null!, context));
    }

    [Test]
    public void Execute_NullContext_ThrowsArgumentNullException()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(plan, null!));
    }

    [Test]
    public void ValidatePlan_DelegatesCorrectly()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.True);
    }

    [Test]
    public void ValidatePlan_NullArgs_ReturnsFalse()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.ValidatePlan(null!, context), Is.False);
        Assert.That(strategy.ValidatePlan(plan, null!), Is.False);
    }

    [Test]
    public async Task ExecuteAsync_WrapsSyncOnBackgroundThread()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void ExecuteAsync_Cancellation_ThrowsOperationCanceledException()
    {
        var strategy = new LazyExecutionStrategy();
        using var cts = new CancellationTokenSource();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        context.CancellationToken = cts.Token;

        cts.Cancel();

        Assert.That(
            async () => await strategy.ExecuteAsync(plan, context),
            Throws.InstanceOf<OperationCanceledException>());
    }

    [Test]
    public void EstimateExecutionCost_ReturnsExpectedCost()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        var cost = strategy.EstimateExecutionCost(plan, context);

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_NullArgs_ReturnsMaxValue()
    {
        var strategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.EstimateExecutionCost(null!, context), Is.EqualTo(long.MaxValue));
        Assert.That(strategy.EstimateExecutionCost(plan, null!), Is.EqualTo(long.MaxValue));
    }
}
