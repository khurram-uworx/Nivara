using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class EagerExecutionStrategyTests
{
    [Test]
    public void Execute_WithValidPlanAndContext_ReturnsFrame()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        Assert.That(result.RowCount, Is.EqualTo(3));
        result.Dispose();
    }

    [Test]
    public void Execute_ProgressReporting_FiresCorrectly()
    {
        var strategy = new EagerExecutionStrategy();
        var tracker = new ProgressTracker();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);
        context.Progress = tracker;

        using var result = strategy.Execute(plan, context);

        Assert.That(tracker.Reports.Count, Is.GreaterThanOrEqualTo(2));
        Assert.That(tracker.Reports[0].OperationName, Does.Contain("Starting"));
        Assert.That(tracker.Reports[^1].IsComplete, Is.True);
    }

    [Test]
    public void Execute_Cancellation_StopsExecutionMidPipeline()
    {
        var strategy = new EagerExecutionStrategy();
        using var cts = new CancellationTokenSource();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[]
            {
                new StubQueryOperation("Filter"),
                new StubQueryOperation("Sort"),
            });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);
        context.CancellationToken = cts.Token;

        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => strategy.Execute(plan, context));
    }

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var strategy = new EagerExecutionStrategy();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(null!, context));
    }

    [Test]
    public void Execute_NullContext_ThrowsArgumentNullException()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(plan, null!));
    }

    [Test]
    public void Execute_EmptyResultColumns_ThrowsQueryExecutionException()
    {
        var strategy = new EagerExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateEmptyColumns();
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.Message, Does.Contain("no columns"));
    }

    [Test]
    public void Execute_OperationFailure_WrapsInQueryExecutionException()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new ThrowingQueryOperation() });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.InnerException, Is.TypeOf<InvalidOperationException>());
    }

    [Test]
    public void ValidatePlan_DelegatesToExecutor()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.True);
    }

    [Test]
    public void ValidatePlan_NullArgs_ReturnsFalse()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.ValidatePlan(null!, context), Is.False);
        Assert.That(strategy.ValidatePlan(plan, null!), Is.False);
    }

    [Test]
    public void EstimateExecutionCost_ReturnsExpectedCost()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        var cost = strategy.EstimateExecutionCost(plan, context);

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_NullArgs_ReturnsMaxValue()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.EstimateExecutionCost(null!, context), Is.EqualTo(long.MaxValue));
        Assert.That(strategy.EstimateExecutionCost(plan, null!), Is.EqualTo(long.MaxValue));
    }

    [Test]
    public async Task ExecuteAsync_ReturnsFrame()
    {
        var strategy = new EagerExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public async Task ExecuteAsync_ProgressReporting_FiresCorrectly()
    {
        var strategy = new EagerExecutionStrategy();
        var tracker = new ProgressTracker();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);
        context.Progress = tracker;

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(tracker.Reports.Count, Is.GreaterThanOrEqualTo(2));
    }
}
