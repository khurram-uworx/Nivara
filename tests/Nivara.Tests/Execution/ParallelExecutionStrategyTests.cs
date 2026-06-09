using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ParallelExecutionStrategyTests
{
    [Test]
    public void Execute_CallsGetAwaiterGetResult()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var strategy = new ParallelExecutionStrategy();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(null!, context));
    }

    [Test]
    public void Execute_NullContext_ThrowsArgumentNullException()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(plan, null!));
    }

    [Test]
    public void Execute_EmptyResultColumns_ThrowsQueryExecutionException()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateEmptyColumns();
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.Message, Does.Contain("no columns"));
    }

    [Test]
    public void ValidatePlan_ValidatesBothPlanAndParallelismConfig()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = 2;

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.True);
    }

    [Test]
    public void ValidatePlan_NullArgs_ReturnsFalse()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.ValidatePlan(null!, context), Is.False);
        Assert.That(strategy.ValidatePlan(plan, null!), Is.False);
    }

    [Test]
    public void ValidatePlan_InvalidParallelism_ReturnsFalse()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = 0;

        Assert.That(strategy.ValidatePlan(plan, context), Is.False);
    }

    [Test]
    public void ValidatePlan_ExcessiveParallelism_ReturnsFalse()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount * 4 + 1;

        Assert.That(strategy.ValidatePlan(plan, context), Is.False);
    }

    [Test]
    public void EstimateExecutionCost_ReturnsExpectedCost()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        var cost = strategy.EstimateExecutionCost(plan, context);

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_NullArgs_ReturnsMaxValue()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.EstimateExecutionCost(null!, context), Is.EqualTo(long.MaxValue));
        Assert.That(strategy.EstimateExecutionCost(plan, null!), Is.EqualTo(long.MaxValue));
    }

    [Test]
    public async Task ExecuteAsync_ReturnsFrame()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void ExecuteAsync_Cancellation_ThrowsOperationCanceledException()
    {
        var strategy = new ParallelExecutionStrategy();
        using var cts = new CancellationTokenSource();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.CancellationToken = cts.Token;

        cts.Cancel();

        Assert.ThrowsAsync<OperationCanceledException>(
            async () => await strategy.ExecuteAsync(plan, context));
    }

    [Test]
    public async Task ExecuteAsync_ProgressReporting_FiresCorrectly()
    {
        var strategy = new ParallelExecutionStrategy();
        var tracker = new ProgressTracker();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.Progress = tracker;

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(tracker.Reports.Count, Is.GreaterThanOrEqualTo(2));
    }

    [Test]
    public void OperationFailure_WrapsInQueryExecutionException()
    {
        var strategy = new ParallelExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new ThrowingQueryOperation() });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.InnerException, Is.TypeOf<InvalidOperationException>());
    }
}
