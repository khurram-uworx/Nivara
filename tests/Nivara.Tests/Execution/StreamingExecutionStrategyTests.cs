using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingExecutionStrategyTests
{
    [Test]
    public void Execute_WithStreamablePlan_ReturnsFrame()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_FallsThroughToExecutor()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Execute_WithNonStreamablePlan_FallsBackToLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Sort") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var strategy = new StreamingExecutionStrategy();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(null!, context));
    }

    [Test]
    public void Execute_NullContext_ThrowsArgumentNullException()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.Throws<ArgumentNullException>(() => strategy.Execute(plan, null!));
    }

    [Test]
    public void ValidatePlan_ValidatesStreamingPlan()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.True);
    }

    [Test]
    public void ValidatePlan_NonStreamableOperation_ReturnsFalse()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Sort") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.False);
    }

    [Test]
    public void ValidatePlan_ZeroMemoryBudget_ReturnsFalse()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 0;

        var result = strategy.ValidatePlan(plan, context);

        Assert.That(result, Is.False);
    }

    [Test]
    public void ValidatePlan_NullArgs_ReturnsFalse()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.ValidatePlan(null!, context), Is.False);
        Assert.That(strategy.ValidatePlan(plan, null!), Is.False);
    }

    [Test]
    public void isSuitableForStreaming_StreamableOps_ReturnsTrue()
    {
        var strategy = new StreamingExecutionStrategy();
        var streamableOps = new[] { "Filter", "Select", "Concatenation" };
        foreach (var opType in streamableOps)
        {
            var plan = ExecutionTestHelpers.CreateTestPlan(
                operations: new IQueryOperation[] { new StubQueryOperation(opType) });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

            // If executor validates, strategy falls through and should execute
            using var result = strategy.Execute(plan, context);
            Assert.That(result, Is.Not.Null, $"Streamable op '{opType}' should succeed");
        }
    }

    [Test]
    public void ChunkSizeCalculation_RespectsMemoryBudgetBounds()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        context.MemoryBudget = 1024 * 1024;
        using (var result = strategy.Execute(plan, context))
        {
            Assert.That(result, Is.Not.Null);
        }

        context.MemoryBudget = 1024L * 1024 * 1024;
        using (var result2 = strategy.Execute(plan, context))
        {
            Assert.That(result2, Is.Not.Null);
        }
    }

    [Test]
    public void EstimateExecutionCost_ReturnsExpectedCost()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        var cost = strategy.EstimateExecutionCost(plan, context);

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_NullArgs_ReturnsMaxValue()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(strategy.EstimateExecutionCost(null!, context), Is.EqualTo(long.MaxValue));
        Assert.That(strategy.EstimateExecutionCost(plan, null!), Is.EqualTo(long.MaxValue));
    }

    [Test]
    public async Task ExecuteAsync_ReturnsFrame()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void OperationFailure_WrapsInQueryExecutionException()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new ThrowingQueryOperation() });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.InnerException, Is.TypeOf<InvalidOperationException>());
    }
}
