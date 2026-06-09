using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ExecutionEngineTests
{
    [Test]
    public void Constructor_RegistersDefaultStrategies()
    {
        var engine = new ExecutionEngine();
        var strategies = engine.GetAvailableStrategies();

        Assert.That(strategies, Contains.Item(ExecutionStrategy.Lazy));
        Assert.That(strategies, Contains.Item(ExecutionStrategy.Eager));
        Assert.That(strategies, Contains.Item(ExecutionStrategy.Streaming));
        Assert.That(strategies, Contains.Item(ExecutionStrategy.Parallel));
    }

    [Test]
    public void Execute_RoutesToCorrectStrategyBasedOnContext()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        using var eagerResult = engine.Execute(plan,
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager));
        using var lazyResult = engine.Execute(plan,
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy));
        using var streamingResult = engine.Execute(plan,
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming));
        using var parallelResult = engine.Execute(plan,
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel));

        Assert.That(eagerResult, Is.Not.Null);
        Assert.That(lazyResult, Is.Not.Null);
        Assert.That(streamingResult, Is.Not.Null);
        Assert.That(parallelResult, Is.Not.Null);
    }

    [Test]
    public void Execute_WithoutExplicitContext_DefaultsToLazy()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        using var result = engine.Execute(plan);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_AppliesOptimizerIfAvailable()
    {
        var engine = new ExecutionEngine();
        var optimizer = new QueryOptimizer();
        engine.SetOptimizer(optimizer);
        var plan = ExecutionTestHelpers.CreateTestPlan();

        using var result = engine.Execute(plan,
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy));

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var engine = new ExecutionEngine();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.Throws<ArgumentNullException>(() => engine.Execute(null!, context));
    }

    [Test]
    public void Execute_NullContext_ThrowsArgumentNullException()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.Throws<ArgumentNullException>(() => engine.Execute(plan, null!));
    }

    [Test]
    public void Execute_WrapsNonQueryExecutionException()
    {
        var engine = new ExecutionEngine();
        var source = new ThrowingQuerySource();
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);

        var ex = Assert.Throws<QueryExecutionException>(() =>
            engine.Execute(plan, ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager)));
        Assert.That(ex!.InnerException, Is.TypeOf<InvalidOperationException>());
    }

    [Test]
    public void ValidatePlan_DelegatesToStrategy()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        var result = engine.ValidatePlan(plan, context);

        Assert.That(result, Is.True);
    }

    [Test]
    public void ValidatePlan_NullArgs_ReturnsFalse()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(engine.ValidatePlan(null!, context), Is.False);
        Assert.That(engine.ValidatePlan(plan, null!), Is.False);
    }

    [Test]
    public void EstimateExecutionCost_DelegatesToStrategy()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        var cost = engine.EstimateExecutionCost(plan, context);

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_NullArgs_ReturnsMaxValue()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext();

        Assert.That(engine.EstimateExecutionCost(null!, context), Is.EqualTo(long.MaxValue));
        Assert.That(engine.EstimateExecutionCost(plan, null!), Is.EqualTo(long.MaxValue));
    }

    [Test]
    public void RegisterStrategy_AddsNewStrategy()
    {
        var engine = new ExecutionEngine();
        var customStrategy = new LazyExecutionStrategy();

        engine.RegisterStrategy(ExecutionStrategy.Lazy, customStrategy);
        var strategies = engine.GetAvailableStrategies();

        Assert.That(strategies, Contains.Item(ExecutionStrategy.Lazy));
    }

    [Test]
    public void RegisterStrategy_UpdatesExistingStrategy()
    {
        var engine = new ExecutionEngine();
        var customStrategy = new EagerExecutionStrategy();

        engine.RegisterStrategy(ExecutionStrategy.Eager, customStrategy);

        using var result = engine.Execute(
            ExecutionTestHelpers.CreateTestPlan(),
            ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager));
        Assert.That(result, Is.Not.Null);
    }

    [Test]
    public void RegisterStrategy_NullStrategy_ThrowsArgumentNullException()
    {
        var engine = new ExecutionEngine();

        Assert.Throws<ArgumentNullException>(() => engine.RegisterStrategy(ExecutionStrategy.Lazy, null!));
    }

    [Test]
    public async Task ExecuteAsync_ReturnsFrame()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        using var result = await engine.ExecuteAsync(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void ToString_ContainsStrategyAndOptimizerInfo()
    {
        var engine = new ExecutionEngine();
        var str = engine.ToString();

        Assert.That(str, Does.Contain("ExecutionEngine"));
        Assert.That(str, Does.Contain("Strategies: 4"));
    }

    [Test]
    public void UnknownStrategy_ThrowsQueryExecutionException()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext((ExecutionStrategy)999);

        Assert.Throws<QueryExecutionException>(() => engine.Execute(plan, context));
    }

    [Test]
    public void Execute_WithDiagnostics_PopulatesOperationTimings()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);

        using var result = engine.Execute(plan, context);

        Assert.That(engine.LastDiagnostics, Is.Not.Null);
        Assert.That(engine.LastDiagnostics!.ExecutionStrategy, Is.EqualTo(ExecutionStrategy.Eager));
        Assert.That(engine.LastDiagnostics.ParallelismDegree, Is.EqualTo(context.MaxDegreeOfParallelism));
        Assert.That(engine.LastDiagnostics.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
        Assert.That(engine.LastDiagnostics.OperationTimings.Count, Is.GreaterThan(0));
    }

    [Test]
    public async Task ExecuteAsync_WithDiagnostics_PopulatesOperationTimings()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);

        using var result = await engine.ExecuteAsync(plan, context);

        Assert.That(engine.LastDiagnostics, Is.Not.Null);
        Assert.That(engine.LastDiagnostics!.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void Execute_DiagnosticsUsesExistingContextDiagnostics()
    {
        var engine = new ExecutionEngine();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Eager);
        var preCreatedDiagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        context.ExecutionDiagnostics = preCreatedDiagnostics;

        using var result = engine.Execute(plan, context);

        Assert.That(engine.LastDiagnostics, Is.SameAs(preCreatedDiagnostics));
        Assert.That(engine.LastDiagnostics!.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void LastDiagnostics_IsPopulatedAfterExecute()
    {
        var engine = new ExecutionEngine();

        Assert.That(engine.LastDiagnostics, Is.Null);

        var plan = ExecutionTestHelpers.CreateTestPlan();
        using var result = engine.Execute(plan);

        Assert.That(engine.LastDiagnostics, Is.Not.Null);
    }
}
