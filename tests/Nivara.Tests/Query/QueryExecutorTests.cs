using Nivara.Exceptions;
using Nivara.Optimization;
using Nivara.Query;
using Nivara.Tests.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class QueryExecutorTests
{
    QueryExecutor CreateExecutor() => new();

    // ── Execute(QueryPlan) ──

    [Test]
    public void Execute_NullPlan_ThrowsArgumentNullException()
    {
        var executor = CreateExecutor();

        Assert.Throws<ArgumentNullException>(() => executor.Execute(null!));
    }

    [Test]
    public void Execute_ValidPlanWithOneOperation_ReturnsFrame()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation() });

        using var result = executor.Execute(plan);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnCount, Is.EqualTo(2));
    }

    [Test]
    public void Execute_PlanWithTwoChainedOps_ProducesCorrectResult()
    {
        var executor = CreateExecutor();
        var op1 = new StubQueryOperation("Filter");
        var op2 = new StubQueryOperation("Select");
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { op1, op2 });

        using var result = executor.Execute(plan);

        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Execute_PlanWithThreeChainedOps_ProducesCorrectResult()
    {
        var executor = CreateExecutor();
        var ops = Enumerable.Range(0, 3)
            .Select(i => new StubQueryOperation("Filter"))
            .ToArray<IQueryOperation>();
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: ops);

        using var result = executor.Execute(plan);

        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Execute_SourceReturnsNullColumns_ThrowsQueryExecutionException()
    {
        var executor = CreateExecutor();
        var source = new StubQuerySource();
        source.ExecuteFn = () => null!;
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("returned null columns"));
    }

    [Test]
    public void Execute_OperationReturnsNullColumns_ThrowsQueryExecutionException()
    {
        var executor = CreateExecutor();
        var op = new StubQueryOperation("Filter");
        op.ExecuteFn = _ => null!;
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: new[] { op });

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("returned null columns"));
    }

    [Test]
    public void Execute_OperationFailure_WrapsInQueryExecutionException()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new ThrowingQueryOperation() });

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("Operation"));
        Assert.That(ex.Message, Does.Contain("Filter"));
        Assert.That(ex.InnerException, Is.TypeOf<InvalidOperationException>());
    }

    [Test]
    public void Execute_SourceFailure_WrapsInQueryExecutionException()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: new ThrowingQuerySource());

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("Data source execution failed"));
        Assert.That(ex.InnerException, Is.TypeOf<InvalidOperationException>());
    }

    [Test]
    public void Execute_EmptyResultColumns_ThrowsQueryExecutionException()
    {
        var executor = CreateExecutor();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateEmptyColumns();
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("no columns"));
    }

    [Test]
    public void Execute_InvalidPlan_ThrowsValidationFailed()
    {
        var executor = CreateExecutor();
        var op = new ThrowingValidateOperation();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { op });

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("validation failed"));
    }

    // ── ExecuteOptimized(QueryPlan, QueryOptimizer?) ──

    [Test]
    public void ExecuteOptimized_NullPlan_ThrowsArgumentNullException()
    {
        var executor = CreateExecutor();

        Assert.Throws<ArgumentNullException>(
            () => executor.ExecuteOptimized(null!));
    }

    [Test]
    public void ExecuteOptimized_WithOptimizer_UsesOptimizedPlan()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var engine = new OptimizationEngine(new[] { new MarkerColumnRule() });
        var optimizer = new QueryOptimizer(engine);

        using var result = executor.ExecuteOptimized(plan, optimizer);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void ExecuteOptimized_WithoutOptimizer_SameAsExecute()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        using var direct = executor.Execute(plan);
        using var optimized = executor.ExecuteOptimized(plan, null);

        Assert.That(optimized.RowCount, Is.EqualTo(direct.RowCount));
        Assert.That(optimized.ColumnCount, Is.EqualTo(direct.ColumnCount));
    }

    [Test]
    public void ExecuteOptimized_OptimizerRuleException_SwallowedAndFallsBackToOriginalPlan()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var engine = new OptimizationEngine(new[] { new ThrowingOptimizationRule() });
        var optimizer = new QueryOptimizer(engine);

        using var unoptimized = executor.Execute(plan);
        using var optimized = executor.ExecuteOptimized(plan, optimizer);

        Assert.That(optimized.ColumnCount, Is.EqualTo(unoptimized.ColumnCount));
        Assert.That(optimized.RowCount, Is.EqualTo(unoptimized.RowCount));
    }

    // ── ValidatePlan(QueryPlan) ──

    [Test]
    public void ValidatePlan_Null_ReturnsFalse()
    {
        var executor = CreateExecutor();

        Assert.That(executor.ValidatePlan(null!), Is.False);
    }

    [Test]
    public void ValidatePlan_ValidPlan_ReturnsTrue()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan();

        Assert.That(executor.ValidatePlan(plan), Is.True);
    }

    [Test]
    public void ValidatePlan_WithThrowingOperation_ReturnsFalse()
    {
        var executor = CreateExecutor();
        var op = new ThrowingValidateOperation();
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: new IQueryOperation[] { op });

        Assert.That(executor.ValidatePlan(plan), Is.False);
    }

    // ── EstimateExecutionCost(QueryPlan) ──

    [Test]
    public void EstimateExecutionCost_Null_ReturnsMaxValue()
    {
        var executor = CreateExecutor();

        Assert.That(executor.EstimateExecutionCost(null!), Is.EqualTo(long.MaxValue));
    }

    [Test]
    public void EstimateExecutionCost_EmptyOperations_ReturnsBaseCost()
    {
        var executor = CreateExecutor();
        var source = new StubQuerySource(isLazy: false);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        var cost = executor.EstimateExecutionCost(plan);

        Assert.That(cost, Is.EqualTo(5));
    }

    [Test]
    public void EstimateExecutionCost_Filter_AddsExpectedCost()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });

        var cost = executor.EstimateExecutionCost(plan);

        Assert.That(cost, Is.EqualTo(10)); // 5 base + 5 filter
    }

    [Test]
    public void EstimateExecutionCost_Select_AddsExpectedCost()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Select") });

        var cost = executor.EstimateExecutionCost(plan);

        Assert.That(cost, Is.EqualTo(8)); // 5 base + 3 select
    }

    [Test]
    public void EstimateExecutionCost_GroupBy_AddsExpectedCost()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("GroupBy") });

        var cost = executor.EstimateExecutionCost(plan);

        Assert.That(cost, Is.EqualTo(25)); // 5 base + 20 groupBy
    }

    [Test]
    public void EstimateExecutionCost_UnknownOperation_DefaultsTo10()
    {
        var executor = CreateExecutor();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("CustomOp") });

        var cost = executor.EstimateExecutionCost(plan);

        Assert.That(cost, Is.EqualTo(15)); // 5 base + 10 unknown
    }

    [Test]
    public void EstimateExecutionCost_LazySource_CostsMore()
    {
        var executor = CreateExecutor();
        var lazyPlan = new QueryPlan(
            new StubQuerySource(isLazy: true),
            Array.Empty<IQueryOperation>());
        var eagerPlan = new QueryPlan(
            new StubQuerySource(isLazy: false),
            Array.Empty<IQueryOperation>());

        var lazyCost = executor.EstimateExecutionCost(lazyPlan);
        var eagerCost = executor.EstimateExecutionCost(eagerPlan);

        Assert.That(lazyCost, Is.EqualTo(10));
        Assert.That(eagerCost, Is.EqualTo(5));
        Assert.That(lazyCost, Is.GreaterThan(eagerCost));
    }
}

sealed class MarkerColumnRule : OptimizationRule
{
    public override string Name => "MarkerColumn";
    public override string Description => "Adds a marker column for testing";

    public override bool CanApply(QueryPlan plan) => true;

    public override QueryPlan Apply(QueryPlan plan)
    {
        var newSchema = plan.Source.Schema.WithColumn("_marker", typeof(int));
        var newSource = new StubQuerySource(schema: newSchema, isLazy: plan.Source.IsLazy);
        newSource.ExecuteFn = () =>
        {
            var data = plan.Source.Execute();
            var count = data.Values.FirstOrDefault()?.Length ?? 0;
            var marker = NivaraColumn<int>.Create(Enumerable.Range(0, count).ToArray());
            var extended = new Dictionary<string, IColumn>(data, StringComparer.OrdinalIgnoreCase)
            {
                ["_marker"] = marker
            };
            return extended;
        };
        return new QueryPlan(newSource, plan.Operations.ToList());
    }
}

sealed class ThrowingOptimizationRule : OptimizationRule
{
    public override string Name => "ThrowingRule";
    public override string Description => "Throws during optimization for testing";

    public override bool CanApply(QueryPlan plan) => true;

    public override QueryPlan Apply(QueryPlan plan)
        => throw new InvalidOperationException("Optimization failed");
}

sealed class ThrowingValidateOperation : IQueryOperation
{
    int _callCount;

    public string OperationType => "ThrowingValidate";

    public Schema TransformSchema(Schema input)
    {
        _callCount++;
        if (_callCount > 1)
            throw new InvalidOperationException("ThrowingValidate failed");
        return input;
    }

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input) => input;
}
