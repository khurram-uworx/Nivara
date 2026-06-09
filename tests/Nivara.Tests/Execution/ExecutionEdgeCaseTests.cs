using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ExecutionEdgeCaseTests
{
    static ExecutionEngine CreateEngine() => new();

    static void AssertExecutesWithStrategy(QueryPlan plan, ExecutionStrategy strategy, Action<NivaraFrame>? assertion = null)
    {
        var engine = CreateEngine();
        var context = new NivaraExecutionContext(strategy);
        if (strategy == ExecutionStrategy.Parallel)
            context.MaxDegreeOfParallelism = 2;
        if (strategy == ExecutionStrategy.Streaming)
            context.MemoryBudget = 1024 * 1024;

        using var result = engine.Execute(plan, context);
        assertion?.Invoke(result);
    }

    static void RunOnAllStrategies(QueryPlan plan, Action<NivaraFrame> assertion)
    {
        foreach (var strategy in new[] { ExecutionStrategy.Eager, ExecutionStrategy.Lazy, ExecutionStrategy.Streaming, ExecutionStrategy.Parallel })
            AssertExecutesWithStrategy(plan, strategy, assertion);
    }

    // ── Empty data source ──

    [Test]
    public void EmptySource_AllStrategies_ReturnsEmptyFrame()
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(Array.Empty<int>())
        };
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        RunOnAllStrategies(plan, result =>
        {
            Assert.That(result.RowCount, Is.EqualTo(0));
            Assert.That(result.ColumnCount, Is.EqualTo(1));
        });
    }

    // ── Single-row source ──

    [Test]
    public void SingleRowSource_AllStrategies_ReturnsCorrectFrame()
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 42 })
        };
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        RunOnAllStrategies(plan, result =>
        {
            Assert.That(result.RowCount, Is.EqualTo(1));
            Assert.That(result.GetColumn<int>("A")[0], Is.EqualTo(42));
        });
    }

    // ── Deep pipeline (20+ operations) ──

    [Test]
    public void DeepPipeline_20StubOps_CompletesSuccessfully()
    {
        var ops = Enumerable.Range(0, 20)
            .Select(i => new StubQueryOperation("Filter"))
            .ToArray<IQueryOperation>();
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: ops);

        RunOnAllStrategies(plan, result =>
        {
            Assert.That(result.RowCount, Is.EqualTo(3));
            Assert.That(result.ColumnCount, Is.EqualTo(2));
        });
    }

    // ── Pre-cancelled token ──

    [Test]
    public void PreCancelledToken_AllStrategies_ThrowsImmediately()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        foreach (var strategy in new[] { ExecutionStrategy.Eager, ExecutionStrategy.Lazy, ExecutionStrategy.Parallel, ExecutionStrategy.Streaming })
        {
            var context = new NivaraExecutionContext(strategy)
            {
                CancellationToken = cts.Token
            };
            var ex = Assert.Throws<QueryExecutionException>(() =>
                CreateEngine().Execute(plan, context),
                $"Strategy {strategy} should throw on pre-cancelled token");
            Assert.That(ex!.InnerException, Is.TypeOf<OperationCanceledException>());
        }
    }

    // ── Max DOP extremes ──

    [Test]
    public void ParallelStrategy_MaxDopIntMax_ThrowsQueryExecutionException()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = new NivaraExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = int.MaxValue
        };

        var ex = Assert.Throws<QueryExecutionException>(() => CreateEngine().Execute(plan, context));
        Assert.That(ex!.Message, Does.Contain("too high"));
    }

    [Test]
    public void ParallelStrategy_MaxDopOne_SequentialFallback()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = new NivaraExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = 1
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void ParallelStrategy_MaxDopBoundary_ProcessorCountTimesFour()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = new NivaraExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount * 4
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
    }

    // ── Streaming edge cases ──

    [Test]
    public void StreamingStrategy_MemoryBudgetZero_FallsBackToLazy()
    {
        var plan = new QueryPlan(
            new StubChunkedQuerySource(100),
            new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
        {
            MemoryBudget = 0
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void StreamingStrategy_MemoryBudgetMaxValue_Works()
    {
        var plan = new QueryPlan(
            new StubChunkedQuerySource(100),
            new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
        {
            MemoryBudget = long.MaxValue
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void StreamingStrategy_EstimatedRowCountZero_Works()
    {
        var plan = new QueryPlan(
            new StubChunkedQuerySource(50, estimatedRowCount: 0),
            new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming);

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
    }

    [Test]
    public void StreamingStrategy_EstimatedRowCountNull_LargeSource_ReadsUntilEmpty()
    {
        var plan = new QueryPlan(
            new StubChunkedQuerySource(750, estimatedRowCount: null),
            new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming);

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(750));
    }

    [Test]
    public void StreamingStrategy_SingleChunk_Works()
    {
        var plan = new QueryPlan(
            new StubChunkedQuerySource(10),
            new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming);

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(10));
    }

    // ── ExecutionEngine edge cases ──

    [Test]
    public void ExecutionEngine_RegisterStrategy_OverwritesExisting()
    {
        var engine = CreateEngine();
        var custom = new EagerExecutionStrategy();

        engine.RegisterStrategy(ExecutionStrategy.Eager, custom);

        var plan = ExecutionTestHelpers.CreateTestPlan();
        using var result = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Eager));
        Assert.That(result, Is.Not.Null);
    }

    [Test]
    public void ExecutionEngine_EstimateCostWith100Ops_DoesNotOverflow()
    {
        var ops = Enumerable.Range(0, 100)
            .Select(i => new StubQueryOperation("Filter"))
            .ToArray<IQueryOperation>();
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: ops);

        var cost = CreateEngine().EstimateExecutionCost(plan, new NivaraExecutionContext(ExecutionStrategy.Lazy));

        Assert.That(cost, Is.GreaterThan(0));
        Assert.That(cost, Is.LessThan(long.MaxValue));
    }

    // ── Large parallel source ──

    [Test]
    [Category("Stress")]
    [CancelAfter(30000)]
    public void ParallelStrategy_MillionRows_ProcessesCorrectly()
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(1_000_000);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());
        var context = new NivaraExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(1_000_000));
    }

    // ── QueryExecutor source returns empty columns ──

    [Test]
    public void QueryExecutor_EmptyColumnsFromSource_ThrowsQueryExecutionException()
    {
        var executor = new QueryExecutor();
        var source = new StubQuerySource();
        source.ExecuteFn = () => new Dictionary<string, IColumn>(0);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        var ex = Assert.Throws<QueryExecutionException>(() => executor.Execute(plan));

        Assert.That(ex!.Message, Does.Contain("no columns"));
    }

    [Test]
    public void QueryExecutor_OperationReturnsSameReference_NoDoubleDispose()
    {
        var executor = new QueryExecutor();
        var op = new StubQueryOperation("Filter");
        op.ExecuteFn = input => input;
        var plan = ExecutionTestHelpers.CreateTestPlan(operations: new IQueryOperation[] { op });

        using var result = executor.Execute(plan);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    // ── CanReadInChunks = false with EstimatedRowCount set ──

    [Test]
    public void CanReadInChunksFalse_WithEstimatedRowCount_Streaming_FallsBack()
    {
        var source = new StubQuerySource { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn, EstimatedRowCount = 100 };
        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming);

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    // ── Unregistered strategy enum value ──

    [Test]
    public void ExecutionEngine_UnregisteredStrategyValue_ThrowsQueryExecutionException()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = new NivaraExecutionContext((ExecutionStrategy)999);

        var ex = Assert.Throws<QueryExecutionException>(() =>
            CreateEngine().Execute(plan, context));

        Assert.That(ex!.Message, Does.Contain("strategy"));
    }

    // ── All-null columns source ──

    [Test]
    public void AllNullColumns_AllStrategies_HandledGracefully()
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.CreateFromNullable(new int?[] { null, null, null })
        };
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        RunOnAllStrategies(plan, result =>
        {
            Assert.That(result.RowCount, Is.EqualTo(3));
            for (int i = 0; i < 3; i++)
                Assert.That(result.GetColumn<int>("A").IsNull(i), Is.True);
        });
    }
}
