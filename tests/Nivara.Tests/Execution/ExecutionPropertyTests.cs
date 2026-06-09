using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ExecutionPropertyTests
{
    static IQueryOperation PassthroughOp()
    {
        var op = new StubQueryOperation("Filter");
        op.ExecuteFn = input => input;
        return op;
    }

    static IQueryOperation FilterOp_GreaterThan500()
    {
        var op = new StubQueryOperation("Filter");
        op.ExecuteFn = input =>
        {
            var col = (NivaraColumn<int>)input["A"];
            var filtered = new List<int>();
            for (int i = 0; i < col.Length; i++)
                if (col[i] > 500) filtered.Add(col[i]);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(filtered.ToArray()) };
        };
        return op;
    }

    static QueryPlan CreatePlanWithRowCount(int rowCount, IQuerySource? source = null, IEnumerable<IQueryOperation>? ops = null)
    {
        var src = (StubQuerySource?)(source as StubQuerySource) ?? new StubQuerySource();
        src.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(rowCount);
        return new QueryPlan(src, ops ?? new[] { PassthroughOp() });
    }

    static void ExecuteStrategies(QueryPlan plan, Action<string, NivaraFrame> assertAction)
    {
        var strategies = new (string Name, ExecutionStrategy Strategy)[]
        {
            ("Eager", ExecutionStrategy.Eager),
            ("Lazy", ExecutionStrategy.Lazy),
            ("Streaming", ExecutionStrategy.Streaming),
            ("Parallel", ExecutionStrategy.Parallel),
        };

        foreach (var (name, strategy) in strategies)
        {
            var context = new NivaraExecutionContext(strategy);
            if (strategy == ExecutionStrategy.Parallel)
                context.MaxDegreeOfParallelism = 2;
            if (strategy == ExecutionStrategy.Streaming)
                context.MemoryBudget = 1024 * 1024;

            var engine = new ExecutionEngine();
            using var result = engine.Execute(plan, context);
            assertAction(name, result);
        }
    }

    // ── Property 1: Strategy determinism ──

    static IEnumerable<int> DeterminismDataSizes() => new[] { 3, 200, 2000 };

    [Test]
    public void Property_StrategyDeterminism_SamePlanTwice_IdenticalResults(
        [ValueSource(nameof(DeterminismDataSizes))] int rowCount)
    {
        var plan = CreatePlanWithRowCount(rowCount);
        var engine = new ExecutionEngine();
        var context = new NivaraExecutionContext(ExecutionStrategy.Eager);

        using var first = engine.Execute(plan, context);
        using var second = engine.Execute(plan, context);

        Assert.That(second.RowCount, Is.EqualTo(first.RowCount));
        foreach (var colName in first.ColumnNames)
        {
            var firstCol = first.GetColumn<int>(colName);
            var secondCol = second.GetColumn<int>(colName);
            for (int i = 0; i < first.RowCount; i++)
                Assert.That(secondCol[i], Is.EqualTo(firstCol[i]));
        }
    }

    // ── Property 2: Cross-strategy equivalence ──

    [Test]
    public void Property_CrossStrategyEquivalence_FilterOnChunkedSource_AllStrategiesMatch()
    {
        var source = new StubChunkedQuerySource(totalRowCount: 500);
        var ops = new IQueryOperation[] { FilterOp_GreaterThan500() };
        var plan = new QueryPlan(source, ops);

        int? eagerCount = null;

        ExecuteStrategies(plan, (name, result) =>
        {
            if (name == "Eager")
                eagerCount = result.RowCount;
            else if (eagerCount.HasValue)
                Assert.That(result.RowCount, Is.EqualTo(eagerCount.Value),
                    $"{name} row count differs from Eager");
        });
    }

    // ── Property 3: Passthrough preserves row count ──

    [Test]
    public void Property_PassthroughPreservesRowCount_AllStrategies(
        [ValueSource(nameof(DeterminismDataSizes))] int rowCount)
    {
        var plan = CreatePlanWithRowCount(rowCount);

        ExecuteStrategies(plan, (name, result) =>
        {
            Assert.That(result.RowCount, Is.EqualTo(rowCount),
                $"{name} should preserve row count for {rowCount} rows");
        });
    }

    // ── Property 4: Identity schema preservation ──

    [Test]
    public void Property_IdentitySchemaPreservation_NoOps_MatchesSource()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema) { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn };
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        ExecuteStrategies(plan, (name, result) =>
        {
            Assert.That(result.ColumnCount, Is.EqualTo(2), $"{name} column count");
            Assert.That(result.ColumnNames, Is.EquivalentTo(new[] { "A", "B" }), $"{name} column names");
        });
    }

    // ── Property 5: Operation isolation ──

    [Test]
    public void Property_OperationIsolation_PipelineColumnsMatchTransformSchema()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema) { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn };
        var op1 = new StubQueryOperation("Select");
        op1.TransformSchemaFn = input => input.WithoutColumn("B");
        op1.ExecuteFn = input =>
        {
            Assert.That(input.ContainsKey("B"), "Op1 input should contain B");
            return new Dictionary<string, IColumn>
            {
                ["A"] = input["A"]
            };
        };
        var op2 = new StubQueryOperation("Filter");
        op2.TransformSchemaFn = input =>
        {
            Assert.That(input.ColumnNames, Is.EquivalentTo(new[] { "A" }), "Op2 input should only have A");
            return input;
        };
        op2.ExecuteFn = input => input;
        var plan = new QueryPlan(source, new IQueryOperation[] { op1, op2 });

        ExecuteStrategies(plan, (name, result) =>
        {
            Assert.That(result.ColumnCount, Is.EqualTo(1), $"{name}: column count");
        });
    }

    // ── Property 6: Cancellation atomicity ──

    [Test]
    public void Property_CancellationAtomicity_MidPipeline_ThrowsNoPartialResult()
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);
        using var cts = new CancellationTokenSource();
        cts.CancelAfter(50);
        var delayOp = new StubQueryOperation("Filter");
        delayOp.ExecuteFn = input =>
        {
            cts.Token.WaitHandle.WaitOne(500);
            cts.Token.ThrowIfCancellationRequested();
            return input;
        };
        var afterOp = new StubQueryOperation("Select");
        afterOp.ExecuteFn = input => input;
        var plan = new QueryPlan(source, new IQueryOperation[] { delayOp, afterOp });
        var context = new NivaraExecutionContext(ExecutionStrategy.Eager)
        {
            CancellationToken = cts.Token
        };

        var ex = Assert.Throws<QueryExecutionException>(() =>
            new ExecutionEngine().Execute(plan, context));
        Assert.That(ex!.InnerException, Is.TypeOf<OperationCanceledException>());
    }

    [Test]
    public void Property_CancellationAtomicity_PreCancelled_ThrowsImmediately()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        foreach (var strategy in new[] { ExecutionStrategy.Eager, ExecutionStrategy.Lazy, ExecutionStrategy.Parallel, ExecutionStrategy.Streaming })
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            var context = new NivaraExecutionContext(strategy)
            {
                CancellationToken = cts.Token
            };

            var ex = Assert.Throws<QueryExecutionException>(() =>
                new ExecutionEngine().Execute(plan, context),
                $"Strategy {strategy} should throw on pre-cancelled token");
            Assert.That(ex!.InnerException, Is.TypeOf<OperationCanceledException>());
        }
    }

    // ── Property 7: Dispose independence ──

    [Test]
    public void Property_DisposeIndependence_SourceDisposed_ResultStillValid()
    {
        var source = new StubQuerySource { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn };
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        ExecuteStrategies(plan, (name, result) =>
        {
            source.Dispose();
            Assert.That(result.RowCount, Is.EqualTo(3), $"{name}: row count after dispose");
            Assert.That(result.GetColumn<int>("A")[0], Is.EqualTo(1), $"{name}: value after dispose");
        });
    }

    // ── Property 8: Collect re-execution ──

    [Test]
    public void Property_CollectReExecution_ReturnsFreshFrame(
        [ValueSource(nameof(DeterminismDataSizes))] int rowCount)
    {
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(rowCount);

        ExecuteStrategies(new QueryPlan(source, Array.Empty<IQueryOperation>()), (name, frame1) =>
        {
            using var frame2 = new ExecutionEngine().Execute(
                new QueryPlan(source, Array.Empty<IQueryOperation>()),
                new NivaraExecutionContext(ParseStrategy(name)));

            Assert.That(frame2.RowCount, Is.EqualTo(frame1.RowCount), $"{name}: row count");
            Assert.That(frame2.ColumnCount, Is.EqualTo(frame1.ColumnCount), $"{name}: column count");
            for (int i = 0; i < frame1.RowCount; i++)
                Assert.That(frame2.GetColumn<int>("A")[i], Is.EqualTo(frame1.GetColumn<int>("A")[i]), $"{name}: row {i}");
        });
    }

    static ExecutionStrategy ParseStrategy(string name) => name switch
    {
        "Eager" => ExecutionStrategy.Eager,
        "Lazy" => ExecutionStrategy.Lazy,
        "Streaming" => ExecutionStrategy.Streaming,
        "Parallel" => ExecutionStrategy.Parallel,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name)
    };
}
