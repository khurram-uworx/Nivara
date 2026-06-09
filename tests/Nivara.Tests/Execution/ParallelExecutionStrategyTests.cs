using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Operations;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ParallelExecutionStrategyTests
{
    [Test]
    public void Execute_WithValidPlanAndContext_ReturnsFrame()
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
        Assert.That(ex!.Message, Does.Contain("Frame must contain at least one column"));
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

    // ── Filter parallel tests ──

    [Test]
    public void ExecuteFilterParallel_ResultsMatchSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var filterOp = new StubQueryOperation("Filter");
        filterOp.ExecuteFn = input =>
        {
            var col = (NivaraColumn<int>)input["A"];
            var filtered = new List<int>();
            for (int i = 0; i < col.Length; i++)
                if (col[i] > 500) filtered.Add(col[i]);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(filtered.ToArray()) };
        };

        var plan = ExecutionTestHelpers.CreateTestPlan(source: source, operations: new[] { filterOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(1499));

        var colA = result.GetColumn<int>("A");
        Assert.That(colA[0], Is.EqualTo(501));
        Assert.That(colA[^1], Is.EqualTo(1999));
    }

    [Test]
    public void ExecuteFilterParallel_ProcessesChunksInParallel()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var trackingOp = new ParallelTrackingOperation("Filter");
        trackingOp.ExecuteFn = input => input; // identity filter

        var plan = ExecutionTestHelpers.CreateTestPlan(source: source, operations: new[] { trackingOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(2000));
        Assert.That(trackingOp.ThreadIds.Count, Is.GreaterThan(1), "Filter should execute on multiple threads");
    }

    [Test]
    public void ExecuteFilterParallel_SmallDataset_FallsBackToSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn };

        var trackingOp = new ParallelTrackingOperation("Filter");
        trackingOp.ExecuteFn = input => input;

        var plan = ExecutionTestHelpers.CreateTestPlan(source: source, operations: new[] { trackingOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(trackingOp.ThreadIds.Count, Is.EqualTo(1), "Small dataset should run on single thread");
    }

    // ── Concatenation parallel tests ──

    [Test]
    public void ExecuteConcatenationParallel_Vertical_WorksCorrectly()
    {
        var strategy = new ParallelExecutionStrategy();

        var extraSource = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 })
        };

        var schema = new Schema(new[] { ("A", typeof(int)) });
        var source = new StubQuerySource(schema: schema);
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var concatOp = new ConcatenationOperation(
            new[] { extraSource },
            ConcatenationDirection.Vertical,
            ConcatenationMismatchHandling.FillWithNulls);

        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { concatOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(2003)); // 2000 + 3
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.HasColumn("A"));
    }

    // ── GroupBy parallel tests ──

    [Test]
    public void ExecuteGroupByParallel_ResultsMatchSequential()
    {
        var strategy = new ParallelExecutionStrategy();

        var schema = new Schema(new[] { ("Key", typeof(int)), ("Value", typeof(int)) });
        var source = new StubQuerySource(schema: schema);
        source.ExecuteFn = () =>
        {
            var data = new int[2000];
            for (int i = 0; i < 2000; i++) data[i] = i;
            return new Dictionary<string, IColumn>
            {
                ["Key"] = NivaraColumn<int>.Create(data),
                ["Value"] = NivaraColumn<int>.Create(data),
            };
        };

        var groupByOp = new GroupByOperation(
            new Nivara.Expressions.ColumnExpression[]
            {
                new Nivara.Expressions.ColumnReference("Key")
            });

        // Sequential via Lazy strategy
        var seqPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { groupByOp });
        var seqContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        using var seqResult = new LazyExecutionStrategy().Execute(seqPlan, seqContext);

        // Parallel via Parallel strategy
        var parPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { groupByOp });
        var parContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        parContext.MaxDegreeOfParallelism = Environment.ProcessorCount;
        using var parResult = strategy.Execute(parPlan, parContext);

        Assert.That(parResult.RowCount, Is.EqualTo(seqResult.RowCount));

        var parKeys = parResult.GetColumn<int>("Key");
        var seqKeys = seqResult.GetColumn<int>("Key");
        var parSorted = parKeys.ToArray().OrderBy(k => k).ToArray();
        var seqSorted = seqKeys.ToArray().OrderBy(k => k).ToArray();
        Assert.That(parSorted, Is.EqualTo(seqSorted));
    }

    [Test]
    public void ExecuteGroupByParallel_EmptyInput_ReturnsEmpty()
    {
        var strategy = new ParallelExecutionStrategy();

        var schema = new Schema(new[] { ("Key", typeof(int)) });
        var source = new StubQuerySource(schema: schema);
        source.ExecuteFn = () => new Dictionary<string, IColumn>
        {
            ["Key"] = NivaraColumn<int>.Create(Array.Empty<int>()),
        };

        var groupByOp = new GroupByOperation(
            new Nivara.Expressions.ColumnExpression[]
            {
                new Nivara.Expressions.ColumnReference("Key")
            });

        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { groupByOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    // ── Join parallel tests ──

    static IReadOnlyDictionary<string, IColumn> CreateJoinDataSet(int rowCount, string columnName)
    {
        var data = new int[rowCount];
        for (int i = 0; i < rowCount; i++) data[i] = i % (rowCount / 2);
        return new Dictionary<string, IColumn>
        {
            [columnName] = NivaraColumn<int>.Create(data),
            ["Value"] = NivaraColumn<int>.Create(Enumerable.Range(0, rowCount).ToArray()),
        };
    }

    static void AssertJoinResultsMatch(NivaraFrame parallelResult, NivaraFrame sequentialResult)
    {
        Assert.That(parallelResult.RowCount, Is.EqualTo(sequentialResult.RowCount));
        foreach (var colName in parallelResult.ColumnNames)
        {
            var parCol = parallelResult.GetColumn<int>(colName);
            var seqCol = sequentialResult.GetColumn<int>(colName);
            var parValues = parCol.ToArray();
            var seqValues = seqCol.ToArray();
            Assert.That(parValues, Is.EqualTo(seqValues), $"Column '{colName}' values don't match");
        }
    }

    [Test]
    public void ExecuteJoinParallel_Inner_MatchesSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var left = CreateJoinDataSet(2000, "Key");
        var right = CreateJoinDataSet(1000, "Key");

        var joinOp = new JoinOperation(left, right, JoinType.Inner,
            new[] { new JoinKey("Key") });

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var seqPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });
        var parPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });

        var seqContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        var parContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        parContext.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var seqResult = new LazyExecutionStrategy().Execute(seqPlan, seqContext);
        using var parResult = strategy.Execute(parPlan, parContext);

        AssertJoinResultsMatch(parResult, seqResult);
    }

    [Test]
    public void ExecuteJoinParallel_Left_MatchesSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var left = CreateJoinDataSet(2000, "Key");
        var right = CreateJoinDataSet(500, "Key");

        var joinOp = new JoinOperation(left, right, JoinType.Left,
            new[] { new JoinKey("Key") });

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var seqPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });
        var parPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });

        var seqContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        var parContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        parContext.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var seqResult = new LazyExecutionStrategy().Execute(seqPlan, seqContext);
        using var parResult = strategy.Execute(parPlan, parContext);

        AssertJoinResultsMatch(parResult, seqResult);
    }

    [Test]
    public void ExecuteJoinParallel_Right_MatchesSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var left = CreateJoinDataSet(500, "Key");
        var right = CreateJoinDataSet(2000, "Key");

        var joinOp = new JoinOperation(left, right, JoinType.Right,
            new[] { new JoinKey("Key") });

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var seqPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });
        var parPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });

        var seqContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        var parContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        parContext.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var seqResult = new LazyExecutionStrategy().Execute(seqPlan, seqContext);
        using var parResult = strategy.Execute(parPlan, parContext);

        AssertJoinResultsMatch(parResult, seqResult);
    }

    [Test]
    public void ExecuteJoinParallel_FullOuter_MatchesSequential()
    {
        var strategy = new ParallelExecutionStrategy();
        var left = CreateJoinDataSet(2000, "Key");
        var right = CreateJoinDataSet(1500, "Key");

        var joinOp = new JoinOperation(left, right, JoinType.FullOuter,
            new[] { new JoinKey("Key") });

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var seqPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });
        var parPlan = ExecutionTestHelpers.CreateTestPlan(
            source: source, operations: new IQueryOperation[] { joinOp });

        var seqContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Lazy);
        var parContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        parContext.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var seqResult = new LazyExecutionStrategy().Execute(seqPlan, seqContext);
        using var parResult = strategy.Execute(parPlan, parContext);

        AssertJoinResultsMatch(parResult, seqResult);
    }

    // ── Sync/Async equivalence ──

    [Test]
    public async Task Execute_SyncPath_EqualsAsyncPath()
    {
        var strategy = new ParallelExecutionStrategy();

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var filterOp = new StubQueryOperation("Filter");
        filterOp.ExecuteFn = input =>
        {
            var col = (NivaraColumn<int>)input["A"];
            var filtered = new List<int>();
            for (int i = 0; i < col.Length; i++)
                if (col[i] > 500) filtered.Add(col[i]);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(filtered.ToArray()) };
        };

        var plan = ExecutionTestHelpers.CreateTestPlan(source: source, operations: new[] { filterOp });
        var syncCtx = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        syncCtx.MaxDegreeOfParallelism = Environment.ProcessorCount;
        var asyncCtx = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        asyncCtx.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var syncResult = strategy.Execute(plan, syncCtx);
        using var asyncResult = await strategy.ExecuteAsync(plan, asyncCtx);

        Assert.That(syncResult.RowCount, Is.EqualTo(asyncResult.RowCount));
        var syncCol = syncResult.GetColumn<int>("A");
        var asyncCol = asyncResult.GetColumn<int>("A");
        Assert.That(syncCol.ToArray(), Is.EqualTo(asyncCol.ToArray()));
    }

    // ── Chunked source test ──

    [Test]
    public async Task ChunkedSource_ReadsInParallel()
    {
        var strategy = new ParallelExecutionStrategy();
        var chunkedSource = new StubChunkedQuerySource(2000);

        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: chunkedSource,
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(2000));
        Assert.That(chunkedSource.ChunksRead.Count, Is.GreaterThan(1), "Should read multiple chunks");
    }

    [Test]
    public void NonChunkedSource_FallsBackToFullRead()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var plan = ExecutionTestHelpers.CreateTestPlan(source: source);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(2000));
    }

    // ── Mixed pipeline test ──

    [Test]
    public void MixedOperationsPipeline_ExecutesWithoutError()
    {
        var strategy = new ParallelExecutionStrategy();

        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var filterOp = new StubQueryOperation("Filter");
        filterOp.ExecuteFn = input =>
        {
            var col = (NivaraColumn<int>)input["A"];
            var filtered = new List<int>();
            for (int i = 0; i < col.Length; i++)
                if (col[i] < 100) filtered.Add(col[i]);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(filtered.ToArray()) };
        };

        var selectOp = new StubQueryOperation("Select");
        selectOp.ExecuteFn = input => input; // pass through

        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { filterOp, selectOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        using var result = strategy.Execute(plan, context);
        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    // ── LSP / custom operation tests (Gap 2) ──

    [Test]
    public void CustomSortOpImplementingInterface_DoesNotThrowInvalidCast()
    {
        var strategy = new ParallelExecutionStrategy();
        var source = new StubQuerySource();
        source.ExecuteFn = () => ExecutionTestHelpers.CreateLargeIntColumn(2000);

        var customOp = new CustomSortOperation("A");
        var plan = ExecutionTestHelpers.CreateTestPlan(
            source: source,
            operations: new IQueryOperation[] { customOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = Environment.ProcessorCount;

        Assert.DoesNotThrow(() => strategy.Execute(plan, context));
    }

    [Test]
    public void Execute_DiagnosticsRecordsParallelismDegree()
    {
        var strategy = new ParallelExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.MaxDegreeOfParallelism = 4;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.OperationTimings.Count, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_DiagnosticsPerOperationTimingCount_MatchOperationCount()
    {
        var strategy = new ParallelExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[]
            {
                new StubQueryOperation("Filter"),
                new StubQueryOperation("Sort"),
            });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Parallel);
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.OperationTimings.Count, Is.EqualTo(4)); // ParallelExecution scope + SourceExecute + Filter + Sort
    }
}

sealed class CustomSortOperation : IQueryOperation, IParallelSortOperation
{
    readonly string columnName;

    public CustomSortOperation(string columnName) => this.columnName = columnName;

    public string OperationType => global::Nivara.Query.OperationType.Sort;
    public IReadOnlyList<SortKey> SortKeys => new[] { new SortKey(columnName) };
    public bool IsStable => true;

    public Schema TransformSchema(Schema inputSchema) => inputSchema;

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input.Count == 0) return input;
        var col = input[columnName];
        var indices = Enumerable.Range(0, col.Length).ToArray();
        Array.Sort(indices, (a, b) =>
        {
            var va = col.GetValue(a) as IComparable;
            var vb = col.GetValue(b) as IComparable;
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            return va.CompareTo(vb);
        });
        var sortedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in input)
            sortedColumns[kvp.Key] = SortOperation.ReorderColumn(kvp.Value, indices);
        return sortedColumns;
    }
}
