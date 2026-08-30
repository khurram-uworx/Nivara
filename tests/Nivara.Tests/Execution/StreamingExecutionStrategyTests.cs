using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.Operations;
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
    public void Execute_NonStreamableOp_NonChunkedSource_ReturnsFrame()
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
        var filter = Nivara.Query.OperationType.Filter;
        var select = Nivara.Query.OperationType.Select;
        var concatPrefix = Nivara.Query.OperationType.ConcatenationPrefix;
        var streamableOps = new[] { filter, select, $"{concatPrefix}Vertical", $"{concatPrefix}Horizontal" };
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
    public void Execute_ExplicitChunkSize_TakesPrecedenceOverBudget()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024L * 1024 * 1024;
        context.ChunkSize = 5;

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(100));
        Assert.That(source.ChunksRead.Count, Is.EqualTo(21),
            "20 data chunks of 5 rows plus the EOF-probe read; the 1GB budget-derived chunk size would read a single chunk");
    }

    [Test]
    public async Task ExecuteAsync_ExplicitChunkSize_Honored()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024L * 1024 * 1024;
        context.ChunkSize = 5;

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(100));
        Assert.That(source.ChunksRead.Count, Is.EqualTo(21),
            "20 data chunks of 5 rows plus the EOF-probe read; the 1GB budget-derived chunk size would read a single chunk");
    }

    [Test]
    public void ValidatePlan_NonPositiveChunkSize_ReturnsFalse()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 0;

        Assert.That(strategy.ValidatePlan(plan, context), Is.False);
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

    // ===== Chunked source execution tests =====

    [Test]
    public void Execute_WithChunkedSource_ReturnsCorrectRowCount()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(sourceRowCount: 2000);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(2000));
        Assert.That(result.ColumnCount, Is.EqualTo(1));
    }

    [Test]
    public void Execute_WithChunkedSource_MergedResultEqualsNonChunked()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(sourceRowCount: 5000);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        var lazyContext = ExecutionTestHelpers.CreateTestContext();

        using var streamingResult = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(plan, lazyContext);

        Assert.That(streamingResult.RowCount, Is.EqualTo(lazyResult.RowCount));
        Assert.That(streamingResult.ColumnCount, Is.EqualTo(lazyResult.ColumnCount));
    }

    [Test]
    public void Execute_ChunkCountMatchesExpected()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2500);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(2500));
        Assert.That(source.ChunksRead.Count, Is.GreaterThanOrEqualTo(1));
    }

    [Test]
    public void Execute_PartialFinalChunk_ReturnsCorrectTotal()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 1050);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(1050));
    }

    [Test]
    public void Execute_ExactMultipleChunks_WorksCorrectly()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 10000);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(10000));
    }

    [Test]
    public void Execute_NonChunkedSource_FallsThrough()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(3));
    }

    [Test]
    public void Execute_EmptySource_ReturnsFrame()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 0);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result, Is.Not.Null);
    }

    [Test]
    public void Execute_EmptyChunkedSourceWithBoundaryOp_RunsBoundaryOpOnce()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 0);
        var boundaryCalls = 0;
        var boundaryOp = new StubQueryOperation("Sort")
        {
            ExecuteFn = input =>
            {
                boundaryCalls++;
                return input;
            },
        };
        var plan = new QueryPlan(source, new IQueryOperation[] { boundaryOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(boundaryCalls, Is.EqualTo(1),
            "The empty-source fallback already applies the full plan; boundary ops must not re-apply");
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    [Test]
    public async Task ExecuteAsync_EmptyChunkedSourceWithBoundaryOp_RunsBoundaryOpOnce()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 0);
        var boundaryCalls = 0;
        var boundaryOp = new StubQueryOperation("Sort")
        {
            ExecuteFn = input =>
            {
                boundaryCalls++;
                return input;
            },
        };
        var plan = new QueryPlan(source, new IQueryOperation[] { boundaryOp });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.ExecuteAsync(plan, context).GetAwaiter().GetResult();

        Assert.That(boundaryCalls, Is.EqualTo(1),
            "The empty-source fallback already applies the full plan; boundary ops must not re-apply");
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    [Test]
    public void Execute_SingleRowSource_Works()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 1);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(1));
    }

    [Test]
    public void Execute_EstimatedRowCountNull_ReadsUntilEmpty()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 750, estimatedRowCount: null);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(750));
    }

    [Test]
    public void Execute_FilterOperationPerChunk_ProducesCorrectResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(
            sourceRowCount: 100,
            operations: new IQueryOperation[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void Execute_SelectOperationPerChunk_ReducesColumns()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(
            sourceRowCount: 100,
            operations: new IQueryOperation[] { new StubQueryOperation("Select") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void ValidatePlan_SelectWithWindowExpression_ReturnsTrue()
    {
        var strategy = new StreamingExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
        });
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(
            sourceRowCount: 2000,
            operations: new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        Assert.That(strategy.ValidatePlan(plan, context), Is.True);
    }

    [Test]
    public void Execute_SelectWithWindowExpression_HandlesAsBoundary_ProducesCorrectResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(plan, ExecutionTestHelpers.CreateTestContext());

        assertIntColumnEqual(lazyResult, result, "RollingSum(A, 2)");
    }

    static void assertIntColumnEqual(NivaraFrame expected, NivaraFrame actual, string columnName)
    {
        var expectedCol = expected.GetColumn<int>(columnName);
        var actualCol = actual.GetColumn<int>(columnName);
        Assert.That(actualCol.Length, Is.EqualTo(expectedCol.Length));
        for (int i = 0; i < expectedCol.Length; i++)
            Assert.That(actualCol[i], Is.EqualTo(expectedCol[i]), $"Row {i} column '{columnName}' mismatch");
    }

    [Test]
    public void ExecuteAsync_SelectWithWindowExpression_HandlesAsBoundary_ProducesCorrectResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.ExecuteAsync(plan, context).GetAwaiter().GetResult();
        using var lazyResult = lazyStrategy.ExecuteAsync(plan, ExecutionTestHelpers.CreateTestContext()).GetAwaiter().GetResult();

        assertIntColumnEqual(lazyResult, result, "RollingSum(A, 2)");
    }

    static readonly long[] chunkEquivalenceBudgets = { 1024 * 1024, 2L * 1024 * 1024, 4L * 1024 * 1024 };

    [Test]
    public void Property_StreamingVsLazy_FilterOnNullableColumn_ValuesAndMasksMatchAcrossChunkSizes()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var lazySource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 100),
        });

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
            var streamingPlan = new QueryPlan(streamingSource, new IQueryOperation[]
            {
                new FilterOperation(ColumnExpressions.Col<int>("A") > 100),
            });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var streamingResult = new StreamingExecutionStrategy().Execute(streamingPlan, context);
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, streamingResult);
        }
    }

    [Test]
    public void Property_StreamingVsLazy_SelectOnNullableColumn_ValuesAndMasksMatchAcrossChunkSizes()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var lazySource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[]
        {
            new SelectOperation(new[] { ColumnExpressions.Col<int>("B") }),
        });

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
            var streamingPlan = new QueryPlan(streamingSource, new IQueryOperation[]
            {
                new SelectOperation(new[] { ColumnExpressions.Col<int>("B") }),
            });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var streamingResult = new StreamingExecutionStrategy().Execute(streamingPlan, context);
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, streamingResult);
        }
    }

    [Test]
    public void Property_StreamingVsLazy_FilterOnNullableConditionColumn_ValuesAndMasksMatchAcrossChunkSizes()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var lazySource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("B") > 1000),
        });

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
            var streamingPlan = new QueryPlan(streamingSource, new IQueryOperation[]
            {
                new FilterOperation(ColumnExpressions.Col<int>("B") > 1000),
            });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var streamingResult = new StreamingExecutionStrategy().Execute(streamingPlan, context);
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, streamingResult);
        }
    }

    [Test]
    public void Property_StreamingVsLazy_WindowSelectHandlesAsBoundary_ValuesAndMasksMatchAcrossChunkSizes()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2),
        });

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var source = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 2000);
            var lazySource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 2000);
            var plan = new QueryPlan(source, new IQueryOperation[] { select });
            var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var result = strategy.Execute(plan, context);
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }
    }

    [Test]
    public void Execute_ChainedStreamableOpsPerChunk_Works()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(
            sourceRowCount: 100,
            operations: new IQueryOperation[]
            {
                new StubQueryOperation("Filter"),
                new StubQueryOperation("Select"),
            });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void Execute_NonStreamableOpInChunkedSource_StreamsPrefixThenAppliesBoundary()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new GroupByOperation(new[] { ColumnExpressions.Col<int>("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(source.ChunksRead.Count, Is.GreaterThan(0),
            "Non-streamable boundary ops must stream the streamable prefix instead of falling back to Lazy");
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_LargeDatasetThroughChunks_DataIntegrityPreserved()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 9876);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(9876));
    }

    [Test]
    public void Execute_SchemaPreserved_AcrossChunks()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Select") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(result.ColumnNames, Is.EquivalentTo(new[] { "A" }));
        Assert.That(result.RowCount, Is.EqualTo(500));
    }

    [Test]
    public void ExecuteAsync_WithChunkedSource_ReturnsFrame()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = ExecutionTestHelpers.CreateChunkedTestPlan(sourceRowCount: 500);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.ExecuteAsync(plan, context).GetAwaiter().GetResult();

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(500));
    }

    [Test]
    public void Execute_ProgressReported_ForEachChunk()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 5000);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        var tracker = new ProgressTracker();
        context.Progress = tracker;

        using var result = strategy.Execute(plan, context);

        Assert.That(tracker.Reports.Count, Is.GreaterThanOrEqualTo(1));
        Assert.That(tracker.Reports.Last().IsComplete, Is.True);
    }

    [Test]
    public void Execute_Cancellation_StopsExecutionBeforeChunkRead()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 10000);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        using var cts = new CancellationTokenSource();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.CancellationToken = cts.Token;

        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => strategy.Execute(plan, context));
    }

    [Test]
    public void Execute_CancellationDuringChunks_Propagates()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 50000);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        using var cts = new CancellationTokenSource();
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.CancellationToken = cts.Token;
        context.MemoryBudget = 1024 * 1024;

        var task = Task.Run(() => strategy.Execute(plan, context));
        cts.Cancel();

        try
        {
            task.Wait(2000);
        }
        catch (AggregateException)
        {
            Assert.That(task.IsFaulted);
            Assert.That(task.Exception!.InnerException, Is.TypeOf<OperationCanceledException>()
                .Or.TypeOf<QueryExecutionException>());
            return;
        }

        // If task completed before cancellation, that's acceptable
        Assert.That(task.IsCompletedSuccessfully);
    }

    [Test]
    public void Execute_ChunkedSourceFailure_WrapsInQueryExecutionException()
    {
        var strategy = new StreamingExecutionStrategy();
        var failingSource = new ThrowingQuerySource();
        var plan = new QueryPlan(failingSource, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        var ex = Assert.Throws<QueryExecutionException>(() => strategy.Execute(plan, context));
        Assert.That(ex!.InnerException, Is.TypeOf<InvalidOperationException>());
    }

    [Test]
    public void Execute_SortInChunkedPlan_StreamsPrefixThenAppliesBoundary()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(source.ChunksRead.Count, Is.GreaterThan(0),
            "Sort as a boundary op must stream the streamable prefix instead of falling back to Lazy");
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_JoinInChunkedPlan_StreamsPrefixThenAppliesBoundary()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 100);
        var joinData = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(Enumerable.Range(0, 100).ToArray()),
        };
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new JoinOperation(joinData, joinData, JoinType.Inner, new[] { new JoinKey("A", "A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);

        using var result = strategy.Execute(plan, context);

        Assert.That(source.ChunksRead.Count, Is.GreaterThan(0),
            "Join as a boundary op must stream the streamable prefix instead of falling back to Lazy");
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.GreaterThan(0));
    }

    [Test]
    public void Execute_FilterThenSortInChunkedSource_StreamsPrefixThenAppliesBoundary()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(plan, ExecutionTestHelpers.CreateTestContext());

        Assert.That(source.ChunksRead.Count, Is.GreaterThan(1),
            "Filter-then-Sort must stream the filter prefix per chunk, not fall back to Lazy");
        assertIntColumnEqual(lazyResult, result, "A");
    }

    [Test]
    public void Execute_DiagnosticsRecordsChunkCount()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2500);
        var plan = new QueryPlan(source, new[] { new StubQueryOperation("Filter") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.OperationTimings.Count, Is.GreaterThanOrEqualTo(1));
    }

    [Test]
    public void Execute_DiagnosticsRecording_NonStreamableFallback()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var plan = ExecutionTestHelpers.CreateTestPlan(
            operations: new IQueryOperation[] { new StubQueryOperation("Sort") });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        // Falls back to Lazy which records diagnostic timings
        Assert.That(result, Is.Not.Null);
    }

    // ===== StreamChunksAsync segmented streaming tests (#307) =====

    [Test]
    public async Task StreamChunksAsync_FilterThenSort_YieldsMultipleFrames()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1),
                "Filter-then-Sort must yield streamable prefix chunks plus a sorted frame");
            Assert.That(source.ChunksRead.Count, Is.GreaterThan(0),
                "Must read chunks from source, not fall back to single-frame");

            var filteredRowCount = frames.Take(frames.Count - 1).Sum(f => f.RowCount);
            var sortedFrame = frames.Last();
            Assert.That(filteredRowCount, Is.EqualTo(1989),
                "Lead chunks: Filter A > 10 over 0..1999 keeps 11..1999 = 1989 rows");
            Assert.That(sortedFrame.RowCount, Is.EqualTo(1989),
                "Boundary Sort result has same row count as filtered input");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_FilterThenSort_MatchesLazyResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var filterThenSort = new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        };
        var streamingPlan = new QueryPlan(source, filterThenSort);
        var lazyPlan = new QueryPlan(lazySource, filterThenSort);
        var streamingContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        streamingContext.MemoryBudget = 1024 * 1024;
        var lazyContext = ExecutionTestHelpers.CreateTestContext();

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(streamingPlan, streamingContext))
            frames.Add(frame);

        using var lazyResult = lazyStrategy.Execute(lazyPlan, lazyContext);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1),
                "Must yield multiple frames (lead chunks + sorted result)");
            var sortedFrame = frames.Last();
            Assert.That(sortedFrame.RowCount, Is.EqualTo(lazyResult.RowCount),
                "Sorted result must match lazy result row count");

            var streamingCol = sortedFrame.GetColumn<int>("A");
            var lazyCol = lazyResult.GetColumn<int>("A");
            for (int i = 0; i < lazyResult.RowCount; i++)
                Assert.That((int)streamingCol.GetValue(i)!, Is.EqualTo((int)lazyCol.GetValue(i)!),
                    $"Row {i} mismatch between streaming and lazy");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_AllNonStreamable_YieldsSingleFrame()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 1000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.EqualTo(1),
                "All-non-streamable plan yields exactly one frame (the full sorted result)");
            Assert.That(frames[0].RowCount, Is.EqualTo(1000));

            var col = frames[0].GetColumn<int>("A");
            for (int i = 1; i < col.Length; i++)
                Assert.That((int)col.GetValue(i)!, Is.GreaterThanOrEqualTo((int)col.GetValue(i - 1)!),
                    "Rows must be in sorted order");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_WindowExpression_YieldsMultipleFrames()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2),
        });
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThanOrEqualTo(1),
                "Window expression must yield at least one frame");
            var totalRows = frames.Sum(f => f.RowCount);
            Assert.That(totalRows, Is.EqualTo(2000),
                "Total rows must equal source row count");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_FilterThenWindow_YieldsMultipleFramesAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var filterThenRolling = new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SelectOperation(new[]
            {
                ColumnExpressions.Col<int>("A"),
                ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2),
            }),
        };
        var streamingPlan = new QueryPlan(source, filterThenRolling);
        var lazyPlan = new QueryPlan(lazySource, filterThenRolling);
        var streamingContext = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        streamingContext.MemoryBudget = 1024 * 1024;
        var lazyContext = ExecutionTestHelpers.CreateTestContext();

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(streamingPlan, streamingContext))
            frames.Add(frame);

        using var lazyResult = lazyStrategy.Execute(lazyPlan, lazyContext);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1),
                "Filter-then-Window must yield per-chunk window frames");
            Assert.That(source.ChunksRead.Count, Is.GreaterThan(0),
                "Must read chunks from source");

            var totalRows = frames.Sum(f => f.RowCount);
            Assert.That(totalRows, Is.EqualTo(lazyResult.RowCount),
                "Per-chunk window totals must match lazy row count");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_StreamableOnly_YieldsOneFramePerChunk()
    {
        var strategy = new StreamingExecutionStrategy();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SelectOperation(new[] { ColumnExpressions.Col<int>("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1),
                "Fully streamable plan must yield one frame per chunk");
            var totalRows = frames.Sum(f => f.RowCount);
            Assert.That(totalRows, Is.EqualTo(1989),
                "Filter A > 10 over 0..1999 keeps 11..1999 = 1989 rows");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public void Execute_OverlapWindow_MatchesLazyExecution()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 5),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        assertIntColumnEqual(lazyResult, result, "RollingSum(A, 5)");
    }

    [Test]
    public async Task ExecuteAsync_OverlapWindow_MatchesLazyExecution()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 5),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = await strategy.ExecuteAsync(plan, context).ConfigureAwait(false);
        using var lazyResult = await lazyStrategy.ExecuteAsync(lazyPlan, ExecutionTestHelpers.CreateTestContext()).ConfigureAwait(false);

        assertIntColumnEqual(lazyResult, result, "RollingSum(A, 5)");
    }

    [Test]
    public void Execute_OverlapWindow_FilterThenRolling_MatchesLazyExecution()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var ops = new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SelectOperation(new[]
            {
                ColumnExpressions.Col<int>("A"),
                ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
            }),
        };
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, ops);
        var lazyPlan = new QueryPlan(lazySource, ops);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        assertIntColumnEqual(lazyResult, result, "RollingSum(A, 3)");
    }

    [Test]
    public async Task StreamChunksAsync_OverlapWindow_YieldsPerChunkAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var lazyPlan = new QueryPlan(lazySource, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1),
                "Overlap window must yield per-chunk frames");
            var totalRows = frames.Sum(f => f.RowCount);
            Assert.That(totalRows, Is.EqualTo(lazyResult.RowCount),
                "Per-chunk total must match lazy row count");

            var allRollingValues = new List<int?>();
            foreach (var f in frames)
            {
                var col = f.GetColumn<int>("RollingSum(A, 3)");
                for (int i = 0; i < col.Length; i++)
                    allRollingValues.Add(col.IsNull(i) ? null : (int)col.GetValue(i)!);
            }
            var lazyCol = lazyResult.GetColumn<int>("RollingSum(A, 3)");
            for (int i = 0; i < lazyCol.Length; i++)
            {
                var expected = lazyCol.IsNull(i) ? (int?)null : (int)lazyCol.GetValue(i)!;
                Assert.That(allRollingValues[i], Is.EqualTo(expected),
                    $"Row {i} RollingSum mismatch between streaming and lazy");
            }
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_OverlapWindow_WithTrailingBoundary_YieldsCorrectResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var ops = new IQueryOperation[]
        {
            new SelectOperation(new[]
            {
                ColumnExpressions.Col<int>("A"),
                ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
            }),
            new SortOperation("RollingSum(A, 3)"),
        };
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500);
        var lazySource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500);
        var plan = new QueryPlan(source, ops);
        var lazyPlan = new QueryPlan(lazySource, ops);
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        try
        {
            Assert.That(frames.Count, Is.GreaterThanOrEqualTo(1));
            var lastFrame = frames.Last();
            Assert.That(lastFrame.RowCount, Is.EqualTo(lazyResult.RowCount),
                "Window + Sort result must match lazy row count");
            var lazyRollingCol = lazyResult.GetColumn<int>("RollingSum(A, 3)");
            var resultRollingCol = lastFrame.GetColumn<int>("RollingSum(A, 3)");
            Assert.That(resultRollingCol.Length, Is.EqualTo(lazyRollingCol.Length));
            for (int i = 0; i < lazyRollingCol.Length; i++)
            {
                var expected = lazyRollingCol.IsNull(i) ? (int?)null : (int)lazyRollingCol.GetValue(i)!;
                var actual = resultRollingCol.IsNull(i) ? (int?)null : (int)resultRollingCol.GetValue(i)!;
                Assert.That(actual, Is.EqualTo(expected),
                    $"Row {i} RollingSum mismatch after Sort");
            }
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    // ===== Boundary materialization diagnostics =====

    [Test]
    public void Execute_MaterializationAtSortBoundary_RecordsDiagnostics()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.StreamMaterializationCount, Is.EqualTo(1),
            "Filter-then-Sort must materialize exactly once at the Sort boundary");
        Assert.That(diagnostics.RowsMaterializedAtBoundaries, Is.EqualTo(1989),
            "Materialized row count must equal the filtered row count");
        Assert.That(diagnostics.Warnings.Any(w => w.Message.Contains("materialized 1,989 rows")),
            Is.True, "A materialization warning must be recorded");
    }

    [Test]
    public async Task ExecuteAsync_MaterializationAtSortBoundary_RecordsDiagnostics()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = await strategy.ExecuteAsync(plan, context).ConfigureAwait(false);

        Assert.That(diagnostics.StreamMaterializationCount, Is.EqualTo(1));
        Assert.That(diagnostics.RowsMaterializedAtBoundaries, Is.EqualTo(1989));
    }

    [Test]
    public async Task StreamChunksAsync_MaterializationAtSortBoundary_RecordsDiagnostics()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
            new SortOperation(new List<SortKey> { new SortKey("A") }),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        var frames = new List<NivaraFrame>();
        try
        {
            await foreach (var frame in strategy.StreamChunksAsync(plan, context))
                frames.Add(frame);

            Assert.That(frames.Count, Is.GreaterThan(1));
            Assert.That(diagnostics.StreamMaterializationCount, Is.EqualTo(1),
                "Chunked streaming must report the Sort-boundary materialization");
            Assert.That(diagnostics.RowsMaterializedAtBoundaries, Is.EqualTo(1989));
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public void Execute_FullyStreamablePlan_RecordsNoMaterializations()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[]
        {
            new FilterOperation(ColumnExpressions.Col<int>("A") > 10),
        });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.StreamMaterializationCount, Is.Zero,
            "Fully streamable plans never materialize at a boundary");
        Assert.That(diagnostics.RowsMaterializedAtBoundaries, Is.Zero);
    }

    [Test]
    public void Execute_OverlapWindowBoundary_RecordsNoMaterializations()
    {
        var strategy = new StreamingExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
        });
        var source = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000);
        var plan = new QueryPlan(source, new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);

        Assert.That(diagnostics.StreamMaterializationCount, Is.Zero,
            "Overlap-streamed window boundaries are not materializations");
        Assert.That(diagnostics.RowsMaterializedAtBoundaries, Is.Zero);
    }

    [Test]
    public void Property_StreamingVsLazy_CumulativeSumSelect_MatchesAcrossChunkSizes()
    {
        // Regression: cumulative windows previously received overlap=1, which carried only
        // the last row of each chunk and dropped all earlier history for multi-row chunks.
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.CumulativeSum(ColumnExpressions.Col("A")),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000);
            var plan = new QueryPlan(streamingSource, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var result = new StreamingExecutionStrategy().Execute(plan, context);

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }
    }

    [Test]
    public void Execute_CumulativeSelect_PinnedChunkSize_MatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.CumulativeMax(ColumnExpressions.Col("A")),
            ColumnExpressions.CumulativeSum(ColumnExpressions.Col("A")),
        });

        var plan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500),
            new IQueryOperation[] { select });
        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 137;

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        Assert.That(plan.Source.EstimatedRowCount / 137, Is.GreaterThan(1));
        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_AllCumulativeKinds_OnDoubleSource_MatchesLazy()
    {
        // Alternating 3/2 and 2/3 values keep the running product bounded near 1.
        var kinds = new (string Name, Func<ColumnExpression, ColumnExpression> Factory)[]
        {
            ("CumulativeSum(V)", s => ColumnExpressions.CumulativeSum(s)),
            ("CumulativeMax(V)", s => ColumnExpressions.CumulativeMax(s)),
            ("CumulativeMin(V)", s => ColumnExpressions.CumulativeMin(s)),
            ("CumulativeProduct(V)", s => ColumnExpressions.CumulativeProduct(s)),
            ("CumulativeCount(V)", s => ColumnExpressions.CumulativeCount(s)),
        };
        var lazyStrategy = new LazyExecutionStrategy();

        foreach (var (name, factory) in kinds)
        {
            var select = new SelectOperation(new[]
            {
                ColumnExpressions.Col<double>("V"),
                factory(ColumnExpressions.Col("V")),
            });

            var lazyPlan = new QueryPlan(
                new DoubleChunkedSource(totalRowCount: 2000),
                new IQueryOperation[] { select });
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

            var source = new DoubleChunkedSource(totalRowCount: 2000);
            var plan = new QueryPlan(source, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.ChunkSize = 333;

            using var result = new StreamingExecutionStrategy().Execute(plan, context);

            Assert.That(source.ChunksRead.Count, Is.GreaterThan(1), $"{name} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }
    }

    [Test]
    public void Property_StreamingVsLazy_CumulativeSum_NullableSource_MatchesWithMasks()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("B"),
            ColumnExpressions.CumulativeSum(ColumnExpressions.Col("B")),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 6000);
            var plan = new QueryPlan(streamingSource, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var result = new StreamingExecutionStrategy().Execute(plan, context);

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }
    }

    [Test]
    public async Task StreamChunksAsync_StandaloneRollingOperation_YieldsPerChunkAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new RollingOperation(ColumnExpressions.Col("A"), "Roll", windowSize: 5);

        var plan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 400;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1), "Standalone rolling boundary must yield per-chunk frames");
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());
            assertNullableIntColumnsMatch(frames, lazyResult, "Roll");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_StandaloneCumulativeOperation_YieldsPerChunkAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new CumulativeOperation(
            ColumnExpressions.Col("A"), "Cum", nullHandler: null,
            NivaraFrameExtensions.CumulativeKind.Sum);

        var plan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 400;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1), "Standalone cumulative boundary must yield per-chunk frames");
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());
            assertNullableIntColumnsMatch(frames, lazyResult, "Cum");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task StreamChunksAsync_StandaloneShiftOperation_YieldsPerChunkAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new ShiftOperation(ColumnExpressions.Col("A"), "Lag", periods: 3);

        var plan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 400;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(plan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.GreaterThan(1), "Standalone shift boundary must yield per-chunk frames");
            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());
            assertNullableIntColumnsMatch(frames, lazyResult, "Lag");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    // ===== Delayed-emission streaming (lead / negative shift) =====

    [Test]
    public void Property_StreamingVsLazy_LeadSelect_MatchesAcrossChunkSizes()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 3),
        });
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000);
            var plan = new QueryPlan(streamingSource, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;
            context.ExecutionDiagnostics = diagnostics;

            using var result = new StreamingExecutionStrategy().Execute(plan, context);

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }

        Assert.That(diagnostics.StreamMaterializationCount, Is.Zero,
            "Lead windows must stream via delayed emission without materializing");
    }

    [Test]
    public void Property_StreamingVsLazy_Lead_NullableSource_MatchesWithMasks()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("B"),
            ColumnExpressions.Lead(ColumnExpressions.Col("B"), 2),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 3000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 3000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 512;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_NegativeShiftSelect_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Shift(ColumnExpressions.Col("A"), -2),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 256;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_LeadWithFillValue_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 3, -7),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 300;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_LeadWithRollingAndLag_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 3),
            ColumnExpressions.Shift(ColumnExpressions.Col("A"), 2),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 3),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 333;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_LeadWithCumulativeSum_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.CumulativeSum(ColumnExpressions.Col("A")),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 2),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 271;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Property_StreamingVsLazy_LeadWithCumulativeCount_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.CumulativeCount(ColumnExpressions.Col("A")),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 2),
        });
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        foreach (var memoryBudget in chunkEquivalenceBudgets)
        {
            var streamingSource = ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 6000);
            var plan = new QueryPlan(streamingSource, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;
            context.ExecutionDiagnostics = diagnostics;

            using var result = new StreamingExecutionStrategy().Execute(plan, context);

            Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
                $"Budget {memoryBudget} should stream multiple chunks");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        }

        Assert.That(diagnostics.StreamMaterializationCount, Is.Zero,
            "Lead windows must stream via delayed emission without materializing");
    }

    [Test]
    public void Property_StreamingVsLazy_LeadWithCumulativeCount_NullableSource_MatchesWithMasks()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("B"),
            ColumnExpressions.CumulativeCount(ColumnExpressions.Col("B")),
            ColumnExpressions.Lead(ColumnExpressions.Col("B"), 2),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 3000),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateNullableChunkedSource(rowCount: 3000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 512;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Execute_LeadSelect_TinyChunksSmallerThanLeadDistance_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 5),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 500),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 7;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Execute_LeadSelect_SingleRowSource_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 3),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 1),
            new IQueryOperation[] { select });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 1),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 100;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
        Assert.That(result.RowCount, Is.EqualTo(1));
    }

    [Test]
    public void Execute_StandaloneNegativeShiftOperation_MatchesLazy()
    {
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new ShiftOperation(ColumnExpressions.Col("A"), "Lead", periods: -2);

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 400;

        using var result = new StreamingExecutionStrategy().Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public async Task StreamChunksAsync_LeadSelect_YieldsPerChunkFramesPlusFlush_AndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.Lead(ColumnExpressions.Col("A"), 3),
        });

        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var streamingPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 400;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(streamingPlan, context))
            frames.Add(frame);

        try
        {
            // 5 full chunks minus the held lead prefix, plus one final flush frame.
            Assert.That(frames.Count, Is.EqualTo(6), "Each chunk yields one frame, plus a final delayed flush frame");
            Assert.That(frames.Last().RowCount, Is.EqualTo(3), "The flush frame carries exactly the held-back tail rows");

            using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());
            assertNullableIntColumnsMatch(frames, lazyResult, "Lead(A, 3)");
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public void Execute_RankWindowSelect_StillMaterializes_AndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();
        var select = new SelectOperation(new[]
        {
            ColumnExpressions.Col<int>("A"),
            ColumnExpressions.RowNumber(orderBy: new[] { new SortExpressionKey(ColumnExpressions.Col("A")) }),
        });

        var plan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var lazyPlan = new QueryPlan(
            ExecutionTestHelpers.CreateLargeChunkedSource(rowCount: 2000),
            new IQueryOperation[] { select });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.MemoryBudget = 1024 * 1024;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(plan, context);
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        Assert.That(diagnostics.StreamMaterializationCount, Is.EqualTo(1),
            "Rank-family windows cannot stream and must be reported as materializations");
        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Execute_PartitionedRollingOperation_MatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new RollingOperation(
            "A", "Roll", 3,
            new WindowSpec().PartitionBy("K").OrderBy("A"));
        var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();

        var lazyPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 500),
            new IQueryOperation[] { op });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingSource = new PartitionedChunkedSource(totalRowCount: 500);
        var streamingPlan = new QueryPlan(streamingSource, new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 137;
        context.ExecutionDiagnostics = diagnostics;

        using var result = strategy.Execute(streamingPlan, context);

        Assert.That(streamingSource.ChunksRead.Count, Is.GreaterThan(1),
            "Partitioned window plan should stream multiple chunks");
        Assert.That(diagnostics.StreamMaterializationCount, Is.Zero,
            "Partitioned windows must be pipelined per-partition, not materialized");
        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public async Task StreamChunksAsync_PartitionedCumulativeOperation_YieldsSingleFrameAndMatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new CumulativeOperation(
            "A", "Cum", new WindowSpec().PartitionBy("K").OrderBy("A"));

        var lazyPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 500),
            new IQueryOperation[] { op });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 500),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 137;

        var frames = new List<NivaraFrame>();
        await foreach (var frame in strategy.StreamChunksAsync(streamingPlan, context))
            frames.Add(frame);

        try
        {
            Assert.That(frames.Count, Is.EqualTo(1),
                "Partitioned window results are only known at drain and yield a single frame");
            ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, frames[0]);
        }
        finally
        {
            foreach (var f in frames) f.Dispose();
        }
    }

    [Test]
    public async Task ExecuteAsync_PartitionedShiftOperation_MatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new ShiftOperation(
            "A", "Lag", 2, new WindowSpec().PartitionBy("K").OrderBy(new[] { "A" }));

        var lazyPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 500),
            new IQueryOperation[] { op });
        using var lazyResult = await lazyStrategy.ExecuteAsync(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 500),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 137;

        using var result = await strategy.ExecuteAsync(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Execute_PartitionedWindow_NullPartitionKeys_MatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new RollingOperation(
            "A", "Roll", 2,
            new WindowSpec().PartitionBy("K").OrderBy("A"));

        var lazyPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 400, nullableKeys: true),
            new IQueryOperation[] { op });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 400, nullableKeys: true),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 97;

        using var result = strategy.Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    [Test]
    public void Execute_HighCardinalityPartitions_MatchesLazy()
    {
        var strategy = new StreamingExecutionStrategy();
        var lazyStrategy = new LazyExecutionStrategy();
        var op = new RollingOperation(
            "A", "Roll", 2,
            new WindowSpec().PartitionBy("K").OrderBy("A"));

        var lazyPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 300, cardinality: 1),
            new IQueryOperation[] { op });
        using var lazyResult = lazyStrategy.Execute(lazyPlan, ExecutionTestHelpers.CreateTestContext());

        var streamingPlan = new QueryPlan(
            new PartitionedChunkedSource(totalRowCount: 300, cardinality: 1),
            new IQueryOperation[] { op });
        var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
        context.ChunkSize = 71;

        using var result = strategy.Execute(streamingPlan, context);

        ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result);
    }

    static void assertNullableIntColumnsMatch(List<NivaraFrame> frames, NivaraFrame lazyResult, string columnName)
    {
        var totalRows = frames.Sum(f => f.RowCount);
        Assert.That(totalRows, Is.EqualTo(lazyResult.RowCount), "Per-chunk total must match lazy row count");

        var allValues = new List<int?>();
        foreach (var f in frames)
        {
            var col = f.GetColumn<int>(columnName);
            for (int i = 0; i < col.Length; i++)
                allValues.Add(col.IsNull(i) ? null : (int)col.GetValue(i)!);
        }

        var lazyCol = lazyResult.GetColumn<int>(columnName);
        for (int i = 0; i < lazyCol.Length; i++)
        {
            var expected = lazyCol.IsNull(i) ? (int?)null : (int)lazyCol.GetValue(i)!;
            Assert.That(allValues[i], Is.EqualTo(expected),
                $"Row {i} '{columnName}' mismatch between streaming and lazy");
        }
    }

    static NivaraColumn<int> concatenateColumn(List<NivaraFrame> frames, string columnName)
    {
        var allValues = new List<int>();
        foreach (var frame in frames)
        {
            var col = frame.GetColumn<int>(columnName);
            for (int i = 0; i < col.Length; i++)
                allValues.Add((int)col.GetValue(i)!);
        }
        return NivaraColumn<int>.Create(allValues.ToArray());
    }
}

sealed class DoubleChunkedSource : IQuerySource
{
    readonly int totalRowCount;

    public DoubleChunkedSource(int totalRowCount)
    {
        this.totalRowCount = totalRowCount;
    }

    public Schema Schema => new(new[] { ("V", typeof(double)) });
    public bool IsLazy => false;
    public bool CanReadInChunks => true;
    public int? EstimatedRowCount => totalRowCount;
    public System.Collections.Concurrent.ConcurrentBag<int> ChunksRead { get; } = new();

    static double ValueAt(int i) => i % 2 == 0 ? 1.5 : 1.0 / 1.5;

    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        var data = new double[totalRowCount];
        for (int i = 0; i < totalRowCount; i++) data[i] = ValueAt(i);
        return new Dictionary<string, IColumn> { ["V"] = NivaraColumn<double>.Create(data) };
    }

    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ChunksRead.Add(chunkIndex);
        var start = chunkIndex * chunkSize;
        if (start >= totalRowCount)
            return new Dictionary<string, IColumn>();
        var length = Math.Min(chunkSize, totalRowCount - start);
        var data = new double[length];
        for (int i = 0; i < length; i++) data[i] = ValueAt(start + i);
        return new Dictionary<string, IColumn> { ["V"] = NivaraColumn<double>.Create(data) };
    }

    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await Task.Yield();
        return ReadChunk(chunkIndex, chunkSize);
    }

    public void Dispose() { }
}

sealed class PartitionedChunkedSource : IQuerySource
{
    readonly int totalRowCount;
    readonly int cardinality;
    readonly bool nullableKeys;

    public PartitionedChunkedSource(int totalRowCount, int cardinality = 3, bool nullableKeys = false)
    {
        this.totalRowCount = totalRowCount;
        this.cardinality = cardinality;
        this.nullableKeys = nullableKeys;
    }

    public Schema Schema => new(new[] { ("A", typeof(int)), ("K", typeof(int)) });
    public bool IsLazy => false;
    public bool CanReadInChunks => true;
    public int? EstimatedRowCount => totalRowCount;
    public System.Collections.Concurrent.ConcurrentBag<int> ChunksRead { get; } = new();

    static int KeyAt(int globalIndex, int cardinality) => globalIndex % cardinality;

    IColumn buildKeyColumn(int start, int count)
    {
        if (!nullableKeys)
        {
            var keys = new int[count];
            for (int i = 0; i < count; i++) keys[i] = KeyAt(start + i, cardinality);
            return NivaraColumn<int>.Create(keys);
        }

        var nullableKeys_ = new int?[count];
        for (int i = 0; i < count; i++)
        {
            var global = start + i;
            nullableKeys_[i] = global % 7 == 0 ? null : KeyAt(global, cardinality);
        }
        return NivaraColumn.CreateFromNullable(nullableKeys_);
    }

    IReadOnlyDictionary<string, IColumn> Build(int start, int count)
    {
        var data = new int[count];
        for (int i = 0; i < count; i++) data[i] = start + i;
        return new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(data),
            ["K"] = buildKeyColumn(start, count),
        };
    }

    public IReadOnlyDictionary<string, IColumn> Execute() => Build(0, totalRowCount);

    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ChunksRead.Add(chunkIndex);
        var start = chunkIndex * chunkSize;
        if (start >= totalRowCount)
            return new Dictionary<string, IColumn>();
        return Build(start, Math.Min(chunkSize, totalRowCount - start));
    }

    public ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        => ValueTask.FromResult(ReadChunk(chunkIndex, chunkSize));

    public void Dispose() { }
}
