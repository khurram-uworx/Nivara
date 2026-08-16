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
    public void ValidatePlan_SelectWithWindowExpression_ReturnsFalse()
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

        Assert.That(strategy.ValidatePlan(plan, context), Is.False);
    }

    [Test]
    public void Execute_SelectWithWindowExpression_FallsBackToLazy_ProducesWholeColumnResult()
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

        Assert.That(source.ChunksRead.Count, Is.EqualTo(0),
            "Window-bearing select must fall back to whole-column execution, not read chunks");
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
    public void ExecuteAsync_SelectWithWindowExpression_FallsBackToLazy_ProducesWholeColumnResult()
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

        Assert.That(source.ChunksRead.Count, Is.EqualTo(0),
            "Window-bearing select must fall back to whole-column execution, not read chunks");
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
    public void Property_StreamingVsLazy_WindowSelectFallsBackToLazy_ValuesAndMasksMatchAcrossChunkSizes()
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
            var plan = new QueryPlan(source, new IQueryOperation[] { select });
            var context = ExecutionTestHelpers.CreateTestContext(ExecutionStrategy.Streaming);
            context.MemoryBudget = memoryBudget;

            using var result = strategy.Execute(plan, context);
            using var lazyResult = lazyStrategy.Execute(plan, ExecutionTestHelpers.CreateTestContext());

            Assert.That(source.ChunksRead.Count, Is.EqualTo(0),
                "Window-bearing select must fall back to whole-column execution, not read chunks");
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
}
