using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;
using System.Collections.Concurrent;

namespace Nivara.Tests.Execution;

sealed class StubQuerySource : IQuerySource
{
    public Schema Schema { get; }
    public bool IsLazy { get; }
    public int? EstimatedRowCount { get; set; }
    public Func<IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public StubQuerySource(Schema? schema = null, bool isLazy = false)
    {
        Schema = schema ?? new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        IsLazy = isLazy;
        EstimatedRowCount = null;
        ExecuteFn = null;
    }

    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        if (ExecuteFn != null) return ExecuteFn();
        throw new InvalidOperationException("StubQuerySource requires ExecuteFn to be set. Use ExecutionTestHelpers.DefaultExecuteFn for 3-row data.");
    }

    public void Dispose() { }
}

sealed class StubQueryOperation : IQueryOperation
{
    public string OperationType { get; }
    public Func<Schema, Schema>? TransformSchemaFn { get; set; }
    public Func<IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public StubQueryOperation(string operationType = global::Nivara.Query.OperationType.Filter)
    {
        OperationType = operationType;
    }

    public Schema TransformSchema(Schema inputSchema)
    {
        if (TransformSchemaFn != null) return TransformSchemaFn(inputSchema);
        return inputSchema;
    }

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (ExecuteFn != null) return ExecuteFn(input);
        return input;
    }
}

sealed class ProgressTracker : IProgress<ExecutionProgress>
{
    public List<ExecutionProgress> Reports { get; } = new();

    public void Report(ExecutionProgress value) => Reports.Add(value);
}

sealed class ThrowingQuerySource : IQuerySource
{
    public Schema Schema => new(new[] { ("A", typeof(int)) });
    public bool IsLazy => false;
    public IReadOnlyDictionary<string, IColumn> Execute()
        => throw new InvalidOperationException("Source failed");
    public void Dispose() { }
}

sealed class ThrowingQueryOperation : IQueryOperation
{
    public string OperationType => global::Nivara.Query.OperationType.Filter;
    public Schema TransformSchema(Schema input) => input;
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
        => throw new InvalidOperationException("Op failed");
}

sealed class ParallelTrackingOperation : IQueryOperation
{
    public string OperationType { get; }
    public ConcurrentBag<int> ThreadIds { get; } = new();
    public Func<IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public ParallelTrackingOperation(string operationType = global::Nivara.Query.OperationType.Filter)
    {
        OperationType = operationType;
    }

    public Schema TransformSchema(Schema input) => input;

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        ThreadIds.Add(Environment.CurrentManagedThreadId);
        if (ExecuteFn != null) return ExecuteFn(input);
        return input;
    }
}

sealed class StubChunkedQuerySource : IQuerySource
{
    readonly int totalRowCount;
    readonly int? estimatedRowCount;

    public StubChunkedQuerySource(int totalRowCount = 2000, int? estimatedRowCount = null)
    {
        this.totalRowCount = totalRowCount;
        this.estimatedRowCount = estimatedRowCount;
    }

    public Schema Schema => new(new[] { ("A", typeof(int)) });
    public bool IsLazy => false;
    public bool CanReadInChunks => true;
    public int? EstimatedRowCount => estimatedRowCount ?? totalRowCount;

    public ConcurrentBag<int> ChunksRead { get; } = new();

    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        var data = new int[totalRowCount];
        for (int i = 0; i < totalRowCount; i++) data[i] = i;
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) };
    }

    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ChunksRead.Add(chunkIndex);
        var start = chunkIndex * chunkSize;
        if (start >= totalRowCount)
            return new Dictionary<string, IColumn>(0);
        var length = Math.Min(chunkSize, totalRowCount - start);
        var data = new int[length];
        for (int i = 0; i < length; i++) data[i] = start + i;
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) };
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

static class ExecutionTestHelpers
{
    public static IReadOnlyDictionary<string, IColumn> CreateLargeIntColumn(int rowCount = 2000)
    {
        var data = new int[rowCount];
        for (int i = 0; i < rowCount; i++) data[i] = i;
        return new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(data),
        };
    }

    public static IReadOnlyDictionary<string, IColumn> DefaultExecuteFn() =>
        new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }),
            ["B"] = NivaraColumn<string>.Create(new[] { "x", "y", "z" }),
        };

    public static QueryPlan CreateTestPlan(
        IQuerySource? source = null,
        IEnumerable<IQueryOperation>? operations = null)
    {
        source ??= new StubQuerySource { ExecuteFn = DefaultExecuteFn };
        operations ??= new[] { new StubQueryOperation(global::Nivara.Query.OperationType.Filter) };
        return new QueryPlan(source, operations);
    }

    public static NivaraExecutionContext CreateTestContext(
        ExecutionStrategy strategy = ExecutionStrategy.Lazy)
    {
        return new NivaraExecutionContext(strategy);
    }

    public static IReadOnlyDictionary<string, IColumn> CreateSimpleColumns()
    {
        return new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 10, 20, 30 }),
        };
    }

    public static IReadOnlyDictionary<string, IColumn> CreateEmptyColumns()
    {
        return new Dictionary<string, IColumn>();
    }

    public static IReadOnlyDictionary<string, IColumn> CreateMultiColumnResult()
    {
        return new Dictionary<string, IColumn>
        {
            ["X"] = NivaraColumn<int>.Create(new[] { 1, 2 }),
            ["Y"] = NivaraColumn<double>.Create(new[] { 1.5, 2.5 }),
        };
    }

    public static StubChunkedQuerySource CreateLargeChunkedSource(int rowCount, int? estimatedRowCount = null)
    {
        return new StubChunkedQuerySource(rowCount, estimatedRowCount);
    }

    public static QueryPlan CreateChunkedTestPlan(
        int sourceRowCount = 2000,
        IEnumerable<IQueryOperation>? operations = null,
        int? estimatedRowCount = null)
    {
        var source = CreateLargeChunkedSource(sourceRowCount, estimatedRowCount);
        operations ??= new[] { new StubQueryOperation(global::Nivara.Query.OperationType.Filter) };
        return new QueryPlan(source, operations);
    }

    public static void AssertFramesEqual(NivaraFrame expected, NivaraFrame actual)
    {
        Assert.That(actual, Is.Not.Null);
        Assert.That(actual.RowCount, Is.EqualTo(expected.RowCount));
        Assert.That(actual.ColumnCount, Is.EqualTo(expected.ColumnCount));
        Assert.That(actual.ColumnNames, Is.EquivalentTo(expected.ColumnNames));

        foreach (var colName in expected.ColumnNames)
        {
            var expectedCol = expected.GetColumn<object>(colName);
            var actualCol = actual.GetColumn<object>(colName);
            for (int i = 0; i < expected.RowCount; i++)
                Assert.That(actualCol[i], Is.EqualTo(expectedCol[i]),
                    $"Row {i} column '{colName}' mismatch");
        }
    }

    public static NivaraFrame CreateTestFrame(int rowCount = 10)
    {
        var intData = new int[rowCount];
        var strData = new string[rowCount];
        for (int i = 0; i < rowCount; i++)
        {
            intData[i] = i * 10;
            strData[i] = $"val{i}";
        }
        return new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<int>.Create(intData)),
            ("Y", (IColumn)NivaraColumn<string>.Create(strData)),
        });
    }
}

sealed class FailingTransformSchemaOperation : IQueryOperation
{
    public string OperationType => global::Nivara.Query.OperationType.Filter;
    public Schema TransformSchema(Schema input)
        => throw new InvalidOperationException("TransformSchema failed");
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
        => throw new InvalidOperationException("TransformSchema failed");
}
