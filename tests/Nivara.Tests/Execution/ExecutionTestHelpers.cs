using Nivara.Execution;
using Nivara.Query;
using System.Collections.Concurrent;

namespace Nivara.Tests.Execution;

sealed class StubQuerySource : IQuerySource
{
    public Schema Schema { get; }
    public bool IsLazy { get; }
    public Func<IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public StubQuerySource(Schema? schema = null, bool isLazy = false)
    {
        Schema = schema ?? new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        IsLazy = isLazy;
        ExecuteFn = null;
    }

    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        if (ExecuteFn != null) return ExecuteFn();
        return new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }),
            ["B"] = NivaraColumn<string>.Create(new[] { "x", "y", "z" }),
        };
    }

    public void Dispose() { }
}

sealed class StubQueryOperation : IQueryOperation
{
    public string OperationType { get; }
    public Func<Schema, Schema>? TransformSchemaFn { get; set; }
    public Func<IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public StubQueryOperation(string operationType = "Filter")
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
    public string OperationType => "Filter";
    public Schema TransformSchema(Schema input) => input;
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
        => throw new InvalidOperationException("Op failed");
}

sealed class ParallelTrackingOperation : IQueryOperation
{
    public string OperationType { get; }
    public ConcurrentBag<int> ThreadIds { get; } = new();
    public Func<IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>? ExecuteFn { get; set; }

    public ParallelTrackingOperation(string operationType = "Filter")
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

    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ChunksRead.Add(chunkIndex);
        await Task.Yield();
        var start = chunkIndex * chunkSize;
        if (start >= totalRowCount)
            return new Dictionary<string, IColumn>(0);
        var length = Math.Min(chunkSize, totalRowCount - start);
        var data = new int[length];
        for (int i = 0; i < length; i++) data[i] = start + i;
        return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) };
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

    public static QueryPlan CreateTestPlan(
        IQuerySource? source = null,
        IEnumerable<IQueryOperation>? operations = null)
    {
        source ??= new StubQuerySource();
        operations ??= new[] { new StubQueryOperation("Filter") };
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
        operations ??= new[] { new StubQueryOperation("Filter") };
        return new QueryPlan(source, operations);
    }
}
