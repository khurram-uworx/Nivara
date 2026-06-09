using Nivara.Execution;
using Nivara.Exceptions;
using Nivara.Query;

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

static class ExecutionTestHelpers
{
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
}
