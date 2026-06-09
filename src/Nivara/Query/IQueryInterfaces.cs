namespace Nivara.Query;

public interface IQueryOperation<T>
{
    QueryPlan Plan { get; }
}

public interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);
}

public interface IQuerySource : IDisposable
{
    Schema Schema { get; }
    bool IsLazy { get; }
    IReadOnlyDictionary<string, IColumn> Execute();

    Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)
        => Task.Run(() => Execute(), cancellationToken);
}
