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

    bool CanReadInChunks => false;

    int? EstimatedRowCount => null;

    ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("This source does not support chunked reading.");
}
