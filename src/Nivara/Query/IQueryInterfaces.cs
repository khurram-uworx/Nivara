using System.Runtime.CompilerServices;

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

    async IAsyncEnumerable<IReadOnlyDictionary<string, IColumn>> ToAsyncEnumerable(
        int chunkSize, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (!CanReadInChunks)
            throw new NotSupportedException("This source does not support chunked reading.");

        var estimated = EstimatedRowCount;
        int maxChunks = estimated.HasValue
            ? (int)((estimated.Value + chunkSize - 1) / chunkSize)
            : int.MaxValue;

        for (int chunkIndex = 0; chunkIndex < maxChunks; chunkIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = await ReadChunkAsync(chunkIndex, chunkSize, cancellationToken).ConfigureAwait(false);
            if (chunk == null || chunk.Count == 0 || chunk.Values.All(c => c.Length == 0))
                yield break;
            yield return chunk;
        }
    }
}
