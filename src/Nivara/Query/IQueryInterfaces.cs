using System.Runtime.CompilerServices;

namespace Nivara.Query;

internal interface IQueryOperation<T>
{
    QueryPlan Plan { get; }
}

internal interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);

    ValueTask<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(
        IReadOnlyDictionary<string, IColumn> input,
        CancellationToken ct = default)
        => new(Execute(input));
}

internal interface IQuerySource : IDisposable
{
    Schema Schema { get; }
    bool IsLazy { get; }
    IReadOnlyDictionary<string, IColumn> Execute();

    Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)
        => Task.Run(() => Execute(), cancellationToken);

    bool CanReadInChunks => false;

    int? EstimatedRowCount => null;

    IReadOnlyDictionary<string, IColumn> ReadChunk(
        int chunkIndex, int chunkSize)
        => throw new NotSupportedException("This source does not support chunked reading.");

    ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
        int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("This source does not support chunked reading.");

    async IAsyncEnumerable<IReadOnlyDictionary<string, IColumn>> ToAsyncEnumerable(
        int chunkSize, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (!CanReadInChunks)
            throw new NotSupportedException("This source does not support chunked reading.");

        // Chunk-capable sources return an empty chunk at EOF, so termination does not depend on
        // EstimatedRowCount (a deliberate heuristic for some sources, e.g. CSV) and never drops data.
        for (int chunkIndex = 0; ; chunkIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = await ReadChunkAsync(chunkIndex, chunkSize, cancellationToken).ConfigureAwait(false);
            if (chunk == null || chunk.Count == 0 || chunk.Values.All(c => c.Length == 0))
                yield break;
            yield return chunk;
        }
    }
}
