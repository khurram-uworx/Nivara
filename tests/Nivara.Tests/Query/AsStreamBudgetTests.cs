using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class AsStreamBudgetTests
{
    [Test]
    public async Task AsStream_ContextWithMemoryBudget_DerivesChunkSizeFromBudget()
    {
        const long budget = 2_000_000;
        var expectedChunkSize = StreamingExecutionStrategy.CalculateChunkSize(budget);
        Assert.That(expectedChunkSize, Is.EqualTo(2_000), "precondition: a 2 MB budget derives 2,000-row chunks");

        using var query = new QueryFrame(new ChunkedIntSource(totalRows: 10_000));
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming) { MemoryBudget = budget };

        var sizes = await CollectChunkSizes(query.AsStream(context));

        Assert.That(sizes, Is.EqualTo(new[] { 2_000, 2_000, 2_000, 2_000, 2_000 }),
            "chunk size must come from the supplied MemoryBudget, not the 1 GB default");
    }

    [Test]
    public async Task AsStream_BareCall_DefaultsToTenThousandRows()
    {
        using var query = new QueryFrame(new ChunkedIntSource(totalRows: 25_000));

        var sizes = await CollectChunkSizes(query.AsStream());

        Assert.That(sizes, Is.EqualTo(new[] { 10_000, 10_000, 5_000 }),
            "the bare overload keeps the documented 10,000-row default");
    }

    [Test]
    public async Task AsStream_ContextWithExplicitChunkSize_UsesChunkSizeNotBudget()
    {
        using var query = new QueryFrame(new ChunkedIntSource(totalRows: 6_000));
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming) { MemoryBudget = 2_000_000 };

        var sizes = await CollectChunkSizes(query.AsStream(context, 3_000));

        Assert.That(sizes, Is.EqualTo(new[] { 3_000, 3_000 }),
            "an explicit chunk size wins over the context's budget-derived value");
    }

    [Test]
    public async Task AsStream_Context_DoesNotMutateCallerContext()
    {
        using var query = new QueryFrame(new ChunkedIntSource(totalRows: 5_000));
        var context = new NivaraExecutionContext(ExecutionStrategy.Lazy) { MemoryBudget = 2_000_000 };

        await CollectChunkSizes(query.AsStream(context));

        Assert.That(context.Strategy, Is.EqualTo(ExecutionStrategy.Lazy), "caller's strategy must be untouched");
        Assert.That(context.ChunkSize, Is.Null, "caller's ChunkSize must be untouched");
        Assert.That(context.ExecutionDiagnostics, Is.Null, "caller's diagnostics slot must be untouched");
    }

    [Test]
    public void AsStream_NullContext_ThrowsArgumentNullException()
    {
        using var query = new QueryFrame(new ChunkedIntSource(totalRows: 10));

        Assert.That(() => query.AsStream((NivaraExecutionContext)null!), Throws.ArgumentNullException);
    }

    static async Task<int[]> CollectChunkSizes(IAsyncEnumerable<NivaraFrame> chunks)
    {
        var sizes = new List<int>();
        await foreach (var chunk in chunks)
        {
            sizes.Add(chunk.RowCount);
            chunk.Dispose();
        }
        return sizes.ToArray();
    }

    sealed class ChunkedIntSource(int totalRows) : IQuerySource
    {
        public Schema Schema { get; } = new(new[] { ("A", typeof(int)) });
        public bool IsLazy => false;
        public bool CanReadInChunks => true;
        public int? EstimatedRowCount => totalRows;

        public IReadOnlyDictionary<string, IColumn> Execute() => ReadChunk(0, totalRows);

        public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
        {
            var start = chunkIndex * chunkSize;
            var length = Math.Min(chunkSize, totalRows - start);
            if (length <= 0)
                return new Dictionary<string, IColumn>(0);

            var data = new int[length];
            for (int i = 0; i < length; i++)
                data[i] = start + i;
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) };
        }

        public ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
            int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(ReadChunk(chunkIndex, chunkSize));
        }

        public void Dispose()
        {
        }
    }
}
