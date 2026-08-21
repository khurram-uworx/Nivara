using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Query;
using Nivara.Tensors;

namespace Nivara.Execution;

/// <summary>
/// Streams a partitioned standalone window boundary operation
/// (<see cref="RollingOperation"/>, <see cref="CumulativeOperation"/>,
/// <see cref="ShiftOperation"/> with a non-empty <see cref="WindowSpec"/>) by grouping rows
/// into per-partition buffers incrementally as chunks arrive, then computing each
/// partition's window once at drain via the operation's own kernel.
/// </summary>
/// <remarks>
/// Mirrors <see cref="PartitionedWindowEngine.Compute"/> semantics: per-partition stable
/// ordering by the spec's order keys (row-index tiebreak), the same raw kernel over each
/// ordered partition slice, and a single scatter pass restoring original row order. The
/// working set during window computation is one partition at a time; input chunks are
/// retained until drain because partition membership is only final then (rows may join a
/// partition from any later chunk). Rank-family and select-based windows are not handled
/// here and remain full materializations.
/// </remarks>
internal sealed class PartitionedWindowStreamer
{
    readonly struct RowRef
    {
        public RowRef(int globalIndex, int chunkId, int localIndex)
        {
            GlobalIndex = globalIndex;
            ChunkId = chunkId;
            LocalIndex = localIndex;
        }

        public int GlobalIndex { get; }
        public int ChunkId { get; }
        public int LocalIndex { get; }
    }

    sealed class PartitionBuffer : List<RowRef> { }

    readonly WindowOperationBase op;
    readonly WindowSpec spec;
    readonly string[] partitionColumns;
    readonly Dictionary<GroupKey, PartitionBuffer> partitions = new();
    readonly List<IReadOnlyDictionary<string, IColumn>> chunks = [];
    readonly List<IColumn> evaluatedSources = [];
    readonly FusedExpressionEvaluator expressionEvaluator = new();
    int globalRowIndex;

    PartitionedWindowStreamer(WindowOperationBase op, WindowSpec spec)
    {
        this.op = op;
        this.spec = spec;
        partitionColumns = [.. spec.PartitionColumns];
    }

    /// <summary>
    /// Creates a streamer for a standalone window operation with a non-empty window
    /// specification, or null for any other operation kind.
    /// </summary>
    public static PartitionedWindowStreamer? TryCreate(IQueryOperation? boundaryOp)
        => boundaryOp is WindowOperationBase windowOp && windowOp.Spec is { IsEmpty: false } spec
            ? new PartitionedWindowStreamer(windowOp, spec)
            : null;

    /// <summary>
    /// Buffers one processed chunk: rows are hashed by the partition key columns into
    /// per-partition row lists. Chunks with no rows are dropped.
    /// </summary>
    public void ProcessChunk(IReadOnlyDictionary<string, IColumn> chunk)
    {
        if (chunk.Count == 0)
            return;

        IColumn? evaluatedSource = null;
        IColumn lengthColumn;
        if (op.SourceExpression is not null)
        {
            evaluatedSource = expressionEvaluator.Evaluate(op.SourceExpression, chunk);
            lengthColumn = evaluatedSource;
        }
        else if (op.Source is not null && chunk.TryGetValue(op.Source, out var sourceColumn))
        {
            lengthColumn = sourceColumn;
        }
        else
        {
            lengthColumn = chunk.Values.First();
        }

        var len = lengthColumn.Length;
        if (len == 0)
            return;

        var chunkId = chunks.Count;
        chunks.Add(chunk);
        if (evaluatedSource is not null)
            evaluatedSources.Add(evaluatedSource);

        var keyValues = new object?[partitionColumns.Length];
        for (var i = 0; i < len; i++)
        {
            for (var k = 0; k < keyValues.Length; k++)
                keyValues[k] = chunk[partitionColumns[k]].GetValue(i);

            var key = GroupKey.FromValues((object?[])keyValues.Clone());
            if (!partitions.TryGetValue(key, out var buffer))
            {
                buffer = new PartitionBuffer();
                partitions[key] = buffer;
            }

            buffer.Add(new RowRef(globalRowIndex++, chunkId, i));
        }
    }

    /// <summary>
    /// Computes every buffered partition through the operation's kernel and returns the
    /// result columns (all input columns plus the window result) in original row order.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> Flush()
    {
        if (chunks.Count == 0)
            throw new InvalidOperationException(
                "PartitionedWindowStreamer.Flush requires at least one non-empty chunk.");

        var totalRows = globalRowIndex;
        var positions = new int[totalRows];
        var parts = new List<IColumn>(partitions.Count);
        var cursor = 0;

        foreach (var buffer in partitions.Values)
        {
            if (buffer.Count == 0)
                continue;

            var n = buffer.Count;
            var sortIndices = new int[n];
            for (var i = 0; i < n; i++)
                sortIndices[i] = i;

            var sortedSource = GatherSourceColumn(buffer);
            if (spec.OrderKeys.Count > 0)
            {
                var orderKeyColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
                foreach (var name in spec.OrderKeys.Select(k => k.ColumnName).Distinct(StringComparer.OrdinalIgnoreCase))
                    orderKeyColumns[name] = GatherColumn(name, buffer);

                Array.Sort(
                    sortIndices,
                    0,
                    n,
                    new RankTieBreakComparer(new MultiColumnComparer(orderKeyColumns, spec.OrderKeys)));
                sortedSource = ColumnFilterHelper.ReorderColumn(sortedSource, sortIndices);
            }

            parts.Add(op.ComputeForPartition(sortedSource));
            for (var j = 0; j < n; j++)
                positions[cursor++] = buffer[sortIndices[j]].GlobalIndex;
        }

        var resultColumn = ColumnFilterHelper.ScatterPartsColumn(parts, positions);

        var passthrough = concatenateChunks();
        passthrough[op.ResultColumn] = resultColumn;
        return passthrough;
    }

    IColumn GatherSourceColumn(List<RowRef> rows)
    {
        if (op.SourceExpression is not null)
        {
            var template = evaluatedSources[rows[0].ChunkId];
            var values = new object?[rows.Count];
            for (var i = 0; i < rows.Count; i++)
                values[i] = template.GetValue(rows[i].LocalIndex);

            return ColumnFactory.Create(template.ElementType, values);
        }

        if (op.Source is null)
            throw new InvalidOperationException("Window operation has neither a source column nor a source expression.");

        return GatherColumn(op.Source, rows);
    }

    IColumn GatherColumn(string columnName, List<RowRef> rows)
    {
        var firstChunk = chunks[rows[0].ChunkId];
        if (!firstChunk.TryGetValue(columnName, out var template))
            throw new ColumnNotFoundException(columnName, firstChunk.Keys);

        var values = new object?[rows.Count];
        for (var i = 0; i < rows.Count; i++)
            values[i] = chunks[rows[i].ChunkId][columnName].GetValue(rows[i].LocalIndex);

        return ColumnFactory.Create(template.ElementType, values);
    }

    Dictionary<string, IColumn> concatenateChunks()
    {
        var frames = new List<NivaraFrame>(chunks.Count);
        foreach (var chunk in chunks)
            frames.Add(NivaraFrame.Create(chunk));

        var concatenated = frames.Count == 1 ? frames[0] : NivaraFrameExtensions.ConcatenateVertical(frames);
        if (frames.Count > 1)
            foreach (var frame in frames)
                frame.Dispose();

        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in concatenated.ColumnNames)
            columns[name] = concatenated.GetColumn(name);

        return columns;
    }
}
