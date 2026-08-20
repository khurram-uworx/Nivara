using Nivara.Helpers;
using Nivara.Query;
using Streamix;
using System.Runtime.CompilerServices;

namespace Nivara.Streamix;

public static class NivaraFlux
{
    public static IFlux<NivaraFrame> ToFlux(
        this QueryFrame queryFrame,
        int chunkSize = 65536,
        ChannelBackpressureMode backpressureMode = ChannelBackpressureMode.Wait,
        int channelCapacity = 2,
        string? name = null)
    {
        ArgumentNullException.ThrowIfNull(queryFrame);

        var stream = queryFrame.AsStream(chunkSize);
        var flux = name is not null ? Flux.From(stream, name) : Flux.From(stream);

        return channelCapacity > 0
            ? flux.PipeThroughChannel(channelCapacity, backpressureMode)
            : flux;
    }

    public static IFlux<NivaraFrame> ToFlux(this NivaraFrame frame, string? name = null)
        => frame.AsQueryFrame().ToFlux(channelCapacity: 0, name: name);

    public static IFlux<NivaraRow> ToFluxRows(
        this QueryFrame queryFrame,
        int chunkSize = 65536,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(queryFrame);

        return Flux.From(EnumerateRows(queryFrame, chunkSize, ct));
    }

    public static IFlux<NivaraRow> ToFluxRows(
        this NivaraFrame frame,
        int chunkSize = 65536,
        CancellationToken ct = default)
        => frame.AsQueryFrame().ToFluxRows(chunkSize, ct);

    public static async Task<NivaraFrame> ToNivaraFrameAsync(
        this IFlux<NivaraFrame> stream,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var frames = await stream.ToListAsync(ct);
        return NivaraFrameExtensions.ConcatenateVertical(frames);
    }

    public static async Task<NivaraFrame> ToNivaraFrameAsync(
        this IFlux<NivaraRow> stream,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var rows = await stream.ToListAsync(ct);
        return RowsToFrame(rows);
    }

    public static IFlux<Timestamped<NivaraRow>> ToFluxWithTimestamp(
        this QueryFrame queryFrame,
        Func<NivaraRow, DateTimeOffset> timestampSelector,
        int chunkSize = 65536,
        string? name = null)
    {
        ArgumentNullException.ThrowIfNull(queryFrame);
        ArgumentNullException.ThrowIfNull(timestampSelector);

        var rows = queryFrame.ToFluxRows(chunkSize);
        var timestamped = rows.Map(row => Timestamped.Create(row, timestampSelector(row)));
        return name is not null ? timestamped.Named(name) : timestamped;
    }

    public static IFlux<Timestamped<NivaraRow>> ToFluxWithTimestamp(
        this NivaraFrame frame,
        Func<NivaraRow, DateTimeOffset> timestampSelector,
        string? name = null)
        => frame.AsQueryFrame().ToFluxWithTimestamp(timestampSelector, name: name);

    public static IFlux<IList<NivaraRow>> BufferByCount(
        this IFlux<NivaraRow> stream,
        int count,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "Batch size must be greater than 0.");

        return stream.Buffer(count);
    }

    public static IFlux<NivaraFrame> BufferFrames(
        this IFlux<NivaraRow> stream,
        int batchSize,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than 0.");

        return stream.Buffer(batchSize).MapAwait(async rowList =>
        {
            var frame = await RowsToFrameAsync(rowList, ct);
            return frame;
        });
    }

    internal static NivaraFrame RowsToFrame(IList<NivaraRow> rows)
    {
        if (rows.Count == 0)
            return NivaraFrame.Create();

        var columnsArray = rows[0].Columns;
        var columnNames = rows[0].ColumnNames;
        var namedColumns = new (string Name, IColumn Column)[columnNames.Length];

        for (int colIdx = 0; colIdx < columnNames.Length; colIdx++)
        {
            var elementType = columnsArray[colIdx].ElementType;
            var values = new object?[rows.Count];
            for (int i = 0; i < rows.Count; i++)
                values[i] = rows[i].Columns[colIdx].GetValue(rows[i].RowIndex);

            namedColumns[colIdx] = (columnNames[colIdx], ColumnFactory.Create(elementType, values));
        }

        return NivaraFrame.Create(namedColumns);
    }

    static Task<NivaraFrame> RowsToFrameAsync(IList<NivaraRow> rows, CancellationToken ct)
        => Task.FromResult(RowsToFrame(rows));

    static async IAsyncEnumerable<NivaraRow> EnumerateRows(
        QueryFrame queryFrame,
        int chunkSize,
        [EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var frame in queryFrame.AsStream(chunkSize, ct).WithCancellation(ct))
        {
            var columns = frame.ColumnNames.Select(name => frame.GetColumn(name)).ToArray();
            var map = new Dictionary<string, int>(frame.ColumnNames.Count, StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < frame.ColumnNames.Count; i++)
                map[frame.ColumnNames[i]] = i;

            for (int i = 0; i < frame.RowCount; i++)
                yield return new NivaraRow(columns, map, i);
        }
    }
}
