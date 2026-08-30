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

    public static async Task<NivaraFrame> ToNivaraFrameAsync(
        this IFlux<Timestamped<NivaraRow>> stream,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var rows = new List<NivaraRow>();
        await foreach (var timestamped in stream.WithCancellation(ct))
            rows.Add(timestamped.Value);
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

    public static IFlux<Timestamped<NivaraRow>> ToFluxWithTimestamp(
        this QueryFrame queryFrame,
        string timestampColumn,
        int chunkSize = 65536,
        string? name = null)
    {
        ArgumentNullException.ThrowIfNull(queryFrame);
        ArgumentException.ThrowIfNullOrWhiteSpace(timestampColumn);

        return queryFrame.ToFluxWithTimestamp(
            row => row.GetValue<DateTimeOffset>(timestampColumn),
            chunkSize,
            name);
    }

    public static IFlux<Timestamped<NivaraRow>> ToFluxWithTimestamp(
        this NivaraFrame frame,
        string timestampColumn,
        string? name = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentException.ThrowIfNullOrWhiteSpace(timestampColumn);

        return frame.AsQueryFrame().ToFluxWithTimestamp(
            row => row.GetValue<DateTimeOffset>(timestampColumn),
            name: name);
    }

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
            throw new InvalidOperationException("Cannot convert an empty row collection to a NivaraFrame: no schema to infer from.");

        var columnsArray = rows[0].Columns;
        var columnNames = rows[0].ColumnNames;
        var namedColumns = new (string Name, IColumn Column)[columnNames.Length];

        for (int colIdx = 0; colIdx < columnNames.Length; colIdx++)
        {
            var col = columnsArray[colIdx];
            var elementType = col.ElementType;
            var count = rows.Count;

            namedColumns[colIdx] = (columnNames[colIdx], elementType switch
            {
                Type t when t == typeof(int) => ReadColumnFast<int>(rows, colIdx, count),
                Type t when t == typeof(int?) => ReadColumnFast<int>(rows, colIdx, count),
                Type t when t == typeof(double) => ReadColumnFast<double>(rows, colIdx, count),
                Type t when t == typeof(double?) => ReadColumnFast<double>(rows, colIdx, count),
                Type t when t == typeof(float) => ReadColumnFast<float>(rows, colIdx, count),
                Type t when t == typeof(float?) => ReadColumnFast<float>(rows, colIdx, count),
                Type t when t == typeof(long) => ReadColumnFast<long>(rows, colIdx, count),
                Type t when t == typeof(long?) => ReadColumnFast<long>(rows, colIdx, count),
                Type t when t == typeof(bool) => ReadColumnFast<bool>(rows, colIdx, count),
                Type t when t == typeof(bool?) => ReadColumnFast<bool>(rows, colIdx, count),
                Type t when t == typeof(short) => ReadColumnFast<short>(rows, colIdx, count),
                Type t when t == typeof(short?) => ReadColumnFast<short>(rows, colIdx, count),
                Type t when t == typeof(byte) => ReadColumnFast<byte>(rows, colIdx, count),
                Type t when t == typeof(byte?) => ReadColumnFast<byte>(rows, colIdx, count),
                Type t when t == typeof(string) => ReadColumnFastRef<string>(rows, colIdx, count),
                _ => ReadColumnBoxed(rows, colIdx, count, elementType),
            });
        }

        return NivaraFrame.Create(namedColumns);
    }

    static NivaraColumn<T> ReadColumnFast<T>(IList<NivaraRow> rows, int colIdx, int count) where T : struct
    {
        var values = new T?[count];
        bool hasNulls = false;
        for (int i = 0; i < count; i++)
        {
            var col = rows[i].Columns[colIdx];
            if (col is IColumn<T?> nullableCol)
            {
                var value = nullableCol[rows[i].RowIndex];
                if (value.HasValue)
                    values[i] = value.GetValueOrDefault();
                else
                    hasNulls = true;
            }
            else if (col.IsNull(rows[i].RowIndex))
            {
                hasNulls = true;
            }
            else
            {
                values[i] = ((IColumn<T>)col)[rows[i].RowIndex];
            }
        }
        if (!hasNulls)
        {
            var data = new T[count];
            for (int i = 0; i < count; i++)
                data[i] = values[i]!.Value;
            return new NivaraColumn<T>(new Storage.ColumnStorage<T>(new ReadOnlyMemory<T>(data)));
        }
        return NivaraColumn.CreateFromNullable(values);
    }

    static NivaraColumn<T> ReadColumnFastRef<T>(IList<NivaraRow> rows, int colIdx, int count) where T : class
    {
        // Reference types cannot be nullable-element columns (Nullable<T> requires a value type), so
        // every reference column here implements IColumn<T> directly and the cast below is always safe.
        var values = new T[count];
        for (int i = 0; i < count; i++)
            values[i] = ((IColumn<T>)rows[i].Columns[colIdx])[rows[i].RowIndex];
        return NivaraColumn<T>.CreateForReferenceType(values);
    }

    static IColumn ReadColumnBoxed(IList<NivaraRow> rows, int colIdx, int count, Type elementType)
    {
        var values = new object?[count];
        for (int i = 0; i < count; i++)
            values[i] = rows[i].Columns[colIdx].GetValue(rows[i].RowIndex);
        return ColumnFactory.Create(elementType, values);
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
