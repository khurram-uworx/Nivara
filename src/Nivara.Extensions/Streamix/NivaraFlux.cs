using System.Runtime.CompilerServices;
using Nivara.Query;
using Streamix;

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
    {
        ArgumentNullException.ThrowIfNull(frame);

        var flux = Flux.From<NivaraFrame>(new[] { frame });
        return name is not null ? flux.Named(name) : flux;
    }

    public static IFlux<NivaraRow> ToFluxRows(
        this QueryFrame queryFrame,
        int chunkSize = 65536,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(queryFrame);

        return Flux.From(EnumerateRows(queryFrame, chunkSize, ct));
    }

    public static async Task<NivaraFrame> ToNivaraFrameAsync(
        this IFlux<NivaraFrame> stream,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var frames = await stream.ToListAsync(ct);
        return NivaraFrameExtensions.ConcatenateVertical(frames);
    }

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
