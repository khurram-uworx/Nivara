using Nivara.IO;
using Nivara.Query;

namespace Nivara.Samples.Incident;

public static class Ingestion
{
    public static QueryFrame LoadParquet(string path)
        => NivaraParquetReader.ScanAsQueryFrame(path);

    public static QueryFrame LoadCsv(string path)
        => Csv.ScanAsQueryFrame(path);

    public static async IAsyncEnumerable<NivaraFrame> StreamChunks(
        string path, int chunkSize, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await using var frame = NivaraParquetReader.ScanAsQueryFrame(path);
        await foreach (var chunk in frame.AsStream(chunkSize, ct))
            yield return chunk;
    }
}
