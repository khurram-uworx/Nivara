using Nivara.Exceptions;
using Nivara.Query;
using Parquet.Schema;

namespace Nivara.IO;

/// <summary>
/// Lazy Parquet data source that defers reading until execution.
/// Supports chunked reading at row-group boundaries for streaming execution.
/// </summary>
sealed class ParquetLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly ParquetReadOptions options;
    private readonly Lazy<Schema> lazySchema;
    private readonly Lazy<int> lazyRowGroupCount;
    private readonly Lazy<int> lazyTotalRowCount;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of ParquetLazySource
    /// </summary>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">The Parquet reading options</param>
    public ParquetLazySource(string filePath, ParquetReadOptions? options = null)
    {
        this.filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        this.options = options ?? new ParquetReadOptions();

        lazySchema = new Lazy<Schema>(InferSchema);
        lazyRowGroupCount = new Lazy<int>(GetRowGroupCount);
        lazyTotalRowCount = new Lazy<int>(GetTotalRowCount);
    }

    /// <inheritdoc />
    public Schema Schema
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return lazySchema.Value;
        }
    }

    /// <inheritdoc />
    public bool IsLazy => true;

    /// <inheritdoc />
    public bool CanReadInChunks => true;

    /// <summary>
    /// Gets the estimated row count from Parquet metadata.
    /// </summary>
    public int? EstimatedRowCount => lazyTotalRowCount.Value > 0 ? lazyTotalRowCount.Value : null;

    /// <summary>
    /// Gets the number of row groups in the Parquet file.
    /// </summary>
    public int RowGroupCount => lazyRowGroupCount.Value;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return ExecuteAsync(CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <inheritdoc />
    public async Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        try
        {
            if (!File.Exists(filePath))
                throw new DataSourceException($"Parquet file not found: '{filePath}'");

            var fileInfo = new FileInfo(filePath);
            if (fileInfo.Length == 0)
                throw new DataSourceException($"Parquet file is empty: '{filePath}'");

            var rowGroupCount = lazyRowGroupCount.Value;
            if (rowGroupCount == 0)
                return new Dictionary<string, IColumn>();

            var frames = new List<NivaraFrame>();
            for (int rg = 0; rg < rowGroupCount; rg++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var frame = await ReadRowGroupAsync(rg, cancellationToken).ConfigureAwait(false);
                frames.Add(frame);
            }

            if (frames.Count == 0)
                return new Dictionary<string, IColumn>();

            if (frames.Count == 1)
            {
                return frames[0].ColumnNames.ToDictionary(
                    name => name, name => frames[0].GetColumn(name), StringComparer.OrdinalIgnoreCase);
            }

            var merged = NivaraParquetWriter.ConcatenateFrames(frames);
            return merged.ColumnNames.ToDictionary(
                name => name, name => merged.GetColumn(name), StringComparer.OrdinalIgnoreCase);
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read Parquet file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return ReadChunkAsync(chunkIndex, chunkSize, CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <inheritdoc />
    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            if (!File.Exists(filePath))
                throw new DataSourceException($"Parquet file not found: '{filePath}'");

            var rowGroupCount = lazyRowGroupCount.Value;
            if (chunkIndex >= rowGroupCount)
                return new Dictionary<string, IColumn>();

            var frame = await ReadRowGroupAsync(chunkIndex, cancellationToken).ConfigureAwait(false);
            return frame.ColumnNames.ToDictionary(
                name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read Parquet chunk from file '{filePath}': {ex.Message}", ex);
        }
    }

    private async Task<NivaraFrame> ReadRowGroupAsync(int rowGroupIndex, CancellationToken ct)
    {
        await using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true);
        await using var reader = await Parquet.ParquetReader.CreateAsync(fileStream).ConfigureAwait(false);

        var fields = reader.Schema.GetDataFields();
        using var rowGroupReader = reader.OpenRowGroupReader(rowGroupIndex);
        var columns = await ReadRowGroupColumnsAsync(rowGroupReader, fields, ct).ConfigureAwait(false);
        return NivaraFrame.Create(columns);
    }

    private static async Task<IReadOnlyDictionary<string, IColumn>> ReadRowGroupColumnsAsync(Parquet.ParquetRowGroupReader rowGroupReader, DataField[] fields, CancellationToken ct)
    {
        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < fields.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            var field = fields[i];
            if (field.Name == "_empty")
                continue;

            var data = await NivaraParquetReader.ReadParquetColumnAsync(rowGroupReader, field, ct).ConfigureAwait(false);
            var column = NivaraParquetReader.CreateNivaraColumnFromParquetData(data, field, null);
            columns[field.Name] = column;
        }

        return columns;
    }

    private Schema InferSchema()
    {
        if (!File.Exists(filePath))
            throw new DataSourceException($"Parquet file not found: '{filePath}'");

        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            var reader = Parquet.ParquetReader.CreateAsync(fileStream).GetAwaiter().GetResult();

            var schema = reader.Schema;
            var dataFields = schema.GetDataFields();
            var clrTypeMetadata = reader.CustomMetadata;

            return BuildSchema(dataFields, clrTypeMetadata);
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to infer Parquet schema from '{filePath}': {ex.Message}", ex);
        }
    }

    private int GetRowGroupCount()
    {
        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            var reader = Parquet.ParquetReader.CreateAsync(fileStream).GetAwaiter().GetResult();
            return reader.RowGroupCount;
        }
        catch
        {
            return 0;
        }
    }

    private int GetTotalRowCount()
    {
        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            var reader = Parquet.ParquetReader.CreateAsync(fileStream).GetAwaiter().GetResult();

            int total = 0;
            for (int rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using var rowGroupReader = reader.OpenRowGroupReader(rg);
                total += (int)rowGroupReader.RowCount;
            }
            return total;
        }
        catch
        {
            return 0;
        }
    }

    private static Schema BuildSchema(DataField[] dataFields, IReadOnlyDictionary<string, string>? clrTypeMetadata)
    {
        if (dataFields.Length == 0)
            return new Schema(new[] { ("_empty", typeof(int)) });

        var columnDefinitions = new List<(string Name, Type Type)>();

        foreach (var field in dataFields)
        {
            if (field.Name == "_empty")
                continue;

            var originalType = TypeMapper.ResolveMetadataClrType(clrTypeMetadata, field.Name);
            if (originalType != null)
            {
                columnDefinitions.Add((field.Name, originalType));
            }
            else
            {
                columnDefinitions.Add((field.Name, field.ClrType));
            }
        }

        if (columnDefinitions.Count == 0)
            return new Schema(new[] { ("_empty", typeof(int)) });

        return new Schema(columnDefinitions);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            disposed = true;
        }
    }
}
