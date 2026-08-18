using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Query;
using Parquet.Data;
using Parquet.Schema;

namespace Nivara.IO;

/// <summary>
/// Lazy Parquet data source that defers reading until execution.
/// Reuses a single <see cref="Parquet.ParquetReader"/> (footer metadata parsed exactly once)
/// across all <see cref="Execute"/>/<see cref="ExecuteAsync"/>/<see cref="ReadChunk"/>/
/// <see cref="ReadChunkAsync"/> calls, seeking row groups by index instead of reopening the
/// file per chunk.
/// </summary>
/// <remarks>
/// Chunks are aligned to native row-group boundaries: <c>chunkIndex</c> maps directly to a row
/// group index and <c>chunkSize</c> is advisory (row-group granularity is the honest replay
/// model). All reads are serialized through a <see cref="SemaphoreSlim"/> because Parquet.Net
/// readers are not thread-safe and <see cref="Nivara.Execution.ParallelExecutionStrategy"/>
/// issues concurrent <see cref="ReadChunkAsync"/> calls.
/// </remarks>
sealed class ParquetLazySource : IQuerySource, IPredicatePushdownSource
{
    private readonly string filePath;
    private readonly ParquetReadOptions options;
    private readonly Lazy<Parquet.ParquetReader> lazyReader;
    private readonly Lazy<Schema> lazySchema;
    private readonly Lazy<int> lazyRowGroupCount;
    private readonly Lazy<int> lazyTotalRowCount;
    private readonly SemaphoreSlim gate = new(1, 1);
    private HashSet<int>? skippedRowGroups;
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

        lazyReader = new Lazy<Parquet.ParquetReader>(CreateReader, LazyThreadSafetyMode.ExecutionAndPublication);
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
    /// Gets the total row count from Parquet metadata.
    /// </summary>
    public int? EstimatedRowCount => lazyTotalRowCount.Value > 0 ? lazyTotalRowCount.Value : null;

    /// <summary>
    /// Gets the number of row groups in the Parquet file.
    /// </summary>
    public int RowGroupCount => lazyRowGroupCount.Value;

    /// <inheritdoc />
    public bool CanPushdownFilter(ColumnExpression condition, Schema sourceSchema)
        => RowGroupFilterEvaluator.CanEvaluate(condition, sourceSchema);

    /// <inheritdoc />
    public void ApplyFilterPredicate(ColumnExpression condition, Schema sourceSchema)
    {
        var reader = lazyReader.Value;
        var rowGroupCount = reader.RowGroupCount;
        if (rowGroupCount == 0)
            return;

        gate.Wait();
        try
        {
            var skipSet = new HashSet<int>();
            var fields = reader.Schema.GetDataFields();

            for (int rg = 0; rg < rowGroupCount; rg++)
            {
                using var rowGroupReader = reader.OpenRowGroupReader(rg);
                bool keep = RowGroupFilterEvaluator.EvaluateRowGroup(
                    condition,
                    columnName => GetColumnStatsForRowGroup(rowGroupReader, columnName, fields),
                    sourceSchema);

                if (!keep)
                    skipSet.Add(rg);
            }

            skippedRowGroups = skipSet.Count > 0 ? skipSet : null;
        }
        finally
        {
            gate.Release();
        }
    }

    static RowGroupFilterEvaluator.RowGroupColumnStats? GetColumnStatsForRowGroup(
        Parquet.ParquetRowGroupReader rowGroupReader,
        string columnName,
        DataField[] allFields)
    {
        for (int i = 0; i < allFields.Length; i++)
        {
            if (string.Equals(allFields[i].Name, columnName, StringComparison.OrdinalIgnoreCase))
            {
                var stats = rowGroupReader.GetStatistics(allFields[i]);
                if (stats == null)
                    return null;

                return new RowGroupFilterEvaluator.RowGroupColumnStats
                {
                    MinValue = stats.MinValue,
                    MaxValue = stats.MaxValue
                };
            }
        }
        return null;
    }

    bool IsSkipped(int rowGroupIndex)
        => skippedRowGroups != null && skippedRowGroups.Contains(rowGroupIndex);

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        try
        {
            var reader = lazyReader.Value;
            var rowGroupCount = reader.RowGroupCount;
            if (rowGroupCount == 0)
                return ToColumnsFromSchema(lazySchema.Value);

            var frames = new List<NivaraFrame>(rowGroupCount);
            for (int rg = 0; rg < rowGroupCount; rg++)
            {
                if (IsSkipped(rg))
                    continue;
                frames.Add(ReadRowGroup(rg));
            }

            return frames.Count > 0 ? ToColumns(frames) : ToColumnsFromSchema(lazySchema.Value);
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
    public async Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        try
        {
            var reader = lazyReader.Value;
            var rowGroupCount = reader.RowGroupCount;
            if (rowGroupCount == 0)
                return ToColumnsFromSchema(lazySchema.Value);

            var frames = new List<NivaraFrame>(rowGroupCount);
            for (int rg = 0; rg < rowGroupCount; rg++)
            {
                if (IsSkipped(rg))
                    continue;
                cancellationToken.ThrowIfCancellationRequested();
                frames.Add(await ReadRowGroupAsync(rg, cancellationToken).ConfigureAwait(false));
            }

            return frames.Count > 0 ? ToColumns(frames) : ToColumnsFromSchema(lazySchema.Value);
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

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        try
        {
            var reader = lazyReader.Value;
            if (chunkIndex >= reader.RowGroupCount)
                return new Dictionary<string, IColumn>();

            if (IsSkipped(chunkIndex))
                return new Dictionary<string, IColumn>();

            return ToColumns(ReadRowGroup(chunkIndex));
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
            var reader = lazyReader.Value;
            if (chunkIndex >= reader.RowGroupCount)
                return new Dictionary<string, IColumn>();

            if (IsSkipped(chunkIndex))
                return new Dictionary<string, IColumn>();

            var frame = await ReadRowGroupAsync(chunkIndex, cancellationToken).ConfigureAwait(false);
            return ToColumns(frame);
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

    private NivaraFrame ReadRowGroup(int rowGroupIndex)
    {
        gate.Wait();
        try
        {
            var reader = lazyReader.Value;
            var fields = reader.Schema.GetDataFields();
            using var rowGroupReader = reader.OpenRowGroupReader(rowGroupIndex);
            var columns = ReadRowGroupColumns(rowGroupReader, fields, reader.CustomMetadata);
            return NivaraFrame.Create(columns);
        }
        finally
        {
            gate.Release();
        }
    }

    private async Task<NivaraFrame> ReadRowGroupAsync(int rowGroupIndex, CancellationToken ct)
    {
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var reader = lazyReader.Value;
            var fields = reader.Schema.GetDataFields();
            using var rowGroupReader = reader.OpenRowGroupReader(rowGroupIndex);
            var columns = await ReadRowGroupColumnsAsync(rowGroupReader, fields, reader.CustomMetadata, ct).ConfigureAwait(false);
            return NivaraFrame.Create(columns);
        }
        finally
        {
            gate.Release();
        }
    }

    private static IReadOnlyDictionary<string, IColumn> ReadRowGroupColumns(Parquet.ParquetRowGroupReader rowGroupReader, DataField[] fields, IReadOnlyDictionary<string, string>? clrTypeMetadata)
    {
        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < fields.Length; i++)
        {
            var field = fields[i];
            if (field.Name == "_empty")
                continue;

            var data = NivaraParquetReader.ReadParquetColumn(rowGroupReader, field);
            var column = NivaraParquetReader.CreateNivaraColumnFromParquetData(data, field, clrTypeMetadata);
            columns[field.Name] = column;
        }

        return columns;
    }

    private static async Task<IReadOnlyDictionary<string, IColumn>> ReadRowGroupColumnsAsync(Parquet.ParquetRowGroupReader rowGroupReader, DataField[] fields, IReadOnlyDictionary<string, string>? clrTypeMetadata, CancellationToken ct)
    {
        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < fields.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            var field = fields[i];
            if (field.Name == "_empty")
                continue;

            var data = await NivaraParquetReader.ReadParquetColumnAsync(rowGroupReader, field, ct).ConfigureAwait(false);
            var column = NivaraParquetReader.CreateNivaraColumnFromParquetData(data, field, clrTypeMetadata);
            columns[field.Name] = column;
        }

        return columns;
    }

    private static IReadOnlyDictionary<string, IColumn> ToColumns(NivaraFrame frame)
        => frame.ColumnNames.ToDictionary(
            name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase);

    private static IReadOnlyDictionary<string, IColumn> ToColumns(List<NivaraFrame> frames)
    {
        if (frames.Count == 0)
            return new Dictionary<string, IColumn>();

        if (frames.Count == 1)
            return ToColumns(frames[0]);

        var merged = NivaraParquetWriter.ConcatenateFrames(frames);
        return ToColumns(merged);
    }

    private static IReadOnlyDictionary<string, IColumn> ToColumnsFromSchema(Schema schema)
    {
        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in schema.ColumnNames)
            columns[name] = CreateZeroLengthColumn(schema.GetColumnType(name));
        return columns;
    }

    static IColumn CreateZeroLengthColumn(Type type)
    {
        if (type == typeof(int)) return NivaraColumn<int>.Create(Array.Empty<int>());
        if (type == typeof(long)) return NivaraColumn<long>.Create(Array.Empty<long>());
        if (type == typeof(float)) return NivaraColumn<float>.Create(Array.Empty<float>());
        if (type == typeof(double)) return NivaraColumn<double>.Create(Array.Empty<double>());
        if (type == typeof(string)) return NivaraColumn<string>.Create(Array.Empty<string>());
        if (type == typeof(bool)) return NivaraColumn<bool>.Create(Array.Empty<bool>());
        if (type == typeof(short)) return NivaraColumn<short>.Create(Array.Empty<short>());
        if (type == typeof(byte)) return NivaraColumn<byte>.Create(Array.Empty<byte>());
        if (type == typeof(decimal)) return NivaraColumn<decimal>.Create(Array.Empty<decimal>());
        if (type == typeof(DateTime)) return NivaraColumn<DateTime>.Create(Array.Empty<DateTime>());
        if (type == typeof(Guid)) return NivaraColumn<Guid>.Create(Array.Empty<Guid>());
        return NivaraColumn<int>.Create(Array.Empty<int>());
    }

    private Parquet.ParquetReader CreateReader()
    {
        if (!File.Exists(filePath))
            throw new DataSourceException($"Parquet file not found: '{filePath}'");

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length == 0)
            throw new DataSourceException($"Parquet file is empty: '{filePath}'");

        var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            return Parquet.ParquetReader.CreateAsync(fileStream, null, leaveStreamOpen: false, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            fileStream.Dispose();
            throw new DataSourceException($"Failed to open Parquet file '{filePath}': {ex.Message}", ex);
        }
    }

    private Schema InferSchema()
    {
        try
        {
            var reader = lazyReader.Value;
            return BuildSchema(reader.Schema.GetDataFields(), reader.CustomMetadata);
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
            return lazyReader.Value.RowGroupCount;
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
            var reader = lazyReader.Value;
            gate.Wait();
            try
            {
                int total = 0;
                for (int rg = 0; rg < reader.RowGroupCount; rg++)
                {
                    using var rowGroupReader = reader.OpenRowGroupReader(rg);
                    total += checked((int)rowGroupReader.RowCount);
                }
                return total;
            }
            finally
            {
                gate.Release();
            }
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
                // Parquet.Net reports string fields as ReadOnlyMemory<char>; data columns are
                // NivaraColumn<string>, so the schema must report string to stay consistent.
                var clrType = TypeMapper.IsStringType(field.ClrType) ? typeof(string) : field.ClrType;
                columnDefinitions.Add((field.Name, clrType));
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
            if (lazyReader.IsValueCreated)
            {
                try
                {
                    lazyReader.Value.DisposeAsync().AsTask().GetAwaiter().GetResult();
                }
                catch
                {
                    // Best-effort disposal; the reader owns the file stream.
                }
            }
            gate.Dispose();
        }
    }
}
