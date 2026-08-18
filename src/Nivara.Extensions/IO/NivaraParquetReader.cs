using Nivara.Linq;
using Nivara.Query;
using Parquet.Schema;

namespace Nivara.IO;

/// <summary>
/// Provides Parquet reading capabilities with columnar compression and complex schema support.
/// </summary>
public static class NivaraParquetReader
{
    /// <summary>
    /// Reads a Parquet file into a NivaraFrame asynchronously.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="NivaraIOException">Thrown when file reading fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
    public static async Task<NivaraFrame> ReadParquetAsync(string filePath, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filePath);

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Parquet file not found: {filePath}");

        options ??= new ParquetReadOptions();

        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return await ReadParquetAsync(fileStream, options, cancellationToken);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is FileNotFoundException))
        {
            throw new NivaraIOException($"Failed to read Parquet file: {ex.Message}", ex)
            {
                FilePath = filePath,
                OperationContext = "ParquetReader.ReadParquetAsync"
            };
        }
    }

    /// <summary>
    /// Reads Parquet data from a stream into a NivaraFrame asynchronously.
    /// </summary>
    /// <param name="stream">The stream containing Parquet data.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="NivaraIOException">Thrown when stream reading fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
    public static async Task<NivaraFrame> ReadParquetAsync(Stream stream, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        options ??= new ParquetReadOptions();

        try
        {
            // Use low-level ParquetReader API
            await using var parquetReader = await Parquet.ParquetReader.CreateAsync(stream);

            if (parquetReader.RowGroupCount == 0)
            {
                // Create an empty frame with a dummy column
                var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
                return NivaraFrame.Create(("_empty", emptyColumn));
            }

            return await ConvertParquetToNivaraFrame(parquetReader, options, cancellationToken);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException))
        {
            throw new NivaraIOException($"Failed to read Parquet stream: {ex.Message}", ex)
            {
                OperationContext = "ParquetReader.ReadParquetAsync"
            };
        }
    }

    /// <summary>
    /// Reads a Parquet file into a NivaraFrame synchronously.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    public static NivaraFrame ReadParquet(string filePath, ParquetReadOptions? options = null)
    {
        return ReadParquetAsync(filePath, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Creates a lazy query frame that scans a Parquet file without immediately reading it.
    /// The returned frame supports chunked reading at row-group boundaries.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <returns>A QueryFrame that will read the Parquet file when executed</returns>
    internal static QueryFrame ScanFrame(string filePath, ParquetReadOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Parquet file not found: {filePath}");

        var source = new ParquetLazySource(filePath, options);
        return new QueryFrame(source);
    }

    /// <summary>
    /// Creates a lazy query frame that scans a Parquet file without immediately reading it.
    /// The frame supports chunked streaming at row-group boundaries via
    /// <see cref="QueryFrame.AsStream"/> and fluent query chains (Filter/Select/Sort/...).
    /// Prefer <see cref="ScanQuery{T}"/> for typed row queries.
    /// </summary>
    /// <remarks>
    /// The source reuses a single Parquet reader (footer metadata parsed once) for the frame's
    /// lifetime, so the file handle stays open until the returned frame is disposed. Use
    /// <c>using</c> (or dispose the frame / <see cref="ScanQuery{T}"/>'s frame via
    /// <c>AsQueryFrame()</c>) to release the file — important before deleting or replacing it.
    /// </remarks>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <returns>A QueryFrame that will read the Parquet file when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    public static QueryFrame ScanAsQueryFrame(string filePath, ParquetReadOptions? options = null)
        => ScanFrame(filePath, options);

    /// <summary>
    /// Creates a lazy typed query that scans a Parquet file without immediately reading it.
    /// </summary>
    /// <remarks>
    /// The underlying frame holds the file open until disposed (reused single reader); dispose it
    /// via <c>AsQueryFrame()</c> when done, e.g. before deleting the file.
    /// </remarks>
    /// <typeparam name="T">The row type. Must be a non-primitive class whose public properties map
    /// (case-insensitively) to the file's columns with exact or nullable-compatible types.</typeparam>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <returns>A lazy typed query that will read the Parquet file when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    public static NivaraQuery<T> ScanQuery<T>(string filePath, ParquetReadOptions? options = null)
        where T : class, new()
    {
        return NivaraTypedLinqExtensions.FromFrame<T>(ScanFrame(filePath, options));
    }

    /// <summary>
    /// Reads Parquet data from a stream into a NivaraFrame synchronously.
    /// </summary>
    /// <param name="stream">The stream containing Parquet data.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    public static NivaraFrame ReadParquet(Stream stream, ParquetReadOptions? options = null)
    {
        return ReadParquetAsync(stream, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Reads a Parquet file in streaming mode for large files with memory management.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="memoryBudget">Maximum memory budget for streaming operations.</param>
    /// <returns>An enumerable of NivaraFrame chunks.</returns>
    public static IEnumerable<NivaraFrame> ReadParquetStreaming(string filePath, ParquetReadOptions? options = null, long memoryBudget = 256L * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(filePath);

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Parquet file not found: {filePath}");

        options ??= new ParquetReadOptions();

        return ReadParquetStreamingInternal(filePath, options, memoryBudget);
    }

    /// <summary>
    /// Internal implementation of streaming Parquet reading.
    /// Yields one frame per row group using a single reader whose footer metadata is parsed once.
    /// </summary>
    private static IEnumerable<NivaraFrame> ReadParquetStreamingInternal(string filePath, ParquetReadOptions options, long memoryBudget)
    {
        using var bufferManager = new StreamingBufferManager(memoryBudget);

        var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        Parquet.ParquetReader reader;
        try
        {
            reader = Parquet.ParquetReader.CreateAsync(fileStream, null, leaveStreamOpen: false, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        catch
        {
            fileStream.Dispose();
            throw;
        }

        try
        {
            var schema = reader.Schema;
            var dataFields = schema.GetDataFields();
            var clrTypeMetadata = reader.CustomMetadata;

            if (options.ValidateSchema)
                ValidateParquetSchema(schema);

            bool yieldedAnyFrame = false;
            for (int rg = 0; rg < reader.RowGroupCount; rg++)
            {
                // Force garbage collection if memory usage is high
                bufferManager.TryCollectGarbage();

                using var rowGroupReader = reader.OpenRowGroupReader(rg);
                var columns = new List<(string Name, IColumn Column)>(dataFields.Length);
                foreach (var field in dataFields)
                {
                    if (field.Name == "_empty")
                        continue;

                    var data = ReadParquetColumn(rowGroupReader, field);
                    var column = CreateNivaraColumnFromParquetData(data, field, clrTypeMetadata);
                    columns.Add((field.Name, column));
                }

                if (columns.Count > 0)
                {
                    yieldedAnyFrame = true;
                    yield return NivaraFrame.Create(columns.ToArray());
                }
            }

            // An empty file (no row groups, or only the dummy column) still yields one empty frame
            // so callers can rely on at least one frame for schema discovery.
            if (!yieldedAnyFrame)
            {
                var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
                yield return NivaraFrame.Create(("_empty", emptyColumn));
            }
        }
        finally
        {
            reader.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    /// <summary>
    /// Converts a Parquet file to a NivaraFrame using low-level API.
    /// </summary>
    private static async Task<NivaraFrame> ConvertParquetToNivaraFrame(Parquet.ParquetReader parquetReader, ParquetReadOptions options, CancellationToken cancellationToken)
    {
        var schema = parquetReader.Schema;
        var dataFields = schema.GetDataFields();
        var clrTypeMetadata = parquetReader.CustomMetadata;

        if (dataFields.Length == 0)
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        // Check if this is an empty file with just the dummy column
        if (dataFields.Length == 1 && dataFields[0].Name == "_empty")
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        // Validate schema if requested
        if (options.ValidateSchema)
        {
            ValidateParquetSchema(schema);
        }

        var columns = new List<(string Name, IColumn Column)>();

        // Read all row groups so multi-row-group files (RowGroupSize) round-trip completely
        var frames = new List<NivaraFrame>(parquetReader.RowGroupCount);
        for (int rowGroupIndex = 0; rowGroupIndex < parquetReader.RowGroupCount; rowGroupIndex++)
        {
            using var rowGroupReader = parquetReader.OpenRowGroupReader(rowGroupIndex);

            for (int columnIndex = 0; columnIndex < dataFields.Length; columnIndex++)
            {
                var field = dataFields[columnIndex];
                var columnName = field.Name;

                // Skip the dummy column used for empty files
                if (columnName == "_empty")
                    continue;

                try
                {
                    // Check for cancellation before processing each column
                    cancellationToken.ThrowIfCancellationRequested();

                    var columnData = await ReadParquetColumnAsync(rowGroupReader, field, cancellationToken);
                    var column = CreateNivaraColumnFromParquetData(columnData, field, clrTypeMetadata);
                    columns.Add((columnName, column));
                }
                catch (Exception ex)
                {
                    throw new DataCorruptionException($"Failed to read column '{columnName}': {ex.Message}", ex)
                    {
                        AffectedColumns = new[] { columnName },
                        AffectedRowRange = new Range(0, 1000) // Use a default range since we can't access ThriftMetadata
                    };
                }
            }

            if (columns.Count > 0)
            {
                frames.Add(NivaraFrame.Create(columns.ToArray()));
                columns.Clear();
            }
        }

        if (frames.Count == 0)
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        if (frames.Count == 1)
            return frames[0];

        return NivaraParquetWriter.ConcatenateFrames(frames);
    }

    /// <summary>
    /// Validates the Parquet schema for compatibility.
    /// </summary>
    private static void ValidateParquetSchema(ParquetSchema schema)
    {
        var dataFields = schema.GetDataFields();
        var unsupportedFields = new List<string>();

        foreach (var field in dataFields)
        {
            if (!IsTypeSupported(field.ClrType))
            {
                unsupportedFields.Add($"{field.Name} ({field.ClrType.Name})");
            }
        }

        if (unsupportedFields.Count > 0)
        {
            var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
            throw new SchemaValidationException($"Unsupported field types found: {string.Join(", ", unsupportedFields)}. Supported types: {supportedTypes}")
            {
                TypeMismatches = unsupportedFields,
                ExpectedSchema = $"Schema with supported types: {supportedTypes}",
                ActualSchema = $"Schema with fields: {string.Join(", ", dataFields.Select(f => $"{f.Name}:{f.ClrType.Name}"))}"
            };
        }
    }

    /// <summary>
    /// Checks if a CLR type is supported for Parquet reading.
    /// </summary>
    private static bool IsTypeSupported(Type clrType)
    {
        // Handle nullable types
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;

        return TypeMapper.IsParquetSupported(actualType);
    }

    /// <summary>
    /// Synchronous version: reads a Parquet column from a row group
    /// </summary>
    internal static Array ReadParquetColumn(Parquet.ParquetRowGroupReader rowGroupReader, DataField field)
    {
        return ReadParquetColumnAsync(rowGroupReader, field, CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Creates a NivaraColumn from Parquet column data.
    /// </summary>
    internal static async Task<Array> ReadParquetColumnAsync(Parquet.ParquetRowGroupReader rowGroupReader, DataField field, CancellationToken cancellationToken)
    {
        var length = checked((int)rowGroupReader.RowCount);
        var elementType = Nullable.GetUnderlyingType(field.ClrType) ?? field.ClrType;

        if (TypeMapper.IsStringType(elementType))
        {
            var values = new string[length];
            await rowGroupReader.ReadAsync(field, values, null, cancellationToken);
            return values;
        }

        return elementType switch
        {
            Type t when t == typeof(bool) => await ReadParquetColumnAsync<bool>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(byte) => await ReadParquetColumnAsync<byte>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(sbyte) => await ReadParquetColumnAsync<sbyte>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(short) => await ReadParquetColumnAsync<short>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(ushort) => await ReadParquetColumnAsync<ushort>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(int) => await ReadParquetColumnAsync<int>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(uint) => await ReadParquetColumnAsync<uint>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(long) => await ReadParquetColumnAsync<long>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(ulong) => await ReadParquetColumnAsync<ulong>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(float) => await ReadParquetColumnAsync<float>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(double) => await ReadParquetColumnAsync<double>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(decimal) => await ReadParquetColumnAsync<decimal>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(DateTime) => await ReadParquetColumnAsync<DateTime>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(DateOnly) => await ReadParquetColumnAsync<DateOnly>(rowGroupReader, field, length, cancellationToken),
            Type t when t == typeof(Guid) => await ReadParquetColumnAsync<Guid>(rowGroupReader, field, length, cancellationToken),
            _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
        };
    }

    internal static async Task<Array> ReadParquetColumnAsync<T>(Parquet.ParquetRowGroupReader rowGroupReader, DataField field, int length, CancellationToken cancellationToken)
        where T : struct
    {
        if (field.IsNullable)
        {
            var nullableValues = new T?[length];
            await rowGroupReader.ReadAsync<T>(field, nullableValues, null, cancellationToken);
            return nullableValues;
        }

        var values = new T[length];
        await rowGroupReader.ReadAsync<T>(field, values, null, cancellationToken);
        return values;
    }

    internal static IColumn CreateNivaraColumnFromParquetData(Array columnData, DataField field, IReadOnlyDictionary<string, string>? clrTypeMetadata)
    {
        var elementType = Nullable.GetUnderlyingType(field.ClrType) ?? field.ClrType;

        // Restore the original CLR type when the file was written by Nivara (widened types)
        var originalType = TypeMapper.ResolveMetadataClrType(clrTypeMetadata, field.Name);

        if (originalType == typeof(Half)) return CreateConvertedColumn<Half>(columnData, static v => v is float f ? (Half)f : null);
        if (originalType == typeof(nint)) return CreateConvertedColumn<nint>(columnData, static v => v is long l ? (nint)l : null);
        if (originalType == typeof(nuint)) return CreateConvertedColumn<nuint>(columnData, static v => v is ulong ul ? (nuint)ul : null);
        if (originalType == typeof(char)) return CreateConvertedColumn<char>(columnData, static v => v is ushort us ? (char)us : null);
        // Parquet.Net reports Date logical fields with ClrType DateTime, so restore DateOnly from the date part
        if (originalType == typeof(DateOnly)) return CreateConvertedColumn<DateOnly>(columnData, static v => v is DateTime dt ? DateOnly.FromDateTime(dt) : null);
        // The stored DateTime is always UTC (the writer converts to UtcDateTime); interpret the
        // read-back value as UTC regardless of the Kind Parquet.Net assigns to it
        if (originalType == typeof(DateTimeOffset)) return CreateConvertedColumn<DateTimeOffset>(columnData, static v => v is DateTime dt ? new DateTimeOffset(DateTime.SpecifyKind(dt, DateTimeKind.Utc), TimeSpan.Zero) : null);
        if (originalType == typeof(TimeSpan)) return CreateConvertedColumn<TimeSpan>(columnData, static v => v is long l ? TimeSpan.FromTicks(l) : null);
        if (originalType == typeof(TimeOnly)) return CreateConvertedColumn<TimeOnly>(columnData, static v => v is long l ? TimeOnly.FromTimeSpan(TimeSpan.FromTicks(l / 100)) : null);

        return elementType switch
        {
            Type t when t == typeof(bool) => CreateNivaraColumn<bool>(columnData),
            Type t when t == typeof(byte) => CreateNivaraColumn<byte>(columnData),
            Type t when t == typeof(sbyte) => CreateNivaraColumn<sbyte>(columnData),
            Type t when t == typeof(short) => CreateNivaraColumn<short>(columnData),
            Type t when t == typeof(ushort) => CreateNivaraColumn<ushort>(columnData),
            Type t when t == typeof(int) => CreateNivaraColumn<int>(columnData),
            Type t when t == typeof(uint) => CreateNivaraColumn<uint>(columnData),
            Type t when t == typeof(long) => CreateNivaraColumn<long>(columnData),
            Type t when t == typeof(ulong) => CreateNivaraColumn<ulong>(columnData),
            Type t when t == typeof(float) => CreateNivaraColumn<float>(columnData),
            Type t when t == typeof(double) => CreateNivaraColumn<double>(columnData),
            Type t when t == typeof(decimal) => CreateNivaraColumn<decimal>(columnData),
            Type t when t == typeof(DateTime) => CreateNivaraColumn<DateTime>(columnData),
            Type t when t == typeof(DateOnly) => CreateNivaraColumn<DateOnly>(columnData),
            Type t when t == typeof(Guid) => CreateNivaraColumn<Guid>(columnData),
            Type t when TypeMapper.IsStringType(t) => CreateStringColumn(columnData),
            _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
        };
    }

    /// <summary>
    /// Creates a typed column from Parquet data using an explicit value converter,
    /// used to restore extended-domain types from their widened on-disk representation.
    /// </summary>
    private static NivaraColumn<T> CreateConvertedColumn<T>(Array columnData, Func<object?, T?> convert)
        where T : struct
    {
        var length = columnData.Length;
        var nullableArray = new T?[length];

        for (int i = 0; i < length; i++)
            nullableArray[i] = convert(columnData.GetValue(i));

        return NivaraColumn.CreateFromNullable(nullableArray);
    }

    /// <summary>
    /// Creates a NivaraColumn for struct types from Parquet data.
    /// Fast-paths when the array is already the correct typed array (no boxing).
    /// Falls back to the unboxed loop only for widened types where Parquet.Net
    /// returns the base CLR type (e.g. float for Half, long for nint).
    /// </summary>
    private static NivaraColumn<T> CreateNivaraColumn<T>(Array columnData) where T : struct
    {
        if (columnData is T?[] nullableArray)
            return NivaraColumn.CreateFromNullable(nullableArray);

        if (columnData is T[] typedArray)
            return NivaraColumn<T>.Create(typedArray);

        var length = columnData.Length;
        nullableArray = new T?[length];

        for (int i = 0; i < length; i++)
        {
            var value = columnData.GetValue(i);
            if (value != null)
            {
                try
                {
                    nullableArray[i] = (T)Convert.ChangeType(value, typeof(T))!;
                }
                catch
                {
                    nullableArray[i] = null;
                }
            }
        }

        return NivaraColumn.CreateFromNullable(nullableArray);
    }

    /// <summary>
    /// Creates a string column from Parquet data.
    /// Fast-paths when the array is already string[] (no boxing).
    /// </summary>
    private static NivaraColumn<string> CreateStringColumn(Array columnData)
    {
        if (columnData is string[] stringArray)
            return NivaraColumn<string>.CreateForReferenceType(stringArray);

        var length = columnData.Length;
        var values = new string[length];

        for (int i = 0; i < length; i++)
        {
            var value = columnData.GetValue(i);
            values[i] = value?.ToString()!;
        }

        return NivaraColumn<string>.CreateForReferenceType(values);
    }
}
