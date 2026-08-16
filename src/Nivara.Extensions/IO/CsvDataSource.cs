using CsvHelper;
using CsvHelper.Configuration;
using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;
using System.Globalization;

namespace Nivara.IO;

/// <summary>
/// Configures whitespace trimming applied to CSV fields while reading
/// </summary>
public enum CsvTrimOptions
{
    /// <summary>
    /// Fields are read verbatim, preserving surrounding whitespace
    /// </summary>
    None,

    /// <summary>
    /// Leading and trailing whitespace is removed from every field
    /// </summary>
    Trim
}

/// <summary>
/// Configuration options for CSV reading operations
/// </summary>
public sealed class CsvOptions
{
    /// <summary>
    /// Gets the default CSV options
    /// </summary>
    public static CsvOptions Default { get; } = new CsvOptions();

    /// <summary>
    /// Gets whether the CSV file has a header row
    /// </summary>
    public bool HasHeaderRecord { get; }

    /// <summary>
    /// Gets the delimiter character
    /// </summary>
    public string Delimiter { get; }

    /// <summary>
    /// Gets the culture info for parsing
    /// </summary>
    public CultureInfo Culture { get; }

    /// <summary>
    /// Gets the number of rows to use for schema inference
    /// </summary>
    public int SchemaInferenceRecords { get; }

    /// <summary>
    /// Gets whether to ignore blank lines
    /// </summary>
    public bool IgnoreBlankLines { get; }

    /// <summary>
    /// Gets the whitespace trimming mode for fields
    /// </summary>
    public CsvTrimOptions TrimOptions { get; }

    /// <summary>
    /// Initializes a new instance of CsvOptions with default values
    /// </summary>
    public CsvOptions()
    {
        HasHeaderRecord = true;
        Delimiter = ",";
        Culture = CultureInfo.InvariantCulture;
        SchemaInferenceRecords = 100;
        IgnoreBlankLines = true;
        TrimOptions = CsvTrimOptions.Trim;
    }

    private CsvOptions(bool hasHeaderRecord, string delimiter, CultureInfo culture, int schemaInferenceRecords, bool ignoreBlankLines, CsvTrimOptions trimOptions)
    {
        HasHeaderRecord = hasHeaderRecord;
        Delimiter = delimiter;
        Culture = culture;
        SchemaInferenceRecords = schemaInferenceRecords;
        IgnoreBlankLines = ignoreBlankLines;
        TrimOptions = trimOptions;
    }

    /// <summary>
    /// Returns a copy of these options with the specified values changed
    /// </summary>
    /// <param name="hasHeaderRecord">New header mode, or null to keep the current value</param>
    /// <param name="delimiter">New delimiter, or null to keep the current value</param>
    /// <param name="culture">New culture, or null to keep the current value</param>
    /// <param name="schemaInferenceRecords">New schema inference record count, or null to keep the current value</param>
    /// <param name="ignoreBlankLines">New blank-line mode, or null to keep the current value</param>
    /// <param name="trimOptions">New trim mode, or null to keep the current value</param>
    /// <returns>A new CsvOptions instance</returns>
    public CsvOptions With(bool? hasHeaderRecord = null, string? delimiter = null, CultureInfo? culture = null, int? schemaInferenceRecords = null, bool? ignoreBlankLines = null, CsvTrimOptions? trimOptions = null)
    {
        return new CsvOptions(
            hasHeaderRecord ?? HasHeaderRecord,
            delimiter ?? Delimiter,
            culture ?? Culture,
            schemaInferenceRecords ?? SchemaInferenceRecords,
            ignoreBlankLines ?? IgnoreBlankLines,
            trimOptions ?? TrimOptions);
    }

    /// <summary>
    /// Creates a CsvHelper configuration from these options
    /// </summary>
    /// <returns>A CsvHelper configuration</returns>
    internal CsvConfiguration ToCsvConfiguration()
    {
        return new CsvConfiguration(Culture)
        {
            HasHeaderRecord = HasHeaderRecord,
            Delimiter = Delimiter,
            IgnoreBlankLines = IgnoreBlankLines,
            TrimOptions = TrimOptions == CsvTrimOptions.Trim
                ? CsvHelper.Configuration.TrimOptions.Trim
                : CsvHelper.Configuration.TrimOptions.None,
            MissingFieldFound = null, // Don't throw on missing fields
            HeaderValidated = null   // Don't validate headers
        };
    }
}

/// <summary>
/// Lazy CSV data source that defers reading until execution
/// </summary>
sealed class CsvLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly CsvOptions options;
    private readonly Lazy<Schema> lazySchema;
    private readonly DeferredErrorHandler errorHandler;
    private StreamReader? chunkStreamReader;
    private CsvReader? chunkCsvReader;
    private int rowsConsumed;
    private bool eofReached;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of CsvLazySource
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">The CSV reading options</param>
    public CsvLazySource(string filePath, CsvOptions options)
    {
        this.filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        this.options = options ?? throw new ArgumentNullException(nameof(options));
        this.errorHandler = new DeferredErrorHandler();

        lazySchema = new Lazy<Schema>(InferSchemaWithErrorHandling);
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
    /// Gets the estimated row count based on file size and average bytes per row
    /// from schema inference, or null if estimation is not possible.
    /// </summary>
    public int? EstimatedRowCount
    {
        get
        {
            if (!File.Exists(filePath))
                return null;

            try
            {
                var fileInfo = new FileInfo(filePath);
                if (fileInfo.Length == 0)
                    return 0;

                // Use sample data from schema inference to estimate average bytes per row
                var schema = lazySchema.Value;
                if (schema == null || !lazySchema.IsValueCreated)
                    return null;

                // Estimate average bytes per row from the header + average field length
                // Sample inference reads up to SchemaInferenceRecords rows; use file size
                // and sample-based heuristic
                var columnCount = schema.ColumnNames.Count;
                if (columnCount == 0)
                    return null;

                // Heuristic: ~10 bytes per field (conservative for CSV)
                var estimatedBytesPerRow = columnCount * 10;
                var estimatedRows = (int)(fileInfo.Length / estimatedBytesPerRow);
                return Math.Max(0, estimatedRows);
            }
            catch
            {
                return null;
            }
        }
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Check for deferred errors first
        errorHandler.ThrowIfHasDeferredErrors("CSV data source execution");

        if (!File.Exists(filePath))
            throw new DataSourceException($"CSV file not found: '{filePath}'");

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length == 0)
            throw new DataSourceException($"CSV file is empty: '{filePath}'");

        var columns = ReadAllChunks();
        return columns;
    }

    /// <inheritdoc />
    public async Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        errorHandler.ThrowIfHasDeferredErrors("CSV data source execution");

        if (!File.Exists(filePath))
            throw new DataSourceException($"CSV file not found: '{filePath}'");

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length == 0)
            throw new DataSourceException($"CSV file is empty: '{filePath}'");

        return await ReadAllChunksAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Reads all data by iterating through chunks and concatenating the results.
    /// </summary>
    private IReadOnlyDictionary<string, IColumn> ReadAllChunks()
    {
        var schema = Schema;
        var columnNames = schema.ColumnNames;
        var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

        var allColumns = new Dictionary<string, List<object?>>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in columnNames)
            allColumns[name] = new List<object?>();

        int chunkIndex = 0;
        int chunkSize = 10000;

        while (true)
        {
            var chunk = ReadChunk(chunkIndex, chunkSize);
            if (chunk == null || chunk.Count == 0)
                break;

            int rowCount = chunk.Values.FirstOrDefault()?.Length ?? 0;
            if (rowCount == 0)
                break;

            foreach (var name in columnNames)
            {
                var columnList = allColumns[name];
                var column = chunk[name];
                CopyColumnToList(column, columnTypes[name], columnList);
            }

            chunkIndex++;
        }

        if (allColumns.Values.All(l => l.Count == 0))
        {
            // Return empty columns based on schema
            var emptyColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                var columnType = columnTypes[columnName];
                emptyColumns[columnName] = ColumnFactory.Create(columnType, Array.Empty<object?>());
            }
            return emptyColumns;
        }

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in columnNames)
        {
            var columnType = columnTypes[name];
            var values = allColumns[name].ToArray();
            result[name] = ColumnFactory.Create(columnType, values);
        }

        return result;
    }

    /// <summary>
    /// Reads all data by iterating through chunks asynchronously and concatenating the results.
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ReadAllChunksAsync(CancellationToken cancellationToken)
    {
        var schema = Schema;
        var columnNames = schema.ColumnNames;
        var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

        var allColumns = new Dictionary<string, List<object?>>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in columnNames)
            allColumns[name] = new List<object?>();

        int chunkIndex = 0;
        int chunkSize = 10000;

        while (true)
        {
            var chunk = await ReadChunkAsync(chunkIndex, chunkSize, cancellationToken).ConfigureAwait(false);
            if (chunk == null || chunk.Count == 0)
                break;

            int rowCount = chunk.Values.FirstOrDefault()?.Length ?? 0;
            if (rowCount == 0)
                break;

            foreach (var name in columnNames)
            {
                var columnList = allColumns[name];
                var column = chunk[name];
                CopyColumnToList(column, columnTypes[name], columnList);
            }

            chunkIndex++;
        }

        if (allColumns.Values.All(l => l.Count == 0))
        {
            // Return empty columns based on schema
            var emptyColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                var columnType = columnTypes[columnName];
                emptyColumns[columnName] = ColumnFactory.Create(columnType, Array.Empty<object?>());
            }
            return emptyColumns;
        }

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in columnNames)
        {
            var columnType = columnTypes[name];
            var values = allColumns[name].ToArray();
            result[name] = ColumnFactory.Create(columnType, values);
        }

        return result;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        errorHandler.ThrowIfHasDeferredErrors("CSV chunk reading");

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        if (!File.Exists(filePath))
            throw new DataSourceException($"CSV file not found: '{filePath}'");

        try
        {
            var schema = Schema;
            var columnNames = schema.ColumnNames;
            var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

            if (!EnsureChunkPosition(chunkIndex, chunkSize, useAsync: false))
                return new Dictionary<string, IColumn>();

            var csv = chunkCsvReader!;

            // Read chunkSize records
            var records = new List<IDictionary<string, object>>(chunkSize);
            int rowsRead = 0;
            while (rowsRead < chunkSize)
            {
                if (!csv.Read())
                {
                    eofReached = true;
                    DisposeChunkReader();
                    break;
                }

                var recordDict = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                for (int i = 0; i < csv.HeaderRecord?.Length; i++)
                {
                    var header = csv.HeaderRecord?[i];
                    if (header != null)
                    {
                        var fieldValue = csv.GetField(i);
                        recordDict[header] = fieldValue ?? string.Empty;
                    }
                }
                records.Add(recordDict);
                rowsRead++;
                rowsConsumed++;
            }

            if (records.Count == 0)
                return new Dictionary<string, IColumn>();

            // Build columns from records
            var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                var columnType = columnTypes[columnName];
                var values = new object?[records.Count];
                for (int i = 0; i < records.Count; i++)
                {
                    object? value = null;
                    if (records[i].TryGetValue(columnName, out var rawValue) && rawValue != null)
                    {
                        var stringValue = rawValue.ToString();
                        value = ConvertValue(stringValue, columnType);
                    }
                    values[i] = value;
                }
                columns[columnName] = ColumnFactory.Create(columnType, values);
            }

            return columns;
        }
        catch (Exception ex) when (ex.GetType().Name.Contains("Csv"))
        {
            throw new DataSourceException($"CSV parsing error in file '{filePath}': {ex.Message}", ex);
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read CSV chunk from file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        errorHandler.ThrowIfHasDeferredErrors("CSV chunk reading");

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        cancellationToken.ThrowIfCancellationRequested();

        if (!File.Exists(filePath))
            throw new DataSourceException($"CSV file not found: '{filePath}'");

        try
        {
            var schema = Schema;
            var columnNames = schema.ColumnNames;
            var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

            cancellationToken.ThrowIfCancellationRequested();

            if (!EnsureChunkPosition(chunkIndex, chunkSize, useAsync: true))
                return new Dictionary<string, IColumn>();

            var csv = chunkCsvReader!;

            // Read chunkSize records
            var records = new List<IDictionary<string, object>>(chunkSize);
            int rowsRead = 0;
            while (rowsRead < chunkSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!await csv.ReadAsync().ConfigureAwait(false))
                {
                    eofReached = true;
                    DisposeChunkReader();
                    break;
                }

                var recordDict = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                for (int i = 0; i < csv.HeaderRecord?.Length; i++)
                {
                    var header = csv.HeaderRecord?[i];
                    if (header != null)
                    {
                        var fieldValue = csv.GetField(i);
                        recordDict[header] = fieldValue ?? string.Empty;
                    }
                }
                records.Add(recordDict);
                rowsRead++;
                rowsConsumed++;
            }

            if (records.Count == 0)
                return new Dictionary<string, IColumn>();

            // Build columns from records
            var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                var columnType = columnTypes[columnName];
                var values = new object?[records.Count];
                for (int i = 0; i < records.Count; i++)
                {
                    object? value = null;
                    if (records[i].TryGetValue(columnName, out var rawValue) && rawValue != null)
                    {
                        var stringValue = rawValue.ToString();
                        value = ConvertValue(stringValue, columnType);
                    }
                    values[i] = value;
                }
                columns[columnName] = ColumnFactory.Create(columnType, values);
            }

            return columns;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex.GetType().Name.Contains("Csv"))
        {
            throw new DataSourceException($"CSV parsing error in file '{filePath}': {ex.Message}", ex);
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read CSV chunk from file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Positions the persistent chunk reader so the next record read is the first data row
    /// of the requested chunk. Sequential chunk reads continue from the current position
    /// (single pass); backward access or re-reads reopen the file and skip forward.
    /// </summary>
    /// <param name="chunkIndex">The zero-based chunk index.</param>
    /// <param name="chunkSize">The number of rows per chunk.</param>
    /// <param name="useAsync">Whether to open the underlying stream with async I/O.</param>
    /// <returns>False when the file has fewer data rows than the requested chunk start.</returns>
    private bool EnsureChunkPosition(int chunkIndex, int chunkSize, bool useAsync)
    {
        var targetRow = (long)chunkIndex * chunkSize;

        if (eofReached && targetRow >= rowsConsumed)
            return false;

        if (chunkCsvReader == null || targetRow < rowsConsumed)
        {
            DisposeChunkReader();
            eofReached = false;
            var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync);
            var streamReader = new StreamReader(stream);
            var csv = new CsvReader(streamReader, options.ToCsvConfiguration());
            if (options.HasHeaderRecord && !csv.Read())
            {
                chunkStreamReader = streamReader;
                chunkCsvReader = csv;
                rowsConsumed = 0;
                eofReached = true;
                return false;
            }
            if (options.HasHeaderRecord)
                csv.ReadHeader();
            chunkStreamReader = streamReader;
            chunkCsvReader = csv;
            rowsConsumed = 0;
        }

        while (rowsConsumed < targetRow)
        {
            if (!chunkCsvReader!.Read())
            {
                eofReached = true;
                return false;
            }
            rowsConsumed++;
        }
        return true;
    }

    private void DisposeChunkReader()
    {
        chunkCsvReader?.Dispose();
        chunkCsvReader = null;
        chunkStreamReader?.Dispose();
        chunkStreamReader = null;
    }

    /// <summary>
    /// Infers schema with deferred error handling for lazy operations
    /// </summary>
    /// <returns>The inferred schema</returns>
    private Schema InferSchemaWithErrorHandling()
    {
        try
        {
            return InferSchema();
        }
        catch (Exception ex)
        {
            // For lazy sources, defer schema inference errors until execution
            errorHandler.AddFileAccessError(filePath, ex, "ScanCsv");

            // Return a minimal schema to allow query building to continue
            // The error will be reported when Execute() is called
            return new Schema(new[] { ("placeholder", typeof(string)) });
        }
    }

    /// <summary>
    /// Infers the schema from the CSV file by reading a sample of rows
    /// </summary>
    /// <returns>The inferred schema</returns>
    private Schema InferSchema()
    {
        try
        {
            // Check file existence and accessibility first
            if (!File.Exists(filePath))
            {
                throw new DataSourceException($"CSV file not found: '{filePath}'");
            }

            string fileContent;
            try
            {
                fileContent = File.ReadAllText(filePath);
            }
            catch (UnauthorizedAccessException ex)
            {
                throw new DataSourceException($"Access denied to CSV file '{filePath}'. Check file permissions.", ex);
            }
            catch (DirectoryNotFoundException ex)
            {
                throw new DataSourceException($"Directory not found for CSV file '{filePath}': {ex.Message}", ex);
            }
            catch (FileNotFoundException ex)
            {
                throw new DataSourceException($"CSV file not found: '{filePath}': {ex.Message}", ex);
            }
            catch (IOException ex)
            {
                throw new DataSourceException($"IO error reading CSV file '{filePath}': {ex.Message}", ex);
            }

            if (string.IsNullOrWhiteSpace(fileContent))
            {
                throw new DataSourceException($"CSV file is empty or contains only whitespace: '{filePath}'");
            }

            using var reader = new StringReader(fileContent);
            using var csv = new CsvReader(reader, options.ToCsvConfiguration());

            // Read header to get column names
            bool headerRead;
            try
            {
                headerRead = csv.Read();
                if (!headerRead)
                {
                    throw new DataSourceException($"CSV file contains no data: '{filePath}'");
                }

                csv.ReadHeader();
            }
            catch (Exception ex) when (ex.GetType().Name.Contains("Csv"))
            {
                throw new DataSourceException($"CSV header parsing error in file '{filePath}': {ex.Message}", ex);
            }

            var headers = csv.HeaderRecord;
            if (headers == null || headers.Length == 0)
            {
                throw new DataSourceException($"No headers found in CSV file: '{filePath}'");
            }

            // Validate headers
            for (int i = 0; i < headers.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(headers[i]))
                {
                    throw new DataSourceException($"Empty or whitespace header found at column {i + 1} in CSV file: '{filePath}'");
                }
            }

            // Check for duplicate headers
            var duplicateHeaders = headers.GroupBy(h => h, StringComparer.OrdinalIgnoreCase)
                                         .Where(g => g.Count() > 1)
                                         .Select(g => g.Key)
                                         .ToList();

            if (duplicateHeaders.Count > 0)
            {
                throw new DataSourceException($"Duplicate headers found in CSV file '{filePath}': {string.Join(", ", duplicateHeaders)}");
            }

            // Read sample rows for type inference
            var sampleRecords = new List<dynamic>();
            int rowsRead = 0;

            try
            {
                while (csv.Read() && rowsRead < options.SchemaInferenceRecords)
                {
                    sampleRecords.Add(csv.GetRecord<dynamic>());
                    rowsRead++;
                }
            }
            catch (Exception ex) when (ex.GetType().Name.Contains("Csv"))
            {
                throw new DataSourceException($"CSV data parsing error in file '{filePath}': {ex.Message}", ex);
            }

            // Infer types for each column
            var columnDefinitions = new List<(string Name, Type Type)>();

            foreach (var header in headers)
            {
                try
                {
                    var inferredType = InferColumnType(sampleRecords, header);
                    columnDefinitions.Add((Name: header, Type: inferredType));
                }
                catch (Exception ex)
                {
                    throw new DataSourceException($"Type inference failed for column '{header}' in CSV file '{filePath}': {ex.Message}", ex);
                }
            }

            return new Schema(columnDefinitions);
        }
        catch (DataSourceException)
        {
            // Re-throw DataSourceException as-is
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to infer schema from CSV file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Infers the type of a column from sample data
    /// </summary>
    /// <param name="sampleRecords">Sample records to analyze</param>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The inferred type</returns>
    private static Type InferColumnType(List<dynamic> sampleRecords, string columnName)
    {
        var values = new List<string>();

        foreach (var record in sampleRecords)
        {
            var dict = (IDictionary<string, object>)record;
            if (dict.TryGetValue(columnName, out var value) && value != null)
            {
                values.Add(value.ToString() ?? string.Empty);
            }
        }

        if (values.Count == 0)
            return typeof(string); // Default to string if no values

        // Try to infer type based on successful parsing
        if (values.All(v => int.TryParse(v, out _)))
            return typeof(int);

        if (values.All(v => double.TryParse(v, out _)))
            return typeof(double);

        if (values.All(v => bool.TryParse(v, out _)))
            return typeof(bool);

        if (values.All(v => DateTime.TryParse(v, out _)))
            return typeof(DateTime);

        return typeof(string); // Default to string
    }

    /// <summary>
    /// Converts a string value to the specified type
    /// </summary>
    /// <param name="value">The string value to convert</param>
    /// <param name="targetType">The target type</param>
    /// <returns>The converted value</returns>
    private static object? ConvertValue(string? value, Type targetType)
    {
        if (string.IsNullOrEmpty(value))
            return GetDefaultValue(targetType);

        try
        {
            if (targetType == typeof(string))
                return value;

            if (targetType == typeof(int))
                return int.Parse(value);

            if (targetType == typeof(double))
                return double.Parse(value);

            if (targetType == typeof(bool))
                return bool.Parse(value);

            if (targetType == typeof(DateTime))
                return DateTime.Parse(value);

            return Convert.ChangeType(value, targetType);
        }
        catch
        {
            return GetDefaultValue(targetType);
        }
    }

    /// <summary>
    /// Gets the default value for a type
    /// </summary>
    /// <param name="type">The type</param>
    /// <returns>The default value</returns>
    private static object? GetDefaultValue(Type type)
    {
        return type.IsValueType ? Activator.CreateInstance(type) : null;
    }

    /// <summary>
    /// Copies values from a typed column into the accumulator list for vertical concatenation.
    /// </summary>
    private static void CopyColumnToList(IColumn column, Type columnType, List<object?> accumulator)
    {
        for (int i = 0; i < column.Length; i++)
            accumulator.Add(column.GetValue(i));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            DisposeChunkReader();
            disposed = true;
        }
    }
}

/// <summary>
/// Eager CSV data source that reads immediately
/// </summary>
sealed class CsvEagerSource : IQuerySource
{
    private readonly CsvLazySource lazySource;
    private readonly Lazy<IReadOnlyDictionary<string, IColumn>> lazyColumns;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of CsvEagerSource
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">The CSV reading options</param>
    public CsvEagerSource(string filePath, CsvOptions options)
    {
        lazySource = new CsvLazySource(filePath, options);
        lazyColumns = new Lazy<IReadOnlyDictionary<string, IColumn>>(lazySource.Execute);
    }

    /// <inheritdoc />
    public Schema Schema
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return lazySource.Schema;
        }
    }

    /// <inheritdoc />
    public bool IsLazy => false;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return lazyColumns.Value;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Dispose the underlying lazy source
            lazySource?.Dispose();

            // Dispose columns if they have been materialized
            if (lazyColumns.IsValueCreated)
            {
                foreach (var column in lazyColumns.Value.Values)
                {
                    column?.Dispose();
                }
            }

            disposed = true;
        }
    }
}
