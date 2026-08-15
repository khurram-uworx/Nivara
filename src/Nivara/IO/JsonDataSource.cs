using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;
using System.Text.Json;
using System.Threading;

namespace Nivara.IO;

/// <summary>
/// Configuration options for JSON reading operations
/// </summary>
public sealed class JsonOptions
{
    /// <summary>
    /// Gets the default JSON options
    /// </summary>
    public static JsonOptions Default { get; } = new JsonOptions();

    /// <summary>
    /// Gets the JSON serializer options
    /// </summary>
    public JsonSerializerOptions SerializerOptions { get; }

    /// <summary>
    /// Gets the number of records to use for schema inference
    /// </summary>
    public int SchemaInferenceRecords { get; }

    /// <summary>
    /// Gets whether to treat the JSON as an array of objects
    /// </summary>
    public bool IsArray { get; }

    /// <summary>
    /// Initializes a new instance of JsonOptions with default values
    /// </summary>
    public JsonOptions()
    {
        SerializerOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };
        SchemaInferenceRecords = 100;
        IsArray = true;
    }

    private JsonOptions(JsonSerializerOptions serializerOptions, int schemaInferenceRecords, bool isArray)
    {
        SerializerOptions = serializerOptions;
        SchemaInferenceRecords = schemaInferenceRecords;
        IsArray = isArray;
    }

    /// <summary>
    /// Returns a copy of these options with the specified values changed
    /// </summary>
    /// <param name="serializerOptions">New serializer options, or null to keep the current value</param>
    /// <param name="schemaInferenceRecords">New schema inference record count, or null to keep the current value</param>
    /// <param name="isArray">New array mode, or null to keep the current value</param>
    /// <returns>A new JsonOptions instance</returns>
    public JsonOptions With(JsonSerializerOptions? serializerOptions = null, int? schemaInferenceRecords = null, bool? isArray = null)
    {
        return new JsonOptions(
            serializerOptions is null ? SerializerOptions : new JsonSerializerOptions(serializerOptions),
            schemaInferenceRecords ?? SchemaInferenceRecords,
            isArray ?? IsArray);
    }
}

/// <summary>
/// Lazy JSON data source that defers reading until execution
/// </summary>
sealed class JsonLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly JsonOptions options;
    private readonly Lazy<Schema> lazySchema;
    private readonly Lazy<JsonElement[]> lazyRecords;
    private readonly DeferredErrorHandler errorHandler;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of JsonLazySource
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">The JSON reading options</param>
    public JsonLazySource(string filePath, JsonOptions options)
    {
        this.filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        this.options = options ?? throw new ArgumentNullException(nameof(options));
        this.errorHandler = new DeferredErrorHandler();

        lazySchema = new Lazy<Schema>(InferSchemaWithErrorHandling);
        lazyRecords = new Lazy<JsonElement[]>(LoadRecords);
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
    /// Gets the estimated row count based on file size and average bytes per record.
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

                var schema = lazySchema.Value;
                if (schema == null || schema.ColumnNames.Count == 0)
                    return null;

                // Heuristic: ~50 bytes per JSON object per field
                var estimatedBytesPerRecord = schema.ColumnNames.Count * 50;
                if (estimatedBytesPerRecord == 0)
                    return null;

                var estimatedRows = (int)(fileInfo.Length / estimatedBytesPerRecord);
                return Math.Max(0, estimatedRows);
            }
            catch
            {
                return null;
            }
        }
    }

    private JsonElement[] LoadRecords()
    {
        var jsonText = File.ReadAllText(filePath);
        if (string.IsNullOrWhiteSpace(jsonText))
            return Array.Empty<JsonElement>();

        try
        {
            if (options.IsArray)
            {
                var records = JsonSerializer.Deserialize<JsonElement[]>(jsonText, options.SerializerOptions);
                return records ?? Array.Empty<JsonElement>();
            }
            else
            {
                var record = JsonSerializer.Deserialize<JsonElement>(jsonText, options.SerializerOptions);
                return new[] { record };
            }
        }
        catch (JsonException)
        {
            throw;
        }
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Check for deferred errors first
        errorHandler.ThrowIfHasDeferredErrors("JSON data source execution");

        if (!File.Exists(filePath))
            throw new DataSourceException($"JSON file not found: '{filePath}'");

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length == 0)
            throw new DataSourceException($"{filePath}: JSON file is empty");

        var columns = ReadAllChunks();
        return columns;
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
                CopyColumnToList(column, columnList);
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
            // If inference failed with a DataSourceException (schema-related problems such
            // as empty arrays or missing properties), surface it now when Schema is accessed.
            // Other exceptions related to transient file access can still be deferred.
            if (ex is DataSourceException)
                throw;

            // For lazy sources, defer non-schema errors until execution
            errorHandler.AddFileAccessError(filePath, ex, "ScanJson");

            // Return a minimal schema to allow query building to continue
            // The error will be reported when Execute() is called
            return new Schema(new[] { ("placeholder", typeof(string)) });
        }
    }

    /// <summary>
    /// Reads a chunk of records from the JSON file.
    /// </summary>
    /// <param name="chunkIndex">The zero-based index of the chunk to read</param>
    /// <param name="chunkSize">The maximum number of records in the chunk</param>
    /// <returns>A dictionary of columns for the chunk</returns>
    public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        errorHandler.ThrowIfHasDeferredErrors("JSON chunk reading");

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        try
        {
            var schema = Schema;
            var columnNames = schema.ColumnNames;
            var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

            var allRecords = lazyRecords.Value;
            var start = chunkIndex * chunkSize;
            if (start >= allRecords.Length)
                return new Dictionary<string, IColumn>();

            var end = Math.Min(start + chunkSize, allRecords.Length);
            var chunkRecords = allRecords[start..end];

            if (chunkRecords.Length == 0)
                return new Dictionary<string, IColumn>();

            var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                var columnType = columnTypes[columnName];
                var values = new object?[chunkRecords.Length];
                for (int i = 0; i < chunkRecords.Length; i++)
                {
                    object? value = null;
                    if (chunkRecords[i].ValueKind == JsonValueKind.Object &&
                        chunkRecords[i].TryGetProperty(columnName, out var property))
                    {
                        value = ConvertJsonValue(property, columnType);
                    }
                    values[i] = value;
                }
                columns[columnName] = ColumnFactory.Create(columnType, values);
            }

            return columns;
        }
        catch (JsonException ex)
        {
            throw new DataSourceException($"JSON parsing error in file '{filePath}': {ex.Message}", ex);
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read JSON chunk from file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Reads a chunk of records from the JSON file asynchronously.
    /// </summary>
    public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        errorHandler.ThrowIfHasDeferredErrors("JSON chunk reading");

        if (chunkIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(chunkIndex));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            var schema = Schema;
            var columnNames = schema.ColumnNames;
            var columnTypes = columnNames.ToDictionary(name => name, schema.GetColumnType, StringComparer.OrdinalIgnoreCase);

            // Ensure records are loaded (lazy, cached)
            var allRecords = lazyRecords.Value;

            var start = chunkIndex * chunkSize;
            if (start >= allRecords.Length)
                return new Dictionary<string, IColumn>();

            var end = Math.Min(start + chunkSize, allRecords.Length);
            var chunkRecords = allRecords[start..end];

            if (chunkRecords.Length == 0)
                return new Dictionary<string, IColumn>();

            var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var columnName in columnNames)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var columnType = columnTypes[columnName];
                var values = new object?[chunkRecords.Length];
                for (int i = 0; i < chunkRecords.Length; i++)
                {
                    object? value = null;
                    if (chunkRecords[i].ValueKind == JsonValueKind.Object &&
                        chunkRecords[i].TryGetProperty(columnName, out var property))
                    {
                        value = ConvertJsonValue(property, columnType);
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
        catch (JsonException ex)
        {
            throw new DataSourceException($"JSON parsing error in file '{filePath}': {ex.Message}", ex);
        }
        catch (DataSourceException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read JSON chunk from file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Infers the schema from the JSON file by reading a sample of records
    /// </summary>
    /// <returns>The inferred schema</returns>
    private Schema InferSchema()
    {
        try
        {
            // Check file existence and accessibility first
            if (!File.Exists(filePath))
            {
                throw new DataSourceException($"JSON file not found: '{filePath}'");
            }

            string jsonText;
            try
            {
                jsonText = File.ReadAllText(filePath);
            }
            catch (UnauthorizedAccessException ex)
            {
                throw new DataSourceException($"Access denied to JSON file '{filePath}'. Check file permissions.", ex);
            }
            catch (DirectoryNotFoundException ex)
            {
                throw new DataSourceException($"Directory not found for JSON file '{filePath}': {ex.Message}", ex);
            }
            catch (FileNotFoundException ex)
            {
                throw new DataSourceException($"JSON file not found: '{filePath}': {ex.Message}", ex);
            }
            catch (IOException ex)
            {
                throw new DataSourceException($"IO error reading JSON file '{filePath}': {ex.Message}", ex);
            }

            if (string.IsNullOrWhiteSpace(jsonText))
            {
                throw new DataSourceException($"JSON file is empty or contains only whitespace: '{filePath}'");
            }

            JsonElement[] sampleRecords;

            try
            {
                if (options.IsArray)
                {
                    var allRecords = JsonSerializer.Deserialize<JsonElement[]>(jsonText, options.SerializerOptions);
                    if (allRecords == null)
                    {
                        throw new DataSourceException($"Failed to deserialize JSON array from file '{filePath}': result was null");
                    }
                    sampleRecords = allRecords.Take(options.SchemaInferenceRecords).ToArray();
                }
                else
                {
                    var record = JsonSerializer.Deserialize<JsonElement>(jsonText, options.SerializerOptions);
                    sampleRecords = new[] { record };
                }
            }
            catch (JsonException ex)
            {
                throw new DataSourceException($"JSON parsing error in file '{filePath}': {ex.Message}", ex);
            }
            catch (NotSupportedException ex)
            {
                throw new DataSourceException($"Unsupported JSON format in file '{filePath}': {ex.Message}", ex);
            }

            if (sampleRecords.Length == 0)
            {
                throw new DataSourceException($"No records found in JSON file for schema inference: '{filePath}'");
            }

            // Validate that records are objects
            for (int i = 0; i < sampleRecords.Length; i++)
            {
                if (sampleRecords[i].ValueKind != JsonValueKind.Object)
                {
                    throw new DataSourceException($"Record {i + 1} in JSON file '{filePath}' is not an object. Expected JSON objects for tabular data.");
                }
            }

            // Get all property names from sample records
            var allPropertyNames = new HashSet<string>();
            foreach (var record in sampleRecords)
            {
                foreach (var property in record.EnumerateObject())
                {
                    allPropertyNames.Add(property.Name);
                }
            }

            if (allPropertyNames.Count == 0)
            {
                throw new DataSourceException($"No properties found in JSON records from file '{filePath}'");
            }

            // Infer types for each property
            var columnDefinitions = new List<(string Name, Type Type)>();

            foreach (var propertyName in allPropertyNames)
            {
                try
                {
                    var inferredType = InferPropertyType(sampleRecords, propertyName);
                    columnDefinitions.Add((propertyName, inferredType));
                }
                catch (Exception ex)
                {
                    throw new DataSourceException($"Type inference failed for property '{propertyName}' in JSON file '{filePath}': {ex.Message}", ex);
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
            throw new DataSourceException($"Failed to infer schema from JSON file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Infers the type of a property from sample JSON records
    /// </summary>
    /// <param name="sampleRecords">Sample records to analyze</param>
    /// <param name="propertyName">The name of the property</param>
    /// <returns>The inferred type</returns>
    private static Type InferPropertyType(JsonElement[] sampleRecords, string propertyName)
    {
        var values = new List<JsonElement>();

        foreach (var record in sampleRecords)
        {
            if (record.ValueKind == JsonValueKind.Object &&
                record.TryGetProperty(propertyName, out var property) &&
                property.ValueKind != JsonValueKind.Null)
            {
                values.Add(property);
            }
        }

        if (values.Count == 0)
            return typeof(string); // Default to string if no values

        // Check if all values are of the same JSON type
        var firstKind = values[0].ValueKind;
        if (values.All(v => v.ValueKind == firstKind))
        {
            return firstKind switch
            {
                JsonValueKind.Number => typeof(double), // Use double for all numbers
                JsonValueKind.String => typeof(string),
                JsonValueKind.True or JsonValueKind.False => typeof(bool),
                _ => typeof(string) // Default for arrays, objects, etc.
            };
        }

        return typeof(string); // Default to string for mixed types
    }

    /// <summary>
    /// Converts a JSON value to the specified type
    /// </summary>
    /// <param name="jsonElement">The JSON element to convert</param>
    /// <param name="targetType">The target type</param>
    /// <returns>The converted value</returns>
    private static object? ConvertJsonValue(JsonElement jsonElement, Type targetType)
    {
        if (jsonElement.ValueKind == JsonValueKind.Null)
            return GetDefaultValue(targetType);

        try
        {
            if (targetType == typeof(string))
                return jsonElement.GetString();

            if (targetType == typeof(int))
                return jsonElement.GetInt32();

            if (targetType == typeof(double))
                return jsonElement.GetDouble();

            if (targetType == typeof(bool))
                return jsonElement.GetBoolean();

            if (targetType == typeof(DateTime))
                return jsonElement.GetDateTime();

            // Fallback to string representation
            return jsonElement.ToString();
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
    private static void CopyColumnToList(IColumn column, List<object?> accumulator)
    {
        for (int i = 0; i < column.Length; i++)
            accumulator.Add(column.GetValue(i));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // JsonLazySource doesn't hold any unmanaged resources
            // The errorHandler and lazySchema don't require disposal
            disposed = true;
        }
    }
}

/// <summary>
/// Eager JSON data source that reads immediately
/// </summary>
sealed class JsonEagerSource : IQuerySource
{
    private readonly JsonLazySource lazySource;
    private readonly Lazy<IReadOnlyDictionary<string, IColumn>> lazyColumns;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of JsonEagerSource
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">The JSON reading options</param>
    public JsonEagerSource(string filePath, JsonOptions options)
    {
        lazySource = new JsonLazySource(filePath, options);
        // Eagerly validate common empty-file/empty-array cases before delegating to lazy execution.
        lazyColumns = new Lazy<IReadOnlyDictionary<string, IColumn>>(() =>
        {
            // Check file existence and basic accessibility
            if (!File.Exists(filePath))
                throw new DataSourceException($"JSON file not found: '{filePath}'");

            FileInfo fileInfo;
            try
            {
                fileInfo = new FileInfo(filePath);
            }
            catch (Exception ex)
            {
                throw new DataSourceException($"Cannot access JSON file '{filePath}': {ex.Message}", ex);
            }

            if (fileInfo.Length == 0)
                throw new DataSourceException($"JSON file is empty: '{filePath}'");

            string jsonText;
            try
            {
                jsonText = File.ReadAllText(filePath);
            }
            catch (Exception ex)
            {
                throw new DataSourceException($"IO error reading JSON file '{filePath}': {ex.Message}", ex);
            }

            if (string.IsNullOrWhiteSpace(jsonText))
                throw new DataSourceException($"JSON file is empty or contains only whitespace: '{filePath}'");

            // If JSON is an array, ensure there is at least one record for schema inference
            if (options.IsArray)
            {
                try
                {
                    var records = JsonSerializer.Deserialize<JsonElement[]>(jsonText, options.SerializerOptions);
                    if (records == null || records.Length == 0)
                    {
                        throw new DataSourceException($"No records found in JSON file for schema inference: '{filePath}'");
                    }
                }
                catch (JsonException ex)
                {
                    throw new DataSourceException($"JSON parsing error in file '{filePath}': {ex.Message}", ex);
                }
            }

            // Delegate to lazy execution for the full processing
            return lazySource.Execute();
        });
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
