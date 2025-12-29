using System.Text.Json;
using Nivara.Exceptions;

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
    /// Gets or sets the JSON serializer options
    /// </summary>
    public JsonSerializerOptions SerializerOptions { get; set; } = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    /// <summary>
    /// Gets or sets the number of records to use for schema inference
    /// </summary>
    public int SchemaInferenceRecords { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to treat the JSON as an array of objects
    /// </summary>
    public bool IsArray { get; set; } = true;
}

/// <summary>
/// Lazy JSON data source that defers reading until execution
/// </summary>
internal sealed class JsonLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly JsonOptions options;
    private readonly Lazy<Schema> lazySchema;

    /// <summary>
    /// Initializes a new instance of JsonLazySource
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">The JSON reading options</param>
    public JsonLazySource(string filePath, JsonOptions options)
    {
        this.filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        this.options = options ?? throw new ArgumentNullException(nameof(options));
        
        lazySchema = new Lazy<Schema>(InferSchema);
    }

    /// <inheritdoc />
    public Schema Schema => lazySchema.Value;

    /// <inheritdoc />
    public bool IsLazy => true;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        try
        {
            var jsonText = File.ReadAllText(filePath);
            
            if (options.IsArray)
            {
                var records = JsonSerializer.Deserialize<JsonElement[]>(jsonText, options.SerializerOptions);
                return ProcessJsonRecords(records ?? Array.Empty<JsonElement>());
            }
            else
            {
                var record = JsonSerializer.Deserialize<JsonElement>(jsonText, options.SerializerOptions);
                return ProcessJsonRecords(new[] { record });
            }
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read JSON file '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Processes JSON records into columns
    /// </summary>
    /// <param name="records">The JSON records to process</param>
    /// <returns>A dictionary of columns</returns>
    private IReadOnlyDictionary<string, IColumn> ProcessJsonRecords(JsonElement[] records)
    {
        if (records.Length == 0)
        {
            // Return empty columns based on schema
            var emptyColumns = new Dictionary<string, IColumn>();
            foreach (var columnName in Schema.ColumnNames)
            {
                var columnType = Schema.GetColumnType(columnName);
                emptyColumns[columnName] = CreateEmptyColumn(columnType);
            }
            return emptyColumns;
        }

        // Convert records to columns
        var columns = new Dictionary<string, IColumn>();
        
        foreach (var columnName in Schema.ColumnNames)
        {
            var columnType = Schema.GetColumnType(columnName);
            var columnData = ExtractColumnData(records, columnName, columnType);
            columns[columnName] = CreateColumn(columnData, columnType);
        }

        return columns;
    }

    /// <summary>
    /// Infers the schema from the JSON file by reading a sample of records
    /// </summary>
    /// <returns>The inferred schema</returns>
    private Schema InferSchema()
    {
        try
        {
            var jsonText = File.ReadAllText(filePath);
            JsonElement[] sampleRecords;
            
            if (options.IsArray)
            {
                var allRecords = JsonSerializer.Deserialize<JsonElement[]>(jsonText, options.SerializerOptions);
                sampleRecords = allRecords?.Take(options.SchemaInferenceRecords).ToArray() ?? Array.Empty<JsonElement>();
            }
            else
            {
                var record = JsonSerializer.Deserialize<JsonElement>(jsonText, options.SerializerOptions);
                sampleRecords = new[] { record };
            }

            if (sampleRecords.Length == 0)
            {
                throw new DataSourceException("No records found in JSON file for schema inference");
            }

            // Get all property names from sample records
            var allPropertyNames = new HashSet<string>();
            foreach (var record in sampleRecords)
            {
                if (record.ValueKind == JsonValueKind.Object)
                {
                    foreach (var property in record.EnumerateObject())
                    {
                        allPropertyNames.Add(property.Name);
                    }
                }
            }

            // Infer types for each property
            var columnDefinitions = new List<(string Name, Type Type)>();
            
            foreach (var propertyName in allPropertyNames)
            {
                var inferredType = InferPropertyType(sampleRecords, propertyName);
                columnDefinitions.Add((propertyName, inferredType));
            }

            return new Schema(columnDefinitions);
        }
        catch (Exception ex) when (!(ex is DataSourceException))
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
    /// Extracts column data from JSON records
    /// </summary>
    /// <param name="records">The records to extract from</param>
    /// <param name="propertyName">The name of the property</param>
    /// <param name="columnType">The type of the column</param>
    /// <returns>Array of column values</returns>
    private static Array ExtractColumnData(JsonElement[] records, string propertyName, Type columnType)
    {
        var array = Array.CreateInstance(columnType, records.Length);
        
        for (int i = 0; i < records.Length; i++)
        {
            object? value = null;
            
            if (records[i].ValueKind == JsonValueKind.Object &&
                records[i].TryGetProperty(propertyName, out var property))
            {
                value = ConvertJsonValue(property, columnType);
            }
            else
            {
                value = GetDefaultValue(columnType);
            }
            
            array.SetValue(value, i);
        }
        
        return array;
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
    /// Creates an empty column of the specified type
    /// </summary>
    /// <param name="columnType">The type of the column</param>
    /// <returns>An empty column</returns>
    private static IColumn CreateEmptyColumn(Type columnType)
    {
        // Create empty typed array
        var emptyArray = Array.CreateInstance(columnType, 0);
        
        // Use reflection to call the appropriate Create method
        var createMethod = typeof(NivaraColumn<>).MakeGenericType(columnType)
            .GetMethod("Create", new[] { columnType.MakeArrayType() });
        
        if (createMethod == null)
            throw new InvalidOperationException($"Could not find Create method for type {columnType.Name}");
        
        return (IColumn)createMethod.Invoke(null, new object[] { emptyArray })!;
    }

    /// <summary>
    /// Creates a column from the specified data
    /// </summary>
    /// <param name="data">The column data</param>
    /// <param name="columnType">The type of the column</param>
    /// <returns>A column containing the data</returns>
    private static IColumn CreateColumn(Array data, Type columnType)
    {
        // Use reflection to call the appropriate Create method
        var createMethod = typeof(NivaraColumn<>).MakeGenericType(columnType)
            .GetMethod("Create", new[] { columnType.MakeArrayType() });
        
        if (createMethod == null)
            throw new InvalidOperationException($"Could not find Create method for type {columnType.Name}");
        
        return (IColumn)createMethod.Invoke(null, new object[] { data })!;
    }
}

/// <summary>
/// Eager JSON data source that reads immediately
/// </summary>
internal sealed class JsonEagerSource : IQuerySource
{
    private readonly JsonLazySource lazySource;
    private readonly Lazy<IReadOnlyDictionary<string, IColumn>> lazyColumns;

    /// <summary>
    /// Initializes a new instance of JsonEagerSource
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">The JSON reading options</param>
    public JsonEagerSource(string filePath, JsonOptions options)
    {
        lazySource = new JsonLazySource(filePath, options);
        lazyColumns = new Lazy<IReadOnlyDictionary<string, IColumn>>(lazySource.Execute);
    }

    /// <inheritdoc />
    public Schema Schema => lazySource.Schema;

    /// <inheritdoc />
    public bool IsLazy => false;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        return lazyColumns.Value;
    }
}