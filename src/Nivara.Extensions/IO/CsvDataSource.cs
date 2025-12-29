using CsvHelper;
using CsvHelper.Configuration;
using Nivara.Exceptions;
using System.Globalization;

namespace Nivara.IO;

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
    /// Gets or sets whether the CSV file has a header row
    /// </summary>
    public bool HasHeaderRecord { get; set; } = true;

    /// <summary>
    /// Gets or sets the delimiter character
    /// </summary>
    public string Delimiter { get; set; } = ",";

    /// <summary>
    /// Gets or sets the culture info for parsing
    /// </summary>
    public CultureInfo Culture { get; set; } = CultureInfo.InvariantCulture;

    /// <summary>
    /// Gets or sets the number of rows to use for schema inference
    /// </summary>
    public int SchemaInferenceRows { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to ignore blank lines
    /// </summary>
    public bool IgnoreBlankLines { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to trim whitespace from fields
    /// </summary>
    public bool TrimOptions { get; set; } = true;

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
            TrimOptions = TrimOptions ? CsvHelper.Configuration.TrimOptions.Trim : CsvHelper.Configuration.TrimOptions.None,
            MissingFieldFound = null, // Don't throw on missing fields
            HeaderValidated = null   // Don't validate headers
        };
    }
}

/// <summary>
/// Lazy CSV data source that defers reading until execution
/// </summary>
internal sealed class CsvLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly CsvOptions options;
    private readonly Lazy<Schema> lazySchema;

    /// <summary>
    /// Initializes a new instance of CsvLazySource
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">The CSV reading options</param>
    public CsvLazySource(string filePath, CsvOptions options)
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
            using var reader = new StringReader(File.ReadAllText(filePath));
            using var csv = new CsvReader(reader, options.ToCsvConfiguration());

            // Read all records as dynamic objects
            var records = csv.GetRecords<dynamic>().ToList();

            if (records.Count == 0)
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
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read CSV file '{filePath}': {ex.Message}", ex);
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
            using var reader = new StringReader(File.ReadAllText(filePath));
            using var csv = new CsvReader(reader, options.ToCsvConfiguration());

            // Read header to get column names
            csv.Read();
            csv.ReadHeader();
            var headers = csv.HeaderRecord ?? throw new DataSourceException("No headers found in CSV file");

            // Read sample rows for type inference
            var sampleRecords = new List<dynamic>();
            int rowsRead = 0;

            while (csv.Read() && rowsRead < options.SchemaInferenceRows)
            {
                sampleRecords.Add(csv.GetRecord<dynamic>());
                rowsRead++;
            }

            // Infer types for each column
            var columnDefinitions = new List<(string Name, Type Type)>();

            foreach (var header in headers)
            {
                var inferredType = InferColumnType(sampleRecords, header);
                columnDefinitions.Add((Name: header, Type: inferredType));
            }

            return new Schema(columnDefinitions);
        }
        catch (Exception ex) when (!(ex is DataSourceException))
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
    /// Extracts column data from records
    /// </summary>
    /// <param name="records">The records to extract from</param>
    /// <param name="columnName">The name of the column</param>
    /// <param name="columnType">The type of the column</param>
    /// <returns>Array of column values</returns>
    private static Array ExtractColumnData(List<dynamic> records, string columnName, Type columnType)
    {
        var array = Array.CreateInstance(columnType, records.Count);

        for (int i = 0; i < records.Count; i++)
        {
            var dict = (IDictionary<string, object>)records[i];
            object? value = null;

            if (dict.TryGetValue(columnName, out var rawValue) && rawValue != null)
            {
                var stringValue = rawValue.ToString();
                value = ConvertValue(stringValue, columnType);
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
/// Eager CSV data source that reads immediately
/// </summary>
internal sealed class CsvEagerSource : IQuerySource
{
    private readonly CsvLazySource lazySource;
    private readonly Lazy<IReadOnlyDictionary<string, IColumn>> lazyColumns;

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
    public Schema Schema => lazySource.Schema;

    /// <inheritdoc />
    public bool IsLazy => false;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        return lazyColumns.Value;
    }
}