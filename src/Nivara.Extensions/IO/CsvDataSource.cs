using CsvHelper;
using CsvHelper.Configuration;
using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;
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
sealed class CsvLazySource : IQuerySource
{
    private readonly string filePath;
    private readonly CsvOptions options;
    private readonly Lazy<Schema> lazySchema;
    private readonly DeferredErrorHandler errorHandler;
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
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Check for deferred errors first
        errorHandler.ThrowIfHasDeferredErrors("CSV data source execution");

        try
        {
            // Check file existence and accessibility first
            if (!File.Exists(filePath))
            {
                throw new DataSourceException($"CSV file not found: '{filePath}'");
            }

            FileInfo fileInfo;
            try
            {
                fileInfo = new FileInfo(filePath);
            }
            catch (Exception ex)
            {
                throw new DataSourceException($"Cannot access CSV file '{filePath}': {ex.Message}", ex);
            }

            if (fileInfo.Length == 0)
            {
                throw new DataSourceException($"CSV file is empty: '{filePath}'");
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

            using var reader = new StringReader(fileContent);
            using var csv = new CsvReader(reader, options.ToCsvConfiguration());

            // Read all records as dynamic objects
            List<dynamic> records;
            try
            {
                records = csv.GetRecords<dynamic>().ToList();
            }
            catch (Exception ex) when (ex.GetType().Name.Contains("Csv"))
            {
                throw new DataSourceException($"CSV parsing error in file '{filePath}': {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new DataSourceException($"Unexpected error parsing CSV file '{filePath}': {ex.Message}", ex);
            }

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
        catch (DataSourceException)
        {
            // Re-throw DataSourceException as-is
            throw;
        }
        catch (Exception ex)
        {
            throw new DataSourceException($"Failed to read CSV file '{filePath}': {ex.Message}", ex);
        }
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
                while (csv.Read() && rowsRead < options.SchemaInferenceRows)
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

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // CsvLazySource doesn't hold any unmanaged resources
            // The errorHandler and lazySchema don't require disposal
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
