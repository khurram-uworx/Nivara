namespace Nivara.IO;

/// <summary>
/// Extension methods for CSV operations that require third-party dependencies
/// </summary>
public static class CsvExtensions
{
    /// <summary>
    /// Creates a lazy query frame that scans a CSV file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A QueryFrame that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static IQuerySource ScanCsv(string filePath, CsvOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"CSV file not found: {filePath}");

        return new CsvLazySource(filePath, options ?? CsvOptions.Default);
    }

    /// <summary>
    /// Reads a CSV file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A Frame containing the CSV data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the CSV file cannot be read</exception>
    public static IReadOnlyDictionary<string, IColumn> ReadCsv(string filePath, CsvOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"CSV file not found: {filePath}");

        var source = new CsvEagerSource(filePath, options ?? CsvOptions.Default);
        return source.Execute();
    }

    /// <summary>
    /// Creates a lazy query frame that scans a CSV file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A QueryFrame that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static QueryFrame ScanCsvAsQueryFrame(string filePath, CsvOptions? options = null)
    {
        var source = ScanCsv(filePath, options);
        return new QueryFrame(source);
    }

    /// <summary>
    /// Reads a CSV file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A NivaraFrame containing the CSV data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the CSV file cannot be read</exception>
    public static NivaraFrame ReadCsvAsFrame(string filePath, CsvOptions? options = null)
    {
        var columns = ReadCsv(filePath, options);
        return NivaraFrame.Create(columns);
    }
}

/// <summary>
/// Static class providing CSV-related factory methods
/// </summary>
public static class Csv
{
    /// <summary>
    /// Creates a lazy query source that scans a CSV file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A query source that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static IQuerySource Scan(string filePath, CsvOptions? options = null)
    {
        return CsvExtensions.ScanCsv(filePath, options);
    }

    /// <summary>
    /// Reads a CSV file immediately and returns the columns
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A dictionary of columns from the CSV data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the CSV file cannot be read</exception>
    public static IReadOnlyDictionary<string, IColumn> Read(string filePath, CsvOptions? options = null)
    {
        return CsvExtensions.ReadCsv(filePath, options);
    }

    /// <summary>
    /// Creates a lazy query frame that scans a CSV file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A QueryFrame that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static QueryFrame ScanAsQueryFrame(string filePath, CsvOptions? options = null)
    {
        return CsvExtensions.ScanCsvAsQueryFrame(filePath, options);
    }

    /// <summary>
    /// Reads a CSV file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A NivaraFrame containing the CSV data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the CSV file cannot be read</exception>
    public static NivaraFrame ReadAsFrame(string filePath, CsvOptions? options = null)
    {
        return CsvExtensions.ReadCsvAsFrame(filePath, options);
    }
}