namespace Nivara.IO;

/// <summary>
/// Extension methods for JSON operations using built-in .NET functionality
/// </summary>
public static class JsonExtensions
{
    /// <summary>
    /// Creates a lazy query frame that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A QueryFrame that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    public static IQuerySource ScanJson(string filePath, JsonOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"JSON file not found: {filePath}");

        return new JsonLazySource(filePath, options ?? JsonOptions.Default);
    }

    /// <summary>
    /// Reads a JSON file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A Frame containing the JSON data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the JSON file cannot be read</exception>
    public static IReadOnlyDictionary<string, IColumn> ReadJson(string filePath, JsonOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"JSON file not found: {filePath}");

        var source = new JsonEagerSource(filePath, options ?? JsonOptions.Default);
        return source.Execute();
    }
}

/// <summary>
/// Static class providing JSON-related factory methods
/// </summary>
public static class Json
{
    /// <summary>
    /// Creates a lazy query source that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A query source that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    public static IQuerySource Scan(string filePath, JsonOptions? options = null)
    {
        return JsonExtensions.ScanJson(filePath, options);
    }

    /// <summary>
    /// Reads a JSON file immediately and returns the columns
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A dictionary of columns from the JSON data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the JSON file cannot be read</exception>
    public static IReadOnlyDictionary<string, IColumn> Read(string filePath, JsonOptions? options = null)
    {
        return JsonExtensions.ReadJson(filePath, options);
    }
}