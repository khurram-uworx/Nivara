using Nivara.Linq;
using Nivara.Query;

namespace Nivara.IO;

/// <summary>
/// Static class providing JSON-related factory methods and operations
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
    internal static IQuerySource Scan(string filePath, JsonOptions? options = null)
    {
        return ScanJson(filePath, options);
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
        return ReadJson(filePath, options);
    }

    /// <summary>
    /// Creates a lazy query frame that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A QueryFrame that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    internal static QueryFrame ScanAsQueryFrame(string filePath, JsonOptions? options = null)
    {
        return ScanJsonAsQueryFrame(filePath, options);
    }

    /// <summary>
    /// Reads a JSON file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A NivaraFrame containing the JSON data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the JSON file cannot be read</exception>
    public static NivaraFrame ReadAsFrame(string filePath, JsonOptions? options = null)
    {
        return ReadJsonAsFrame(filePath, options);
    }

    /// <summary>
    /// Creates a lazy query frame that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A QueryFrame that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    internal static IQuerySource ScanJson(string filePath, JsonOptions? options = null)
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

    /// <summary>
    /// Creates a lazy query frame that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A QueryFrame that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    internal static QueryFrame ScanJsonAsQueryFrame(string filePath, JsonOptions? options = null)
    {
        var source = ScanJson(filePath, options);
        return new QueryFrame(source);
    }

    /// <summary>
    /// Reads a JSON file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A NivaraFrame containing the JSON data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the JSON file cannot be read</exception>
    public static NivaraFrame ReadJsonAsFrame(string filePath, JsonOptions? options = null)
    {
        var columns = ReadJson(filePath, options);
        return NivaraFrame.Create(columns);
    }

    /// <summary>
    /// Creates a lazy typed query that scans a JSON file without immediately reading it
    /// </summary>
    /// <typeparam name="T">The row type. Must be a non-primitive class whose public properties map
    /// (case-insensitively) to the file's columns with exact or nullable-compatible types.</typeparam>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A lazy typed query that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    public static NivaraQuery<T> ScanAsQuery<T>(string filePath, JsonOptions? options = null)
        where T : class, new()
    {
        return ScanJsonAsQuery<T>(filePath, options);
    }

    /// <summary>
    /// Creates a lazy typed query that scans a JSON file without immediately reading it
    /// </summary>
    /// <typeparam name="T">The row type. Must be a non-primitive class whose public properties map
    /// (case-insensitively) to the file's columns with exact or nullable-compatible types.</typeparam>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A lazy typed query that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    public static NivaraQuery<T> ScanJsonAsQuery<T>(string filePath, JsonOptions? options = null)
        where T : class, new()
    {
        return NivaraTypedLinqExtensions.FromFrame<T>(ScanJsonAsQueryFrame(filePath, options));
    }
}
