using Nivara.Linq;
using Nivara.Query;

namespace Nivara.IO;

/// <summary>
/// Static class providing JSON-related factory methods and operations
/// </summary>
public static class Json
{
    /// <summary>
    /// Reads a JSON file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A NivaraFrame containing the JSON data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the JSON file cannot be read</exception>
    public static NivaraFrame ReadFrame(string filePath, JsonOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"JSON file not found: {filePath}");

        var source = new JsonEagerSource(filePath, options ?? JsonOptions.Default);
        return NivaraFrame.Create(source.Execute());
    }

    /// <summary>
    /// Creates a lazy query frame that scans a JSON file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the JSON file</param>
    /// <param name="options">Optional JSON reading options</param>
    /// <returns>A QueryFrame that will read the JSON when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the JSON file doesn't exist</exception>
    internal static QueryFrame ScanFrame(string filePath, JsonOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"JSON file not found: {filePath}");

        var source = new JsonLazySource(filePath, options ?? JsonOptions.Default);
        return new QueryFrame(source);
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
    public static NivaraQuery<T> ScanQuery<T>(string filePath, JsonOptions? options = null)
        where T : class, new()
    {
        return NivaraTypedLinqExtensions.FromFrame<T>(ScanFrame(filePath, options));
    }
}
