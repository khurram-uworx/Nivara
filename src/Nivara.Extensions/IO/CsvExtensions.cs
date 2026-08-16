using Nivara.Linq;
using Nivara.Query;

namespace Nivara.IO;

/// <summary>
/// Static class providing CSV-related factory methods and operations
/// </summary>
public static class Csv
{
    /// <summary>
    /// Reads a CSV file immediately and returns a frame with the data
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A NivaraFrame containing the CSV data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    /// <exception cref="DataSourceException">Thrown when the CSV file cannot be read</exception>
    public static NivaraFrame ReadFrame(string filePath, CsvOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"CSV file not found: {filePath}");

        var source = new CsvEagerSource(filePath, options ?? CsvOptions.Default);
        return NivaraFrame.Create(source.Execute());
    }

    /// <summary>
    /// Creates a lazy query frame that scans a CSV file without immediately reading it
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A QueryFrame that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    internal static QueryFrame ScanFrame(string filePath, CsvOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentNullException(nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"CSV file not found: {filePath}");

        var source = new CsvLazySource(filePath, options ?? CsvOptions.Default);
        return new QueryFrame(source);
    }

    /// <summary>
    /// Creates a lazy query frame that scans a CSV file without immediately reading it.
    /// The frame supports chunked streaming via <see cref="QueryFrame.AsStream"/> and
    /// fluent query chains (Filter/Select/Sort/...). Prefer <see cref="ScanQuery{T}"/> for
    /// typed row queries.
    /// </summary>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A QueryFrame that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static QueryFrame ScanAsQueryFrame(string filePath, CsvOptions? options = null)
        => ScanFrame(filePath, options);

    /// <summary>
    /// Creates a lazy typed query that scans a CSV file without immediately reading it
    /// </summary>
    /// <typeparam name="T">The row type. Must be a non-primitive class whose public properties map
    /// (case-insensitively) to the file's columns with exact or nullable-compatible types.</typeparam>
    /// <param name="filePath">The path to the CSV file</param>
    /// <param name="options">Optional CSV reading options</param>
    /// <returns>A lazy typed query that will read the CSV when executed</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file doesn't exist</exception>
    public static NivaraQuery<T> ScanQuery<T>(string filePath, CsvOptions? options = null)
        where T : class, new()
    {
        return NivaraTypedLinqExtensions.FromFrame<T>(ScanFrame(filePath, options));
    }
}
