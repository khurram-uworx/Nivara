namespace Nivara;

/// <summary>
/// Represents a data source that can be queried lazily or eagerly.
/// Provides the foundation for all data input operations in the query engine.
/// </summary>
public interface IQuerySource : IDisposable
{
    /// <summary>
    /// Gets the schema of the data source
    /// </summary>
    Schema Schema { get; }

    /// <summary>
    /// Gets a value indicating whether this source supports lazy evaluation
    /// </summary>
    bool IsLazy { get; }

    /// <summary>
    /// Executes the data source and returns the resulting columns
    /// </summary>
    /// <returns>A dictionary of column names to their corresponding NivaraColumn instances</returns>
    /// <exception cref="DataSourceException">Thrown when the data source cannot be executed</exception>
    IReadOnlyDictionary<string, IColumn> Execute();
}