namespace Nivara;

/// <summary>
/// A query source that wraps in-memory column data
/// </summary>
internal sealed class MemoryQuerySource : IQuerySource
{
    readonly IReadOnlyDictionary<string, IColumn> columns;
    readonly Schema schema;
    bool disposed;

    /// <summary>
    /// Initializes a new instance of MemoryQuerySource
    /// </summary>
    /// <param name="columns">The in-memory columns</param>
    /// <param name="schema">The schema of the columns</param>
    /// <exception cref="ArgumentNullException">Thrown when columns or schema is null</exception>
    public MemoryQuerySource(IReadOnlyDictionary<string, IColumn> columns, Schema schema)
    {
        this.columns = columns ?? throw new ArgumentNullException(nameof(columns));
        this.schema = schema ?? throw new ArgumentNullException(nameof(schema));
    }

    /// <inheritdoc />
    public Schema Schema
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return schema;
        }
    }

    /// <inheritdoc />
    public bool IsLazy => false; // Memory sources are already materialized

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        
        // Memory sources just return their columns directly
        return columns;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Dispose all columns
            foreach (var column in columns.Values)
            {
                column?.Dispose();
            }
            disposed = true;
        }
    }
}