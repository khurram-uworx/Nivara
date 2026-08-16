using Nivara.Query;

namespace Nivara.Helpers;

/// <summary>
/// A query source that wraps in-memory column data as a non-owning view.
/// It never owns the wrapped columns: <see cref="Dispose"/> only invalidates the
/// source, and <see cref="Execute"/> returns fresh column instances (zero-copy
/// slices over the same backing storage) so result frames own independent
/// disposal. See issue #279.
/// </summary>
sealed class MemoryQuerySource : IQuerySource
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

        // Return fresh column instances (zero-copy slices over the same backing storage)
        // so result frames built from this source own independent disposal. Sharing the
        // caller's column instances would let a collected result's Dispose() mark the
        // source frame's columns disposed. See #279.
        var fresh = new Dictionary<string, IColumn>(columns.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var (name, column) in columns)
            fresh[name] = column.Slice(0, column.Length);
        return fresh;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        disposed = true;
    }
}
