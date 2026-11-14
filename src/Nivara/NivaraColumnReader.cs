namespace Nivara;

/// <summary>
/// A pre-resolved, typed accessor for a single column of a <see cref="NivaraFrame"/>. The
/// nullable-element probe and the read delegate are resolved once at construction time, so
/// per-cell reads are a single delegate invocation with no dictionary lookup or reflection.
/// </summary>
/// <typeparam name="T">The expected column element type</typeparam>
public readonly struct NivaraColumnReader<T>
{
    readonly IColumn column;
    readonly Func<int, T> read;

    public NivaraColumnReader(IColumn column, string? columnName = null)
    {
        this.column = column ?? throw new ArgumentNullException(nameof(column));
        this.read = NivaraRow.CreateColumnReader<T>(column, columnName);
    }

    /// <summary>
    /// Gets the underlying column.
    /// </summary>
    public IColumn Column => column;

    /// <summary>
    /// Reads the value at the specified row index.
    /// </summary>
    public T Read(int rowIndex) => read(rowIndex);

    /// <summary>
    /// Reads the value at the specified row index.
    /// </summary>
    public T this[int rowIndex] => read(rowIndex);

    /// <summary>
    /// Determines whether the cell at the specified row index is null.
    /// </summary>
    public bool IsNull(int rowIndex) => column.IsNull(rowIndex);
}
