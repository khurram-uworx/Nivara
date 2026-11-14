using Nivara.Exceptions;
using System.Collections.Concurrent;
using System.Reflection;

namespace Nivara;

/// <summary>
/// A typed, allocation-free view over a single row of a <see cref="NivaraFrame"/>. Constructed by
/// the framework and passed to the predicate of <see cref="NivaraFrameExtensions.Where"/>; the
/// struct is read-only and carries the row's position plus a shared column array and name map.
/// </summary>
/// <remarks>
/// The zero state (<c>default(NivaraRow)</c>) is valid and throws a clear
/// <see cref="InvalidOperationException"/> on access so a defaulted value is never misread as data.
/// </remarks>
public readonly struct NivaraRow
{
    readonly IColumn[] columns;
    readonly string[]? columnNames;
    readonly IReadOnlyDictionary<string, int>? map;
    readonly int rowIndex;

    internal NivaraRow(IColumn[] columns, string[] columnNames, IReadOnlyDictionary<string, int> map, int rowIndex)
    {
        this.columns = columns;
        this.columnNames = columnNames;
        this.map = map;
        this.rowIndex = rowIndex;
    }

    // Legacy constructor: column names are materialised lazily (allocating) on access. Retained for
    // backwards compatibility with callers that only have a name->index map.
    internal NivaraRow(IColumn[] columns, IReadOnlyDictionary<string, int> map, int rowIndex)
    {
        this.columns = columns;
        this.columnNames = null;
        this.map = map;
        this.rowIndex = rowIndex;
    }

    /// <summary>
    /// Gets the zero-based position of this row within the source frame
    /// </summary>
    public int RowIndex => rowIndex;

    internal IColumn[] Columns => columns;

    /// <summary>
    /// Gets the column names for this row's source frame. On the framework's hot path (e.g.
    /// <see cref="NivaraFrameExtensions.Where"/>) this returns the shared, pre-built array and
    /// does not allocate; otherwise it materialises a fresh array.
    /// </summary>
    public string[] ColumnNames => columnNames is not null
        ? columnNames
        : (map is not null ? map.Keys.ToArray() : Array.Empty<string>());

    /// <summary>
    /// Gets the value at the specified column as an object, or <c>null</c> for a null cell.
    /// Boxes value-type cells; prefer <see cref="GetValue{T}"/> in hot paths.
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The value at the column, or <c>null</c> when the cell is null</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public object? this[string columnName] => GetColumn(columnName).GetValue(rowIndex);

    /// <summary>
    /// Gets the value at the specified column strongly typed. Returns the stored value on null
    /// cells (matching the <c>NivaraColumn{T}</c> indexer contract); use <see cref="IsNull"/> to
    /// detect nulls.
    /// </summary>
    /// <typeparam name="T">The expected column element type</typeparam>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The value at the column</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    /// <exception cref="ColumnTypeMismatchException">Thrown when the column element type does not match <typeparamref name="T"/></exception>
    public T GetValue<T>(string columnName)
    {
        var column = GetColumn(columnName);

        // Fast path — ordinary columns (mask-based value columns, nullable-element columns
        // requested as T?, reference-type columns) all implement IColumn<T> directly.
        if (column is IColumn<T> typed)
            return typed[rowIndex];

        // Nullable-element columns (NivaraColumn<T?>) implement IColumn<T?>, not IColumn<T>.
        // When the caller requests the underlying value type, read through a cached typed
        // delegate (built once per T) rather than reflection so predicate hot loops stay
        // allocation-free. Reference-type requesters can never satisfy this branch; the mismatch
        // throw below keeps the old contract.
        if (NullableColumnReader.IsNullableColumnFor<T>(column))
            return NullableColumnReader.Read<T>(column, rowIndex);

        throw new ColumnTypeMismatchException(columnName, typeof(T), column.ElementType);
    }

    /// <summary>
    /// Attempts to get the value at the specified column strongly typed
    /// </summary>
    /// <typeparam name="T">The expected column element type</typeparam>
    /// <param name="columnName">The name of the column</param>
    /// <param name="value">When this method returns <c>true</c>, the value at the column</param>
    /// <returns><c>true</c> when the column exists and its element type matches <typeparamref name="T"/>; otherwise <c>false</c></returns>
    public bool TryGetValue<T>(string columnName, out T value)
    {
        if (!TryGetColumn(columnName, out var column))
        {
            value = default!;
            return false;
        }

        if (column is IColumn<T> typed)
        {
            value = typed[rowIndex];
            return true;
        }

        // Nullable-element columns (NivaraColumn<T?>) implement IColumn<T?>, not IColumn<T>.
        // When the caller requests the underlying value type, treat it as a match and read via
        // the cached typed delegate.
        if (NullableColumnReader.IsNullableColumnFor<T>(column!))
        {
            value = NullableColumnReader.Read<T>(column!, rowIndex);
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>
    /// Determines whether the cell at the specified column is null
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns><c>true</c> when the cell is null; otherwise <c>false</c></returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public bool IsNull(string columnName) => GetColumn(columnName).IsNull(rowIndex);

    /// <summary>
    /// Resolves a column name to its zero-based index within this row's source frame.
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The zero-based column index</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public int GetColumnIndex(string columnName)
    {
        if (columns is null)
            throw new InvalidOperationException("NivaraRow is the default value and does not reference a frame row.");

        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        if (map is null || !map.TryGetValue(columnName, out var index))
            throw new ColumnNotFoundException(columnName, map?.Keys ?? []);

        return index;
    }

    /// <summary>
    /// Attempts to resolve a column name to its zero-based index within this row's source frame.
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <param name="index">When this method returns <c>true</c>, the zero-based column index</param>
    /// <returns><c>true</c> when the column exists; otherwise <c>false</c></returns>
    public bool TryGetColumnIndex(string columnName, out int index)
    {
        index = -1;

        if (columns is null || map is null || string.IsNullOrWhiteSpace(columnName))
            return false;

        return map.TryGetValue(columnName, out index);
    }

    /// <summary>
    /// Gets the column at the specified zero-based index.
    /// </summary>
    /// <param name="columnIndex">The zero-based column index</param>
    /// <returns>The column at that index</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when columnIndex is out of range</exception>
    public IColumn GetColumn(int columnIndex)
    {
        if (columns is null)
            throw new InvalidOperationException("NivaraRow is the default value and does not reference a frame row.");

        if (columnIndex < 0 || columnIndex >= columns.Length)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        return columns[columnIndex];
    }

    /// <summary>
    /// Gets the column with the specified name.
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The column with that name</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public IColumn GetColumn(string columnName)
    {
        if (columns is null)
            throw new InvalidOperationException("NivaraRow is the default value and does not reference a frame row.");

        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        if (map is null || !map.TryGetValue(columnName, out var index))
            throw new ColumnNotFoundException(columnName, map?.Keys ?? []);

        return columns[index];
    }

    /// <summary>
    /// Attempts to get the column with the specified name.
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <param name="column">When this method returns <c>true</c>, the column with that name</param>
    /// <returns><c>true</c> when the column exists; otherwise <c>false</c></returns>
    public bool TryGetColumn(string columnName, out IColumn? column)
    {
        column = null;

        if (columns is null || map is null || string.IsNullOrWhiteSpace(columnName))
            return false;

        if (!map.TryGetValue(columnName, out var index))
            return false;

        column = columns[index];
        return true;
    }

    /// <summary>
    /// Creates a cached read delegate for the specified column, resolving the nullable-element
    /// probe once. The returned delegate performs no dictionary lookup or reflection per call.
    /// </summary>
    public static Func<int, T> CreateColumnReader<T>(IColumn column, string? columnName = null)
        => NullableColumnReader.CreateReader<T>(column, columnName);

    /// <summary>
    /// Wraps this row's columns in a <see cref="NivaraRowReader"/> so columns can be pre-resolved
    /// and read without per-cell string lookups.
    /// </summary>
    public NivaraRowReader AsReader()
    {
        if (columns is null)
            throw new InvalidOperationException("NivaraRow is the default value and does not reference a frame row.");

        var names = columnNames ?? ColumnNames;
        return new NivaraRowReader(names, columns);
    }

    // Reads a nullable-element column (NivaraColumn<T?>, which implements IColumn<T?> rather than
    // IColumn<T>) through the nullable indexer and returns the underlying T. Unconstrained
    // GetValue<T> cannot form IColumn<T?> directly (T? collapses to T for an unconstrained T), so
    // dispatch goes through a TValue:struct core via a cached closed generic delegate. The delegate
    // is built once per T (small cached reflection) but invoked without boxing or reflection, so
    // Where()/predicate loops over nullable-element columns stay allocation-free.
    static class NullableColumnReader
    {
        static readonly MethodInfo readCore = typeof(NullableColumnReader)
            .GetMethod(nameof(ReadCore), BindingFlags.NonPublic | BindingFlags.Static)!;
        static readonly ConcurrentDictionary<Type, Delegate> cache = new();
        static readonly ConcurrentDictionary<Type, Type> nullableInterfaceCache = new();

        public static bool IsNullableColumnFor<T>(IColumn column)
        {
            if (!typeof(T).IsValueType || Nullable.GetUnderlyingType(typeof(T)) != null)
                return false;

            var nullableInterface = nullableInterfaceCache.GetOrAdd(typeof(T), static t =>
                typeof(IColumn<>).MakeGenericType(typeof(Nullable<>).MakeGenericType(t)));
            return nullableInterface.IsInstanceOfType(column);
        }

        public static T Read<T>(IColumn column, int rowIndex)
        {
            if (!cache.TryGetValue(typeof(T), out var reader))
                reader = cache.GetOrAdd(typeof(T), static t =>
                    readCore.MakeGenericMethod(t).CreateDelegate(typeof(Func<,,>)
                        .MakeGenericType(typeof(IColumn), typeof(int), t)));
            return ((Func<IColumn, int, T>)reader)(column, rowIndex);
        }

        public static Func<int, T> CreateReader<T>(IColumn column, string? columnName = null)
        {
            if (column is IColumn<T> typed)
                return i => typed[i];

            if (IsNullableColumnFor<T>(column))
            {
                var reader = (Func<IColumn, int, T>)cache.GetOrAdd(typeof(T), static t =>
                    readCore.MakeGenericMethod(t).CreateDelegate(typeof(Func<,,>)
                        .MakeGenericType(typeof(IColumn), typeof(int), t)));
                return i => reader(column, i);
            }

            throw new ColumnTypeMismatchException(columnName ?? string.Empty, typeof(T), column.ElementType);
        }

        static TValue ReadCore<TValue>(IColumn column, int rowIndex)
            where TValue : struct
            => ((IColumn<TValue?>)column)[rowIndex].GetValueOrDefault();
    }
}
