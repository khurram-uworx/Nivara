using System.Collections.Concurrent;
using System.Reflection;

namespace Nivara;

/// <summary>
/// Typed per-column key reader used to de-box group-by/rank/distinct key hashing and equality.
/// One reader is created per key column; rows are hashed and compared without boxing or per-row
/// key objects (issue #251).
/// </summary>
internal interface IGroupKeyReader
{
    /// <summary>Gets the element type of the key column.</summary>
    Type ElementType { get; }

    /// <summary>Gets the composite hash of the value at <paramref name="row"/>.</summary>
    int GetHashCode(int row);

    /// <summary>Compares this column's value at <paramref name="rowA"/> with <paramref name="other"/>'s value at <paramref name="rowB"/>.</summary>
    bool ValuesEqual(int rowA, IGroupKeyReader other, int rowB);

    /// <summary>Gets the value at <paramref name="row"/> as a boxed object (null for null positions).</summary>
    object? GetValue(int row);
}

/// <summary>
/// Typed reader over a <see cref="NivaraColumn{T}"/> using <see cref="EqualityComparer{T}.Default"/>
/// and the column's null mask (null equals null; null hashes to 0).
/// </summary>
internal sealed class GroupKeyReader<T> : IGroupKeyReader
{
    readonly NivaraColumn<T> column;

    public GroupKeyReader(NivaraColumn<T> column) => this.column = column;

    public Type ElementType => typeof(T);

    public int GetHashCode(int row) => column.IsNull(row) ? 0 : EqualityComparer<T>.Default.GetHashCode(column[row]!);

    public bool ValuesEqual(int rowA, IGroupKeyReader other, int rowB)
    {
        if (other is GroupKeyReader<T> typed)
        {
            bool aNull = column.IsNull(rowA);
            bool bNull = typed.column.IsNull(rowB);
            return aNull == bNull && (aNull || EqualityComparer<T>.Default.Equals(column[rowA], typed.column[rowB]));
        }

        return Equals(GetValue(rowA), other.GetValue(rowB));
    }

    public object? GetValue(int row) => column.IsNull(row) ? null : column[row];
}

/// <summary>
/// Boxed fallback reader over the non-generic <see cref="IColumn"/> surface, for exotic or
/// custom column implementations.
/// </summary>
internal sealed class BoxedGroupKeyReader : IGroupKeyReader
{
    readonly IColumn column;

    public BoxedGroupKeyReader(IColumn column) => this.column = column;

    public Type ElementType => column.ElementType;

    public int GetHashCode(int row) => column.IsNull(row) ? 0 : (column.GetValue(row)?.GetHashCode() ?? 0);

    public bool ValuesEqual(int rowA, IGroupKeyReader other, int rowB)
        => Equals(GetValue(rowA), other.GetValue(rowB));

    public object? GetValue(int row) => column.GetValue(row);
}

/// <summary>
/// Creates typed <see cref="IGroupKeyReader"/>s via cached <see cref="MethodInfo.MakeGenericMethod(Type)"/>,
/// falling back to a boxed reader when the column is not a <see cref="NivaraColumn{T}"/>.
/// </summary>
internal static class GroupKeyReaderFactory
{
    static readonly MethodInfo createKernel = typeof(GroupKeyReaderFactory)
        .GetMethod(nameof(createReader), BindingFlags.Static | BindingFlags.NonPublic)!;
    static readonly ConcurrentDictionary<Type, MethodInfo> cache = new();

    public static IGroupKeyReader Create(IColumn column)
    {
        ArgumentNullException.ThrowIfNull(column);
        var kernel = cache.GetOrAdd(column.ElementType, static t => createKernel.MakeGenericMethod(t));
        return (IGroupKeyReader)kernel.Invoke(null, new object?[] { column })!;
    }

    static IGroupKeyReader createReader<T>(IColumn column)
        => column is NivaraColumn<T> typed ? new GroupKeyReader<T>(typed) : new BoxedGroupKeyReader(column);
}

/// <summary>
/// Composite row hashing over a set of <see cref="IGroupKeyReader"/>s. The combining scheme is
/// deterministic and value-based, so equal keys produced from different column instances (parallel
/// chunks) hash identically.
/// </summary>
internal static class TypedGroupHash
{
    public static int ComputeRowHash(IReadOnlyList<IGroupKeyReader> readers, int row)
    {
        int hash = 17;
        for (int i = 0; i < readers.Count; i++)
            hash = unchecked(hash * 31 + readers[i].GetHashCode(row));
        return hash;
    }

    public static void ComputeRowHashes(IReadOnlyList<IGroupKeyReader> readers, int rowCount, Span<int> dest)
    {
        for (int row = 0; row < rowCount; row++)
            dest[row] = ComputeRowHash(readers, row);
    }

    public static bool RowsEqual(IReadOnlyList<IGroupKeyReader> readers, int rowA, int rowB)
    {
        for (int i = 0; i < readers.Count; i++)
            if (!readers[i].ValuesEqual(rowA, readers[i], rowB))
                return false;
        return true;
    }
}
