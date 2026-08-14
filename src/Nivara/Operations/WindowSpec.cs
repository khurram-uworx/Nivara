namespace Nivara.Operations;

/// <summary>
/// A reusable window specification capturing partition-by and order-by keys, built via
/// <see cref="NivaraFrameExtensions.Over"/> (or <c>new WindowSpec()</c>) and passed to the
/// window-function methods on <see cref="Nivara.NivaraFrame"/> and the query pipeline.
/// <para>
/// The spec is immutable: the fluent <see cref="PartitionBy"/> and <see cref="OrderBy"/>
/// methods return a new spec, so a base spec can be reused with different partition or
/// order keys. Chain order is irrelevant (<c>PartitionBy(...).OrderBy(...)</c> and
/// <c>OrderBy(...).PartitionBy(...)</c> are equivalent).
/// </para>
/// </summary>
/// <remarks>Added as part of issue #162 Over/WindowSpec builder delivery.</remarks>
public sealed class WindowSpec
{
    readonly string[] partitionBy;
    readonly SortKey[] orderBy;

    /// <summary>
    /// Initializes an empty window specification (single partition, row order).
    /// </summary>
    public WindowSpec()
        : this(Array.Empty<string>(), Array.Empty<SortKey>())
    {
    }

    WindowSpec(string[] partitionBy, SortKey[] orderBy)
    {
        this.partitionBy = partitionBy;
        this.orderBy = orderBy;
    }

    /// <summary>
    /// Gets the partition key column names (empty = a single partition over all rows).
    /// </summary>
    public IReadOnlyList<string> PartitionColumns => partitionBy;

    /// <summary>
    /// Gets the order keys (empty = row order within each partition).
    /// </summary>
    public IReadOnlyList<SortKey> OrderKeys => orderBy;

    /// <summary>
    /// Gets a value indicating whether the spec has no partition or order keys.
    /// </summary>
    public bool IsEmpty => partitionBy.Length == 0 && orderBy.Length == 0;

    /// <summary>
    /// Returns a new spec with the given partition key column names added.
    /// </summary>
    /// <param name="partitionBy">The partition key column names</param>
    /// <returns>A new spec with the partition keys</returns>
    /// <exception cref="ArgumentNullException">Thrown when partitionBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when a partition column name is null or whitespace</exception>
    public WindowSpec PartitionBy(params string[] partitionBy)
    {
        ArgumentNullException.ThrowIfNull(partitionBy);

        for (int i = 0; i < partitionBy.Length; i++)
            if (string.IsNullOrWhiteSpace(partitionBy[i]))
                throw new ArgumentException("Partition column name cannot be null or whitespace", nameof(partitionBy));

        return new WindowSpec(partitionBy, orderBy);
    }

    /// <summary>
    /// Returns a new spec with the given order keys added.
    /// </summary>
    /// <param name="orderBy">The order keys</param>
    /// <returns>A new spec with the order keys</returns>
    /// <exception cref="ArgumentNullException">Thrown when orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when an order key is null</exception>
    public WindowSpec OrderBy(params SortKey[] orderBy)
    {
        ArgumentNullException.ThrowIfNull(orderBy);

        for (int i = 0; i < orderBy.Length; i++)
            if (orderBy[i] is null)
                throw new ArgumentException("Order key cannot be null", nameof(orderBy));

        return new WindowSpec(partitionBy, orderBy);
    }

    /// <summary>
    /// Returns a new spec with the given column names ordered ascending (nulls last).
    /// </summary>
    /// <param name="columns">The column names to order by</param>
    /// <returns>A new spec with ascending order keys</returns>
    /// <exception cref="ArgumentNullException">Thrown when columns is null</exception>
    /// <exception cref="ArgumentException">Thrown when a column name is null or whitespace</exception>
    public WindowSpec OrderBy(params string[] columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        var keys = new SortKey[columns.Length];
        for (int i = 0; i < columns.Length; i++)
            keys[i] = new SortKey(columns[i], SortDirection.Ascending);

        return new WindowSpec(partitionBy, keys);
    }

    /// <summary>
    /// Returns a new spec ordered by a single column with the given direction and null ordering.
    /// </summary>
    /// <param name="column">The column name to order by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <returns>A new spec with the single order key</returns>
    /// <exception cref="ArgumentException">Thrown when column is null or whitespace</exception>
    public WindowSpec OrderBy(string column, SortDirection direction, NullOrdering nullOrdering = NullOrdering.NullsLast)
        => new WindowSpec(partitionBy, new[] { new SortKey(column, direction, nullOrdering) });

    /// <summary>
    /// Returns a string representation of the window specification.
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var orderStr = orderBy.Length > 0 ? string.Join(", ", orderBy.Select(k => k.ColumnName)) : "row order";
        var partitionStr = partitionBy.Length > 0 ? $" PARTITION BY {string.Join(", ", partitionBy)}" : "";
        return $"OVER (ORDER BY {orderStr}){partitionStr}";
    }
}