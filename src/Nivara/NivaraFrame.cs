using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Query;
using Nivara.Storage;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics.Tensors;

namespace Nivara;

/// <summary>
/// Immutable DataFrame-like structure that provides schema management and multi-column data operations.
/// Serves as the primary interface for working with structured, multi-column datasets in Nivara.
/// </summary>
public sealed class NivaraFrame : IFrame
{
    /// <summary>
    /// Estimates the memory usage of a column (rough approximation)
    /// </summary>
    /// <param name="column">The column to estimate</param>
    /// <returns>Estimated memory usage in bytes</returns>
    static long estimateColumnMemoryUsage(IColumn column)
    {
        if (column == null) return 0;

        var elementType = column.ElementType;
        var elementSize = getTypeSize(elementType);
        var baseMemory = column.Length * elementSize;

        // Add overhead for null mask if present
        if (column.HasNulls)
        {
            baseMemory += column.Length; // 1 byte per boolean in null mask
        }

        // Add some overhead for object structure
        return baseMemory + 64; // 64 bytes overhead estimate
    }

    /// <summary>
    /// Slices a column using the column's built-in Slice method
    /// </summary>
    /// <param name="column">The column to slice</param>
    /// <param name="start">The starting index</param>
    /// <param name="length">The number of elements to include</param>
    /// <returns>A sliced column</returns>
    static IColumn sliceColumn(IColumn column, int start, int length)
    {
        // Use reflection to call the Slice method on the typed column
        var columnType = column.GetType();
        var sliceMethod = columnType.GetMethod("Slice", new[] { typeof(int), typeof(int) });

        if (sliceMethod != null)
        {
            return (IColumn)sliceMethod.Invoke(column, new object[] { start, length })!;
        }

        // Fallback: create filtered column using indices
        var indices = Enumerable.Range(start, length).ToList();
        return ColumnFilterHelper.CreateFilteredColumn(column, indices);
    }

    static int getTypeSize(Type type)
    {
        if (type == typeof(bool)) return 1;
        if (type == typeof(byte) || type == typeof(sbyte)) return 1;
        if (type == typeof(short) || type == typeof(ushort)) return 2;
        if (type == typeof(int) || type == typeof(uint)) return 4;
        if (type == typeof(long) || type == typeof(ulong)) return 8;
        if (type == typeof(float)) return 4;
        if (type == typeof(double)) return 8;
        if (type == typeof(decimal)) return 16;
        if (type == typeof(DateTime)) return 8;
        if (type == typeof(Guid)) return 16;

        // For reference types, estimate pointer size + average string length
        if (!type.IsValueType)
        {
            if (type == typeof(string)) return 50; // Average string estimate
            return 8; // Pointer size on 64-bit systems
        }

        return 8; // Default estimate for unknown types
    }

    /// <summary>
    /// Creates a new NivaraFrame from a collection of named NivaraColumn instances
    /// </summary>
    /// <param name="namedColumns">The named columns to include in the frame</param>
    /// <returns>A new NivaraFrame instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when namedColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths or invalid names</exception>
    public static NivaraFrame Create(params (string Name, IColumn Column)[] namedColumns)
        => new NivaraFrame(namedColumns);

    /// <summary>
    /// Creates a new NivaraFrame from a dictionary of named columns
    /// </summary>
    /// <param name="columnDictionary">Dictionary mapping column names to columns</param>
    /// <returns>A new NivaraFrame instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnDictionary is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths or invalid names</exception>
    public static NivaraFrame Create(IReadOnlyDictionary<string, IColumn> columnDictionary)
    {
        if (columnDictionary == null)
            throw new ArgumentNullException(nameof(columnDictionary));

        var namedColumns = columnDictionary.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Creates a new NivaraFrame with a single typed column, eliminating the need for (IColumn) cast.
    /// </summary>
    /// <typeparam name="T">The column element type</typeparam>
    /// <param name="name">The column name</param>
    /// <param name="column">The typed column</param>
    /// <returns>A new NivaraFrame with the specified column</returns>
    /// <exception cref="ArgumentNullException">Thrown when name or column is null</exception>
    public static NivaraFrame Create<T>(string name, NivaraColumn<T> column)
    {
        if (name == null)
            throw new ArgumentNullException(nameof(name));
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return Create((name, (IColumn)column));
    }

    /// <summary>
    /// Creates a frame from a 2D tensor using row-major tensor orientation [rows, columns].
    /// </summary>
    public static NivaraFrame FromMatrix<T>(
        Tensor<T> matrix,
        string[]? columnNames = null,
        object[]? rowLabels = null,
        string rowLabelColumnName = "Label")
        where T : unmanaged
        => TensorInteropExtensions.FromTensor(matrix, columnNames, rowLabels, rowLabelColumnName);

    /// <summary>
    /// Creates a frame from labeled row vectors. Labels are stored as a normal column.
    /// </summary>
    public static NivaraFrame FromRows<T>(
        IEnumerable<(string Label, T[] Vector)> rows,
        string[]? columnNames = null,
        string labelColumnName = "Label")
    {
        ArgumentNullException.ThrowIfNull(rows);

        if (string.IsNullOrWhiteSpace(labelColumnName))
            throw new ArgumentException("Label column name cannot be null or whitespace", nameof(labelColumnName));

        var rowList = rows.ToList();
        if (rowList.Count == 0)
            throw new ArgumentException("At least one row is required to infer vector width.", nameof(rows));

        var width = rowList[0].Vector?.Length
            ?? throw new ArgumentException("Row vectors cannot be null.", nameof(rows));

        foreach (var row in rowList)
        {
            if (row.Vector == null)
                throw new ArgumentException("Row vectors cannot be null.", nameof(rows));
            if (row.Vector.Length != width)
                throw new ArgumentException("All row vectors must have the same length.", nameof(rows));
        }

        columnNames ??= Enumerable.Range(0, width).Select(i => $"Col_{i}").ToArray();
        if (columnNames.Length != width)
            throw new ArgumentException($"Column names count ({columnNames.Length}) must match vector width ({width}).", nameof(columnNames));

        if (columnNames.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Column names cannot contain null or whitespace values.", nameof(columnNames));

        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!names.Add(labelColumnName))
            throw new ArgumentException($"Duplicate column name '{labelColumnName}' found.", nameof(labelColumnName));

        foreach (var columnName in columnNames)
        {
            if (!names.Add(columnName))
                throw new ArgumentException($"Duplicate column name '{columnName}' found. Column names must be unique (case-insensitive).", nameof(columnNames));
        }

        var columns = new List<(string Name, IColumn Column)>
        {
            (labelColumnName, NivaraColumn<string>.CreateForReferenceType(rowList.Select(row => row.Label).ToArray()))
        };

        for (int col = 0; col < width; col++)
        {
            var values = new T[rowList.Count];
            for (int row = 0; row < rowList.Count; row++)
                values[row] = rowList[row].Vector[col];

            columns.Add((columnNames[col], NivaraColumn<T>.Create(values)));
        }

        return new NivaraFrame(columns);
    }

    readonly IReadOnlyDictionary<string, IColumn> columns;
    readonly Schema schema;
    bool disposed;

    /// <summary>
    /// Gets the sort keys that produced this frame's row order, when it was produced by
    /// <c>OrderBy</c> or <c>ThenBy</c>. Used to compose secondary sorts lexicographically.
    /// </summary>
    internal IReadOnlyList<SortKey>? SortKeys { get; set; }

    /// <summary>
    /// Initializes a new instance of NivaraFrame with the specified columns
    /// </summary>
    /// <param name="namedColumns">The named columns to include in the frame</param>
    /// <exception cref="ArgumentNullException">Thrown when namedColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths or invalid names</exception>
    public NivaraFrame(IEnumerable<(string Name, IColumn Column)> namedColumns)
    {
        if (namedColumns == null)
            throw new ArgumentNullException(nameof(namedColumns));

        var columnList = namedColumns.ToList();

        // Validate we have at least one column
        if (columnList.Count == 0)
            throw new ArgumentException("Frame must contain at least one column", nameof(namedColumns));

        // Validate column names and build dictionaries
        var columnDict = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        var schemaColumns = new List<(string Name, Type Type)>();
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        int? expectedLength = null;
        long estimatedMemoryUsage = 0;

        foreach (var (name, column) in columnList)
        {
            // Validate column name
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Column names cannot be null or whitespace", nameof(namedColumns));

            // Validate column is not null
            if (column == null)
                throw new ArgumentException($"Column '{name}' cannot be null", nameof(namedColumns));

            // Check for duplicate names (case-insensitive)
            if (!names.Add(name))
                throw new ArgumentException($"Duplicate column name '{name}' found. Column names must be unique (case-insensitive)", nameof(namedColumns));

            // Validate column lengths match
            if (expectedLength == null)
            {
                expectedLength = column.Length;
            }
            else if (column.Length != expectedLength.Value)
            {
                var existingColumns = string.Join(", ", names.Where(n => !string.Equals(n, name, StringComparison.OrdinalIgnoreCase)));
                throw new ArgumentException(
                    $"Column length mismatch: Column '{name}' has length {column.Length}, but expected {expectedLength.Value} " +
                    $"to match existing columns [{existingColumns}]. All columns in a frame must have the same length.",
                    nameof(namedColumns));
            }

            columnDict[name] = column;
            schemaColumns.Add((name, column.ElementType));

            // Estimate memory usage (rough approximation)
            estimatedMemoryUsage += estimateColumnMemoryUsage(column);
        }

        columns = columnDict;
        schema = new Schema(schemaColumns);

        // Track this frame for resource management
        NivaraResourceManager.TrackResource(this, "NivaraFrame", estimatedMemoryUsage);
    }

    /// <summary>
    /// Validates that a column type is compatible for addition to this frame
    /// </summary>
    /// <param name="columnName">The name of the column being added</param>
    /// <param name="columnType">The type of the column being added</param>
    void validateColumnTypeForAddition(string columnName, Type columnType)
    {
        // Basic type validation - ensure it's a supported type
        var supportedTypes = TypeCompatibilityValidator.GetNumericTypes()
            .Concat(new[] { typeof(string), typeof(bool), typeof(DateTime), typeof(object) })
            .ToList();

        if (!supportedTypes.Contains(columnType) && !columnType.IsEnum)
        {
            var supportedTypeNames = string.Join(", ", supportedTypes.Select(t => t.Name));
            throw new ColumnTypeMismatchException(
                columnName,
                typeof(object), // Expected type (generic)
                columnType);
        }
    }

    /// <summary>
    /// Estimates the total memory usage of this frame
    /// </summary>
    /// <returns>Estimated memory usage in bytes</returns>
    long estimateFrameMemoryUsage()
    {
        long totalSize = 0;
        foreach (var column in columns.Values)
        {
            totalSize += estimateColumnMemoryUsage(column);
        }
        return totalSize;
    }

    /// <summary>
    /// Creates a QueryFrame from an existing NivaraFrame for lazy query operations
    /// </summary>
    /// <returns>A QueryFrame that can be used to build query chains</returns>
    public QueryFrame AsQueryFrame()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var memorySource = new MemoryQuerySource(columns, schema);
        return new QueryFrame(memorySource);
    }

    /// <inheritdoc />
    public int RowCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return columns.Values.FirstOrDefault()?.Length ?? 0;
        }
    }

    /// <inheritdoc />
    public int ColumnCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return columns.Count;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return schema.ColumnNames;
        }
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
    public NivaraColumn<T> GetColumn<T>(string name)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(name));

        if (!columns.TryGetValue(name, out var column))
            throw new ColumnNotFoundException(name, ColumnNames);

        if (column is not NivaraColumn<T> typedColumn)
        {
            var actualType = column.ElementType;
            var expectedType = typeof(T);
            throw new ColumnTypeMismatchException(name, expectedType, actualType);
        }

        return typedColumn;
    }

    /// <inheritdoc />
    public bool HasColumn(string name)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(name))
            return false;

        return columns.ContainsKey(name);
    }

    /// <summary>
    /// Gets a column by name without type checking (returns the base IColumn interface)
    /// </summary>
    /// <param name="name">The name of the column</param>
    /// <returns>The column as IColumn</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public IColumn GetColumn(string name)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(name));

        if (!columns.TryGetValue(name, out var column))
            throw new ColumnNotFoundException(name, ColumnNames);

        return column;
    }

    /// <summary>
    /// Converts all columns of type T to an array of Tensor&lt;T&gt;, one per column.
    /// </summary>
    /// <typeparam name="T">The numeric type of the columns to convert</typeparam>
    /// <returns>An array of Tensor&lt;T&gt;, one per column</returns>
    /// <exception cref="InvalidOperationException">Thrown when any column contains nulls or is not of type T</exception>
    public Tensor<T>[] ToTensors<T>()
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        return DiagnosticsTracker.MeasureOperation(
            "FrameToTensors",
            KernelSelector.DetermineBatchKernelType<T>(),
            RowCount,
            typeof(T),
            columns.Values.Any(column => column.HasNulls),
            () =>
            {
                var tensors = new Tensor<T>[ColumnCount];
                for (int i = 0; i < ColumnCount; i++)
                {
                    var name = ColumnNames[i];
                    var col = GetColumn<T>(name);
                    tensors[i] = col.ToTensor();
                }
                return tensors;
            },
            "TensorConversion=FrameColumns");
    }

    /// <summary>
    /// Attempts to get a zero-copy row-major span over the frame data.
    /// Returns true only when the frame has a single column with no nulls (the only case
    /// where row-major data is structurally contiguous in memory). For multi-column frames,
    /// callers should use <see cref="CopyToRowMajor{T}"/> instead.
    /// </summary>
    /// <typeparam name="T">The unmanaged numeric type</typeparam>
    /// <param name="span">The read-only span over the row-major data, if zero-copy is available</param>
    /// <returns>true if a zero-copy row-major span was obtained; false otherwise</returns>
    public bool TryGetRowMajorSpan<T>(out ReadOnlySpan<T> span)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (ColumnCount == 0 || RowCount == 0)
        {
            span = default;
            return false;
        }

        // Zero-copy row-major is only structurally possible for single-column frames
        // where the column's flattened data IS the row-major layout.
        if (ColumnCount == 1)
        {
            var col = GetColumn<T>(ColumnNames[0]);
            if (col.TryGetSpan(out span))
                return true;

            span = default;
            return false;
        }

        // Multi-column frames require interleaving separate column tensors,
        // which cannot be zero-copy. Use CopyToRowMajor instead.
        span = default;
        return false;
    }

    /// <summary>
    /// Copies all columns of type T into a row-major destination span, filling nulls with <paramref name="fillValue"/>.
    /// Uses span-level copy for null-free columns and bulk-fill for nullable columns to avoid per-element virtual dispatch.
    /// </summary>
    /// <typeparam name="T">The unmanaged numeric type</typeparam>
    /// <param name="destination">The destination span. Must have length >= RowCount * ColumnCount.</param>
    /// <param name="fillValue">The value to write in place of nulls.</param>
    /// <exception cref="ArgumentException">Thrown when destination is too short.</exception>
    public void CopyToRowMajor<T>(Span<T> destination, T fillValue = default)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var needed = RowCount * ColumnCount;
        if (destination.Length < needed)
            throw new ArgumentException($"Destination span length ({destination.Length}) must be at least {needed}", nameof(destination));

        T[]? pooledTemp = null;
        Span<T> temp = default;
        var rowStride = ColumnCount;

        try
        {
            for (int col = 0; col < ColumnCount; col++)
            {
                var colData = GetColumn<T>(ColumnNames[col]);

                if (colData.TryGetSpan(out var span))
                {
                    for (int row = 0; row < RowCount; row++)
                        destination[row * rowStride + col] = span[row];
                }
                else
                {
                    if (temp.IsEmpty)
                    {
                        temp = RowCount >= 1024
                            ? (pooledTemp = ArrayPool<T>.Shared.Rent(RowCount))
                            : new T[RowCount];
                    }

                    colData.CopyTo(temp, fillValue);
                    for (int row = 0; row < RowCount; row++)
                        destination[row * rowStride + col] = temp[row];
                }
            }
        }
        finally
        {
            if (pooledTemp != null)
                ArrayPool<T>.Shared.Return(pooledTemp);
        }
    }

    /// <summary>
    /// Converts this frame to a 2D tensor plus an optional explicit null mask.
    /// All columns must be of the same unmanaged type T. Tensor orientation is [rows, columns].
    /// </summary>
    public NullableTensor<T> ToNullableTensor<T>()
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (ColumnCount == 0)
            throw new ArgumentException("Frame must have at least one column");

        foreach (var columnType in Schema.ColumnTypes.Values)
            if (columnType != typeof(T))
                throw new ArgumentException($"All columns must be of type {typeof(T).Name}. Found column of type {columnType.Name}");

        var length = RowCount * ColumnCount;
        var data = new T[length];
        var mask = new bool[length];
        var hasNulls = false;
        var rowStride = ColumnCount;

        for (int col = 0; col < ColumnCount; col++)
        {
            var column = GetColumn<T>(ColumnNames[col]);
            var rawValues = column.AsSpan();
            column.TryGetNullMask(out var columnMask);

            for (int row = 0; row < RowCount; row++)
            {
                var index = row * rowStride + col;
                data[index] = rawValues[row];

                if (columnMask.Length > 0 && columnMask[row])
                {
                    mask[index] = true;
                    hasNulls = true;
                }
            }
        }

        var dimensions = new nint[] { RowCount, ColumnCount };
        return new NullableTensor<T>(
            Tensor.Create(data, dimensions),
            hasNulls ? Tensor.Create(mask, dimensions) : null);
    }

    /// <summary>
    /// Creates a new frame with an additional column
    /// </summary>
    /// <param name="name">The name of the new column</param>
    /// <param name="column">The column to add</param>
    /// <returns>A new NivaraFrame with the additional column</returns>
    /// <exception cref="ArgumentException">Thrown when the column name already exists or column length doesn't match</exception>
    public NivaraFrame WithColumn(string name, IColumn column)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(name));

        if (column == null)
            throw new ArgumentNullException(nameof(column));

        if (HasColumn(name))
            throw new ArgumentException($"Column '{name}' already exists in the frame. Use a different name or remove the existing column first.", nameof(name));

        if (column.Length != RowCount)
            throw new ArgumentException(
                $"Column length mismatch: New column '{name}' has length {column.Length}, but frame has {RowCount} rows. " +
                $"All columns must have the same length.",
                nameof(column));

        // Validate column type compatibility if needed for operations
        validateColumnTypeForAddition(name, column.ElementType);

        var newColumns = columns.Concat(new[] { new KeyValuePair<string, IColumn>(name, column) });
        var namedColumns = newColumns.Select(kvp => (kvp.Key, kvp.Value));

        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Creates a new frame without the specified column
    /// </summary>
    /// <param name="name">The name of the column to remove</param>
    /// <returns>A new NivaraFrame without the specified column</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    /// <exception cref="InvalidOperationException">Thrown when trying to remove the last column</exception>
    public NivaraFrame WithoutColumn(string name)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(name));

        if (!HasColumn(name))
            throw new ColumnNotFoundException(name, ColumnNames);

        if (ColumnCount == 1)
            throw new InvalidOperationException("Cannot remove the last column from a frame. A frame must contain at least one column.");

        var newColumns = columns.Where(kvp => !string.Equals(kvp.Key, name, StringComparison.OrdinalIgnoreCase));
        var namedColumns = newColumns.Select(kvp => (kvp.Key, kvp.Value));

        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Creates a new frame with only the specified columns
    /// </summary>
    /// <param name="columnNames">The names of the columns to include</param>
    /// <returns>A new NivaraFrame with only the specified columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no column names are provided</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any of the specified columns is not found</exception>
    public NivaraFrame SelectColumns(params string[] columnNames)
    {
        return SelectColumns((IEnumerable<string>)columnNames);
    }

    /// <summary>
    /// Creates a new frame with only the specified columns
    /// </summary>
    /// <param name="columnNames">The names of the columns to include</param>
    /// <returns>A new NivaraFrame with only the specified columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no column names are provided</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any of the specified columns is not found</exception>
    public NivaraFrame SelectColumns(IEnumerable<string> columnNames)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        var selectedNames = columnNames.ToList();

        if (selectedNames.Count == 0)
            throw new ArgumentException("Must specify at least one column name", nameof(columnNames));

        // Validate all columns exist
        foreach (var name in selectedNames)
        {
            if (!HasColumn(name))
                throw new ColumnNotFoundException(name, ColumnNames);
        }

        var selectedColumns = selectedNames.Select(name => (name, columns[name]));
        return new NivaraFrame(selectedColumns);
    }

    /// <summary>
    /// Creates a new frame by filtering rows using a boolean mask
    /// </summary>
    /// <param name="mask">Boolean mask indicating which rows to keep (true = keep, false = exclude)</param>
    /// <returns>A new NivaraFrame containing only the rows where mask is true</returns>
    /// <exception cref="ArgumentNullException">Thrown when mask is null</exception>
    /// <exception cref="ArgumentException">Thrown when mask length doesn't match frame row count</exception>
    public NivaraFrame FilterByMask(NivaraColumn<bool> mask)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (mask == null)
            throw new ArgumentNullException(nameof(mask));

        if (mask.Length != RowCount)
            throw new ArgumentException(
                $"Mask length ({mask.Length}) must match frame row count ({RowCount})",
                nameof(mask));

        // Collect indices where mask is true
        var filteredIndices = new List<int>();
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] == true) // Only include rows where mask is true
            {
                filteredIndices.Add(i);
            }
        }

        // If no rows match, return empty frame with same schema
        if (filteredIndices.Count == 0)
        {
            var emptyColumns = new List<(string Name, IColumn Column)>();
            foreach (var columnName in ColumnNames)
            {
                var originalColumn = columns[columnName];
                var emptyColumn = ColumnFilterHelper.CreateEmptyColumn(originalColumn.ElementType);
                emptyColumns.Add((columnName, emptyColumn));
            }
            return new NivaraFrame(emptyColumns);
        }

        // Create filtered columns
        var filteredColumns = new List<(string Name, IColumn Column)>();
        foreach (var kvp in columns)
        {
            var filteredColumn = ColumnFilterHelper.CreateFilteredColumn(kvp.Value, filteredIndices);
            filteredColumns.Add((kvp.Key, filteredColumn));
        }

        return new NivaraFrame(filteredColumns);
    }

    /// <summary>
    /// Creates a new frame containing the first n rows
    /// </summary>
    /// <param name="count">The number of rows to take from the beginning</param>
    /// <returns>A new NivaraFrame containing the first n rows</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when count is negative</exception>
    public NivaraFrame Take(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative");

        if (count == 0)
        {
            // Return empty frame with same schema
            var emptyColumns = new List<(string Name, IColumn Column)>();
            foreach (var columnName in ColumnNames)
            {
                var originalColumn = columns[columnName];
                var emptyColumn = ColumnFilterHelper.CreateEmptyColumn(originalColumn.ElementType);
                emptyColumns.Add((columnName, emptyColumn));
            }
            return new NivaraFrame(emptyColumns);
        }

        // Clamp count to actual row count
        var actualCount = Math.Min(count, RowCount);

        // Slice all columns from 0 to actualCount
        var slicedColumns = new List<(string Name, IColumn Column)>();
        foreach (var kvp in columns)
        {
            var slicedColumn = sliceColumn(kvp.Value, 0, actualCount);
            slicedColumns.Add((kvp.Key, slicedColumn));
        }

        return new NivaraFrame(slicedColumns);
    }

    /// <summary>
    /// Creates a new frame by skipping the first n rows
    /// </summary>
    /// <param name="count">The number of rows to skip from the beginning</param>
    /// <returns>A new NivaraFrame containing all rows except the first n</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when count is negative</exception>
    public NivaraFrame Skip(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative");

        if (count >= RowCount)
        {
            // Return empty frame with same schema
            var emptyColumns = new List<(string Name, IColumn Column)>();
            foreach (var columnName in ColumnNames)
            {
                var originalColumn = columns[columnName];
                var emptyColumn = ColumnFilterHelper.CreateEmptyColumn(originalColumn.ElementType);
                emptyColumns.Add((columnName, emptyColumn));
            }
            return new NivaraFrame(emptyColumns);
        }

        if (count == 0)
        {
            // Return copy of entire frame
            var allColumns = columns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(allColumns);
        }

        // Slice all columns from count to end
        var remainingCount = RowCount - count;
        var slicedColumns = new List<(string Name, IColumn Column)>();
        foreach (var kvp in columns)
        {
            var slicedColumn = sliceColumn(kvp.Value, count, remainingCount);
            slicedColumns.Add((kvp.Key, slicedColumn));
        }

        return new NivaraFrame(slicedColumns);
    }

    /// <summary>
    /// Creates a new frame by slicing rows using start and length parameters
    /// </summary>
    /// <param name="start">The starting index (inclusive)</param>
    /// <param name="length">The number of rows to include</param>
    /// <returns>A new NivaraFrame containing the specified slice of rows</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when start or length are invalid</exception>
    public NivaraFrame Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (start < 0)
            throw new ArgumentOutOfRangeException(nameof(start), "Start index cannot be negative");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative");
        if (start + length > RowCount)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"Start + length ({start + length}) exceeds frame row count ({RowCount})");

        if (length == 0)
        {
            // Return empty frame with same schema
            var emptyColumns = new List<(string Name, IColumn Column)>();
            foreach (var columnName in ColumnNames)
            {
                var originalColumn = columns[columnName];
                var emptyColumn = ColumnFilterHelper.CreateEmptyColumn(originalColumn.ElementType);
                emptyColumns.Add((columnName, emptyColumn));
            }
            return new NivaraFrame(emptyColumns);
        }

        // Slice all columns
        var slicedColumns = new List<(string Name, IColumn Column)>();
        foreach (var kvp in columns)
        {
            var slicedColumn = sliceColumn(kvp.Value, start, length);
            slicedColumns.Add((kvp.Key, slicedColumn));
        }

        return new NivaraFrame(slicedColumns);
    }

    /// <summary>
    /// Creates a new frame by reordering rows using the specified indices
    /// </summary>
    /// <param name="indices">The indices specifying the new row order</param>
    /// <returns>A new NivaraFrame with rows reordered according to the indices</returns>
    /// <exception cref="ArgumentNullException">Thrown when indices is null</exception>
    /// <exception cref="ArgumentException">Thrown when indices length doesn't match row count or contains invalid indices</exception>
    public NivaraFrame ReorderByIndices(int[] indices)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        if (indices.Length != RowCount)
            throw new ArgumentException(
                $"Indices length ({indices.Length}) must match frame row count ({RowCount})",
                nameof(indices));

        // Validate all indices are within bounds
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= RowCount)
            {
                throw new ArgumentException(
                    $"Index {indices[i]} at position {i} is out of bounds. Valid range is 0 to {RowCount - 1}",
                    nameof(indices));
            }
        }

        // If indices are already in order, return a copy
        if (indices.SequenceEqual(Enumerable.Range(0, RowCount)))
        {
            var identityColumns = columns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(identityColumns);
        }

        // Reorder all columns using the indices
        var reorderedColumns = new List<(string Name, IColumn Column)>();
        foreach (var kvp in columns)
        {
            var reorderedColumn = ColumnFilterHelper.ReorderColumn(kvp.Value, indices);
            reorderedColumns.Add((kvp.Key, reorderedColumn));
        }

        return new NivaraFrame(reorderedColumns);
    }

    /// <summary>
    /// Vertically concatenates this frame with another frame with identical schema.
    /// </summary>
    /// <param name="other">The frame to append</param>
    /// <param name="mismatchHandling">How to handle schema mismatches (default Error)</param>
    /// <returns>A new frame containing rows from both frames</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible and mismatchHandling is Error</exception>
    public NivaraFrame Concat(NivaraFrame other, ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.Error)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(other);

        return this.ConcatenateVertical(other, mismatchHandling);
    }

    /// <summary>
    /// Vertically concatenates a sequence of frames with identical schema.
    /// </summary>
    /// <param name="frames">The frames to concatenate</param>
    /// <param name="mismatchHandling">How to handle schema mismatches (default Error)</param>
    /// <returns>A new frame containing rows from all input frames</returns>
    /// <exception cref="ArgumentNullException">Thrown when frames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no frames are provided</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible and mismatchHandling is Error</exception>
    public static NivaraFrame Concat(IEnumerable<NivaraFrame> frames, ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.Error)
    {
        ArgumentNullException.ThrowIfNull(frames);

        return NivaraFrameExtensions.ConcatenateVertical(frames, mismatchHandling);
    }

    /// <summary>
    /// Validates schema compatibility between this frame and another frame
    /// </summary>
    /// <param name="other">The other frame to validate against</param>
    /// <param name="operationName">The name of the operation for error messages</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible</exception>
    public void ValidateSchemaCompatibility(NivaraFrame other, string operationName)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (!Schema.IsCompatibleWith(other.Schema, requireExactMatch: false))
        {
            throw new SchemaValidationException(
                $"Schema incompatibility detected for operation '{operationName}'. " +
                $"This frame has schema: {Schema}. Other frame has schema: {other.Schema}.",
                Schema,
                other.Schema);
        }
    }

    /// <summary>
    /// Gets memory management recommendations for operations on this frame
    /// </summary>
    /// <param name="operationType">The type of operation being performed</param>
    /// <returns>Memory management recommendations</returns>
    public MemoryRecommendations GetMemoryRecommendations(string operationType = "general")
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var estimatedSize = estimateFrameMemoryUsage();
        return NivaraResourceManager.GetMemoryRecommendations(estimatedSize, operationType);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Untrack from resource manager
            NivaraResourceManager.UntrackResource(this);

            // Dispose all columns
            foreach (var column in columns.Values)
            {
                column?.Dispose();
            }
            disposed = true;
        }
    }

    /// <summary>
    /// Returns a string representation of the frame showing its structure
    /// </summary>
    /// <returns>A formatted string describing the frame</returns>
    public override string ToString()
    {
        if (disposed)
            return "NivaraFrame [Disposed]";

        return $"NivaraFrame [{RowCount} rows × {ColumnCount} columns] {{ {string.Join(", ", ColumnNames)} }}";
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current frame
    /// </summary>
    /// <param name="obj">The object to compare</param>
    /// <returns>True if the objects are equal, false otherwise</returns>
    public override bool Equals(object? obj)
    {
        return ReferenceEquals(this, obj);
    }

    /// <summary>
    /// Returns a hash code for the frame
    /// </summary>
    /// <returns>A hash code for the frame</returns>
    public override int GetHashCode()
    {
        return base.GetHashCode();
    }
}
