using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Immutable DataFrame-like structure that provides schema management and multi-column data operations.
/// Serves as the primary interface for working with structured, multi-column datasets in Nivara.
/// </summary>
public sealed class NivaraFrame : IFrame
{
    readonly IReadOnlyDictionary<string, IColumn> columns;
    readonly Schema schema;
    bool disposed;

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
            estimatedMemoryUsage += EstimateColumnMemoryUsage(column);
        }

        columns = columnDict;
        schema = new Schema(schemaColumns);

        // Track this frame for resource management
        ResourceManager.TrackResource(this, "NivaraFrame", estimatedMemoryUsage);
    }

    /// <summary>
    /// Creates a new NivaraFrame from a collection of named NivaraColumn instances
    /// </summary>
    /// <param name="namedColumns">The named columns to include in the frame</param>
    /// <returns>A new NivaraFrame instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when namedColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths or invalid names</exception>
    public static NivaraFrame Create(params (string Name, IColumn Column)[] namedColumns)
    {
        return new NivaraFrame(namedColumns);
    }

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
        ValidateColumnTypeForAddition(name, column.ElementType);

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

    /// <summary>
    /// Validates that a column type is compatible for addition to this frame
    /// </summary>
    /// <param name="columnName">The name of the column being added</param>
    /// <param name="columnType">The type of the column being added</param>
    private void ValidateColumnTypeForAddition(string columnName, Type columnType)
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
    /// Validates schema compatibility between two frames for operations like joins
    /// </summary>
    /// <param name="other">The other frame to validate compatibility with</param>
    /// <param name="operationName">The name of the operation for error messages</param>
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

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Untrack from resource manager
            ResourceManager.UntrackResource(this);

            // Dispose all columns
            foreach (var column in columns.Values)
            {
                column?.Dispose();
            }
            disposed = true;
        }
    }

    /// <summary>
    /// Estimates the memory usage of a column (rough approximation)
    /// </summary>
    /// <param name="column">The column to estimate</param>
    /// <returns>Estimated memory usage in bytes</returns>
    private static long EstimateColumnMemoryUsage(IColumn column)
    {
        if (column == null) return 0;

        var elementType = column.ElementType;
        var elementSize = GetTypeSize(elementType);
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
    /// Gets memory management recommendations for operations on this frame
    /// </summary>
    /// <param name="operationType">The type of operation being performed</param>
    /// <returns>Memory management recommendations</returns>
    public MemoryRecommendations GetMemoryRecommendations(string operationType = "general")
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var estimatedSize = EstimateFrameMemoryUsage();
        return ResourceManager.GetMemoryRecommendations(estimatedSize, operationType);
    }

    /// <summary>
    /// Estimates the total memory usage of this frame
    /// </summary>
    /// <returns>Estimated memory usage in bytes</returns>
    private long EstimateFrameMemoryUsage()
    {
        long totalSize = 0;
        foreach (var column in columns.Values)
        {
            totalSize += EstimateColumnMemoryUsage(column);
        }
        return totalSize;
    }

    /// <summary>
    /// Gets the approximate size of a type in bytes
    /// </summary>
    /// <param name="type">The type to get size for</param>
    /// <returns>Size in bytes</returns>
    private static int GetTypeSize(Type type)
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
}