using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Represents the schema of a frame, including column names, types, and metadata.
/// Provides immutable schema management with transformation capabilities.
/// </summary>
public sealed class Schema
{
    /// <summary>
    /// Checks if two types are compatible for operations
    /// </summary>
    /// <param name="type1">The first type</param>
    /// <param name="type2">The second type</param>
    /// <returns>True if the types are compatible, false otherwise</returns>
    static bool areTypesCompatible(Type type1, Type type2)
    {
        if (type1 == type2)
            return true;

        // Numeric type compatibility
        var numericTypes = new[]
        {
            typeof(byte), typeof(sbyte), typeof(short), typeof(ushort),
            typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double), typeof(decimal)
        };

        return numericTypes.Contains(type1) && numericTypes.Contains(type2);
    }

    readonly IReadOnlyDictionary<string, Type> columnTypes;
    readonly IReadOnlyDictionary<string, ColumnMetadata> metadata;

    /// <summary>
    /// Initializes a new instance of Schema with columns and metadata
    /// </summary>
    /// <param name="columns">The column definitions</param>
    /// <param name="columnMetadata">The metadata for each column</param>
    internal Schema(
        IEnumerable<(string Name, Type Type)> columns,
        IReadOnlyDictionary<string, ColumnMetadata> columnMetadata)
    {
        if (columns == null)
            throw new ArgumentNullException(nameof(columns));

        var columnList = columns.ToList();
        var names = new List<string>();
        var typeDict = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);

        foreach (var (name, type) in columnList)
        {
            names.Add(name);
            typeDict[name] = type;
        }

        ColumnNames = names;
        columnTypes = typeDict;
        metadata = columnMetadata ?? new Dictionary<string, ColumnMetadata>();
    }

    /// <summary>
    /// Initializes a new instance of Schema with the specified columns
    /// </summary>
    /// <param name="columns">The column definitions as (name, type) pairs</param>
    /// <exception cref="ArgumentNullException">Thrown when columns is null</exception>
    /// <exception cref="ArgumentException">Thrown when column names are duplicated or invalid</exception>
    public Schema(IEnumerable<(string Name, Type Type)> columns)
    {
        if (columns == null)
            throw new ArgumentNullException(nameof(columns));

        var columnList = columns.ToList();

        // Validate column names
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var typeDict = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
        var metadataDict = new Dictionary<string, ColumnMetadata>(StringComparer.OrdinalIgnoreCase);

        foreach (var (name, type) in columnList)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Column names cannot be null or whitespace", nameof(columns));

            if (type == null)
                throw new ArgumentException($"Column type cannot be null for column '{name}'", nameof(columns));

            if (!names.Add(name))
                throw new ArgumentException($"Duplicate column name '{name}' found", nameof(columns));

            typeDict[name] = type;
            metadataDict[name] = new ColumnMetadata();
        }

        ColumnNames = names.ToList();
        columnTypes = typeDict;
        metadata = metadataDict;
    }

    /// <summary>
    /// Gets the names of all columns in the schema
    /// </summary>
    public IReadOnlyList<string> ColumnNames { get; }

    /// <summary>
    /// Gets the types of all columns in the schema
    /// </summary>
    public IReadOnlyDictionary<string, Type> ColumnTypes => columnTypes;

    /// <summary>
    /// Gets the metadata for all columns in the schema
    /// </summary>
    public IReadOnlyDictionary<string, ColumnMetadata> Metadata => metadata;

    /// <summary>
    /// Checks if a column with the specified name exists
    /// </summary>
    /// <param name="name">The name of the column</param>
    /// <returns>True if the column exists, false otherwise</returns>
    public bool HasColumn(string name)
    {
        return columnTypes.ContainsKey(name);
    }

    /// <summary>
    /// Gets the type of the specified column
    /// </summary>
    /// <param name="name">The name of the column</param>
    /// <returns>The type of the column</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public Type GetColumnType(string name)
    {
        if (!columnTypes.TryGetValue(name, out var type))
            throw new ColumnNotFoundException(name, ColumnNames);

        return type;
    }

    /// <summary>
    /// Gets the metadata for the specified column
    /// </summary>
    /// <param name="name">The name of the column</param>
    /// <returns>The metadata for the column</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public ColumnMetadata GetColumnMetadata(string name)
    {
        if (!metadata.TryGetValue(name, out var meta))
            throw new ColumnNotFoundException(name, ColumnNames);

        return meta;
    }

    /// <summary>
    /// Creates a new schema with an additional column
    /// </summary>
    /// <param name="name">The name of the new column</param>
    /// <param name="type">The type of the new column</param>
    /// <param name="columnMetadata">Optional metadata for the new column</param>
    /// <returns>A new schema with the additional column</returns>
    /// <exception cref="ArgumentException">Thrown when the column name already exists</exception>
    public Schema WithColumn(string name, Type type, ColumnMetadata? columnMetadata = null)
    {
        if (HasColumn(name))
            throw new ArgumentException($"Column '{name}' already exists in the schema", nameof(name));

        var newColumns = ColumnNames.Concat(new[] { name })
            .Zip(columnTypes.Values.Concat(new[] { type }));

        var newMetadata = new Dictionary<string, ColumnMetadata>(metadata, StringComparer.OrdinalIgnoreCase);
        newMetadata[name] = columnMetadata ?? new ColumnMetadata();

        return new Schema(newColumns, newMetadata);
    }

    /// <summary>
    /// Creates a new schema without the specified column
    /// </summary>
    /// <param name="name">The name of the column to remove</param>
    /// <returns>A new schema without the specified column</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public Schema WithoutColumn(string name)
    {
        if (!HasColumn(name))
            throw new ColumnNotFoundException(name, ColumnNames);

        var newColumns = ColumnNames.Where(n => !string.Equals(n, name, StringComparison.OrdinalIgnoreCase))
            .Select(n => (n, columnTypes[n]));

        var newMetadata = metadata.Where(kvp => !string.Equals(kvp.Key, name, StringComparison.OrdinalIgnoreCase))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);

        return new Schema(newColumns, newMetadata);
    }

    /// <summary>
    /// Creates a new schema with only the specified columns
    /// </summary>
    /// <param name="columnNames">The names of the columns to include</param>
    /// <returns>A new schema with only the specified columns</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when any of the specified columns is not found</exception>
    public Schema SelectColumns(IEnumerable<string> columnNames)
    {
        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        var selectedNames = columnNames.ToList();

        // Validate all columns exist
        foreach (var name in selectedNames)
        {
            if (!HasColumn(name))
                throw new ColumnNotFoundException(name, ColumnNames);
        }

        var newColumns = selectedNames.Select(name => (name, columnTypes[name]));
        var newMetadata = selectedNames.ToDictionary(
            name => name,
            name => metadata.TryGetValue(name, out var meta) ? meta : new ColumnMetadata(),
            StringComparer.OrdinalIgnoreCase);

        return new Schema(newColumns, newMetadata);
    }

    /// <summary>
    /// Validates that this schema is compatible with another schema for operations like joins
    /// </summary>
    /// <param name="other">The other schema to compare against</param>
    /// <param name="requireExactMatch">Whether to require exact type matches</param>
    /// <returns>True if the schemas are compatible, false otherwise</returns>
    public bool IsCompatibleWith(Schema other, bool requireExactMatch = true)
    {
        if (other == null)
            return false;

        if (ColumnNames.Count != other.ColumnNames.Count)
            return false;

        for (int i = 0; i < ColumnNames.Count; i++)
        {
            var thisName = ColumnNames[i];
            var otherName = other.ColumnNames[i];

            if (!string.Equals(thisName, otherName, StringComparison.OrdinalIgnoreCase))
                return false;

            var thisType = columnTypes[thisName];
            var otherType = other.columnTypes[otherName];

            if (requireExactMatch)
            {
                if (thisType != otherType)
                    return false;
            }
            else
            {
                // Allow compatible types (e.g., int and long)
                if (!areTypesCompatible(thisType, otherType))
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Returns a string representation of the schema
    /// </summary>
    /// <returns>A formatted string describing the schema</returns>
    public override string ToString()
    {
        var columns = ColumnNames.Select(name => $"{name}: {columnTypes[name].Name}");
        return $"Schema {{ {string.Join(", ", columns)} }}";
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current schema
    /// </summary>
    /// <param name="obj">The object to compare</param>
    /// <returns>True if the objects are equal, false otherwise</returns>
    public override bool Equals(object? obj)
    {
        return obj is Schema other && IsCompatibleWith(other, requireExactMatch: true);
    }

    /// <summary>
    /// Returns a hash code for the schema
    /// </summary>
    /// <returns>A hash code for the schema</returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var name in ColumnNames)
        {
            hash.Add(name.ToLowerInvariant());
            hash.Add(columnTypes[name]);
        }
        return hash.ToHashCode();
    }
}

/// <summary>
/// Represents metadata for a column in a schema
/// </summary>
public sealed class ColumnMetadata
{
    /// <summary>
    /// Initializes a new instance of ColumnMetadata with default values
    /// </summary>
    public ColumnMetadata()
    {
        IsNullable = true;
        DefaultValue = null;
        Description = null;
        Properties = new Dictionary<string, object>();
    }

    /// <summary>
    /// Initializes a new instance of ColumnMetadata with specified values
    /// </summary>
    /// <param name="isNullable">Whether the column can contain null values</param>
    /// <param name="defaultValue">The default value for the column</param>
    /// <param name="description">A description of the column</param>
    /// <param name="properties">Additional properties for the column</param>
    public ColumnMetadata(
        bool isNullable = true,
        object? defaultValue = null,
        string? description = null,
        IReadOnlyDictionary<string, object>? properties = null)
    {
        IsNullable = isNullable;
        DefaultValue = defaultValue;
        Description = description;
        Properties = properties ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Gets a value indicating whether the column can contain null values
    /// </summary>
    public bool IsNullable { get; }

    /// <summary>
    /// Gets the default value for the column
    /// </summary>
    public object? DefaultValue { get; }

    /// <summary>
    /// Gets the description of the column
    /// </summary>
    public string? Description { get; }

    /// <summary>
    /// Gets additional properties for the column
    /// </summary>
    public IReadOnlyDictionary<string, object> Properties { get; }

    /// <summary>
    /// Creates a new ColumnMetadata with updated properties
    /// </summary>
    /// <param name="isNullable">Whether the column can contain null values</param>
    /// <param name="defaultValue">The default value for the column</param>
    /// <param name="description">A description of the column</param>
    /// <param name="properties">Additional properties for the column</param>
    /// <returns>A new ColumnMetadata instance</returns>
    public ColumnMetadata With(
        bool? isNullable = null,
        object? defaultValue = null,
        string? description = null,
        IReadOnlyDictionary<string, object>? properties = null)
    {
        return new ColumnMetadata(
            isNullable ?? IsNullable,
            defaultValue ?? DefaultValue,
            description ?? Description,
            properties ?? Properties);
    }

    /// <summary>
    /// Returns a string representation of the column metadata
    /// </summary>
    /// <returns>A formatted string describing the metadata</returns>
    public override string ToString()
    {
        var parts = new List<string>();

        if (!IsNullable)
            parts.Add("NOT NULL");

        if (DefaultValue != null)
            parts.Add($"DEFAULT {DefaultValue}");

        if (!string.IsNullOrEmpty(Description))
            parts.Add($"DESC '{Description}'");

        return parts.Count > 0 ? $"{{ {string.Join(", ", parts)} }}" : "{}";
    }
}
