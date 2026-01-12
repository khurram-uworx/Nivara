using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara;

/// <summary>
/// Specifies how to handle schema mismatches during concatenation
/// </summary>
public enum ConcatenationMismatchHandling
{
    /// <summary>
    /// Throw an error if schemas don't match exactly
    /// </summary>
    Error,

    /// <summary>
    /// Fill missing columns with null values
    /// </summary>
    FillWithNulls
}

/// <summary>
/// Specifies the direction of concatenation
/// </summary>
public enum ConcatenationDirection
{
    /// <summary>
    /// Vertical concatenation (append rows)
    /// </summary>
    Vertical,

    /// <summary>
    /// Horizontal concatenation (append columns)
    /// </summary>
    Horizontal
}

/// <summary>
/// Represents a concatenation operation that combines multiple DataFrames
/// </summary>
internal sealed class ConcatenationOperation : IQueryOperation
{
    readonly IReadOnlyList<IReadOnlyDictionary<string, IColumn>> sources;
    readonly ConcatenationDirection direction;
    readonly ConcatenationMismatchHandling mismatchHandling;

    /// <summary>
    /// Initializes a new instance of ConcatenationOperation for vertical concatenation
    /// </summary>
    /// <param name="sources">The source column dictionaries to concatenate</param>
    /// <param name="direction">The direction of concatenation</param>
    /// <param name="mismatchHandling">How to handle schema mismatches</param>
    /// <exception cref="ArgumentNullException">Thrown when sources is null</exception>
    /// <exception cref="ArgumentException">Thrown when sources is empty</exception>
    public ConcatenationOperation(
        IReadOnlyList<IReadOnlyDictionary<string, IColumn>> sources,
        ConcatenationDirection direction = ConcatenationDirection.Vertical,
        ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.FillWithNulls)
    {
        this.sources = sources ?? throw new ArgumentNullException(nameof(sources));
        this.direction = direction;
        this.mismatchHandling = mismatchHandling;

        if (sources.Count == 0)
            throw new ArgumentException("Must provide at least one source for concatenation", nameof(sources));
    }

    /// <inheritdoc />
    public string OperationType => $"Concatenate{direction}";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // For concatenation, we need to validate all source schemas
        var schemas = new List<Schema> { inputSchema };

        // Extract schemas from sources (this is a simplified approach)
        // In a real implementation, we'd need a way to get schemas from the sources
        foreach (var source in sources.Skip(1))
        {
            var sourceSchema = ExtractSchemaFromColumns(source);
            schemas.Add(sourceSchema);
        }

        return direction switch
        {
            ConcatenationDirection.Vertical => TransformSchemaVertical(schemas),
            ConcatenationDirection.Horizontal => TransformSchemaHorizontal(schemas),
            _ => throw new ArgumentException($"Unknown concatenation direction: {direction}")
        };
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        try
        {
            // Combine input with other sources
            var allSources = new List<IReadOnlyDictionary<string, IColumn>> { input };
            allSources.AddRange(sources);

            return direction switch
            {
                ConcatenationDirection.Vertical => ExecuteVerticalConcatenation(allSources),
                ConcatenationDirection.Horizontal => ExecuteHorizontalConcatenation(allSources),
                _ => throw new ArgumentException($"Unknown concatenation direction: {direction}")
            };
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Concatenation operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Transforms schemas for vertical concatenation
    /// </summary>
    Schema TransformSchemaVertical(List<Schema> schemas)
    {
        if (mismatchHandling == ConcatenationMismatchHandling.Error)
        {
            // All schemas must match exactly
            var firstSchema = schemas[0];
            for (int i = 1; i < schemas.Count; i++)
            {
                if (!firstSchema.IsCompatibleWith(schemas[i], requireExactMatch: true))
                {
                    throw new SchemaValidationException(
                        $"Schema mismatch in vertical concatenation at source {i}. " +
                        $"Expected: {firstSchema}, Actual: {schemas[i]}",
                        firstSchema,
                        schemas[i]);
                }
            }
            return firstSchema;
        }
        else
        {
            // Create union schema with all columns from all sources
            return CreateUnionSchema(schemas);
        }
    }

    /// <summary>
    /// Transforms schemas for horizontal concatenation
    /// </summary>
    Schema TransformSchemaHorizontal(List<Schema> schemas)
    {
        // Validate that all sources have the same row count (this would need to be checked at execution time)
        // For now, just combine all columns
        var allColumns = new List<(string Name, Type Type)>();
        var seenColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var schema in schemas)
        {
            foreach (var columnName in schema.ColumnNames)
            {
                var columnType = schema.ColumnTypes[columnName];
                if (seenColumns.Contains(columnName))
                {
                    throw new SchemaValidationException(
                        $"Column name conflict in horizontal concatenation: '{columnName}' appears in multiple sources",
                        schemas[0],
                        schema);
                }
                seenColumns.Add(columnName);
                allColumns.Add((columnName, columnType));
            }
        }

        return new Schema(allColumns);
    }

    /// <summary>
    /// Creates a union schema that includes all columns from all schemas
    /// </summary>
    Schema CreateUnionSchema(List<Schema> schemas)
    {
        var allColumns = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);

        foreach (var schema in schemas)
        {
            foreach (var columnName in schema.ColumnNames)
            {
                var columnType = schema.ColumnTypes[columnName];
                if (allColumns.TryGetValue(columnName, out var existingType))
                {
                    // Validate type compatibility
                    if (existingType != columnType)
                    {
                        throw new SchemaValidationException(
                            $"Type mismatch for column '{columnName}': {existingType.Name} vs {columnType.Name}",
                            schemas[0],
                            schema);
                    }
                }
                else
                {
                    allColumns[columnName] = columnType;
                }
            }
        }

        var schemaColumns = allColumns.Select(kvp => (kvp.Key, kvp.Value)).ToList();
        return new Schema(schemaColumns);
    }

    /// <summary>
    /// Executes vertical concatenation (row append)
    /// </summary>
    IReadOnlyDictionary<string, IColumn> ExecuteVerticalConcatenation(List<IReadOnlyDictionary<string, IColumn>> allSources)
    {
        if (allSources.Count == 1)
            return allSources[0];

        // Handle empty DataFrames
        var nonEmptySources = allSources.Where(s => s.Values.FirstOrDefault()?.Length > 0).ToList();
        if (nonEmptySources.Count == 0)
        {
            return allSources[0]; // Return first (empty) source
        }
        if (nonEmptySources.Count == 1)
        {
            return nonEmptySources[0]; // Return the only non-empty source
        }

        // Get union of all column names
        var allColumnNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var source in allSources)
        {
            foreach (var columnName in source.Keys)
            {
                allColumnNames.Add(columnName);
            }
        }

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        foreach (var columnName in allColumnNames)
        {
            var columnsToConcat = new List<IColumn>();

            foreach (var source in allSources)
            {
                if (source.TryGetValue(columnName, out var column))
                {
                    columnsToConcat.Add(column);
                }
                else if (mismatchHandling == ConcatenationMismatchHandling.FillWithNulls)
                {
                    // Create null column with same length as source
                    var sourceLength = source.Values.FirstOrDefault()?.Length ?? 0;
                    if (sourceLength > 0)
                    {
                        // Determine column type from other sources
                        var referenceColumn = GetReferenceColumnForType(allSources, columnName);
                        var nullColumn = CreateNullColumn(referenceColumn.ElementType, sourceLength);
                        columnsToConcat.Add(nullColumn);
                    }
                }
                else
                {
                    throw new SchemaValidationException(
                        $"Column '{columnName}' is missing from one of the sources and MismatchHandling is set to Error");
                }
            }

            if (columnsToConcat.Count > 0)
            {
                result[columnName] = ConcatenateColumns(columnsToConcat);
            }
        }

        return result;
    }

    /// <summary>
    /// Executes horizontal concatenation (column append)
    /// </summary>
    IReadOnlyDictionary<string, IColumn> ExecuteHorizontalConcatenation(List<IReadOnlyDictionary<string, IColumn>> allSources)
    {
        if (allSources.Count == 1)
            return allSources[0];

        // Validate that all sources have the same row count
        var expectedRowCount = allSources[0].Values.FirstOrDefault()?.Length ?? 0;
        for (int i = 1; i < allSources.Count; i++)
        {
            var actualRowCount = allSources[i].Values.FirstOrDefault()?.Length ?? 0;
            if (actualRowCount != expectedRowCount)
            {
                throw new ArgumentException(
                    $"Row count mismatch in horizontal concatenation: source 0 has {expectedRowCount} rows, " +
                    $"but source {i} has {actualRowCount} rows");
            }
        }

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        foreach (var source in allSources)
        {
            foreach (var kvp in source)
            {
                if (result.ContainsKey(kvp.Key))
                {
                    throw new ArgumentException(
                        $"Column name conflict in horizontal concatenation: '{kvp.Key}' appears in multiple sources");
                }
                result[kvp.Key] = kvp.Value;
            }
        }

        return result;
    }

    /// <summary>
    /// Gets a reference column to determine the type for null columns
    /// </summary>
    IColumn GetReferenceColumnForType(List<IReadOnlyDictionary<string, IColumn>> allSources, string columnName)
    {
        foreach (var source in allSources)
        {
            if (source.TryGetValue(columnName, out var column))
            {
                return column;
            }
        }
        throw new InvalidOperationException($"No reference column found for '{columnName}'");
    }

    /// <summary>
    /// Creates a column filled with null values of the specified type
    /// </summary>
    IColumn CreateNullColumn(Type elementType, int length)
    {
        return elementType switch
        {
            Type t when t == typeof(int) => CreateNullColumnTyped<int>(length),
            Type t when t == typeof(double) => CreateNullColumnTyped<double>(length),
            Type t when t == typeof(float) => CreateNullColumnTyped<float>(length),
            Type t when t == typeof(long) => CreateNullColumnTyped<long>(length),
            Type t when t == typeof(string) => CreateNullColumnTyped<string>(length),
            Type t when t == typeof(bool) => CreateNullColumnTyped<bool>(length),
            Type t when t == typeof(decimal) => CreateNullColumnTyped<decimal>(length),
            Type t when t == typeof(byte) => CreateNullColumnTyped<byte>(length),
            Type t when t == typeof(short) => CreateNullColumnTyped<short>(length),
            Type t when t == typeof(DateTime) => CreateNullColumnTyped<DateTime>(length),
            _ => CreateNullColumnTyped<object>(length)
        };
    }

    /// <summary>
    /// Creates a typed null column
    /// </summary>
    static IColumn CreateNullColumnTyped<T>(int length)
    {
        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array filled with nulls
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var nullArray = System.Array.CreateInstance(nullableType, length);
            // Array is already filled with nulls by default

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { nullArray })!;
        }
        else
        {
            // For reference types, create array filled with nulls
            var nullArray = new T[length];
            // Array is already filled with nulls by default

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { nullArray })!;
        }
    }

    /// <summary>
    /// Concatenates multiple columns of the same type
    /// </summary>
    IColumn ConcatenateColumns(List<IColumn> columns)
    {
        if (columns.Count == 1)
            return columns[0];

        var elementType = columns[0].ElementType;

        // Validate all columns have the same type
        foreach (var column in columns)
        {
            if (column.ElementType != elementType)
            {
                throw new ArgumentException(
                    $"Cannot concatenate columns of different types: {elementType.Name} vs {column.ElementType.Name}");
            }
        }

        return elementType switch
        {
            Type t when t == typeof(int) => ConcatenateColumnsTyped<int>(columns),
            Type t when t == typeof(double) => ConcatenateColumnsTyped<double>(columns),
            Type t when t == typeof(float) => ConcatenateColumnsTyped<float>(columns),
            Type t when t == typeof(long) => ConcatenateColumnsTyped<long>(columns),
            Type t when t == typeof(string) => ConcatenateColumnsTyped<string>(columns),
            Type t when t == typeof(bool) => ConcatenateColumnsTyped<bool>(columns),
            Type t when t == typeof(decimal) => ConcatenateColumnsTyped<decimal>(columns),
            Type t when t == typeof(byte) => ConcatenateColumnsTyped<byte>(columns),
            Type t when t == typeof(short) => ConcatenateColumnsTyped<short>(columns),
            Type t when t == typeof(DateTime) => ConcatenateColumnsTyped<DateTime>(columns),
            _ => ConcatenateColumnsGeneric(columns)
        };
    }

    /// <summary>
    /// Concatenates columns of a specific type
    /// </summary>
    static IColumn ConcatenateColumnsTyped<T>(List<IColumn> columns)
    {
        var totalLength = columns.Sum(c => c.Length);

        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var concatenatedArray = System.Array.CreateInstance(nullableType, totalLength);

            int currentIndex = 0;
            foreach (var column in columns)
            {
                for (int i = 0; i < column.Length; i++)
                {
                    var value = column.GetValue(i);
                    if (value != null)
                    {
                        var nullableInstance = Activator.CreateInstance(nullableType, value);
                        concatenatedArray.SetValue(nullableInstance, currentIndex);
                    }
                    currentIndex++;
                }
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { concatenatedArray })!;
        }
        else
        {
            // For reference types, create regular array
            var concatenatedArray = new T[totalLength];

            int currentIndex = 0;
            foreach (var column in columns)
            {
                for (int i = 0; i < column.Length; i++)
                {
                    var value = column.GetValue(i);
                    concatenatedArray[currentIndex] = (T)value!;
                    currentIndex++;
                }
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { concatenatedArray })!;
        }
    }

    /// <summary>
    /// Concatenates columns of unknown type using object columns
    /// </summary>
    static IColumn ConcatenateColumnsGeneric(List<IColumn> columns)
    {
        var totalLength = columns.Sum(c => c.Length);
        var concatenatedArray = new object[totalLength];

        int currentIndex = 0;
        foreach (var column in columns)
        {
            for (int i = 0; i < column.Length; i++)
            {
                concatenatedArray[currentIndex] = column.GetValue(i)!;
                currentIndex++;
            }
        }

        return NivaraColumn<object>.Create(concatenatedArray);
    }

    /// <summary>
    /// Extracts schema from a column dictionary
    /// </summary>
    static Schema ExtractSchemaFromColumns(IReadOnlyDictionary<string, IColumn> columns)
    {
        var schemaColumns = columns.Select(kvp => (kvp.Key, kvp.Value.ElementType)).ToList();
        return new Schema(schemaColumns);
    }

    /// <summary>
    /// Returns a string representation of the concatenation operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        return $"Concatenate{direction}({sources.Count} sources, {mismatchHandling})";
    }
}