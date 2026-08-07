using Nivara.Exceptions;
using System.Reflection;

namespace Nivara.Linq;

/// <summary>
/// Builds the mapping between typed-row properties and frame columns, and validates that a row
/// type can be used as a typed-query row type (IDEA §6.1). All validation is fail-fast: it runs at
/// query-build time, before any data is touched.
/// </summary>
internal static class TypedLinqMetadata
{
    /// <summary>
    /// Creates a translator for the given row type against the given schema
    /// </summary>
    /// <param name="rowType">The row type</param>
    /// <param name="schema">The schema the row type maps to</param>
    /// <returns>A configured TypedExpressionTranslator</returns>
    public static TypedExpressionTranslator CreateTranslator(Type rowType, Schema schema)
    {
        return new TypedExpressionTranslator(BuildPropertyToColumn(rowType, schema));
    }

    /// <summary>
    /// Builds a case-insensitive property-name to column-name mapping. Every readable property that
    /// matches a schema column (by name, case-insensitively) is included; unmatched properties are
    /// left out and produce a clear diagnostic when referenced by a predicate.
    /// </summary>
    /// <param name="rowType">The row type</param>
    /// <param name="schema">The schema to map against</param>
    /// <returns>The property-to-column mapping</returns>
    public static IReadOnlyDictionary<string, string> BuildPropertyToColumn(Type rowType, Schema schema)
    {
        ArgumentNullException.ThrowIfNull(rowType);
        ArgumentNullException.ThrowIfNull(schema);

        var columnLookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in schema.ColumnNames)
            columnLookup[name] = name;

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var property in rowType.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.GetMethod is null)
                continue;

            if (columnLookup.TryGetValue(property.Name, out var columnName))
                map[property.Name] = columnName;
        }

        return map;
    }

    /// <summary>
    /// Validates that a type can serve as a typed-query row type: a non-primitive class with at
    /// least one readable public property, where every property maps to a schema column whose type
    /// is compatible (exact, or differing only by nullability).
    /// </summary>
    /// <param name="rowType">The row type to validate</param>
    /// <param name="schema">The schema the row type must map to</param>
    /// <exception cref="SchemaValidationException">Thrown when the row type is invalid</exception>
    public static void ValidateRowType(Type rowType, Schema schema)
    {
        ArgumentNullException.ThrowIfNull(rowType);
        ArgumentNullException.ThrowIfNull(schema);

        if (rowType == typeof(string) || rowType.IsPrimitive || rowType.IsValueType || rowType.IsInterface)
            throw new SchemaValidationException(
                $"Row type '{rowType.Name}' must be a non-primitive class with at least one readable public property.");

        var readable = GetReadableProperties(rowType);
        if (readable.Length == 0)
            throw new SchemaValidationException(
                $"Row type '{rowType.Name}' has no readable public properties; a typed query requires at least one mapping property.");

        var columnLookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in schema.ColumnNames)
            columnLookup[name] = name;

        foreach (var property in readable)
        {
            if (!columnLookup.TryGetValue(property.Name, out var columnName))
                throw new SchemaValidationException(
                    $"Property '{property.Name}' on row type '{rowType.Name}' does not map to any column in the frame schema. " +
                    $"Available columns: {string.Join(", ", schema.ColumnNames)}.");

            var columnType = schema.GetColumnType(columnName);
            if (!ArePropertyTypesCompatible(property.PropertyType, columnType))
                throw new SchemaValidationException(
                    $"Property '{property.Name}' of type '{property.PropertyType.Name}' is incompatible with column '{columnName}' " +
                    $"of type '{columnType.Name}'. Properties must match the column type exactly or differ only by nullability.");
        }
    }

    /// <summary>
    /// Determines whether a property type and a column type are compatible for row materialization:
    /// equal, or equal after unwrapping nullable annotations on either side.
    /// </summary>
    public static bool ArePropertyTypesCompatible(Type propertyType, Type columnType)
    {
        var propertyUnderlying = Nullable.GetUnderlyingType(propertyType);
        var columnUnderlying = Nullable.GetUnderlyingType(columnType);

        return propertyType == columnType
            || (propertyUnderlying is not null && propertyUnderlying == columnType)
            || (columnUnderlying is not null && propertyType == columnUnderlying)
            || (propertyUnderlying is not null && columnUnderlying is not null && propertyUnderlying == columnUnderlying);
    }

    /// <summary>
    /// Gets the readable public instance properties of a type, ordered by their declaration offset
    /// within the type to keep factory generation deterministic.
    /// </summary>
    public static PropertyInfo[] GetReadableProperties(Type rowType)
    {
        return rowType
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.GetMethod is not null)
            .OrderBy(p => p.MetadataToken)
            .ToArray();
    }
}
