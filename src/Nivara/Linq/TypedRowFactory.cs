using Nivara.Exceptions;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Reflection;

namespace Nivara.Linq;

/// <summary>
/// Materializes typed rows from a result frame using a compiled, cached row factory. Anonymous
/// types are built through their constructor (parameter names match the result column names);
/// regular classes are built through the default constructor plus property assignments. One
/// factory is compiled per (row type, result schema) signature and cached.
/// </summary>
/// <typeparam name="T">The row type to materialize</typeparam>
internal static class TypedRowFactory<T>
{
    static readonly ConcurrentDictionary<string, Func<IColumn[], int, T>> cache = new();

    /// <summary>
    /// Gets (building and caching if necessary) a row factory for the given result schema
    /// </summary>
    /// <param name="schema">The result schema</param>
    /// <returns>A delegate that materializes a row from an aligned column array and a row index</returns>
    /// <exception cref="SchemaValidationException">Thrown when the row type cannot be materialized from the schema</exception>
    public static Func<IColumn[], int, T> GetFactory(Schema schema)
    {
        ArgumentNullException.ThrowIfNull(schema);

        var key = BuildKey(schema);
        return cache.GetOrAdd(key, _ => BuildFactory(schema));
    }

    static string BuildKey(Schema schema)
    {
        return string.Join("|", schema.ColumnNames.Select(name => $"{name}:{schema.GetColumnType(name).AssemblyQualifiedName}"));
    }

    static Func<IColumn[], int, T> BuildFactory(Schema schema)
    {
        var type = typeof(T);
        var readable = TypedLinqMetadata.GetReadableProperties(type);

        // Map every readable property to its column index; every property must map.
        var columnLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < schema.ColumnNames.Count; i++)
            columnLookup[schema.ColumnNames[i]] = i;

        var propertyToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var property in readable)
        {
            if (!columnLookup.TryGetValue(property.Name, out var index))
                throw new SchemaValidationException(
                    $"Property '{property.Name}' on row type '{type.Name}' does not map to any column in the result schema. " +
                    $"Available columns: {string.Join(", ", schema.ColumnNames)}.");

            var columnType = schema.GetColumnType(schema.ColumnNames[index]);
            if (!TypedLinqMetadata.ArePropertyTypesCompatible(property.PropertyType, columnType))
                throw new SchemaValidationException(
                    $"Property '{property.Name}' of type '{property.PropertyType.Name}' is incompatible with column " +
                    $"'{schema.ColumnNames[index]}' of type '{columnType.Name}'.");

            propertyToIndex[property.Name] = index;
        }

        var columnsParameter = Expression.Parameter(typeof(IColumn[]), "columns");
        var indexParameter = Expression.Parameter(typeof(int), "rowIndex");

        Expression ReadColumn(int columnIndex, Type targetType)
        {
            var column = Expression.ArrayIndex(columnsParameter, Expression.Constant(columnIndex));
            var getValue = Expression.Call(column, nameof(IColumn.GetValue), Type.EmptyTypes, indexParameter);
            var readMethod = typeof(TypedRowFactory<T>).GetMethod(nameof(ReadValue), BindingFlags.NonPublic | BindingFlags.Static)!;
            return Expression.Call(readMethod.MakeGenericMethod(targetType), getValue, Expression.Constant(schema.ColumnNames[columnIndex]));
        }

        var constructor = FindColumnMatchingConstructor(readable, propertyToIndex);
        Expression body;

        if (constructor is not null)
        {
            var arguments = constructor.GetParameters()
                .Select(parameter => ReadColumn(propertyToIndex[parameter.Name!], parameter.ParameterType))
                .ToArray();
            body = Expression.New(constructor, arguments);
        }
        else
        {
            var defaultConstructor = type.GetConstructor(BindingFlags.Public | BindingFlags.Instance, binder: null, Type.EmptyTypes, modifiers: null)
                ?? throw new SchemaValidationException(
                    $"Row type '{type.Name}' has no parameterless constructor and no constructor whose parameters map to the result columns; " +
                    "it cannot be materialized from a query result.");

            var instance = Expression.New(defaultConstructor);
            var bindings = readable
                .Where(property => property.SetMethod is not null && propertyToIndex.ContainsKey(property.Name))
                .Select(property =>
                    (MemberBinding)Expression.Bind(
                        property,
                        ReadColumn(propertyToIndex[property.Name], property.PropertyType)))
                .ToArray();

            if (bindings.Length == 0)
                throw new SchemaValidationException(
                    $"Row type '{type.Name}' has no settable properties that map to the result schema; it cannot be materialized from a query result.");

            body = Expression.MemberInit(instance, bindings);
        }

        var lambda = Expression.Lambda<Func<IColumn[], int, T>>(body, columnsParameter, indexParameter);
        return lambda.Compile();
    }

    /// <summary>
    /// Finds a public constructor whose parameters cover exactly the readable mapped properties
    /// (matched by name). This is the shape produced by compiler-generated anonymous types and by
    /// immutable records/classes with a full constructor; it avoids double-binding properties.
    /// </summary>
    static ConstructorInfo? FindColumnMatchingConstructor(PropertyInfo[] readable, IReadOnlyDictionary<string, int> propertyToIndex)
    {
        var readableNames = new HashSet<string>(readable.Select(p => p.Name), StringComparer.OrdinalIgnoreCase);

        foreach (var constructor in typeof(T).GetConstructors(BindingFlags.Public | BindingFlags.Instance))
        {
            var parameters = constructor.GetParameters();
            if (parameters.Length == 0)
                continue;

            var parameterNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var allMapped = true;

            foreach (var parameter in parameters)
            {
                if (parameter.Name is null || !propertyToIndex.ContainsKey(parameter.Name))
                {
                    allMapped = false;
                    break;
                }
                parameterNames.Add(parameter.Name);
            }

            if (allMapped && parameterNames.SetEquals(readableNames))
                return constructor;
        }

        return null;
    }

    /// <summary>
    /// Reads and converts a boxed column value to the target property type, producing a clear
    /// diagnostic when a null cannot be assigned to a non-nullable value type.
    /// </summary>
    static TResult ReadValue<TResult>(object? value, string columnName)
    {
        if (value is null)
        {
            if (typeof(TResult).IsValueType && Nullable.GetUnderlyingType(typeof(TResult)) is null)
                throw new SchemaValidationException(
                    $"Column '{columnName}' contains a null value that cannot be assigned to the non-nullable property type '{typeof(TResult)}'.");

            return default!;
        }

        return (TResult)value;
    }
}
