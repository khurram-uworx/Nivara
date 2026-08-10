using System.Collections.Concurrent;
using System.Reflection;

namespace Nivara.Helpers;

/// <summary>
/// Creates typed <see cref="NivaraColumn{T}"/> instances from a runtime element type and
/// boxed values. This is the single authoritative dynamic column-creation dispatch used by
/// query operations (aggregation, group-by, join coalesce/gather) and the fused evaluator.
/// </summary>
/// <remarks>
/// Dispatch uses cached <see cref="MethodInfo.MakeGenericMethod(Type)"/> over null-safe
/// kernels that handle any element type, so the extended CLR domain (<c>Half</c>,
/// <c>nint</c>/<c>nuint</c>, <c>Int128</c>/<c>UInt128</c>, <c>DateOnly</c>/<c>TimeOnly</c>,
/// <c>DateTimeOffset</c>, etc.) never falls through to an object column. Added as part of
/// issue #158.
/// </remarks>
static class ColumnFactory
{
    static readonly MethodInfo createValueTypeKernel = getMethod(nameof(createValueTypeColumn));
    static readonly MethodInfo createReferenceKernel = getMethod(nameof(createReferenceColumn));

    static readonly ConcurrentDictionary<Type, MethodInfo> valueTypeCache = new();
    static readonly ConcurrentDictionary<Type, MethodInfo> referenceCache = new();

    static MethodInfo getMethod(string name)
        => typeof(ColumnFactory).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic)!;

    /// <summary>
    /// Creates a typed column from an array of boxed values.
    /// </summary>
    /// <param name="elementType">The element type (a <c>Nullable&lt;T&gt;</c> is unwrapped)</param>
    /// <param name="values">The values; null entries become null positions in the column</param>
    /// <returns>A typed column whose element type matches <paramref name="elementType"/></returns>
    public static IColumn Create(Type elementType, object?[] values)
    {
        ArgumentNullException.ThrowIfNull(elementType);
        ArgumentNullException.ThrowIfNull(values);

        var targetType = Nullable.GetUnderlyingType(elementType) ?? elementType;
        var kernel = targetType.IsValueType
            ? valueTypeCache.GetOrAdd(targetType, static t => createValueTypeKernel.MakeGenericMethod(t))
            : referenceCache.GetOrAdd(targetType, static t => createReferenceKernel.MakeGenericMethod(t));
        return (IColumn)kernel.Invoke(null, new object?[] { values })!;
    }

    static IColumn createValueTypeColumn<T>(object?[] values)
        where T : struct
    {
        var nullable = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (value is not null)
                nullable[i] = (T)value;
        }

        return NivaraColumn<T>.CreateFromNullable(nullable);
    }

    static IColumn createReferenceColumn<T>(object?[] values)
        => NivaraColumn<T>.CreateForReferenceType(values.Cast<T>().ToArray());
}
