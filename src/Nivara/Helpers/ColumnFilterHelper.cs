using System.Reflection;

namespace Nivara.Helpers;

static class ColumnFilterHelper
{
    static readonly MethodInfo s_createFilteredColumnTyped = getMethod(nameof(createFilteredColumnTyped));
    static readonly MethodInfo s_reorderColumnTyped = getMethod(nameof(reorderColumnTyped));
    static readonly MethodInfo s_createEmptyColumnTyped = getMethod(nameof(createEmptyColumnTyped));
    static readonly MethodInfo s_concatenateColumnsTyped = getMethod(nameof(concatenateColumnsTyped));
    static readonly MethodInfo s_createNullColumnTyped = getMethod(nameof(createNullColumnTyped));

    static MethodInfo getMethod(string name)
        => typeof(ColumnFilterHelper).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!;

    static Type unwrapNullable(Type elementType)
        => Nullable.GetUnderlyingType(elementType) ?? elementType;

    /// <summary>
    /// Creates a new column containing only the values at the specified indices,
    /// preserving the source column's element type.
    /// </summary>
    public static IColumn CreateFilteredColumn(IColumn column, List<int> indices)
    {
        var elementType = unwrapNullable(column.ElementType);
        return (IColumn)s_createFilteredColumnTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { column, indices })!;
    }

    /// <summary>
    /// Reorders a column using the specified indices,
    /// preserving the source column's element type.
    /// </summary>
    public static IColumn ReorderColumn(IColumn column, int[] indices)
    {
        var elementType = unwrapNullable(column.ElementType);
        return (IColumn)s_reorderColumnTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { column, indices })!;
    }

    /// <summary>
    /// Creates an empty column of the specified element type.
    /// </summary>
    public static IColumn CreateEmptyColumn(Type elementType)
    {
        var targetType = unwrapNullable(elementType);
        return (IColumn)s_createEmptyColumnTyped
            .MakeGenericMethod(targetType)
            .Invoke(null, null)!;
    }

    /// <summary>
    /// Concatenates columns of the same element type,
    /// preserving the source element type.
    /// </summary>
    public static IColumn ConcatenateColumns(List<IColumn> columns)
    {
        if (columns.Count == 1)
            return columns[0];

        var elementType = unwrapNullable(columns[0].ElementType);

        foreach (var column in columns)
            if (unwrapNullable(column.ElementType) != elementType)
                throw new ArgumentException(
                    $"Cannot concatenate columns of different types: {column.ElementType.Name} vs {columns[0].ElementType.Name}");

        return (IColumn)s_concatenateColumnsTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { columns })!;
    }

    /// <summary>
    /// Creates a column filled with null values of the specified element type.
    /// </summary>
    public static IColumn CreateNullColumn(Type elementType, int length)
    {
        var targetType = unwrapNullable(elementType);
        return (IColumn)s_createNullColumnTyped
            .MakeGenericMethod(targetType)
            .Invoke(null, new object[] { length })!;
    }

    static IColumn createFilteredColumnTyped<T>(IColumn column, List<int> indices)
    {
        if (typeof(T).IsValueType)
        {
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var filteredArray = System.Array.CreateInstance(nullableType, indices.Count);

            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    var nullableInstance = Activator.CreateInstance(nullableType, value);
                    filteredArray.SetValue(nullableInstance, i);
                }
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { filteredArray })!;
        }
        else
        {
            var filteredArray = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                filteredArray[i] = (T)value!;
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { filteredArray })!;
        }
    }

    static IColumn reorderColumnTyped<T>(IColumn column, int[] indices)
    {
        if (typeof(T).IsValueType)
        {
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var reorderedArray = System.Array.CreateInstance(nullableType, indices.Length);

            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    var nullableInstance = Activator.CreateInstance(nullableType, value);
                    reorderedArray.SetValue(nullableInstance, i);
                }
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { reorderedArray })!;
        }
        else
        {
            var reorderedArray = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                reorderedArray[i] = (T)value!;
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { reorderedArray })!;
        }
    }

    static IColumn createEmptyColumnTyped<T>()
    {
        if (typeof(T).IsValueType)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        return NivaraColumn<T>.CreateForReferenceType(Array.Empty<T>());
    }

    static IColumn concatenateColumnsTyped<T>(List<IColumn> columns)
    {
        var totalLength = columns.Sum(c => c.Length);

        if (typeof(T).IsValueType)
        {
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var concatenatedArray = System.Array.CreateInstance(nullableType, totalLength);

            int currentIndex = 0;
            foreach (var column in columns)
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

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { concatenatedArray })!;
        }
        else
        {
            var concatenatedArray = new T[totalLength];

            int currentIndex = 0;
            foreach (var column in columns)
                for (int i = 0; i < column.Length; i++)
                {
                    concatenatedArray[currentIndex] = (T)column.GetValue(i)!;
                    currentIndex++;
                }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { concatenatedArray })!;
        }
    }

    static IColumn createNullColumnTyped<T>(int length)
    {
        if (typeof(T).IsValueType)
        {
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var nullArray = System.Array.CreateInstance(nullableType, length);

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { nullArray })!;
        }
        else
        {
            var nullArray = new T[length];

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { nullArray })!;
        }
    }
}
