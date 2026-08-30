using System.Reflection;

namespace Nivara.Helpers;

static class ColumnFilterHelper
{
    static readonly MethodInfo s_createFilteredColumnTyped = getMethod(nameof(createFilteredColumnTyped));
    static readonly MethodInfo s_reorderColumnTyped = getMethod(nameof(reorderColumnTyped));
    static readonly MethodInfo s_createEmptyColumnTyped = getMethod(nameof(createEmptyColumnTyped));
    static readonly MethodInfo s_concatenateColumnsTyped = getMethod(nameof(concatenateColumnsTyped));
    static readonly MethodInfo s_createNullColumnTyped = getMethod(nameof(createNullColumnTyped));
    static readonly MethodInfo s_scatterPartsTyped = getMethod(nameof(scatterPartsTyped));

    static MethodInfo getMethod(string name)
        => typeof(ColumnFilterHelper).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!;

    /// <summary>
    /// Creates a new column containing only the values at the specified indices,
    /// preserving the source column's element type (including nullable element types such as
    /// <c>NivaraColumn&lt;int?&gt;</c>).
    /// </summary>
    public static IColumn CreateFilteredColumn(IColumn column, List<int> indices)
    {
        var elementType = column.ElementType;
        return (IColumn)s_createFilteredColumnTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { column, indices })!;
    }

    /// <summary>
    /// Reorders a column using the specified indices,
    /// preserving the source column's element type (including nullable element types).
    /// </summary>
    public static IColumn ReorderColumn(IColumn column, int[] indices)
    {
        var elementType = column.ElementType;
        return (IColumn)s_reorderColumnTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { column, indices })!;
    }

    /// <summary>
    /// Creates an empty column of the specified element type.
    /// </summary>
    public static IColumn CreateEmptyColumn(Type elementType)
    {
        var targetType = elementType;
        return (IColumn)s_createEmptyColumnTyped
            .MakeGenericMethod(targetType)
            .Invoke(null, null)!;
    }

    /// <summary>
    /// Concatenates columns of the same element type,
    /// preserving the source element type (including nullable element types).
    /// </summary>
    public static IColumn ConcatenateColumns(List<IColumn> columns)
    {
        if (columns.Count == 1)
            return columns[0];

        var elementType = columns[0].ElementType;

        foreach (var column in columns)
            if (column.ElementType != elementType)
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
        var targetType = elementType;
        return (IColumn)s_createNullColumnTyped
            .MakeGenericMethod(targetType)
            .Invoke(null, new object[] { length })!;
    }

    /// <summary>
    /// Scatters concatenated partition results back to the original row order in a single pass.
    /// <paramref name="positions"/> maps each concatenated position to its target row
    /// (<c>result[positions[i]] = value[i]</c>); partition boundaries are derived from the part lengths.
    /// </summary>
    public static IColumn ScatterPartsColumn(IReadOnlyList<IColumn> parts, int[] positions)
    {
        var elementType = parts[0].ElementType;

        foreach (var column in parts)
            if (column.ElementType != elementType)
                throw new ArgumentException(
                    $"Cannot scatter columns of different types: {column.ElementType.Name} vs {parts[0].ElementType.Name}");

        return (IColumn)s_scatterPartsTyped
            .MakeGenericMethod(elementType)
            .Invoke(null, new object[] { parts, positions })!;
    }

    static IColumn createFilteredColumnTyped<T>(IColumn column, List<int> indices)
    {
        if (typeof(T).IsValueType && column is NivaraColumn<T> typed)
        {
            var filteredValues = new T[indices.Count];
            var nullMask = new bool[indices.Count];
            bool hasNulls = typed.HasNulls;
            bool anyNull = false;

            for (int i = 0; i < indices.Count; i++)
            {
                int index = indices[i];
                if (hasNulls && typed.IsNull(index))
                {
                    nullMask[i] = true;
                    anyNull = true;
                }
                else
                {
                    filteredValues[i] = typed[index];
                }
            }

            if (!anyNull)
                return NivaraColumn<T>.CreateFromOwnedArray(filteredValues);
            return NivaraColumn<T>.CreateFromSpans(filteredValues, nullMask);
        }

        if (typeof(T).IsValueType)
        {
            var filteredValues = new T[indices.Count];
            var nullMask = new bool[indices.Count];

            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    filteredValues[i] = (T)value;
                }
                else
                {
                    nullMask[i] = true;
                }
            }

            return NivaraColumn<T>.CreateFromSpans(filteredValues, nullMask);
        }
        else
        {
            var filteredArray = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                filteredArray[i] = (T)value!;
            }

            return NivaraColumn<T>.CreateForReferenceType(filteredArray);
        }
    }

    static IColumn reorderColumnTyped<T>(IColumn column, int[] indices)
    {
        if (typeof(T).IsValueType && column is NivaraColumn<T> typed)
        {
            var reorderedValues = new T[indices.Length];
            var nullMask = new bool[indices.Length];
            bool hasNulls = typed.HasNulls;
            bool anyNull = false;
            for (int i = 0; i < indices.Length; i++)
            {
                int index = indices[i];
                if (hasNulls && typed.IsNull(index))
                {
                    nullMask[i] = true;
                    anyNull = true;
                }
                else
                {
                    reorderedValues[i] = typed[index];
                }
            }

            if (anyNull)
                return NivaraColumn<T>.CreateFromSpans(reorderedValues, nullMask);
            return NivaraColumn<T>.CreateFromOwnedArray(reorderedValues);
        }

        if (typeof(T).IsValueType)
        {
            var reorderedValues = new T[indices.Length];
            var nullMask = new bool[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    reorderedValues[i] = (T)value;
                }
                else
                {
                    nullMask[i] = true;
                }
            }

            return NivaraColumn<T>.CreateFromSpans(reorderedValues, nullMask);
        }
        else
        {
            var reorderedArray = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                reorderedArray[i] = (T)value!;
            }

            return NivaraColumn<T>.CreateForReferenceType(reorderedArray);
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
        if (columns.Count == 1)
            return columns[0];

        var totalLength = columns.Sum(c => c.Length);

        if (typeof(T).IsValueType && columns.All(c => c is NivaraColumn<T>))
        {
            var concatenatedValues = new T[totalLength];
            var nullMask = new bool[totalLength];

            int currentIndex = 0;
            foreach (var column in columns)
            {
                var typedColumn = (NivaraColumn<T>)column;
                for (int i = 0; i < column.Length; i++)
                {
                    if (typedColumn.IsNull(i))
                        nullMask[currentIndex] = true;
                    else
                        concatenatedValues[currentIndex] = typedColumn[i];
                    currentIndex++;
                }
            }

            if (!nullMask.Any())
                return NivaraColumn<T>.CreateFromOwnedArray(concatenatedValues);
            return NivaraColumn<T>.CreateFromSpans(concatenatedValues, nullMask);
        }

        if (typeof(T).IsValueType)
        {
            var concatenatedValues = new T[totalLength];
            var nullMask = new bool[totalLength];

            int currentIndex = 0;
            foreach (var column in columns)
            {
                for (int i = 0; i < column.Length; i++)
                {
                    var value = column.GetValue(i);
                    if (value != null)
                    {
                        concatenatedValues[currentIndex] = (T)value;
                    }
                    else
                    {
                        nullMask[currentIndex] = true;
                    }
                    currentIndex++;
                }
            }

            return NivaraColumn<T>.CreateFromSpans(concatenatedValues, nullMask);
        }
        else
        {
            var concatenatedArray = new T[totalLength];

            int currentIndex = 0;
            foreach (var column in columns)
            {
                for (int i = 0; i < column.Length; i++)
                {
                    concatenatedArray[currentIndex] = (T)column.GetValue(i)!;
                    currentIndex++;
                }
            }

            return NivaraColumn<T>.CreateForReferenceType(concatenatedArray);
        }
    }

    static IColumn createNullColumnTyped<T>(int length)
    {
        if (typeof(T).IsValueType)
        {
            var nullMask = new bool[length];
            Array.Fill(nullMask, true);
            return NivaraColumn<T>.CreateFromSpans(new T[length], nullMask);
        }
        else
        {
            var nullArray = new T[length];
            return NivaraColumn<T>.CreateForReferenceType(nullArray);
        }
    }

    static IColumn scatterPartsTyped<T>(IReadOnlyList<IColumn> parts, int[] positions)
    {
        if (typeof(T).IsValueType && parts.All(p => p is NivaraColumn<T>))
        {
            var result = new T[positions.Length];
            var nullMask = new bool[positions.Length];
            bool anyNull = false;
            int pos = 0;
            foreach (NivaraColumn<T> part in parts)
            {
                bool hasNulls = part.HasNulls;
                for (int i = 0; i < part.Length; i++)
                {
                    int target = positions[pos];
                    if (hasNulls && part.IsNull(i))
                    {
                        nullMask[target] = true;
                        anyNull = true;
                    }
                    else
                    {
                        result[target] = part[i];
                    }
                    pos++;
                }
            }

            if (anyNull)
                return NivaraColumn<T>.CreateFromSpans(result, nullMask);
            return NivaraColumn<T>.CreateFromOwnedArray(result);
        }

        if (typeof(T).IsValueType)
        {
            var result = new T[positions.Length];
            var nullMask = new bool[positions.Length];

            int pos = 0;
            foreach (var part in parts)
            {
                for (int i = 0; i < part.Length; i++)
                {
                    var value = part.GetValue(i);
                    if (value != null)
                        result[positions[pos]] = (T)value;
                    else
                        nullMask[positions[pos]] = true;
                    pos++;
                }
            }

            return NivaraColumn<T>.CreateFromSpans(result, nullMask);
        }
        else
        {
            var result = new T[positions.Length];

            int pos = 0;
            foreach (var part in parts)
            {
                for (int i = 0; i < part.Length; i++)
                {
                    result[positions[pos]] = (T)part.GetValue(i)!;
                    pos++;
                }
            }

            return NivaraColumn<T>.CreateForReferenceType(result);
        }
    }
}
