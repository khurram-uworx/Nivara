namespace Nivara.Helpers;

static class ColumnFilterHelper
{
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

    static IColumn createFilteredColumnGeneric(IColumn column, List<int> indices)
    {
        var filteredArray = new object[indices.Count];
        for (int i = 0; i < indices.Count; i++)
            filteredArray[i] = column.GetValue(indices[i])!;

        return NivaraColumn<object>.Create(filteredArray);
    }

    public static IColumn CreateFilteredColumn(IColumn column, List<int> indices)
    {
        var elementType = column.ElementType;
        return elementType switch
        {
            Type t when t == typeof(int) => createFilteredColumnTyped<int>(column, indices),
            Type t when t == typeof(double) => createFilteredColumnTyped<double>(column, indices),
            Type t when t == typeof(float) => createFilteredColumnTyped<float>(column, indices),
            Type t when t == typeof(long) => createFilteredColumnTyped<long>(column, indices),
            Type t when t == typeof(string) => createFilteredColumnTyped<string>(column, indices),
            Type t when t == typeof(bool) => createFilteredColumnTyped<bool>(column, indices),
            Type t when t == typeof(decimal) => createFilteredColumnTyped<decimal>(column, indices),
            Type t when t == typeof(byte) => createFilteredColumnTyped<byte>(column, indices),
            Type t when t == typeof(short) => createFilteredColumnTyped<short>(column, indices),
            Type t when t == typeof(DateTime) => createFilteredColumnTyped<DateTime>(column, indices),
            _ => createFilteredColumnGeneric(column, indices)
        };
    }

    public static IColumn CreateEmptyColumn(Type elementType)
    {
        return elementType switch
        {
            Type t when t == typeof(int) => NivaraColumn<int>.Create([]),
            Type t when t == typeof(double) => NivaraColumn<double>.Create([]),
            Type t when t == typeof(float) => NivaraColumn<float>.Create([]),
            Type t when t == typeof(long) => NivaraColumn<long>.Create([]),
            Type t when t == typeof(string) => NivaraColumn<string>.CreateForReferenceType([]),
            Type t when t == typeof(bool) => NivaraColumn<bool>.Create([]),
            Type t when t == typeof(decimal) => NivaraColumn<decimal>.Create([]),
            Type t when t == typeof(byte) => NivaraColumn<byte>.Create([]),
            Type t when t == typeof(short) => NivaraColumn<short>.Create([]),
            Type t when t == typeof(DateTime) => NivaraColumn<DateTime>.Create([]),
            _ => NivaraColumn<object>.Create([])
        };
    }
}
