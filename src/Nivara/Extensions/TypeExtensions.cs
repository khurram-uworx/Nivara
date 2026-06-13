namespace Nivara.Extensions;

static class TypeExtensions
{
    /// <summary>
    /// Helper method to check if a type is numeric and supports arithmetic operations
    /// </summary>
    public static bool IsNumericType(this Type type)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;
        return underlying == typeof(int) ||
            underlying == typeof(float) ||
            underlying == typeof(double) ||
            underlying == typeof(long) ||
            underlying == typeof(short) ||
            underlying == typeof(byte) ||
            underlying == typeof(sbyte) ||
            underlying == typeof(uint) ||
            underlying == typeof(ulong) ||
            underlying == typeof(ushort) ||
            underlying == typeof(decimal) ||
            underlying == typeof(bool);
    }

    /// <summary>
    /// Helper method to check if a type supports comparison operations
    /// </summary>
    public static bool IsComparableType(this Type type)
    {
        // All numeric types support comparison
        if (IsNumericType(type))
            return true;

        // String supports comparison
        if (type == typeof(string))
            return true;

        // DateTime and other common comparable types
        if (type == typeof(DateTime) || type == typeof(DateTimeOffset) || type == typeof(TimeSpan))
            return true;

        // Guid supports comparison
        if (type == typeof(Guid))
            return true;

        // Check if type implements IComparable<T> or IComparable
        return typeof(IComparable<>).MakeGenericType(type).IsAssignableFrom(type) ||
               typeof(IComparable).IsAssignableFrom(type);
    }

}
