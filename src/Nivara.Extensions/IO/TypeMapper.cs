using Apache.Arrow;
using Apache.Arrow.Types;
using Parquet.Schema;
using Nivara.IO;

namespace Nivara.IO;

/// <summary>
/// Provides type mapping between CLR types, Apache Arrow types, and Parquet types
/// </summary>
internal static class TypeMapper
{
    // CLR to Arrow type mapping
    private static readonly Dictionary<Type, IArrowType> ClrToArrowMap = new()
    {
        { typeof(bool), BooleanType.Default },
        { typeof(int), Int32Type.Default },
        { typeof(long), Int64Type.Default },
        { typeof(float), FloatType.Default },
        { typeof(double), DoubleType.Default },
        { typeof(DateTime), new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc) },
        { typeof(string), StringType.Default },
        { typeof(byte), UInt8Type.Default },
        { typeof(short), Int16Type.Default },
        { typeof(uint), UInt32Type.Default },
        { typeof(ulong), UInt64Type.Default },
        { typeof(ushort), UInt16Type.Default },
        { typeof(sbyte), Int8Type.Default }
    };

    // Arrow to CLR type mapping
    private static readonly Dictionary<Type, Type> ArrowToClrMap = new()
    {
        { typeof(BooleanType), typeof(bool) },
        { typeof(Int32Type), typeof(int) },
        { typeof(Int64Type), typeof(long) },
        { typeof(FloatType), typeof(float) },
        { typeof(DoubleType), typeof(double) },
        { typeof(TimestampType), typeof(DateTime) },
        { typeof(StringType), typeof(string) },
        { typeof(UInt8Type), typeof(byte) },
        { typeof(Int16Type), typeof(short) },
        { typeof(UInt32Type), typeof(uint) },
        { typeof(UInt64Type), typeof(ulong) },
        { typeof(UInt16Type), typeof(ushort) },
        { typeof(Int8Type), typeof(sbyte) }
    };

    /// <summary>
    /// Maps a CLR type to the corresponding Arrow type
    /// </summary>
    /// <param name="clrType">The CLR type to map</param>
    /// <returns>The corresponding Arrow type</returns>
    /// <exception cref="UnsupportedTypeException">Thrown when the CLR type is not supported</exception>
    public static IArrowType MapClrToArrow(Type clrType)
    {
        // Handle nullable types by extracting the underlying type
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;

        if (ClrToArrowMap.TryGetValue(actualType, out var arrowType))
        {
            return arrowType;
        }

        // Provide suggestions for common unsupported types
        var suggestions = GetTypeSuggestions(actualType);
        throw new UnsupportedTypeException(actualType, suggestions);
    }

    /// <summary>
    /// Maps an Arrow type to the corresponding CLR type
    /// </summary>
    /// <param name="arrowType">The Arrow type to map</param>
    /// <returns>The corresponding CLR type</returns>
    /// <exception cref="UnsupportedTypeException">Thrown when the Arrow type is not supported</exception>
    public static Type MapArrowToClr(IArrowType arrowType)
    {
        var arrowTypeType = arrowType.GetType();

        if (ArrowToClrMap.TryGetValue(arrowTypeType, out var clrType))
        {
            return clrType;
        }

        // Handle special cases
        if (arrowType is TimestampType)
        {
            return typeof(DateTime);
        }

        throw new UnsupportedTypeException(arrowTypeType, new[] { "bool", "int", "long", "float", "double", "DateTime", "string" });
    }

    /// <summary>
    /// Creates a Parquet field for the specified CLR type
    /// </summary>
    /// <param name="name">The field name</param>
    /// <param name="clrType">The CLR type</param>
    /// <returns>A Parquet DataField</returns>
    /// <exception cref="UnsupportedTypeException">Thrown when the CLR type is not supported for Parquet</exception>
    public static DataField CreateParquetField(string name, Type clrType)
    {
        // Handle nullable types
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        var isNullable = Nullable.GetUnderlyingType(clrType) != null || !actualType.IsValueType;

        return actualType switch
        {
            Type t when t == typeof(bool) => new DataField<bool>(name, isNullable),
            Type t when t == typeof(int) => new DataField<int>(name, isNullable),
            Type t when t == typeof(long) => new DataField<long>(name, isNullable),
            Type t when t == typeof(float) => new DataField<float>(name, isNullable),
            Type t when t == typeof(double) => new DataField<double>(name, isNullable),
            Type t when t == typeof(DateTime) => new DataField<DateTime>(name, isNullable),
            Type t when t == typeof(string) => new DataField<string>(name, isNullable),
            Type t when t == typeof(byte) => new DataField<byte>(name, isNullable),
            Type t when t == typeof(short) => new DataField<short>(name, isNullable),
            Type t when t == typeof(uint) => new DataField<uint>(name, isNullable),
            Type t when t == typeof(ulong) => new DataField<ulong>(name, isNullable),
            Type t when t == typeof(ushort) => new DataField<ushort>(name, isNullable),
            Type t when t == typeof(sbyte) => new DataField<sbyte>(name, isNullable),
            Type t when t == typeof(decimal) => new DataField<decimal>(name, isNullable),
            _ => throw new UnsupportedTypeException(actualType, GetTypeSuggestions(actualType))
        };
    }

    /// <summary>
    /// Checks if a CLR type is supported for Arrow conversion
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True if the type is supported, false otherwise</returns>
    public static bool IsArrowSupported(Type clrType)
    {
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        return ClrToArrowMap.ContainsKey(actualType);
    }

    /// <summary>
    /// Checks if a CLR type is supported for Parquet conversion
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True if the type is supported, false otherwise</returns>
    public static bool IsParquetSupported(Type clrType)
    {
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        
        return actualType == typeof(bool) ||
               actualType == typeof(int) ||
               actualType == typeof(long) ||
               actualType == typeof(float) ||
               actualType == typeof(double) ||
               actualType == typeof(DateTime) ||
               actualType == typeof(string) ||
               actualType == typeof(byte) ||
               actualType == typeof(short) ||
               actualType == typeof(uint) ||
               actualType == typeof(ulong) ||
               actualType == typeof(ushort) ||
               actualType == typeof(sbyte) ||
               actualType == typeof(decimal);
    }

    /// <summary>
    /// Gets all supported CLR types for I/O operations
    /// </summary>
    /// <returns>A collection of supported CLR types</returns>
    public static IEnumerable<Type> GetSupportedTypes()
    {
        return ClrToArrowMap.Keys;
    }

    /// <summary>
    /// Gets suggested alternative types for unsupported types
    /// </summary>
    /// <param name="unsupportedType">The unsupported type</param>
    /// <returns>A list of suggested alternative type names</returns>
    public static List<string> GetTypeSuggestions(Type unsupportedType)
    {
        return unsupportedType switch
        {
            Type t when t == typeof(Guid) => new List<string> { "string", "byte[]" },
            Type t when t == typeof(TimeSpan) => new List<string> { "long (ticks)", "double (seconds)" },
            Type t when t == typeof(DateOnly) => new List<string> { "DateTime", "string" },
            Type t when t == typeof(TimeOnly) => new List<string> { "TimeSpan", "string" },
            Type t when t == typeof(char) => new List<string> { "string", "ushort" },
            Type t when t.IsEnum => new List<string> { "int", "string" },
            Type t when t.IsArray => new List<string> { "Use individual columns for array elements" },
            Type t when t.IsGenericType => new List<string> { "Break down into primitive components" },
            _ => new List<string> { "bool", "int", "long", "float", "double", "DateTime", "string" }
        };
    }
}
