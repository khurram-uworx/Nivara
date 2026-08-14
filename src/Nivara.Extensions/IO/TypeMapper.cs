using Apache.Arrow.Types;
using Parquet.Schema;

namespace Nivara.IO;

/// <summary>
/// Provides type mapping between CLR types, Apache Arrow types, and Parquet types
/// </summary>
static class TypeMapper
{
    /// <summary>
    /// Prefix for schema metadata keys that preserve the original CLR column type across a
    /// Parquet or Arrow round-trip when the on-disk representation uses a widened type
    /// (e.g. <see cref="Half"/> stored as <see cref="float"/>, <see cref="nint"/> stored as
    /// <see cref="long"/>). The key suffix is the column name.
    /// </summary>
    internal const string ClrTypeMetadataKeyPrefix = "nivara.clrType.";

    // CLR to Arrow type mapping
    private static readonly Dictionary<Type, IArrowType> ClrToArrowMap = new()
    {
        { typeof(bool), BooleanType.Default },
        { typeof(int), Int32Type.Default },
        { typeof(long), Int64Type.Default },
        { typeof(float), FloatType.Default },
        { typeof(double), DoubleType.Default },
        { typeof(DateTime), new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc) },
        { typeof(DateTimeOffset), new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc) },
        { typeof(string), StringType.Default },
        { typeof(byte), UInt8Type.Default },
        { typeof(short), Int16Type.Default },
        { typeof(uint), UInt32Type.Default },
        { typeof(ulong), UInt64Type.Default },
        { typeof(ushort), UInt16Type.Default },
        { typeof(sbyte), Int8Type.Default },
        { typeof(Half), new HalfFloatType() },
        { typeof(nint), Int64Type.Default },
        { typeof(nuint), UInt64Type.Default },
        { typeof(char), StringType.Default },
        { typeof(DateOnly), new Date32Type() },
        { typeof(TimeOnly), new Time64Type(TimeUnit.Nanosecond) },
        { typeof(Guid), new FixedSizeBinaryType(16) },
        { typeof(TimeSpan), DurationType.Nanosecond }
    };

    // Arrow to CLR type mapping. Types with a shared Arrow representation (nint/nuint/char
    // via Int64/UInt64/String, DateTimeOffset via Timestamp) intentionally map to the base
    // CLR type at schema level; the original type is restored from metadata when present.
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
        { typeof(Int8Type), typeof(sbyte) },
        { typeof(HalfFloatType), typeof(Half) },
        { typeof(Date32Type), typeof(DateOnly) },
        { typeof(Date64Type), typeof(DateOnly) },
        { typeof(Time32Type), typeof(TimeOnly) },
        { typeof(Time64Type), typeof(TimeOnly) },
        { typeof(DurationType), typeof(TimeSpan) },
        { typeof(FixedSizeBinaryType), typeof(Guid) }
    };

    /// <summary>
    /// Maps a CLR type to the corresponding Arrow type
    /// </summary>
    /// <param name="clrType">The CLR type to map</param>
    /// <returns>The corresponding Arrow type</returns>
    /// <exception cref="ArgumentNullException">Thrown when clrType is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the CLR type is not supported</exception>
    public static IArrowType MapClrToArrow(Type clrType)
    {
        ArgumentNullException.ThrowIfNull(clrType);

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
    /// <exception cref="ArgumentNullException">Thrown when arrowType is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the Arrow type is not supported</exception>
    public static Type MapArrowToClr(IArrowType arrowType)
    {
        ArgumentNullException.ThrowIfNull(arrowType);

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
    /// <exception cref="ArgumentNullException">Thrown when name or clrType is null</exception>
    /// <exception cref="ArgumentException">Thrown when name is empty or whitespace</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the CLR type is not supported for Parquet</exception>
    public static DataField CreateParquetField(string name, Type clrType)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(clrType);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Field name cannot be empty or whitespace", nameof(name));

        // Handle nullable types
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        var isNullable = Nullable.GetUnderlyingType(clrType) != null || !actualType.IsValueType;

        return CreateParquetField(name, actualType, isNullable);
    }

    /// <summary>
    /// Creates a Parquet field for the specified CLR type with explicit nullability.
    /// Extended-domain types either map to a native Parquet.Net <see cref="DataField{T}"/>
    /// (<see cref="DateOnly"/>, <see cref="TimeOnly"/>, <see cref="Guid"/>) or to a lossless
    /// widened representation (<see cref="Half"/> as <see cref="float"/>, <see cref="nint"/>
    /// as <see cref="long"/>, <see cref="nuint"/> as <see cref="ulong"/>, <see cref="char"/>
    /// as <see cref="ushort"/>, <see cref="DateTimeOffset"/> as <see cref="DateTime"/>,
    /// <see cref="TimeSpan"/> as <see cref="long"/>).
    /// </summary>
    /// <param name="name">The field name</param>
    /// <param name="actualType">The non-nullable CLR type</param>
    /// <param name="isNullable">Whether the field permits nulls</param>
    /// <returns>A Parquet DataField</returns>
    /// <exception cref="UnsupportedTypeException">Thrown when the CLR type is not supported for Parquet</exception>
    internal static DataField CreateParquetField(string name, Type actualType, bool isNullable)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(actualType);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Field name cannot be empty or whitespace", nameof(name));

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
            Type t when t == typeof(DateOnly) => new DataField<DateOnly>(name, isNullable),
            Type t when t == typeof(TimeOnly) => new DataField<TimeOnly>(name, isNullable),
            Type t when t == typeof(Guid) => new DataField<Guid>(name, isNullable),
            Type t when t == typeof(Half) => new DataField<float>(name, isNullable),
            Type t when t == typeof(nint) => new DataField<long>(name, isNullable),
            Type t when t == typeof(nuint) => new DataField<ulong>(name, isNullable),
            Type t when t == typeof(char) => new DataField<ushort>(name, isNullable),
            Type t when t == typeof(DateTimeOffset) => new DataField<DateTime>(name, isNullable),
            Type t when t == typeof(TimeSpan) => new DataField<long>(name, isNullable),
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
    /// Checks if a CLR type is supported for Parquet conversion. Extended-domain types are
    /// supported either natively (<see cref="DateOnly"/>, <see cref="TimeOnly"/>,
    /// <see cref="Guid"/>) or via a lossless widened representation (<see cref="Half"/> as
    /// <see cref="float"/>, <see cref="nint"/> as <see cref="long"/>, etc.).
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True if the type is supported, false otherwise</returns>
    public static bool IsParquetSupported(Type clrType)
    {
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;

        return IsStringType(actualType) ||
               actualType == typeof(bool) ||
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
               actualType == typeof(decimal) ||
               actualType == typeof(Half) ||
               actualType == typeof(nint) ||
               actualType == typeof(nuint) ||
               actualType == typeof(char) ||
               actualType == typeof(DateOnly) ||
               actualType == typeof(TimeOnly) ||
               actualType == typeof(DateTimeOffset) ||
               actualType == typeof(Guid) ||
               actualType == typeof(TimeSpan);
    }

    /// <summary>
    /// Checks if a CLR type can be represented as an ML.NET <see cref="Microsoft.ML.Data.PrimitiveDataViewType"/>
    /// column. Covers boolean, every numeric DataView width, text, and the date/time types
    /// ML.NET understands (<see cref="DateTime"/>, <see cref="DateTimeOffset"/>, <see cref="TimeSpan"/>).
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True if the type is supported, false otherwise</returns>
    public static bool IsMLNetSupported(Type clrType)
    {
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;

        return actualType == typeof(bool) ||
               actualType == typeof(byte) ||
               actualType == typeof(sbyte) ||
               actualType == typeof(short) ||
               actualType == typeof(ushort) ||
               actualType == typeof(int) ||
               actualType == typeof(uint) ||
               actualType == typeof(long) ||
               actualType == typeof(ulong) ||
               actualType == typeof(float) ||
               actualType == typeof(double) ||
               actualType == typeof(string) ||
               actualType == typeof(DateTime) ||
               actualType == typeof(DateTimeOffset) ||
               actualType == typeof(TimeSpan);
    }

    /// <summary>
    /// Checks whether a CLR type is part of the extended domain from issue #158
    /// (<see cref="Half"/>, <see cref="nint"/>/<see cref="nuint"/>, <see cref="char"/>,
    /// <see cref="DateOnly"/>/<see cref="TimeOnly"/>, <see cref="DateTimeOffset"/>,
    /// <see cref="Guid"/>, <see cref="TimeSpan"/>). These types use <c>nivara.clrType</c>
    /// schema metadata so a round-trip restores the original column type.
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True when the type belongs to the extended domain</returns>
    public static bool IsExtendedDomainType(Type clrType)
    {
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;

        return actualType == typeof(Half) ||
               actualType == typeof(nint) ||
               actualType == typeof(nuint) ||
               actualType == typeof(char) ||
               actualType == typeof(DateOnly) ||
               actualType == typeof(TimeOnly) ||
               actualType == typeof(DateTimeOffset) ||
               actualType == typeof(Guid) ||
               actualType == typeof(TimeSpan);
    }

    /// <summary>
    /// Gets the schema-metadata key under which the original CLR type of a column is stored.
    /// </summary>
    /// <param name="columnName">The column name</param>
    /// <returns>The metadata key</returns>
    internal static string GetClrTypeMetadataKey(string columnName) => ClrTypeMetadataKeyPrefix + columnName;

    /// <summary>
    /// Checks whether a schema-metadata key carries a preserved CLR type.
    /// </summary>
    internal static bool IsClrTypeMetadataKey(string key)
        => key.StartsWith(ClrTypeMetadataKeyPrefix, StringComparison.Ordinal);

    /// <summary>
    /// Resolves the preserved CLR type for a column from schema metadata.
    /// </summary>
    /// <param name="metadata">The schema metadata dictionary, or null</param>
    /// <param name="columnName">The column name</param>
    /// <returns>The preserved CLR type, or null when no metadata is present for the column</returns>
    internal static Type? ResolveMetadataClrType(IReadOnlyDictionary<string, string>? metadata, string columnName)
    {
        if (metadata is not null && metadata.TryGetValue(GetClrTypeMetadataKey(columnName), out var fullName))
            return Type.GetType(fullName, throwOnError: false);

        return null;
    }

    /// <summary>
    /// Checks whether a CLR type represents a string in Parquet.
    /// Parquet.Net 6.1.0 reports string fields with <see cref="ReadOnlyMemory{T}"/> of char.
    /// </summary>
    /// <param name="clrType">The CLR type to check</param>
    /// <returns>True if the type is a string or its Parquet.Net string representation</returns>
    internal static bool IsStringType(Type clrType)
    {
        return clrType == typeof(string) || clrType == typeof(ReadOnlyMemory<char>);
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
            Type t when t == typeof(Int128) || t == typeof(UInt128) => new List<string>
            {
                "Parquet/Arrow/ML.NET have no lossless 128-bit integer representation; store as string or decimal with documented precision loss"
            },
            Type t when t.IsEnum => new List<string> { "int", "string" },
            Type t when t.IsArray => new List<string> { "Use individual columns for array elements" },
            Type t when t.IsGenericType => new List<string> { "Break down into primitive components" },
            _ => new List<string> { "bool", "int", "long", "float", "double", "DateTime", "string" }
        };
    }
}
