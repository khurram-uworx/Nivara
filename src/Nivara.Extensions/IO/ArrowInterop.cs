using Apache.Arrow;
using Apache.Arrow.Arrays;
using Apache.Arrow.Types;
using Nivara.IO;

namespace Nivara.IO;

/// <summary>
/// Provides interoperability between Nivara data structures and Apache Arrow
/// </summary>
/// <remarks>
/// This class handles bidirectional conversion between NivaraFrame/NivaraSeries
/// and Apache Arrow Table/Array structures, with support for zero-copy operations
/// when possible.
/// </remarks>
public static class ArrowInterop
{
    /// <summary>
    /// Converts a NivaraFrame to an Apache Arrow Table
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <param name="options">Optional conversion options</param>
    /// <returns>An Apache Arrow Table</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when a column type is not supported</exception>
    public static Table ToArrowTable(NivaraFrame frame, ArrowConversionOptions? options = null)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));

        options ??= new ArrowConversionOptions();

        // Validate types if requested
        if (options.ValidateTypes)
        {
            ValidateFrameTypesForArrowConversion(frame);
        }

        // Handle empty frames
        if (frame.ColumnCount == 0)
        {
            var emptySchema = new Apache.Arrow.Schema(new Field[0], null);
            var emptyBatch = new RecordBatch(emptySchema, new IArrowArray[0], 0);
            return Table.TableFromRecordBatches(emptySchema, new[] { emptyBatch });
        }

        var fields = new List<Field>();
        var arrowArrays = new List<IArrowArray>();

        foreach (var columnName in frame.ColumnNames)
        {
            var column = frame.GetColumn(columnName);
            var arrowType = TypeMapper.MapClrToArrow(column.ElementType);
            
            // For DateTime columns, use the timezone from options
            if (column.ElementType == typeof(DateTime) && arrowType is TimestampType)
            {
                arrowType = new TimestampType(TimeUnit.Microsecond, options.TimeZone);
            }
            
            // Create field with proper nullability
            var field = new Field(columnName, arrowType, nullable: true);
            fields.Add(field);

            // Convert column to Arrow array
            var arrowArray = ConvertColumnToArrowArray(column, options);
            arrowArrays.Add(arrowArray);
        }

        var schema = new Apache.Arrow.Schema(fields, null);
        var recordBatch = new RecordBatch(schema, arrowArrays, frame.RowCount);
        
        return Table.TableFromRecordBatches(schema, new[] { recordBatch });
    }

    /// <summary>
    /// Converts an Apache Arrow Table to a NivaraFrame
    /// </summary>
    /// <param name="arrowTable">The Apache Arrow Table to convert</param>
    /// <param name="options">Optional conversion options</param>
    /// <returns>A NivaraFrame</returns>
    /// <exception cref="ArgumentNullException">Thrown when arrowTable is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when an Arrow type is not supported</exception>
    public static NivaraFrame FromArrowTable(Table arrowTable, ArrowConversionOptions? options = null)
    {
        if (arrowTable == null)
            throw new ArgumentNullException(nameof(arrowTable));

        options ??= new ArrowConversionOptions();

        // Validate types if requested
        if (options.ValidateTypes)
        {
            ValidateArrowTableTypesForNivaraConversion(arrowTable);
        }

        // Handle empty tables
        if (arrowTable.ColumnCount == 0)
        {
            // Create a frame with a single empty column to satisfy NivaraFrame requirements
            var emptyColumn = NivaraColumn<object>.Create(System.Array.Empty<object>());
            return NivaraFrame.Create(("EmptyColumn", emptyColumn));
        }

        var columns = new List<(string Name, IColumn Column)>();

        for (int fieldIndex = 0; fieldIndex < arrowTable.ColumnCount; fieldIndex++)
        {
            var field = arrowTable.Schema.GetFieldByIndex(fieldIndex);
            var columnName = field.Name;
            var arrowColumn = arrowTable.Column(fieldIndex);
            
            // Convert Arrow column to Nivara column
            var column = ConvertArrowColumnToNivaraColumn(arrowColumn, field.DataType, options);
            columns.Add((columnName, column));
        }

        return new NivaraFrame(columns);
    }

    /// <summary>
    /// Converts a NivaraSeries to an Apache Arrow Array
    /// </summary>
    /// <typeparam name="T">The type of data in the series</typeparam>
    /// <param name="series">The NivaraSeries to convert</param>
    /// <param name="options">Optional conversion options</param>
    /// <returns>An Apache Arrow Array</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the series type is not supported</exception>
    public static IArrowArray ToArrowArray<T>(NivaraSeries<T> series, ArrowConversionOptions? options = null)
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        options ??= new ArrowConversionOptions();

        // Validate types if requested
        if (options.ValidateTypes)
        {
            ValidateSeriesTypeForArrowConversion<T>();
        }

        // Handle empty series
        if (series.Length == 0)
        {
            return CreateEmptyArrowArray<T>();
        }

        // Convert series to array and create a temporary column
        var data = new T[series.Length];
        for (int i = 0; i < series.Length; i++)
        {
            data[i] = series[i];
        }
        
        var column = NivaraColumn<T>.Create(data);
        return ConvertColumnToArrowArray(column, options);
    }

    /// <summary>
    /// Converts an Apache Arrow Array to a NivaraSeries
    /// </summary>
    /// <typeparam name="T">The type of data for the series</typeparam>
    /// <param name="arrowArray">The Apache Arrow Array to convert</param>
    /// <param name="options">Optional conversion options</param>
    /// <returns>A NivaraSeries</returns>
    /// <exception cref="ArgumentNullException">Thrown when arrowArray is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the Arrow array type is not supported</exception>
    public static NivaraSeries<T> FromArrowArray<T>(IArrowArray arrowArray, ArrowConversionOptions? options = null)
    {
        if (arrowArray == null)
            throw new ArgumentNullException(nameof(arrowArray));

        options ??= new ArrowConversionOptions();

        // Validate types if requested
        if (options.ValidateTypes)
        {
            ValidateArrowArrayTypeForNivaraConversion<T>(arrowArray);
        }

        // Handle empty arrays
        if (arrowArray.Length == 0)
        {
            return NivaraSeries<T>.Create(System.Array.Empty<T>());
        }

        // Extract data directly from the Arrow array
        var data = new List<T>();
        
        for (int i = 0; i < arrowArray.Length; i++)
        {
            if (arrowArray.IsNull(i))
            {
                data.Add(default(T)!);
            }
            else
            {
                var value = ExtractValueFromArrowArray<T>(arrowArray, i, options);
                data.Add(value);
            }
        }
        
        return NivaraSeries<T>.Create(data.ToArray());
    }

    /// <summary>
    /// Creates an empty Arrow array for the specified type
    /// </summary>
    private static IArrowArray CreateEmptyArrowArray<T>()
    {
        return typeof(T) switch
        {
            Type t when t == typeof(bool) => new BooleanArray.Builder().Build(),
            Type t when t == typeof(int) => new Int32Array.Builder().Build(),
            Type t when t == typeof(long) => new Int64Array.Builder().Build(),
            Type t when t == typeof(float) => new FloatArray.Builder().Build(),
            Type t when t == typeof(double) => new DoubleArray.Builder().Build(),
            Type t when t == typeof(string) => new StringArray.Builder().Build(),
            Type t when t == typeof(DateTime) => new TimestampArray.Builder(new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc)).Build(),
            Type t when t == typeof(byte) => new UInt8Array.Builder().Build(),
            Type t when t == typeof(short) => new Int16Array.Builder().Build(),
            Type t when t == typeof(uint) => new UInt32Array.Builder().Build(),
            Type t when t == typeof(ulong) => new UInt64Array.Builder().Build(),
            Type t when t == typeof(ushort) => new UInt16Array.Builder().Build(),
            Type t when t == typeof(sbyte) => new Int8Array.Builder().Build(),
            _ => throw new UnsupportedTypeException(typeof(T), TypeMapper.GetTypeSuggestions(typeof(T)))
        };
    }

    /// <summary>
    /// Converts a column to an Arrow array using dynamic dispatch
    /// </summary>
    private static IArrowArray ConvertColumnToArrowArray(IColumn column, ArrowConversionOptions options)
    {
        var elementType = column.ElementType;
        
        return elementType switch
        {
            Type t when t == typeof(bool) => ConvertColumnToArrowArrayTyped<bool>(column, options),
            Type t when t == typeof(int) => ConvertColumnToArrowArrayTyped<int>(column, options),
            Type t when t == typeof(long) => ConvertColumnToArrowArrayTyped<long>(column, options),
            Type t when t == typeof(float) => ConvertColumnToArrowArrayTyped<float>(column, options),
            Type t when t == typeof(double) => ConvertColumnToArrowArrayTyped<double>(column, options),
            Type t when t == typeof(string) => ConvertColumnToArrowArrayTyped<string>(column, options),
            Type t when t == typeof(DateTime) => ConvertColumnToArrowArrayTyped<DateTime>(column, options),
            Type t when t == typeof(byte) => ConvertColumnToArrowArrayTyped<byte>(column, options),
            Type t when t == typeof(short) => ConvertColumnToArrowArrayTyped<short>(column, options),
            Type t when t == typeof(uint) => ConvertColumnToArrowArrayTyped<uint>(column, options),
            Type t when t == typeof(ulong) => ConvertColumnToArrowArrayTyped<ulong>(column, options),
            Type t when t == typeof(ushort) => ConvertColumnToArrowArrayTyped<ushort>(column, options),
            Type t when t == typeof(sbyte) => ConvertColumnToArrowArrayTyped<sbyte>(column, options),
            _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
        };
    }

    /// <summary>
    /// Converts a typed column to an Arrow array
    /// </summary>
    private static IArrowArray ConvertColumnToArrowArrayTyped<T>(IColumn column, ArrowConversionOptions options)
    {
        var typedColumn = (NivaraColumn<T>)column;

        return typeof(T) switch
        {
            Type t when t == typeof(bool) => CreateBooleanArray((NivaraColumn<bool>)(object)typedColumn, options),
            Type t when t == typeof(int) => CreateInt32Array((NivaraColumn<int>)(object)typedColumn, options),
            Type t when t == typeof(long) => CreateInt64Array((NivaraColumn<long>)(object)typedColumn, options),
            Type t when t == typeof(float) => CreateFloatArray((NivaraColumn<float>)(object)typedColumn, options),
            Type t when t == typeof(double) => CreateDoubleArray((NivaraColumn<double>)(object)typedColumn, options),
            Type t when t == typeof(string) => CreateStringArray((NivaraColumn<string>)(object)typedColumn, options),
            Type t when t == typeof(DateTime) => CreateTimestampArray((NivaraColumn<DateTime>)(object)typedColumn, options),
            Type t when t == typeof(byte) => CreateUInt8Array((NivaraColumn<byte>)(object)typedColumn, options),
            Type t when t == typeof(short) => CreateInt16Array((NivaraColumn<short>)(object)typedColumn, options),
            Type t when t == typeof(uint) => CreateUInt32Array((NivaraColumn<uint>)(object)typedColumn, options),
            Type t when t == typeof(ulong) => CreateUInt64Array((NivaraColumn<ulong>)(object)typedColumn, options),
            Type t when t == typeof(ushort) => CreateUInt16Array((NivaraColumn<ushort>)(object)typedColumn, options),
            Type t when t == typeof(sbyte) => CreateInt8Array((NivaraColumn<sbyte>)(object)typedColumn, options),
            _ => throw new UnsupportedTypeException(typeof(T))
        };
    }

    /// <summary>
    /// Creates a Boolean Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateBooleanArray(NivaraColumn<bool> column, ArrowConversionOptions options)
    {
        // Try zero-copy optimization first if enabled
        if (options.UseZeroCopy)
        {
            var zeroCopyArray = TryCreateZeroCopyBooleanArray(column);
            if (zeroCopyArray != null)
                return zeroCopyArray;
        }

        // Fallback to copying approach
        var builder = new BooleanArray.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates an Int32 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt32Array(NivaraColumn<int> column, ArrowConversionOptions options)
    {
        // Try zero-copy optimization first if enabled
        if (options.UseZeroCopy)
        {
            var zeroCopyArray = TryCreateZeroCopyInt32Array(column);
            if (zeroCopyArray != null)
                return zeroCopyArray;
        }

        // Fallback to copying approach
        var builder = new Int32Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates an Int64 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt64Array(NivaraColumn<long> column, ArrowConversionOptions options)
    {
        var builder = new Int64Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a Float Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateFloatArray(NivaraColumn<float> column, ArrowConversionOptions options)
    {
        var builder = new FloatArray.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a Double Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateDoubleArray(NivaraColumn<double> column, ArrowConversionOptions options)
    {
        // Try zero-copy optimization first if enabled
        if (options.UseZeroCopy)
        {
            var zeroCopyArray = TryCreateZeroCopyDoubleArray(column);
            if (zeroCopyArray != null)
                return zeroCopyArray;
        }

        // Fallback to copying approach
        var builder = new DoubleArray.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a String Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateStringArray(NivaraColumn<string> column, ArrowConversionOptions options)
    {
        var builder = new StringArray.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a Timestamp Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateTimestampArray(NivaraColumn<DateTime> column, ArrowConversionOptions options)
    {
        var timestampType = new TimestampType(TimeUnit.Microsecond, options.TimeZone);
        var builder = new TimestampArray.Builder(timestampType);
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
            {
                builder.AppendNull();
            }
            else
            {
                var dateTime = column[i];
                // Convert DateTime to microseconds since Unix epoch
                if (dateTime.Kind == DateTimeKind.Unspecified)
                {
                    dateTime = DateTime.SpecifyKind(dateTime, DateTimeKind.Utc);
                }
                else if (dateTime.Kind == DateTimeKind.Local)
                {
                    dateTime = dateTime.ToUniversalTime();
                }

                var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
                var microseconds = (long)(dateTime - unixEpoch).TotalMicroseconds;
                builder.Append(DateTimeOffset.FromUnixTimeMilliseconds(microseconds / 1000));
            }
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a UInt8 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt8Array(NivaraColumn<byte> column, ArrowConversionOptions options)
    {
        var builder = new UInt8Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates an Int16 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt16Array(NivaraColumn<short> column, ArrowConversionOptions options)
    {
        var builder = new Int16Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a UInt32 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt32Array(NivaraColumn<uint> column, ArrowConversionOptions options)
    {
        var builder = new UInt32Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a UInt64 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt64Array(NivaraColumn<ulong> column, ArrowConversionOptions options)
    {
        var builder = new UInt64Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a UInt16 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt16Array(NivaraColumn<ushort> column, ArrowConversionOptions options)
    {
        var builder = new UInt16Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates an Int8 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt8Array(NivaraColumn<sbyte> column, ArrowConversionOptions options)
    {
        var builder = new Int8Array.Builder();
        
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i]);
        }

        return builder.Build();
    }

    /// <summary>
    /// Converts an Arrow column to a Nivara column using dynamic dispatch
    /// </summary>
    private static IColumn ConvertArrowColumnToNivaraColumn(Column arrowColumn, IArrowType arrowType, ArrowConversionOptions options)
    {
        var clrType = TypeMapper.MapArrowToClr(arrowType);
        
        return clrType switch
        {
            Type t when t == typeof(bool) => ConvertArrowColumnToNivaraColumnTyped<bool>(arrowColumn, options),
            Type t when t == typeof(int) => ConvertArrowColumnToNivaraColumnTyped<int>(arrowColumn, options),
            Type t when t == typeof(long) => ConvertArrowColumnToNivaraColumnTyped<long>(arrowColumn, options),
            Type t when t == typeof(float) => ConvertArrowColumnToNivaraColumnTyped<float>(arrowColumn, options),
            Type t when t == typeof(double) => ConvertArrowColumnToNivaraColumnTyped<double>(arrowColumn, options),
            Type t when t == typeof(string) => ConvertArrowColumnToNivaraColumnTyped<string>(arrowColumn, options),
            Type t when t == typeof(DateTime) => ConvertArrowColumnToNivaraColumnTyped<DateTime>(arrowColumn, options),
            Type t when t == typeof(byte) => ConvertArrowColumnToNivaraColumnTyped<byte>(arrowColumn, options),
            Type t when t == typeof(short) => ConvertArrowColumnToNivaraColumnTyped<short>(arrowColumn, options),
            Type t when t == typeof(uint) => ConvertArrowColumnToNivaraColumnTyped<uint>(arrowColumn, options),
            Type t when t == typeof(ulong) => ConvertArrowColumnToNivaraColumnTyped<ulong>(arrowColumn, options),
            Type t when t == typeof(ushort) => ConvertArrowColumnToNivaraColumnTyped<ushort>(arrowColumn, options),
            Type t when t == typeof(sbyte) => ConvertArrowColumnToNivaraColumnTyped<sbyte>(arrowColumn, options),
            _ => throw new UnsupportedTypeException(clrType)
        };
    }

    /// <summary>
    /// Converts a typed Arrow column to a Nivara column
    /// </summary>
    private static IColumn ConvertArrowColumnToNivaraColumnTyped<T>(Column arrowColumn, ArrowConversionOptions options)
    {
        // Calculate total length across all chunks
        int totalLength = (int)arrowColumn.Length;
        
        if (totalLength == 0)
        {
            return NivaraColumn<T>.Create(System.Array.Empty<T>());
        }

        var values = new List<T?>();

        // Get the ChunkedArray
        var chunkedArray = arrowColumn.Data;
        
        // Try to get the number of chunks using ArrayCount property
        int chunkCount = chunkedArray.ArrayCount;
        
        if (chunkCount == 0)
        {
            // No chunks, return empty column
            return NivaraColumn<T>.Create(System.Array.Empty<T>());
        }

        // Process each chunk in the column
        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
        {
            var chunk = chunkedArray.Array(chunkIndex);
            
            for (int i = 0; i < chunk.Length; i++)
            {
                if (chunk.IsNull(i))
                {
                    values.Add(default(T?));
                }
                else
                {
                    values.Add(ExtractValueFromArrowArray<T>(chunk, i, options));
                }
            }
        }

        // Create appropriate Nivara column
        if (typeof(T).IsValueType)
        {
            // For value types, we need to create an array of nullable values with the correct type
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var nullableArray = System.Array.CreateInstance(nullableType, values.Count);
            
            for (int i = 0; i < values.Count; i++)
            {
                var nullableValue = values[i];
                if (nullableValue != null)
                {
                    // Create a Nullable<T> instance with the value
                    var nullableInstance = Activator.CreateInstance(nullableType, nullableValue);
                    nullableArray.SetValue(nullableInstance, i);
                }
                else
                {
                    // Set null value
                    nullableArray.SetValue(null, i);
                }
            }
            
            return NivaraColumn<T>.CreateFromNullable(nullableArray);
        }
        else
        {
            // For reference types, convert nulls to actual null values
            var referenceValues = new T[totalLength];
            for (int i = 0; i < totalLength; i++)
            {
                referenceValues[i] = values[i] == null ? default(T)! : values[i]!;
            }
            return NivaraColumn<T>.CreateForReferenceType(referenceValues);
        }
    }

    /// <summary>
    /// Extracts a typed value from an Arrow array at the specified index
    /// </summary>
    private static T ExtractValueFromArrowArray<T>(IArrowArray array, int index, ArrowConversionOptions options)
    {
        return array switch
        {
            BooleanArray boolArray when typeof(T) == typeof(bool) => (T)(object)boolArray.GetValue(index)!.Value,
            Int32Array intArray when typeof(T) == typeof(int) => (T)(object)intArray.GetValue(index)!.Value,
            Int64Array longArray when typeof(T) == typeof(long) => (T)(object)longArray.GetValue(index)!.Value,
            FloatArray floatArray when typeof(T) == typeof(float) => (T)(object)floatArray.GetValue(index)!.Value,
            DoubleArray doubleArray when typeof(T) == typeof(double) => (T)(object)doubleArray.GetValue(index)!.Value,
            StringArray stringArray when typeof(T) == typeof(string) => (T)(object)stringArray.GetString(index),
            TimestampArray timestampArray when typeof(T) == typeof(DateTime) => (T)(object)ConvertTimestampToDateTime(timestampArray, index, options),
            UInt8Array byteArray when typeof(T) == typeof(byte) => (T)(object)byteArray.GetValue(index)!.Value,
            Int16Array shortArray when typeof(T) == typeof(short) => (T)(object)shortArray.GetValue(index)!.Value,
            UInt32Array uintArray when typeof(T) == typeof(uint) => (T)(object)uintArray.GetValue(index)!.Value,
            UInt64Array ulongArray when typeof(T) == typeof(ulong) => (T)(object)ulongArray.GetValue(index)!.Value,
            UInt16Array ushortArray when typeof(T) == typeof(ushort) => (T)(object)ushortArray.GetValue(index)!.Value,
            Int8Array sbyteArray when typeof(T) == typeof(sbyte) => (T)(object)sbyteArray.GetValue(index)!.Value,
            _ => throw new UnsupportedTypeException(typeof(T), new[] { $"Arrow array type {array.GetType().Name} not supported for CLR type {typeof(T).Name}" })
        };
    }

    /// <summary>
    /// Converts a timestamp value from Arrow to DateTime
    /// </summary>
    private static DateTime ConvertTimestampToDateTime(TimestampArray timestampArray, int index, ArrowConversionOptions options)
    {
        var timestampValue = timestampArray.GetValue(index)!.Value;
        var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        
        // Convert from microseconds since Unix epoch to DateTime
        var dateTime = unixEpoch.AddMicroseconds(timestampValue);
        
        // Convert to the specified timezone if needed
        if (options.TimeZone != TimeZoneInfo.Utc)
        {
            dateTime = TimeZoneInfo.ConvertTimeFromUtc(dateTime, options.TimeZone);
        }
        
        return dateTime;
    }

    /// <summary>
    /// Validates that all column types in a NivaraFrame are supported for Arrow conversion
    /// </summary>
    /// <param name="frame">The frame to validate</param>
    /// <exception cref="UnsupportedTypeException">Thrown when unsupported types are found</exception>
    private static void ValidateFrameTypesForArrowConversion(NivaraFrame frame)
    {
        var unsupportedColumns = new List<string>();

        foreach (var columnName in frame.ColumnNames)
        {
            var column = frame.GetColumn(columnName);
            if (!TypeMapper.IsArrowSupported(column.ElementType))
            {
                unsupportedColumns.Add($"{columnName} ({column.ElementType.Name})");
            }
        }

        if (unsupportedColumns.Count > 0)
        {
            var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
            throw new UnsupportedTypeException(
                typeof(object), // Generic type since multiple types are involved
                new[] { $"Unsupported column types found: {string.Join(", ", unsupportedColumns)}. Supported types: {supportedTypes}" }
            );
        }
    }

    /// <summary>
    /// Validates that all Arrow types in a Table are supported for Nivara conversion
    /// </summary>
    /// <param name="arrowTable">The Arrow table to validate</param>
    /// <exception cref="UnsupportedTypeException">Thrown when unsupported types are found</exception>
    private static void ValidateArrowTableTypesForNivaraConversion(Table arrowTable)
    {
        var unsupportedFields = new List<string>();

        for (int i = 0; i < arrowTable.Schema.FieldsList.Count; i++)
        {
            var field = arrowTable.Schema.FieldsList[i];
            try
            {
                TypeMapper.MapArrowToClr(field.DataType);
            }
            catch (UnsupportedTypeException)
            {
                unsupportedFields.Add($"{field.Name} ({field.DataType.Name})");
            }
        }

        if (unsupportedFields.Count > 0)
        {
            var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
            throw new UnsupportedTypeException(
                typeof(object), // Generic type since multiple types are involved
                new[] { $"Unsupported Arrow field types found: {string.Join(", ", unsupportedFields)}. Supported types: {supportedTypes}" }
            );
        }
    }

    /// <summary>
    /// Validates that a series type is supported for Arrow conversion
    /// </summary>
    /// <typeparam name="T">The series type to validate</typeparam>
    /// <exception cref="UnsupportedTypeException">Thrown when the type is not supported</exception>
    private static void ValidateSeriesTypeForArrowConversion<T>()
    {
        if (!TypeMapper.IsArrowSupported(typeof(T)))
        {
            throw new UnsupportedTypeException(typeof(T), TypeMapper.GetTypeSuggestions(typeof(T)));
        }
    }

    /// <summary>
    /// Validates that an Arrow array type is compatible with the target Nivara type
    /// </summary>
    /// <typeparam name="T">The target Nivara type</typeparam>
    /// <param name="arrowArray">The Arrow array to validate</param>
    /// <exception cref="UnsupportedTypeException">Thrown when types are incompatible</exception>
    private static void ValidateArrowArrayTypeForNivaraConversion<T>(IArrowArray arrowArray)
    {
        // Check if the target type is supported
        if (!TypeMapper.IsArrowSupported(typeof(T)))
        {
            throw new UnsupportedTypeException(typeof(T), TypeMapper.GetTypeSuggestions(typeof(T)));
        }

        // Check if the Arrow array type is compatible with the target type
        var expectedArrowType = TypeMapper.MapClrToArrow(typeof(T));
        var actualArrowType = arrowArray.Data.DataType;

        // Simple type compatibility check - in practice, this could be more sophisticated
        if (expectedArrowType.TypeId != actualArrowType.TypeId)
        {
            throw new UnsupportedTypeException(
                typeof(T),
                new[] { $"Arrow array type {actualArrowType.Name} is not compatible with target type {typeof(T).Name}. Expected Arrow type: {expectedArrowType.Name}" }
            );
        }
    }

    /// <summary>
    /// Attempts to create a zero-copy Boolean Arrow array from a NivaraColumn
    /// </summary>
    /// <param name="column">The NivaraColumn to convert</param>
    /// <returns>A zero-copy Arrow array if possible, null if zero-copy is not feasible</returns>
    private static IArrowArray? TryCreateZeroCopyBooleanArray(NivaraColumn<bool> column)
    {
        // Zero-copy is only possible if:
        // 1. The column has no nulls (or we can handle the null mask efficiently)
        // 2. The underlying storage is compatible with Arrow's memory layout
        // 3. The data is contiguous in memory
        
        // For now, zero-copy is not implemented for boolean arrays due to bit-packing complexity
        // This is a fallback that returns null to use the copying approach
        return null;
    }

    /// <summary>
    /// Attempts to create a zero-copy Int32 Arrow array from a NivaraColumn
    /// </summary>
    /// <param name="column">The NivaraColumn to convert</param>
    /// <returns>A zero-copy Arrow array if possible, null if zero-copy is not feasible</returns>
    private static IArrowArray? TryCreateZeroCopyInt32Array(NivaraColumn<int> column)
    {
        // Zero-copy is only possible if:
        // 1. The column has no nulls (or we can handle the null mask efficiently)
        // 2. The underlying storage is compatible with Arrow's memory layout
        // 3. The data is contiguous in memory
        
        // Check if column has nulls - zero-copy is more complex with nulls
        if (column.HasNulls)
        {
            return null; // Fallback to copying for now
        }

        // For tensor-backed columns, we might be able to share memory
        // This is a simplified implementation - in practice, we'd need to check
        // memory layout compatibility and create Arrow arrays from existing buffers
        try
        {
            // This is a placeholder for actual zero-copy implementation
            // In a real implementation, we would:
            // 1. Get the underlying memory buffer from the column
            // 2. Create an Arrow array that shares this buffer
            // 3. Handle memory ownership and lifecycle properly
            
            // For now, return null to use copying approach
            return null;
        }
        catch
        {
            // If zero-copy fails for any reason, return null to fallback to copying
            return null;
        }
    }

    /// <summary>
    /// Attempts to create a zero-copy Double Arrow array from a NivaraColumn
    /// </summary>
    /// <param name="column">The NivaraColumn to convert</param>
    /// <returns>A zero-copy Arrow array if possible, null if zero-copy is not feasible</returns>
    private static IArrowArray? TryCreateZeroCopyDoubleArray(NivaraColumn<double> column)
    {
        // Zero-copy is only possible if:
        // 1. The column has no nulls (or we can handle the null mask efficiently)
        // 2. The underlying storage is compatible with Arrow's memory layout
        // 3. The data is contiguous in memory
        
        // Check if column has nulls - zero-copy is more complex with nulls
        if (column.HasNulls)
        {
            return null; // Fallback to copying for now
        }

        // For tensor-backed columns, we might be able to share memory
        // This is a simplified implementation - in practice, we'd need to check
        // memory layout compatibility and create Arrow arrays from existing buffers
        try
        {
            // This is a placeholder for actual zero-copy implementation
            // In a real implementation, we would:
            // 1. Get the underlying memory buffer from the column
            // 2. Create an Arrow array that shares this buffer
            // 3. Handle memory ownership and lifecycle properly
            
            // For now, return null to use copying approach
            return null;
        }
        catch
        {
            // If zero-copy fails for any reason, return null to fallback to copying
            return null;
        }
    }
}
