using Apache.Arrow;
using Apache.Arrow.Arrays;
using Apache.Arrow.Types;
using Nivara.Storage;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.IO;

/// <summary>
/// Provides interoperability between Nivara data structures and Apache Arrow
/// </summary>
/// <remarks>
/// This class handles bidirectional conversion between NivaraFrame/NivaraSeries
/// and Apache Arrow Table/Array structures.
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
    /// <exception cref="NivaraIOException">Thrown when conversion fails</exception>
    public static Table ToArrowTable(NivaraFrame frame, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frame);

        options ??= new ArrowConversionOptions();

        try
        {
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
            Dictionary<string, string>? clrTypeMetadata = null;

            foreach (var columnName in frame.ColumnNames)
            {
                try
                {
                    var column = frame.GetColumn(columnName);
                    var arrowType = TypeMapper.MapClrToArrow(column.ElementType);

                    // For DateTime columns, use the timezone from options
                    if (column.ElementType == typeof(DateTime) && arrowType is TimestampType)
                    {
                        arrowType = new TimestampType(TimeUnit.Microsecond, options.TimeZone);
                    }

                    // Preserve the original CLR type for extended-domain columns so a round-trip
                    // restores nint/nuint/char/DateTimeOffset/Half/TimeSpan/DateOnly/TimeOnly/Guid
                    // even though the Arrow representation is widened (or shared with a base type).
                    if (TypeMapper.IsExtendedDomainType(column.ElementType))
                    {
                        (clrTypeMetadata ??= new Dictionary<string, string>())[TypeMapper.GetClrTypeMetadataKey(columnName)]
                            = column.ElementType.FullName!;
                    }

                    // Create field with proper nullability
                    var field = new Field(columnName, arrowType, nullable: true);
                    fields.Add(field);

                    // Convert column to Arrow array
                    var arrowArray = ConvertColumnToArrowArray(column, options, columnName);
                    arrowArrays.Add(arrowArray);
                }
                catch (Exception ex) when (ex is not UnsupportedTypeException)
                {
                    throw new DataCorruptionException($"Failed to convert column '{columnName}' to Arrow format: {ex.Message}", ex)
                    {
                        AffectedColumns = new[] { columnName },
                        AffectedRowRange = new Range(0, frame.RowCount),
                        OperationContext = "ArrowInterop.ToArrowTable"
                    };
                }
            }

            var schema = new Apache.Arrow.Schema(fields, clrTypeMetadata);
            var recordBatch = new RecordBatch(schema, arrowArrays, frame.RowCount);

            return Table.TableFromRecordBatches(schema, new[] { recordBatch });
        }
        catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException and not DataCorruptionException)
        {
            throw new NivaraIOException($"Failed to convert NivaraFrame to Arrow Table: {ex.Message}", ex)
            {
                OperationContext = "ArrowInterop.ToArrowTable"
            };
        }
    }

    /// <summary>
    /// Converts an Apache Arrow Table to a NivaraFrame
    /// </summary>
    /// <param name="arrowTable">The Apache Arrow Table to convert</param>
    /// <param name="options">Optional conversion options</param>
    /// <returns>A NivaraFrame</returns>
    /// <exception cref="ArgumentNullException">Thrown when arrowTable is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when an Arrow type is not supported</exception>
    /// <exception cref="NivaraIOException">Thrown when conversion fails</exception>
    public static NivaraFrame FromArrowTable(Table arrowTable, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(arrowTable);

        options ??= new ArrowConversionOptions();

        try
        {
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

            System.Diagnostics.Debug.WriteLine($"Converting Arrow table with {arrowTable.ColumnCount} columns");

            for (int fieldIndex = 0; fieldIndex < arrowTable.ColumnCount; fieldIndex++)
            {
                var field = arrowTable.Schema.GetFieldByIndex(fieldIndex);
                var columnName = field.Name;

                System.Diagnostics.Debug.WriteLine($"Converting column '{columnName}' of type {field.DataType.GetType().Name}");

                try
                {
                    var arrowColumn = arrowTable.Column(fieldIndex);

                    // Convert Arrow column to Nivara column
                    var column = ConvertArrowColumnToNivaraColumn(arrowColumn, field.DataType, options, columnName, arrowTable.Schema.Metadata);

                    System.Diagnostics.Debug.WriteLine($"Created column '{columnName}' with HasNulls: {column.HasNulls}");

                    columns.Add((columnName, column));
                }
                catch (Exception ex) when (ex is not UnsupportedTypeException)
                {
                    throw new DataCorruptionException($"Failed to convert Arrow column '{columnName}' to Nivara format: {ex.Message}", ex)
                    {
                        AffectedColumns = new[] { columnName },
                        AffectedRowRange = new Range(0, (int)arrowTable.RowCount),
                        OperationContext = "ArrowInterop.FromArrowTable"
                    };
                }
            }

            return new NivaraFrame(columns);
        }
        catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException and not DataCorruptionException)
        {
            throw new NivaraIOException($"Failed to convert Arrow Table to NivaraFrame: {ex.Message}", ex)
            {
                OperationContext = "ArrowInterop.FromArrowTable"
            };
        }
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
    /// <exception cref="NivaraIOException">Thrown when conversion fails</exception>
    public static IArrowArray ToArrowArray<T>(NivaraSeries<T> series, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(series);

        options ??= new ArrowConversionOptions();

        try
        {
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

            // Use the series' underlying column directly so its null mask is preserved
            // (the previous rebuild dropped nulls and produced an all-valid Arrow array).
            var column = series.Values;
            return ConvertColumnToArrowArray(column, options);
        }
        catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException)
        {
            throw new NivaraIOException($"Failed to convert NivaraSeries<{typeof(T).Name}> to Arrow Array: {ex.Message}", ex)
            {
                OperationContext = "ArrowInterop.ToArrowArray"
            };
        }
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
    /// <exception cref="NivaraIOException">Thrown when conversion fails</exception>
    public static NivaraSeries<T> FromArrowArray<T>(IArrowArray arrowArray, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(arrowArray);

        options ??= new ArrowConversionOptions();

        try
        {
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

            // Extract data directly from the Arrow array, preserving null positions via a mask
            // (the previous path dropped nulls and produced an all-valid Nivara series).
            var validityBytes = (arrowArray.Data.Buffers.Length > 0)
                ? arrowArray.Data.Buffers[0].Span
                : ReadOnlySpan<byte>.Empty;
            var nullMask = ArrowValidityMask.ToNullMask(validityBytes, (int)arrowArray.Data.Offset, (int)arrowArray.Length);

            var data = new T[arrowArray.Length];
            for (int i = 0; i < arrowArray.Length; i++)
            {
                if (!nullMask[i])
                    data[i] = ExtractValueFromArrowArray<T>(arrowArray, i, options);
            }

            var column = NivaraColumn<T>.CreateFromSpans(data.AsSpan(), nullMask.AsSpan());
            return new NivaraSeries<T>(column);
        }
        catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException)
        {
            throw new NivaraIOException($"Failed to convert Arrow Array to NivaraSeries<{typeof(T).Name}>: {ex.Message}", ex)
            {
                OperationContext = "ArrowInterop.FromArrowArray"
            };
        }
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
            Type t when t == typeof(Half) => new HalfFloatArray.Builder().Build(),
            Type t when t == typeof(nint) => new Int64Array.Builder().Build(),
            Type t when t == typeof(nuint) => new UInt64Array.Builder().Build(),
            Type t when t == typeof(char) => new StringArray.Builder().Build(),
            Type t when t == typeof(DateOnly) => new Date32Array.Builder().Build(),
            Type t when t == typeof(TimeOnly) => new Time64Array.Builder(new Time64Type(TimeUnit.Nanosecond)).Build(),
            Type t when t == typeof(Guid) => CreateEmptyFixedSizeBinaryArray(),
            Type t when t == typeof(DateTimeOffset) => new TimestampArray.Builder(new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc)).Build(),
            Type t when t == typeof(TimeSpan) => new DurationArray.Builder(DurationType.Nanosecond).Build(),
            _ => throw new UnsupportedTypeException(typeof(T), TypeMapper.GetTypeSuggestions(typeof(T)))
        };
    }

    /// <summary>
    /// Creates an empty fixed-size-binary Arrow array for a 16-byte Guid representation
    /// </summary>
    private static IArrowArray CreateEmptyFixedSizeBinaryArray()
    {
        var arrayData = new ArrayData(
            new FixedSizeBinaryType(16),
            0,
            0,
            0,
            new[] { ArrowBuffer.Empty, ArrowBuffer.Empty },
            System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Converts a column to an Arrow array using dynamic dispatch
    /// </summary>
    private static IArrowArray ConvertColumnToArrowArray(IColumn column, ArrowConversionOptions options, string? columnName = null)
    {
        var elementType = column.ElementType;

        return elementType switch
        {
            Type t when t == typeof(bool) => ConvertColumnToArrowArrayTyped<bool>(column, options, columnName),
            Type t when t == typeof(int) => ConvertColumnToArrowArrayTyped<int>(column, options, columnName),
            Type t when t == typeof(long) => ConvertColumnToArrowArrayTyped<long>(column, options, columnName),
            Type t when t == typeof(float) => ConvertColumnToArrowArrayTyped<float>(column, options, columnName),
            Type t when t == typeof(double) => ConvertColumnToArrowArrayTyped<double>(column, options, columnName),
            Type t when t == typeof(string) => ConvertColumnToArrowArrayTyped<string>(column, options, columnName),
            Type t when t == typeof(DateTime) => ConvertColumnToArrowArrayTyped<DateTime>(column, options, columnName),
            Type t when t == typeof(byte) => ConvertColumnToArrowArrayTyped<byte>(column, options, columnName),
            Type t when t == typeof(short) => ConvertColumnToArrowArrayTyped<short>(column, options, columnName),
            Type t when t == typeof(uint) => ConvertColumnToArrowArrayTyped<uint>(column, options, columnName),
            Type t when t == typeof(ulong) => ConvertColumnToArrowArrayTyped<ulong>(column, options, columnName),
            Type t when t == typeof(ushort) => ConvertColumnToArrowArrayTyped<ushort>(column, options, columnName),
            Type t when t == typeof(sbyte) => ConvertColumnToArrowArrayTyped<sbyte>(column, options, columnName),
            Type t when t == typeof(Half) => ConvertColumnToArrowArrayTyped<Half>(column, options, columnName),
            Type t when t == typeof(nint) => ConvertColumnToArrowArrayTyped<nint>(column, options, columnName),
            Type t when t == typeof(nuint) => ConvertColumnToArrowArrayTyped<nuint>(column, options, columnName),
            Type t when t == typeof(char) => ConvertColumnToArrowArrayTyped<char>(column, options, columnName),
            Type t when t == typeof(DateOnly) => ConvertColumnToArrowArrayTyped<DateOnly>(column, options, columnName),
            Type t when t == typeof(TimeOnly) => ConvertColumnToArrowArrayTyped<TimeOnly>(column, options, columnName),
            Type t when t == typeof(Guid) => ConvertColumnToArrowArrayTyped<Guid>(column, options, columnName),
            Type t when t == typeof(DateTimeOffset) => ConvertColumnToArrowArrayTyped<DateTimeOffset>(column, options, columnName),
            Type t when t == typeof(TimeSpan) => ConvertColumnToArrowArrayTyped<TimeSpan>(column, options, columnName),
            _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
        };
    }

    /// <summary>
    /// Converts a typed column to an Arrow array
    /// </summary>
    private static IArrowArray ConvertColumnToArrowArrayTyped<T>(IColumn column, ArrowConversionOptions options, string? columnName = null)
    {
        var typedColumn = (NivaraColumn<T>)column;

        if (options.UseZeroCopy)
            return CreateZeroCopyArrowArray(typedColumn, columnName);

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
            Type t when t == typeof(Half) => CreateHalfFloatArray((NivaraColumn<Half>)(object)typedColumn, options),
            Type t when t == typeof(nint) => CreateInt64Array((NivaraColumn<nint>)(object)typedColumn, options),
            Type t when t == typeof(nuint) => CreateUInt64Array((NivaraColumn<nuint>)(object)typedColumn, options),
            Type t when t == typeof(char) => CreateStringArray((NivaraColumn<char>)(object)typedColumn, options),
            Type t when t == typeof(DateOnly) => CreateDate32Array((NivaraColumn<DateOnly>)(object)typedColumn, options),
            Type t when t == typeof(TimeOnly) => CreateTime64Array((NivaraColumn<TimeOnly>)(object)typedColumn, options),
            Type t when t == typeof(Guid) => CreateFixedSizeBinaryArray((NivaraColumn<Guid>)(object)typedColumn, options),
            Type t when t == typeof(DateTimeOffset) => CreateTimestampArray((NivaraColumn<DateTimeOffset>)(object)typedColumn, options),
            Type t when t == typeof(TimeSpan) => CreateDurationArray((NivaraColumn<TimeSpan>)(object)typedColumn, options),
            _ => throw new UnsupportedTypeException(typeof(T))
        };
    }

    /// <summary>
    /// Creates a Boolean Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateBooleanArray(NivaraColumn<bool> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        // Arrow stores boolean data bit-packed, one bit per slot (LSB first).
        var dataBytes = new byte[(length + 7) / 8];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            if (column[i])
                dataBytes[i >> 3] |= (byte)(1 << (i & 7));
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(dataBytes));
        var arrayData = new ArrayData(BooleanType.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates an Int32 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt32Array(NivaraColumn<int> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new int[length];
        column.CopyTo(data.AsSpan(), 0);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(Int32Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates an Int64 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt64Array(NivaraColumn<long> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new long[length];
        column.CopyTo(data.AsSpan(), 0L);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(Int64Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a Float Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateFloatArray(NivaraColumn<float> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new float[length];
        column.CopyTo(data.AsSpan(), 0f);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(FloatType.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a Double Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateDoubleArray(NivaraColumn<double> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new double[length];
        column.CopyTo(data.AsSpan(), 0d);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(DoubleType.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a String Arrow array from a NivaraColumn
    /// </summary>
    /// <remarks>
    /// Variable-binary (offsets + byte data) layout requires a per-value pass to encode each
    /// string's bytes and advance the offset buffer, so this path uses the Apache.Arrow builder
    /// (element-wise) and is the only remaining element-wise export path. Validity is produced
    /// by the builder itself; the bulk <see cref="ArrowValidityMask"/> helper cannot avoid the
    /// per-value offset pass for this layout.
    /// </remarks>
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

        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var minSafeDateTime = new DateTime(1677, 9, 21, 0, 0, 0, DateTimeKind.Utc);
        var maxSafeDateTime = new DateTime(2262, 4, 11, 23, 47, 16, DateTimeKind.Utc);
        var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        var data = new long[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;

            var dateTime = column[i];
            if (dateTime.Kind == DateTimeKind.Unspecified)
                dateTime = DateTime.SpecifyKind(dateTime, DateTimeKind.Utc);
            else if (dateTime.Kind == DateTimeKind.Local)
                dateTime = dateTime.ToUniversalTime();

            if (dateTime < minSafeDateTime)
                dateTime = minSafeDateTime;
            else if (dateTime > maxSafeDateTime)
                dateTime = maxSafeDateTime;

            data[i] = (dateTime - unixEpoch).Ticks / 10;
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(timestampType, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a UInt8 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt8Array(NivaraColumn<byte> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new byte[length];
        column.CopyTo(data.AsSpan(), (byte)0);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(data));
        var arrayData = new ArrayData(UInt8Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates an Int16 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt16Array(NivaraColumn<short> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new short[length];
        column.CopyTo(data.AsSpan(), (short)0);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(Int16Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a UInt32 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt32Array(NivaraColumn<uint> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new uint[length];
        column.CopyTo(data.AsSpan(), 0u);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(UInt32Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a UInt64 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt64Array(NivaraColumn<ulong> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new ulong[length];
        column.CopyTo(data.AsSpan(), 0ul);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(UInt64Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a UInt16 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateUInt16Array(NivaraColumn<ushort> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new ushort[length];
        column.CopyTo(data.AsSpan(), (ushort)0);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(UInt16Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates an Int8 Arrow array from a NivaraColumn
    /// </summary>
    private static IArrowArray CreateInt8Array(NivaraColumn<sbyte> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new sbyte[length];
        column.CopyTo(data.AsSpan(), (sbyte)0);

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(Int8Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a HalfFloat Arrow array from a NivaraColumn of <see cref="Half"/>
    /// </summary>
    private static IArrowArray CreateHalfFloatArray(NivaraColumn<Half> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new ushort[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            var h = column[i];
            data[i] = Unsafe.As<Half, ushort>(ref h);
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(new HalfFloatType(), length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates an Int64 Arrow array from a NivaraColumn of <see cref="nint"/> (widened representation)
    /// </summary>
    private static IArrowArray CreateInt64Array(NivaraColumn<nint> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new long[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            data[i] = (long)column[i];
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(Int64Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a UInt64 Arrow array from a NivaraColumn of <see cref="nuint"/> (widened representation)
    /// </summary>
    private static IArrowArray CreateUInt64Array(NivaraColumn<nuint> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new ulong[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            data[i] = (ulong)column[i];
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(UInt64Type.Default, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a String Arrow array from a NivaraColumn of <see cref="char"/> (widened representation)
    /// </summary>
    /// <remarks>
    /// See <see cref="CreateStringArray(NivaraColumn{string}, ArrowConversionOptions)"/>: the
    /// variable-binary layout needs a per-value pass, so this path stays element-wise.
    /// </remarks>
    private static IArrowArray CreateStringArray(NivaraColumn<char> column, ArrowConversionOptions options)
    {
        var builder = new StringArray.Builder();

        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                builder.AppendNull();
            else
                builder.Append(column[i].ToString());
        }

        return builder.Build();
    }

    /// <summary>
    /// Creates a Date32 Arrow array from a NivaraColumn of <see cref="DateOnly"/>
    /// </summary>
    private static IArrowArray CreateDate32Array(NivaraColumn<DateOnly> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var epoch = new DateOnly(1970, 1, 1);
        var data = new int[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            data[i] = column[i].DayNumber - epoch.DayNumber;
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(new Date32Type(), length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a Time64 (nanosecond) Arrow array from a NivaraColumn of <see cref="TimeOnly"/>
    /// </summary>
    private static IArrowArray CreateTime64Array(NivaraColumn<TimeOnly> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new long[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            data[i] = column[i].ToTimeSpan().Ticks * 100L;
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(new Time64Type(TimeUnit.Nanosecond), length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a FixedSizeBinary(16) Arrow array from a NivaraColumn of <see cref="Guid"/>
    /// </summary>
    private static IArrowArray CreateFixedSizeBinaryArray(NivaraColumn<Guid> column, ArrowConversionOptions options)
    {
        const int byteLength = 16;
        var values = new byte[column.Length * byteLength];
        int nullCount = column.NullCount;

        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                continue;

            column[i].ToByteArray().CopyTo(values, i * byteLength);
        }

        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validity = ArrowValidityMask.ToValidityBitmap(nullMask);

        var arrayData = new ArrayData(
            new FixedSizeBinaryType(byteLength),
            column.Length,
            nullCount,
            0,
            new[] { new ArrowBuffer(validity), new ArrowBuffer(values) },
            System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a Timestamp Arrow array from a NivaraColumn of <see cref="DateTimeOffset"/>,
    /// storing each value's UTC instant
    /// </summary>
    private static IArrowArray CreateTimestampArray(NivaraColumn<DateTimeOffset> column, ArrowConversionOptions options)
    {
        var timestampType = new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc);

        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var minSafeDateTime = new DateTime(1677, 9, 21, 0, 0, 0, DateTimeKind.Utc);
        var maxSafeDateTime = new DateTime(2262, 4, 11, 23, 47, 16, DateTimeKind.Utc);
        var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        var data = new long[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;

            var utcDateTime = column[i].ToUniversalTime().UtcDateTime;
            if (utcDateTime < minSafeDateTime)
                utcDateTime = minSafeDateTime;
            else if (utcDateTime > maxSafeDateTime)
                utcDateTime = maxSafeDateTime;

            data[i] = (utcDateTime - unixEpoch).Ticks / 10;
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(timestampType, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a Duration (nanosecond) Arrow array from a NivaraColumn of <see cref="TimeSpan"/>
    /// </summary>
    private static IArrowArray CreateDurationArray(NivaraColumn<TimeSpan> column, ArrowConversionOptions options)
    {
        int length = column.Length;
        var nullMask = column.TryGetNullMask(out var nm) ? nm : ReadOnlySpan<bool>.Empty;
        var validityBytes = ArrowValidityMask.ToValidityBitmap(nullMask);
        var validityBuffer = validityBytes.Length == 0 ? ArrowBuffer.Empty : new ArrowBuffer(new ReadOnlyMemory<byte>(validityBytes));

        var data = new long[length];
        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
                continue;
            data[i] = column[i].Ticks * 100L;
        }

        var dataBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));
        var arrayData = new ArrayData(DurationType.Nanosecond, length, column.NullCount, 0,
            new[] { validityBuffer, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    /// <summary>
    /// Creates a zero-copy Arrow array sharing the column's backing data buffer.
    /// Only flat, null-free columns of supported unmanaged primitive types are eligible.
    /// </summary>
    private static IArrowArray CreateZeroCopyArrowArray<T>(NivaraColumn<T> column, string? columnName)
    {
        var label = columnName is null ? "the source column" : $"column '{columnName}'";

        if (column.HasNulls)
            throw new UnsupportedTypeException(typeof(T),
                $"{label} contains null values; zero-copy export requires a null-free column. " +
                $"Set {nameof(ArrowConversionOptions.UseZeroCopy)} = false to use the copying path.");

        return typeof(T) switch
        {
            Type t when t == typeof(byte) => CreateZeroCopyPrimitive((NivaraColumn<byte>)(object)column, UInt8Type.Default, sizeof(byte)),
            Type t when t == typeof(sbyte) => CreateZeroCopyPrimitive((NivaraColumn<sbyte>)(object)column, Int8Type.Default, sizeof(sbyte)),
            Type t when t == typeof(short) => CreateZeroCopyPrimitive((NivaraColumn<short>)(object)column, Int16Type.Default, sizeof(short)),
            Type t when t == typeof(ushort) => CreateZeroCopyPrimitive((NivaraColumn<ushort>)(object)column, UInt16Type.Default, sizeof(ushort)),
            Type t when t == typeof(int) => CreateZeroCopyPrimitive((NivaraColumn<int>)(object)column, Int32Type.Default, sizeof(int)),
            Type t when t == typeof(uint) => CreateZeroCopyPrimitive((NivaraColumn<uint>)(object)column, UInt32Type.Default, sizeof(uint)),
            Type t when t == typeof(long) => CreateZeroCopyPrimitive((NivaraColumn<long>)(object)column, Int64Type.Default, sizeof(long)),
            Type t when t == typeof(ulong) => CreateZeroCopyPrimitive((NivaraColumn<ulong>)(object)column, UInt64Type.Default, sizeof(ulong)),
            Type t when t == typeof(float) => CreateZeroCopyPrimitive((NivaraColumn<float>)(object)column, FloatType.Default, sizeof(float)),
            Type t when t == typeof(double) => CreateZeroCopyPrimitive((NivaraColumn<double>)(object)column, DoubleType.Default, sizeof(double)),
            Type t when t == typeof(Half) => CreateZeroCopyHalf((NivaraColumn<Half>)(object)column),
            _ => throw new UnsupportedTypeException(typeof(T),
                $"{label} has element type '{typeof(T).Name}', which has no zero-copy export layout. " +
                $"The type is supported by the copying path; set {nameof(ArrowConversionOptions.UseZeroCopy)} = false to use it.")
        };
    }

    private static IArrowArray CreateZeroCopyPrimitive<T>(NivaraColumn<T> column, IArrowType arrowType, int elementSize) where T : unmanaged
    {
        if (column.Storage is not ColumnStorage<T> storage)
            throw new InvalidOperationException("Zero-copy Arrow conversion requires array-backed column storage.");

        if (!MemoryMarshal.TryGetArray(storage.Data, out var segment) || segment.Array is null)
            throw new InvalidOperationException("Zero-copy Arrow conversion requires contiguous array-backed storage.");

        int count = column.Length;
        var manager = new UnmanagedMemoryManager<T>(segment.Array, segment.Offset, count);
        var byteMemory = manager.Memory.Slice(0, count * elementSize);
        var dataBuffer = new ArrowBuffer(byteMemory);

        var arrayData = new ArrayData(arrowType, count, 0, 0,
            new[] { ArrowBuffer.Empty, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    private static IArrowArray CreateZeroCopyHalf(NivaraColumn<Half> column)
    {
        if (column.Storage is not ColumnStorage<Half> storage)
            throw new InvalidOperationException("Zero-copy Arrow conversion requires array-backed column storage.");

        if (!MemoryMarshal.TryGetArray(storage.Data, out var segment) || segment.Array is null)
            throw new InvalidOperationException("Zero-copy Arrow conversion requires contiguous array-backed storage.");

        int count = column.Length;
        var manager = new UnmanagedMemoryManager<Half>(segment.Array, segment.Offset, count);
        var byteMemory = manager.Memory.Slice(0, count * 2);
        var dataBuffer = new ArrowBuffer(byteMemory);

        var arrayData = new ArrayData(new HalfFloatType(), count, 0, 0,
            new[] { ArrowBuffer.Empty, dataBuffer }, System.Array.Empty<ArrayData>());

        return ArrowArrayFactory.BuildArray(arrayData);
    }

    private sealed class UnmanagedMemoryManager<T> : MemoryManager<byte> where T : unmanaged
    {
        private readonly T[] _array;
        private readonly int _start;
        private readonly int _count;

        public UnmanagedMemoryManager(T[] array, int start, int count)
        {
            _array = array;
            _start = start;
            _count = count;
        }

        public override Span<byte> GetSpan() =>
            MemoryMarshal.AsBytes(_array.AsSpan(_start, _count));

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            var handle = GCHandle.Alloc(_array, GCHandleType.Pinned);
            unsafe
            {
                void* ptr = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_array));
                ptr = (byte*)ptr + (_start + elementIndex) * Unsafe.SizeOf<T>();
                return new MemoryHandle(ptr, handle, this);
            }
        }

        public override void Unpin() { }

        protected override void Dispose(bool disposing) { }
    }

    /// <summary>
    /// Converts an Arrow column to a Nivara column using dynamic dispatch
    /// </summary>
    private static IColumn ConvertArrowColumnToNivaraColumn(Column arrowColumn, IArrowType arrowType, ArrowConversionOptions options, string columnName, IReadOnlyDictionary<string, string>? metadata)
    {
        var clrType = TypeMapper.MapArrowToClr(arrowType);

        // Restore the original CLR type for columns written by Nivara. Types with a shared
        // Arrow representation (nint via Int64, nuint via UInt64, char via String,
        // DateTimeOffset via Timestamp) depend on this metadata to round-trip correctly.
        var restoredType = TypeMapper.ResolveMetadataClrType(metadata, columnName);
        if (restoredType is not null)
        {
            clrType = restoredType;
        }

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
            Type t when t == typeof(Half) => ConvertArrowColumnToNivaraColumnTyped<Half>(arrowColumn, options),
            Type t when t == typeof(nint) => ConvertArrowColumnToNivaraColumnTyped<nint>(arrowColumn, options),
            Type t when t == typeof(nuint) => ConvertArrowColumnToNivaraColumnTyped<nuint>(arrowColumn, options),
            Type t when t == typeof(char) => ConvertArrowColumnToNivaraColumnTyped<char>(arrowColumn, options),
            Type t when t == typeof(DateOnly) => ConvertArrowColumnToNivaraColumnTyped<DateOnly>(arrowColumn, options),
            Type t when t == typeof(TimeOnly) => ConvertArrowColumnToNivaraColumnTyped<TimeOnly>(arrowColumn, options),
            Type t when t == typeof(Guid) => ConvertArrowColumnToNivaraColumnTyped<Guid>(arrowColumn, options),
            Type t when t == typeof(DateTimeOffset) => ConvertArrowColumnToNivaraColumnTyped<DateTimeOffset>(arrowColumn, options),
            Type t when t == typeof(TimeSpan) => ConvertArrowColumnToNivaraColumnTyped<TimeSpan>(arrowColumn, options),
            _ => throw new UnsupportedTypeException(clrType)
        };
    }

    /// <summary>
    /// Converts an Arrow column to a NivaraColumn<int> with proper null handling
    /// </summary>
    private static IColumn ConvertArrowColumnToNivaraColumnInt(Column arrowColumn, ArrowConversionOptions options)
    {
        int totalLength = (int)arrowColumn.Length;

        if (totalLength == 0)
        {
            return NivaraColumn<int>.Create(System.Array.Empty<int>());
        }

        var chunkedArray = arrowColumn.Data;
        int chunkCount = chunkedArray.ArrayCount;

        var values = new int[totalLength];
        var nullMask = new bool[totalLength];

        int offset = 0;
        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
        {
            var chunk = chunkedArray.Array(chunkIndex);
            int chunkLength = (int)chunk.Length;

            var validityBytes = (chunk.Data.Buffers.Length > 0)
                ? chunk.Data.Buffers[0].Span
                : ReadOnlySpan<byte>.Empty;
            ArrowValidityMask.ToNullMask(validityBytes, (int)chunk.Offset, nullMask.AsSpan(offset, chunkLength));

            var intArray = (Int32Array)chunk;
            for (int i = 0; i < chunkLength; i++)
            {
                if (!nullMask[offset + i])
                    values[offset + i] = intArray.GetValue(i)!.Value;
            }

            offset += chunkLength;
        }

        var nullable = new int?[totalLength];
        for (int i = 0; i < totalLength; i++)
            if (!nullMask[i])
                nullable[i] = values[i];

        return NivaraColumnFactory.CreateFromNullable(nullable);
    }

    /// <summary>
    /// Creates a null value for the specified type
    /// </summary>
    private static T? CreateNullValue<T>()
    {
        if (typeof(T).IsValueType)
        {
            return default(T?); // This should be null for nullable value types
        }
        else
        {
            return default(T); // This should be null for reference types
        }
    }

    /// <summary>
    /// Converts a typed Arrow column to a Nivara column
    /// </summary>
    private static IColumn ConvertArrowColumnToNivaraColumnTyped<T>(Column arrowColumn, ArrowConversionOptions options)
    {
        int totalLength = (int)arrowColumn.Length;

        if (totalLength == 0)
        {
            return NivaraColumn<T>.Create(System.Array.Empty<T>());
        }

        var chunkedArray = arrowColumn.Data;
        int chunkCount = chunkedArray.ArrayCount;

        if (chunkCount == 0)
        {
            return NivaraColumn<T>.Create(System.Array.Empty<T>());
        }

        var nullMask = new bool[totalLength];
        var values = new T[totalLength];

        int offset = 0;
        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
        {
            var chunk = chunkedArray.Array(chunkIndex);
            int chunkLength = (int)chunk.Length;

            var validityBytes = (chunk.Data.Buffers.Length > 0)
                ? chunk.Data.Buffers[0].Span
                : ReadOnlySpan<byte>.Empty;
            ArrowValidityMask.ToNullMask(validityBytes, (int)chunk.Offset, nullMask.AsSpan(offset, chunkLength));

            for (int i = 0; i < chunkLength; i++)
            {
                if (!nullMask[offset + i])
                    values[offset + i] = ExtractValueFromArrowArray<T>(chunk, i, options);
            }

            offset += chunkLength;
        }

        if (typeof(T).IsValueType)
        {
            return typeof(T) switch
            {
                Type t when t == typeof(int) => ConvertArrowColumnToNivaraColumnInt(arrowColumn, options),
                Type t when t == typeof(long) => CreateNullableColumn<long>((long[])(object)values, nullMask),
                Type t when t == typeof(float) => CreateNullableColumn<float>((float[])(object)values, nullMask),
                Type t when t == typeof(double) => CreateNullableColumn<double>((double[])(object)values, nullMask),
                Type t when t == typeof(bool) => CreateNullableColumn<bool>((bool[])(object)values, nullMask),
                Type t when t == typeof(DateTime) => CreateNullableColumn<DateTime>((DateTime[])(object)values, nullMask),
                Type t when t == typeof(byte) => CreateNullableColumn<byte>((byte[])(object)values, nullMask),
                Type t when t == typeof(short) => CreateNullableColumn<short>((short[])(object)values, nullMask),
                Type t when t == typeof(uint) => CreateNullableColumn<uint>((uint[])(object)values, nullMask),
                Type t when t == typeof(ulong) => CreateNullableColumn<ulong>((ulong[])(object)values, nullMask),
                Type t when t == typeof(ushort) => CreateNullableColumn<ushort>((ushort[])(object)values, nullMask),
                Type t when t == typeof(sbyte) => CreateNullableColumn<sbyte>((sbyte[])(object)values, nullMask),
                Type t when t == typeof(Half) => CreateNullableColumn<Half>((Half[])(object)values, nullMask),
                Type t when t == typeof(nint) => CreateNullableColumn<nint>((nint[])(object)values, nullMask),
                Type t when t == typeof(nuint) => CreateNullableColumn<nuint>((nuint[])(object)values, nullMask),
                Type t when t == typeof(char) => CreateNullableColumn<char>((char[])(object)values, nullMask),
                Type t when t == typeof(DateOnly) => CreateNullableColumn<DateOnly>((DateOnly[])(object)values, nullMask),
                Type t when t == typeof(TimeOnly) => CreateNullableColumn<TimeOnly>((TimeOnly[])(object)values, nullMask),
                Type t when t == typeof(Guid) => CreateNullableColumn<Guid>((Guid[])(object)values, nullMask),
                Type t when t == typeof(DateTimeOffset) => CreateNullableColumn<DateTimeOffset>((DateTimeOffset[])(object)values, nullMask),
                Type t when t == typeof(TimeSpan) => CreateNullableColumn<TimeSpan>((TimeSpan[])(object)values, nullMask),
                _ => throw new UnsupportedTypeException(typeof(T))
            };
        }
        else
        {
            var referenceValues = new T[totalLength];
            for (int i = 0; i < totalLength; i++)
            {
                referenceValues[i] = nullMask[i] ? default(T)! : values[i];
            }
            return NivaraColumn<T>.CreateForReferenceType(referenceValues);
        }
    }

    /// <summary>
    /// Creates a nullable column from a list of object values
    /// </summary>
    private static IColumn CreateNullableColumn<T>(T[] values, bool[] nullMask) where T : struct
    {
        var nullableArray = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            nullableArray[i] = nullMask[i] ? null : values[i];
        }
        return NivaraColumnFactory.CreateFromNullable(nullableArray);
    }

    /// <summary>
    /// Creates a null value for the specified type
    /// </summary>
    private static T ExtractValueFromArrowArray<T>(IArrowArray array, int index, ArrowConversionOptions options)
    {
        // This method should only be called for non-null values
        if (array.IsNull(index))
        {
            throw new InvalidOperationException($"ExtractValueFromArrowArray called for null value at index {index}");
        }

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
            HalfFloatArray halfFloatArray when typeof(T) == typeof(Half) => (T)(object)halfFloatArray.GetValue(index)!.Value,
            Int64Array nintArray when typeof(T) == typeof(nint) => (T)(object)(nint)nintArray.GetValue(index)!.Value,
            UInt64Array nuintArray when typeof(T) == typeof(nuint) => (T)(object)(nuint)nuintArray.GetValue(index)!.Value,
            StringArray charArray when typeof(T) == typeof(char) => (T)(object)ExtractCharFromString(charArray, index),
            Date32Array date32Array when typeof(T) == typeof(DateOnly) => (T)(object)date32Array.GetDateOnly(index)!.Value,
            Date64Array date64Array when typeof(T) == typeof(DateOnly) => (T)(object)date64Array.GetDateOnly(index)!.Value,
            Time64Array time64Array when typeof(T) == typeof(TimeOnly) => (T)(object)Time64ToTimeOnly(time64Array, index),
            Time32Array time32Array when typeof(T) == typeof(TimeOnly) => (T)(object)Time32ToTimeOnly(time32Array, index),
            DurationArray durationArray when typeof(T) == typeof(TimeSpan) => (T)(object)durationArray.GetTimeSpan(index)!.Value,
            FixedSizeBinaryArray fixedSizeBinaryArray when typeof(T) == typeof(Guid) => (T)(object)new Guid(fixedSizeBinaryArray.GetBytes(index)),
            TimestampArray timestampArray when typeof(T) == typeof(DateTimeOffset) => (T)(object)TimestampToDateTimeOffset(timestampArray, index),
            _ => throw new UnsupportedTypeException(typeof(T), new[] { $"Arrow array type {array.GetType().Name} not supported for CLR type {typeof(T).Name}" })
        };
    }

    /// <summary>
    /// Extracts a single <see cref="char"/> from a widened string representation
    /// </summary>
    private static char ExtractCharFromString(StringArray stringArray, int index)
    {
        var value = stringArray.GetString(index);
        if (value.Length != 1)
        {
            throw new UnsupportedTypeException(
                typeof(char),
                new[] { $"String value at index {index} has length {value.Length}; a char column requires exactly one character" });
        }

        return value[0];
    }

    /// <summary>
    /// Converts a Time64 (nanoseconds since midnight) value to <see cref="TimeOnly"/>
    /// </summary>
    private static TimeOnly Time64ToTimeOnly(Time64Array time64Array, int index)
    {
        var nanoseconds = time64Array.GetValue(index)!.Value;
        return new TimeOnly(nanoseconds / 100);
    }

    /// <summary>
    /// Converts a Time32 (seconds or milliseconds since midnight) value to <see cref="TimeOnly"/>
    /// </summary>
    private static TimeOnly Time32ToTimeOnly(Time32Array time32Array, int index)
    {
        var value = time32Array.GetValue(index)!.Value;
        var unit = ((Time32Type)time32Array.Data.DataType).Unit;

        return unit == TimeUnit.Second
            ? TimeOnly.FromTimeSpan(TimeSpan.FromSeconds(value))
            : TimeOnly.FromTimeSpan(TimeSpan.FromMilliseconds(value));
    }

    /// <summary>
    /// Converts a timestamp value to <see cref="DateTimeOffset"/>, preserving the UTC instant
    /// regardless of the configured timezone
    /// </summary>
    private static DateTimeOffset TimestampToDateTimeOffset(TimestampArray timestampArray, int index)
    {
        var unit = ((TimestampType)timestampArray.Data.DataType).Unit;
        var utcDateTime = ConvertTimestampValueToDateTime(timestampArray.GetValue(index)!.Value, unit);
        return new DateTimeOffset(DateTime.SpecifyKind(utcDateTime, DateTimeKind.Utc), TimeSpan.Zero);
    }

    /// <summary>
    /// Converts a timestamp value (in the given unit since the Unix epoch) to a UTC <see cref="DateTime"/>
    /// </summary>
    private static DateTime ConvertTimestampValueToDateTime(long value, TimeUnit unit)
    {
        var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        return unit switch
        {
            TimeUnit.Second => unixEpoch.AddSeconds(value),
            TimeUnit.Millisecond => unixEpoch.AddMilliseconds(value),
            TimeUnit.Nanosecond => unixEpoch.AddTicks(value / 100),
            _ => unixEpoch.AddMicroseconds(value)
        };
    }

    /// <summary>
    /// Converts a timestamp value from Arrow to DateTime
    /// </summary>
    private static DateTime ConvertTimestampToDateTime(TimestampArray timestampArray, int index, ArrowConversionOptions options)
    {
        var timestampValue = timestampArray.GetValue(index)!.Value;
        var unit = ((TimestampType)timestampArray.Data.DataType).Unit;

        // Convert from the field's unit since Unix epoch to DateTime (UTC)
        var dateTime = ConvertTimestampValueToDateTime(timestampValue, unit);

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
}
