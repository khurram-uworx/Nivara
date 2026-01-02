using Parquet.Data;
using Parquet.Schema;
using Nivara.Exceptions;

namespace Nivara.IO;

/// <summary>
/// Provides functionality for writing Nivara data structures to Parquet files
/// </summary>
/// <remarks>
/// This class supports both file-based and stream-based writing operations,
/// with configurable compression and row group sizes for optimal performance.
/// </remarks>
public static class ParquetWriter
{
    /// <summary>
    /// Writes a NivaraFrame to a Parquet file asynchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <exception cref="ArgumentNullException">Thrown when frame or filePath is null</exception>
    /// <exception cref="NivaraIOException">Thrown when file writing fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static async Task WriteParquetAsync(NivaraFrame frame, string filePath, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(filePath);

        options ??= new ParquetWriteOptions();

        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            await WriteParquetAsync(frame, fileStream, options, cancellationToken);
        }
        catch (Exception ex) when (ex is not NivaraIOException)
        {
            throw new NivaraIOException($"Failed to write Parquet file: {ex.Message}", ex)
            {
                FilePath = filePath,
                OperationContext = "ParquetWriter.WriteParquetAsync"
            };
        }
    }

    /// <summary>
    /// Writes a NivaraFrame to a Parquet stream asynchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <exception cref="ArgumentNullException">Thrown when frame or stream is null</exception>
    /// <exception cref="NivaraIOException">Thrown when stream writing fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static async Task WriteParquetAsync(NivaraFrame frame, Stream stream, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(stream);

        options ??= new ParquetWriteOptions();

        try
        {
            if (frame.ColumnCount == 0)
            {
                await WriteEmptyParquetFile(stream, cancellationToken);
                return;
            }

            if (options.ValidateSchema)
            {
                ValidateFrameSchema(frame);
            }

            await ConvertNivaraFrameToParquet(frame, stream, options, cancellationToken);
        }
        catch (Exception ex) when (ex is not NivaraIOException and not UnsupportedTypeException and not SchemaValidationException)
        {
            throw new NivaraIOException($"Failed to write Parquet stream: {ex.Message}", ex)
            {
                OperationContext = "ParquetWriter.WriteParquetAsync"
            };
        }
    }

    /// <summary>
    /// Writes a NivaraFrame to a Parquet file synchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    public static void WriteParquet(NivaraFrame frame, string filePath, ParquetWriteOptions? options = null)
    {
        WriteParquetAsync(frame, filePath, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Writes a NivaraFrame to a Parquet stream synchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    public static void WriteParquet(NivaraFrame frame, Stream stream, ParquetWriteOptions? options = null)
    {
        WriteParquetAsync(frame, stream, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Writes multiple NivaraFrames to a single Parquet file asynchronously
    /// </summary>
    /// <param name="frames">The NivaraFrames to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <exception cref="ArgumentNullException">Thrown when frames or filePath is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when frames have incompatible schemas</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static async Task WriteParquetBatchAsync(IEnumerable<NivaraFrame> frames, string filePath, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(filePath);

        options ??= new ParquetWriteOptions();

        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            await WriteParquetBatchAsync(frames, fileStream, options, cancellationToken);
        }
        catch (Exception ex) when (ex is not NivaraIOException and not SchemaValidationException)
        {
            throw new NivaraIOException($"Failed to write Parquet batch file: {ex.Message}", ex)
            {
                FilePath = filePath,
                OperationContext = "ParquetWriter.WriteParquetBatchAsync"
            };
        }
    }

    /// <summary>
    /// Writes multiple NivaraFrames to a Parquet stream asynchronously
    /// </summary>
    /// <param name="frames">The NivaraFrames to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <exception cref="ArgumentNullException">Thrown when frames or stream is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when frames have incompatible schemas</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static async Task WriteParquetBatchAsync(IEnumerable<NivaraFrame> frames, Stream stream, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(stream);

        options ??= new ParquetWriteOptions();

        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            await WriteEmptyParquetFile(stream, cancellationToken);
            return;
        }

        if (frameList.Count == 1)
        {
            await WriteParquetAsync(frameList[0], stream, options, cancellationToken);
            return;
        }

        // Validate schema compatibility
        if (options.ValidateSchema)
        {
            ValidateFrameSchemaCompatibility(frameList);
        }

        // Concatenate frames and write as single frame
        var concatenatedFrame = ConcatenateFrames(frameList);
        try
        {
            await WriteParquetAsync(concatenatedFrame, stream, options, cancellationToken);
        }
        finally
        {
            concatenatedFrame.Dispose();
        }
    }

    /// <summary>
    /// Writes multiple NivaraFrames to a single Parquet file synchronously
    /// </summary>
    /// <param name="frames">The NivaraFrames to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    public static void WriteParquetBatch(IEnumerable<NivaraFrame> frames, string filePath, ParquetWriteOptions? options = null)
    {
        WriteParquetBatchAsync(frames, filePath, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Writes multiple NivaraFrames to a Parquet stream synchronously
    /// </summary>
    /// <param name="frames">The NivaraFrames to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    public static void WriteParquetBatch(IEnumerable<NivaraFrame> frames, Stream stream, ParquetWriteOptions? options = null)
    {
        WriteParquetBatchAsync(frames, stream, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Writes an empty Parquet file with a dummy column
    /// </summary>
    private static async Task WriteEmptyParquetFile(Stream stream, CancellationToken cancellationToken = default)
    {
        // Parquet requires at least one field, so create a dummy field for empty frames
        var dummyField = new DataField<int>("_empty");
        var emptySchema = new ParquetSchema(dummyField);

        using var emptyWriter = await Parquet.ParquetWriter.CreateAsync(emptySchema, stream);
        using var emptyRowGroup = emptyWriter.CreateRowGroup();
        var emptyColumn = new DataColumn(dummyField, Array.Empty<int>());
        await emptyRowGroup.WriteColumnAsync(emptyColumn);
    }

    /// <summary>
    /// Validates that a frame's schema is compatible with Parquet format
    /// </summary>
    private static void ValidateFrameSchema(NivaraFrame frame)
    {
        var unsupportedColumns = new List<string>();

        for (int i = 0; i < frame.ColumnCount; i++)
        {
            var columnName = frame.ColumnNames[i];
            var columnType = frame.Schema.GetColumnType(columnName);

            if (!TypeMapper.IsParquetSupported(columnType))
            {
                unsupportedColumns.Add($"{columnName} ({columnType.Name})");
            }
        }

        if (unsupportedColumns.Count > 0)
        {
            var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
            throw new SchemaValidationException(
                $"Frame contains unsupported column types for Parquet: {string.Join(", ", unsupportedColumns)}. " +
                $"Supported types: {supportedTypes}")
            {
                ExpectedSchema = $"Columns with types: {supportedTypes}",
                ActualSchema = $"Columns with types: {string.Join(", ", unsupportedColumns)}"
            };
        }
    }

    /// <summary>
    /// Validates that multiple frames have compatible schemas
    /// </summary>
    private static void ValidateFrameSchemaCompatibility(List<NivaraFrame> frames)
    {
        if (frames.Count <= 1) return;

        var firstFrame = frames[0];
        for (int i = 1; i < frames.Count; i++)
        {
            var currentFrame = frames[i];

            if (!AreFramesSchemaCompatible(firstFrame, currentFrame))
            {
                throw new SchemaValidationException(
                    $"Frame {i} has incompatible schema with the first frame. All frames must have the same column names and types.")
                {
                    ExpectedSchema = GetFrameSchemaDescription(firstFrame),
                    ActualSchema = GetFrameSchemaDescription(currentFrame)
                };
            }
        }
    }

    /// <summary>
    /// Creates a Parquet DataField with explicit nullability based on actual column data
    /// </summary>
    private static DataField CreateParquetFieldWithNullability(string name, Type clrType, bool hasNulls)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(clrType);

        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Field name cannot be empty or whitespace", nameof(name));

        // Handle nullable types
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        
        // For value types, use hasNulls to determine nullability
        // For reference types, they are inherently nullable
        var isNullable = hasNulls || !actualType.IsValueType;

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
            _ => throw new UnsupportedTypeException(actualType, TypeMapper.GetTypeSuggestions(actualType))
        };
    }

    /// <summary>
    /// Checks if two frames have compatible schemas
    /// </summary>
    private static bool AreFramesSchemaCompatible(NivaraFrame frame1, NivaraFrame frame2)
    {
        if (frame1.ColumnCount != frame2.ColumnCount)
            return false;

        for (int i = 0; i < frame1.ColumnCount; i++)
        {
            if (!string.Equals(frame1.ColumnNames[i], frame2.ColumnNames[i], StringComparison.OrdinalIgnoreCase) ||
                frame1.Schema.GetColumnType(frame1.ColumnNames[i]) != frame2.Schema.GetColumnType(frame2.ColumnNames[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Gets a description of a frame's schema for error messages
    /// </summary>
    private static string GetFrameSchemaDescription(NivaraFrame frame)
    {
        var columns = new List<string>();
        for (int i = 0; i < frame.ColumnCount; i++)
        {
            var columnName = frame.ColumnNames[i];
            var columnType = frame.Schema.GetColumnType(columnName);
            columns.Add($"{columnName}:{columnType.Name}");
        }
        return $"[{string.Join(", ", columns)}]";
    }

    /// <summary>
    /// Converts a NivaraFrame to Parquet format using the Parquet.Net API
    /// </summary>
    private static async Task ConvertNivaraFrameToParquet(NivaraFrame frame, Stream stream, ParquetWriteOptions options, CancellationToken cancellationToken)
    {
        // Create Parquet schema
        var fields = new List<DataField>();

        for (int i = 0; i < frame.ColumnCount; i++)
        {
            var columnName = frame.ColumnNames[i];
            var columnType = frame.Schema.GetColumnType(columnName);
            var column = frame.GetColumn(columnName);

            // Check if the column actually has nulls to determine nullability
            var hasNulls = column.HasNulls;
            var field = CreateParquetFieldWithNullability(columnName, columnType, hasNulls);
            fields.Add(field);
        }

        var schema = new ParquetSchema(fields.ToArray());

        // Create Parquet writer
        using var parquetWriter = await Parquet.ParquetWriter.CreateAsync(schema, stream);
        using var rowGroupWriter = parquetWriter.CreateRowGroup();

        // Write column data
        for (int i = 0; i < frame.ColumnCount; i++)
        {
            // Check for cancellation before processing each column
            cancellationToken.ThrowIfCancellationRequested();

            var columnName = frame.ColumnNames[i];
            var columnType = frame.Schema.GetColumnType(columnName);
            var field = fields[i];

            try
            {
                var columnData = ExtractColumnDataArray(frame, columnName, columnType);
                var dataColumn = new DataColumn(field, columnData);

                await rowGroupWriter.WriteColumnAsync(dataColumn);
            }
            catch (Exception ex)
            {
                throw new DataCorruptionException($"Failed to write column '{columnName}': {ex.Message}", ex)
                {
                    AffectedColumns = new[] { columnName },
                    AffectedRowRange = new Range(0, frame.RowCount)
                };
            }
        }
    }

    /// <summary>
    /// Extracts column data from a NivaraFrame as an Array for Parquet writing
    /// </summary>
    private static Array ExtractColumnDataArray(NivaraFrame frame, string columnName, Type columnType)
    {
        // Handle nullable types by extracting the underlying type
        var actualType = Nullable.GetUnderlyingType(columnType) ?? columnType;
        var column = frame.GetColumn(columnName);

        return actualType switch
        {
            Type t when t == typeof(bool) => ExtractColumnDataArrayTyped<bool>(frame, columnName, column.HasNulls),
            Type t when t == typeof(int) => ExtractColumnDataArrayTyped<int>(frame, columnName, column.HasNulls),
            Type t when t == typeof(long) => ExtractColumnDataArrayTyped<long>(frame, columnName, column.HasNulls),
            Type t when t == typeof(float) => ExtractColumnDataArrayTyped<float>(frame, columnName, column.HasNulls),
            Type t when t == typeof(double) => ExtractColumnDataArrayTyped<double>(frame, columnName, column.HasNulls),
            Type t when t == typeof(DateTime) => ExtractColumnDataArrayTyped<DateTime>(frame, columnName, column.HasNulls),
            Type t when t == typeof(byte) => ExtractColumnDataArrayTyped<byte>(frame, columnName, column.HasNulls),
            Type t when t == typeof(short) => ExtractColumnDataArrayTyped<short>(frame, columnName, column.HasNulls),
            Type t when t == typeof(uint) => ExtractColumnDataArrayTyped<uint>(frame, columnName, column.HasNulls),
            Type t when t == typeof(ulong) => ExtractColumnDataArrayTyped<ulong>(frame, columnName, column.HasNulls),
            Type t when t == typeof(ushort) => ExtractColumnDataArrayTyped<ushort>(frame, columnName, column.HasNulls),
            Type t when t == typeof(sbyte) => ExtractColumnDataArrayTyped<sbyte>(frame, columnName, column.HasNulls),
            Type t when t == typeof(decimal) => ExtractColumnDataArrayTyped<decimal>(frame, columnName, column.HasNulls),
            Type t when t == typeof(string) => ExtractStringColumnDataArray(frame, columnName),
            _ => throw new UnsupportedTypeException(actualType, TypeMapper.GetTypeSuggestions(actualType))
        };
    }

    /// <summary>
    /// Extracts typed column data from a NivaraFrame as a non-nullable or nullable array for Parquet writing with buffer reuse.
    /// </summary>
    private static Array ExtractColumnDataArrayTyped<T>(NivaraFrame frame, string columnName, bool hasNulls) where T : struct
    {
        var column = frame.GetColumn<T>(columnName);

        if (hasNulls)
        {
            // Create nullable array when column has nulls
            var nullableValues = new T?[column.Length];

            // Use buffer pool for large columns to reduce memory pressure
            if (column.Length > 1024)
            {
                var buffer = BufferPool.RentIntBuffer(column.Length);
                try
                {
                    for (int i = 0; i < column.Length; i++)
                    {
                        nullableValues[i] = column.IsNull(i) ? null : column[i];
                    }
                }
                finally
                {
                    BufferPool.ReturnIntBuffer(buffer);
                }
            }
            else
            {
                for (int i = 0; i < column.Length; i++)
                {
                    nullableValues[i] = column.IsNull(i) ? null : column[i];
                }
            }

            return nullableValues;
        }
        else
        {
            // Create non-nullable array when column has no nulls
            var values = new T[column.Length];

            // Use buffer pool for large columns to reduce memory pressure
            if (column.Length > 1024)
            {
                var buffer = BufferPool.RentIntBuffer(column.Length);
                try
                {
                    for (int i = 0; i < column.Length; i++)
                    {
                        values[i] = column[i];
                    }
                }
                finally
                {
                    BufferPool.ReturnIntBuffer(buffer);
                }
            }
            else
            {
                for (int i = 0; i < column.Length; i++)
                {
                    values[i] = column[i];
                }
            }

            return values;
        }
    }

    /// <summary>
    /// Extracts string column data from a NivaraFrame as a string array with memory optimization.
    /// </summary>
    private static Array ExtractStringColumnDataArray(NivaraFrame frame, string columnName)
    {
        var column = frame.GetColumn<string>(columnName);
        var stringArray = new string[column.Length];

        // Use buffer pool for large string columns
        if (column.Length > 1024)
        {
            var buffer = BufferPool.RentByteBuffer(column.Length * 32); // Estimate for string processing
            try
            {
                for (int i = 0; i < column.Length; i++)
                {
                    // For strings, null is a valid value that Parquet.Net can handle
                    stringArray[i] = column.IsNull(i) ? null! : column[i];
                }
            }
            finally
            {
                BufferPool.ReturnByteBuffer(buffer);
            }
        }
        else
        {
            for (int i = 0; i < column.Length; i++)
            {
                // For strings, null is a valid value that Parquet.Net can handle
                stringArray[i] = column.IsNull(i) ? null! : column[i];
            }
        }

        return stringArray;
    }

    /// <summary>
    /// Concatenates multiple NivaraFrames into a single frame
    /// </summary>
    private static NivaraFrame ConcatenateFrames(List<NivaraFrame> frames)
    {
        if (frames.Count == 0)
            throw new ArgumentException("Cannot concatenate empty frame list");

        if (frames.Count == 1)
            return frames[0];

        var firstFrame = frames[0];
        var concatenatedColumns = new List<(string Name, IColumn Column)>();

        // Concatenate each column
        for (int columnIndex = 0; columnIndex < firstFrame.ColumnCount; columnIndex++)
        {
            var columnName = firstFrame.ColumnNames[columnIndex];
            var columnType = firstFrame.Schema.GetColumnType(columnName);

            var concatenatedColumn = ConcatenateColumn(frames, columnName, columnType);
            concatenatedColumns.Add((columnName, concatenatedColumn));
        }

        return new NivaraFrame(concatenatedColumns);
    }

    /// <summary>
    /// Concatenates a specific column from multiple frames
    /// </summary>
    private static IColumn ConcatenateColumn(List<NivaraFrame> frames, string columnName, Type columnType)
    {
        // Handle nullable types by extracting the underlying type
        var actualType = Nullable.GetUnderlyingType(columnType) ?? columnType;

        return actualType switch
        {
            Type t when t == typeof(bool) => ConcatenateColumnTyped<bool>(frames, columnName),
            Type t when t == typeof(int) => ConcatenateColumnTyped<int>(frames, columnName),
            Type t when t == typeof(long) => ConcatenateColumnTyped<long>(frames, columnName),
            Type t when t == typeof(float) => ConcatenateColumnTyped<float>(frames, columnName),
            Type t when t == typeof(double) => ConcatenateColumnTyped<double>(frames, columnName),
            Type t when t == typeof(DateTime) => ConcatenateColumnTyped<DateTime>(frames, columnName),
            Type t when t == typeof(byte) => ConcatenateColumnTyped<byte>(frames, columnName),
            Type t when t == typeof(short) => ConcatenateColumnTyped<short>(frames, columnName),
            Type t when t == typeof(uint) => ConcatenateColumnTyped<uint>(frames, columnName),
            Type t when t == typeof(ulong) => ConcatenateColumnTyped<ulong>(frames, columnName),
            Type t when t == typeof(ushort) => ConcatenateColumnTyped<ushort>(frames, columnName),
            Type t when t == typeof(sbyte) => ConcatenateColumnTyped<sbyte>(frames, columnName),
            Type t when t == typeof(decimal) => ConcatenateColumnTyped<decimal>(frames, columnName),
            Type t when t == typeof(string) => ConcatenateStringColumn(frames, columnName),
            _ => throw new UnsupportedTypeException(actualType, TypeMapper.GetTypeSuggestions(actualType))
        };
    }

    /// <summary>
    /// Concatenates a typed column from multiple frames with memory optimization.
    /// </summary>
    private static IColumn ConcatenateColumnTyped<T>(List<NivaraFrame> frames, string columnName) where T : struct
    {
        var totalLength = frames.Sum(f => f.RowCount);
        var nullableArray = new T?[totalLength];

        // Use buffer pool for large concatenations
        if (totalLength > 1024)
        {
            var buffer = BufferPool.RentIntBuffer(totalLength);
            try
            {
                var currentIndex = 0;
                foreach (var frame in frames)
                {
                    var column = frame.GetColumn<T>(columnName);

                    for (int i = 0; i < column.Length; i++)
                    {
                        nullableArray[currentIndex] = column.IsNull(i) ? null : column[i];
                        currentIndex++;
                    }
                }
            }
            finally
            {
                BufferPool.ReturnIntBuffer(buffer);
            }
        }
        else
        {
            var currentIndex = 0;
            foreach (var frame in frames)
            {
                var column = frame.GetColumn<T>(columnName);

                for (int i = 0; i < column.Length; i++)
                {
                    nullableArray[currentIndex] = column.IsNull(i) ? null : column[i];
                    currentIndex++;
                }
            }
        }

        return NivaraColumn<T>.CreateFromNullable(nullableArray);
    }

    /// <summary>
    /// Concatenates a string column from multiple frames with memory optimization.
    /// </summary>
    private static IColumn ConcatenateStringColumn(List<NivaraFrame> frames, string columnName)
    {
        var totalLength = frames.Sum(f => f.RowCount);
        var values = new string[totalLength];

        // Use buffer pool for large string concatenations
        if (totalLength > 1024)
        {
            var buffer = BufferPool.RentByteBuffer(totalLength * 32); // Estimate for string processing
            try
            {
                var currentIndex = 0;
                foreach (var frame in frames)
                {
                    var column = frame.GetColumn<string>(columnName);

                    for (int i = 0; i < column.Length; i++)
                    {
                        values[currentIndex] = column.IsNull(i) ? null! : column[i];
                        currentIndex++;
                    }
                }
            }
            finally
            {
                BufferPool.ReturnByteBuffer(buffer);
            }
        }
        else
        {
            var currentIndex = 0;
            foreach (var frame in frames)
            {
                var column = frame.GetColumn<string>(columnName);

                for (int i = 0; i < column.Length; i++)
                {
                    values[currentIndex] = column.IsNull(i) ? null! : column[i];
                    currentIndex++;
                }
            }
        }

        return NivaraColumn<string>.CreateForReferenceType(values);
    }
}
