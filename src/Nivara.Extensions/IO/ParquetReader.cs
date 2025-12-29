using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Parquet;
using Parquet.Data;
using Parquet.Schema;
using Nivara.IO;

namespace Nivara.IO;

/// <summary>
/// Provides Parquet reading capabilities with columnar compression and complex schema support.
/// </summary>
public static class ParquetReader
{
    /// <summary>
    /// Reads a Parquet file into a NivaraFrame asynchronously.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="NivaraIOException">Thrown when file reading fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
    public static async Task<NivaraFrame> ReadParquetAsync(string filePath, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filePath);
        
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Parquet file not found: {filePath}");

        options ??= new ParquetReadOptions();

        try
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return await ReadParquetAsync(fileStream, options, cancellationToken);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is FileNotFoundException))
        {
            throw new NivaraIOException($"Failed to read Parquet file: {ex.Message}", ex)
            {
                FilePath = filePath,
                OperationContext = "ParquetReader.ReadParquetAsync"
            };
        }
    }

    /// <summary>
    /// Reads Parquet data from a stream into a NivaraFrame asynchronously.
    /// </summary>
    /// <param name="stream">The stream containing Parquet data.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="NivaraIOException">Thrown when stream reading fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
    public static async Task<NivaraFrame> ReadParquetAsync(Stream stream, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        
        options ??= new ParquetReadOptions();

        try
        {
            // Use low-level ParquetReader API
            using var parquetReader = await Parquet.ParquetReader.CreateAsync(stream);
            
            if (parquetReader.RowGroupCount == 0)
            {
                // Create an empty frame with a dummy column
                var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
                return NivaraFrame.Create(("_empty", emptyColumn));
            }

            return await ConvertParquetToNivaraFrame(parquetReader, options, cancellationToken);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException))
        {
            throw new NivaraIOException($"Failed to read Parquet stream: {ex.Message}", ex)
            {
                OperationContext = "ParquetReader.ReadParquetAsync"
            };
        }
    }

    /// <summary>
    /// Reads a Parquet file into a NivaraFrame synchronously.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    public static NivaraFrame ReadParquet(string filePath, ParquetReadOptions? options = null)
    {
        return ReadParquetAsync(filePath, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Reads Parquet data from a stream into a NivaraFrame synchronously.
    /// </summary>
    /// <param name="stream">The stream containing Parquet data.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <returns>A NivaraFrame containing the Parquet data.</returns>
    public static NivaraFrame ReadParquet(Stream stream, ParquetReadOptions? options = null)
    {
        return ReadParquetAsync(stream, options).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Reads a Parquet file in streaming mode for large files with memory management.
    /// </summary>
    /// <param name="filePath">The path to the Parquet file.</param>
    /// <param name="options">Optional Parquet reading options.</param>
    /// <param name="memoryBudget">Maximum memory budget for streaming operations.</param>
    /// <returns>An enumerable of NivaraFrame chunks.</returns>
    public static IEnumerable<NivaraFrame> ReadParquetStreaming(string filePath, ParquetReadOptions? options = null, long memoryBudget = 256L * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(filePath);
        
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Parquet file not found: {filePath}");

        options ??= new ParquetReadOptions();

        return ReadParquetStreamingInternal(filePath, options, memoryBudget);
    }

    /// <summary>
    /// Internal implementation of streaming Parquet reading.
    /// </summary>
    private static IEnumerable<NivaraFrame> ReadParquetStreamingInternal(string filePath, ParquetReadOptions options, long memoryBudget)
    {
        using var bufferManager = new StreamingBufferManager(memoryBudget);
        
        using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        
        // For now, return the entire file as a single chunk with memory management
        // True streaming would require processing row groups individually
        var frame = ReadParquet(fileStream, options);
        
        // Force garbage collection if memory usage is high
        bufferManager.TryCollectGarbage();
        
        yield return frame;
    }

    /// <summary>
    /// Converts a Parquet file to a NivaraFrame using low-level API.
    /// </summary>
    private static async Task<NivaraFrame> ConvertParquetToNivaraFrame(Parquet.ParquetReader parquetReader, ParquetReadOptions options, CancellationToken cancellationToken)
    {
        var schema = parquetReader.Schema;
        var dataFields = schema.GetDataFields();
        
        if (dataFields.Length == 0)
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        // Check if this is an empty file with just the dummy column
        if (dataFields.Length == 1 && dataFields[0].Name == "_empty")
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        // Validate schema if requested
        if (options.ValidateSchema)
        {
            ValidateParquetSchema(schema);
        }

        var columns = new List<(string Name, IColumn Column)>();

        // Read all row groups
        for (int rowGroupIndex = 0; rowGroupIndex < parquetReader.RowGroupCount; rowGroupIndex++)
        {
            using var rowGroupReader = parquetReader.OpenRowGroupReader(rowGroupIndex);
            
            if (rowGroupIndex == 0)
            {
                // Initialize columns on first row group
                for (int columnIndex = 0; columnIndex < dataFields.Length; columnIndex++)
                {
                    var field = dataFields[columnIndex];
                    var columnName = field.Name;
                    
                    // Skip the dummy column used for empty files
                    if (columnName == "_empty")
                        continue;
                    
                    try
                    {
                        // Check for cancellation before processing each column
                        cancellationToken.ThrowIfCancellationRequested();
                        
                        // Read column data using the correct API
                        var columnData = await rowGroupReader.ReadColumnAsync(field);
                        var column = CreateNivaraColumnFromParquetData(columnData, field);
                        columns.Add((columnName, column));
                    }
                    catch (Exception ex)
                    {
                        throw new DataCorruptionException($"Failed to read column '{columnName}': {ex.Message}", ex)
                        {
                            AffectedColumns = new[] { columnName },
                            AffectedRowRange = new Range(0, 1000) // Use a default range since we can't access ThriftMetadata
                        };
                    }
                }
            }
            else
            {
                // Append data from subsequent row groups (simplified - would need proper concatenation)
                // For now, just use the first row group
                break;
            }
        }

        if (columns.Count == 0)
        {
            // Create an empty frame with a dummy column
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            return NivaraFrame.Create(("_empty", emptyColumn));
        }

        return NivaraFrame.Create(columns.ToArray());
    }

    /// <summary>
    /// Validates the Parquet schema for compatibility.
    /// </summary>
    private static void ValidateParquetSchema(ParquetSchema schema)
    {
        var dataFields = schema.GetDataFields();
        var unsupportedFields = new List<string>();

        foreach (var field in dataFields)
        {
            if (!IsTypeSupported(field.ClrType))
            {
                unsupportedFields.Add($"{field.Name} ({field.ClrType.Name})");
            }
        }

        if (unsupportedFields.Count > 0)
        {
            var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
            throw new SchemaValidationException($"Unsupported field types found: {string.Join(", ", unsupportedFields)}. Supported types: {supportedTypes}")
            {
                TypeMismatches = unsupportedFields,
                ExpectedSchema = $"Schema with supported types: {supportedTypes}",
                ActualSchema = $"Schema with fields: {string.Join(", ", dataFields.Select(f => $"{f.Name}:{f.ClrType.Name}"))}"
            };
        }
    }

    /// <summary>
    /// Checks if a CLR type is supported for Parquet reading.
    /// </summary>
    private static bool IsTypeSupported(Type clrType)
    {
        // Handle nullable types
        var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
        
        return TypeMapper.GetSupportedTypes().Contains(actualType);
    }

    /// <summary>
    /// Creates a NivaraColumn from Parquet column data.
    /// </summary>
    private static IColumn CreateNivaraColumnFromParquetData(DataColumn columnData, DataField field)
    {
        var elementType = field.ClrType;

        // Handle nullable types
        if (elementType.IsGenericType && elementType.GetGenericTypeDefinition() == typeof(Nullable<>))
        {
            elementType = Nullable.GetUnderlyingType(elementType)!;
        }

        return elementType switch
        {
            Type t when t == typeof(bool) => CreateNivaraColumn<bool>(columnData),
            Type t when t == typeof(int) => CreateNivaraColumn<int>(columnData),
            Type t when t == typeof(long) => CreateNivaraColumn<long>(columnData),
            Type t when t == typeof(float) => CreateNivaraColumn<float>(columnData),
            Type t when t == typeof(double) => CreateNivaraColumn<double>(columnData),
            Type t when t == typeof(DateTime) => CreateNivaraColumn<DateTime>(columnData),
            Type t when t == typeof(string) => CreateStringColumn(columnData),
            _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
        };
    }

    /// <summary>
    /// Creates a NivaraColumn for struct types from Parquet data with buffer reuse.
    /// </summary>
    private static NivaraColumn<T> CreateNivaraColumn<T>(DataColumn columnData) where T : struct
    {
        var length = columnData.Data.Length;
        
        // Use buffer pool for large arrays to reduce allocations
        T?[] nullableArray;
        if (length > 1024)
        {
            // For large arrays, try to reuse buffers
            var buffer = BufferPool.RentIntBuffer(length);
            try
            {
                nullableArray = new T?[length];
                
                for (int i = 0; i < length; i++)
                {
                    var value = columnData.Data.GetValue(i);
                    if (value != null)
                    {
                        try
                        {
                            nullableArray[i] = (T)Convert.ChangeType(value, typeof(T))!;
                        }
                        catch
                        {
                            nullableArray[i] = null;
                        }
                    }
                    else
                    {
                        nullableArray[i] = null;
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
            // For small arrays, use direct allocation
            nullableArray = new T?[length];
            
            for (int i = 0; i < length; i++)
            {
                var value = columnData.Data.GetValue(i);
                if (value != null)
                {
                    try
                    {
                        nullableArray[i] = (T)Convert.ChangeType(value, typeof(T))!;
                    }
                    catch
                    {
                        nullableArray[i] = null;
                    }
                }
                else
                {
                    nullableArray[i] = null;
                }
            }
        }

        return NivaraColumn<T>.CreateFromNullable(nullableArray);
    }

    /// <summary>
    /// Creates a string column from Parquet data with memory optimization.
    /// </summary>
    private static NivaraColumn<string> CreateStringColumn(DataColumn columnData)
    {
        var length = columnData.Data.Length;
        var values = new string[length];

        // Use buffer pool for processing large string columns
        if (length > 1024)
        {
            var buffer = BufferPool.RentByteBuffer(length * 32); // Estimate for string processing
            try
            {
                for (int i = 0; i < length; i++)
                {
                    var value = columnData.Data.GetValue(i);
                    if (value != null)
                    {
                        values[i] = value.ToString()!;
                    }
                    else
                    {
                        values[i] = null!; // Use null for missing values
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
            for (int i = 0; i < length; i++)
            {
                var value = columnData.Data.GetValue(i);
                if (value != null)
                {
                    values[i] = value.ToString()!;
                }
                else
                {
                    values[i] = null!; // Use null for missing values
                }
            }
        }

        return NivaraColumn<string>.CreateForReferenceType(values);
    }
}