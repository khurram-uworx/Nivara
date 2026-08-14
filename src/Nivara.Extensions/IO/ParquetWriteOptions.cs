namespace Nivara.IO;

/// <summary>
/// Configuration options for Parquet writing operations
/// </summary>
public sealed class ParquetWriteOptions
{
    /// <summary>
    /// Gets the default Parquet write options
    /// </summary>
    public static ParquetWriteOptions Default { get; } = new ParquetWriteOptions();

    /// <summary>
    /// Gets the row group size for Parquet files
    /// </summary>
    /// <remarks>
    /// Specifies the number of rows to include in each row group.
    /// Larger row groups provide better compression but use more memory.
    /// Default is 10000.
    /// </remarks>
    public int RowGroupSize { get; }

    /// <summary>
    /// Gets the compression algorithm to use
    /// </summary>
    /// <remarks>
    /// Supported compression algorithms: "none", "snappy", "gzip", "lz4", "brotli", "zstd".
    /// Default is "snappy" for good balance of compression ratio and speed.
    /// </remarks>
    public string Compression { get; }

    /// <summary>
    /// Gets whether to validate the schema before writing
    /// </summary>
    /// <remarks>
    /// When enabled, validates the frame schema against Parquet requirements before writing.
    /// Can be disabled for performance-critical scenarios. Default is true.
    /// </remarks>
    public bool ValidateSchema { get; }

    /// <summary>
    /// Gets whether to write metadata to the Parquet file
    /// </summary>
    /// <remarks>
    /// When enabled, includes additional metadata in the Parquet file such as
    /// creation time, software version, and custom properties.
    /// Default is true.
    /// </remarks>
    public bool WriteMetadata { get; }

    /// <summary>
    /// Initializes a new instance of ParquetWriteOptions with default values
    /// </summary>
    public ParquetWriteOptions()
    {
        RowGroupSize = 10000;
        Compression = "snappy";
        ValidateSchema = true;
        WriteMetadata = true;
    }

    private ParquetWriteOptions(int rowGroupSize, string compression, bool validateSchema, bool writeMetadata)
    {
        RowGroupSize = rowGroupSize;
        Compression = compression;
        ValidateSchema = validateSchema;
        WriteMetadata = writeMetadata;
    }

    /// <summary>
    /// Returns a copy of these options with the specified values changed
    /// </summary>
    /// <param name="rowGroupSize">New row group size, or null to keep the current value</param>
    /// <param name="compression">New compression algorithm, or null to keep the current value</param>
    /// <param name="validateSchema">New schema validation mode, or null to keep the current value</param>
    /// <param name="writeMetadata">New metadata writing mode, or null to keep the current value</param>
    /// <returns>A new ParquetWriteOptions instance</returns>
    public ParquetWriteOptions With(int? rowGroupSize = null, string? compression = null, bool? validateSchema = null, bool? writeMetadata = null)
    {
        return new ParquetWriteOptions(
            rowGroupSize ?? RowGroupSize,
            compression ?? Compression,
            validateSchema ?? ValidateSchema,
            writeMetadata ?? WriteMetadata);
    }
}
