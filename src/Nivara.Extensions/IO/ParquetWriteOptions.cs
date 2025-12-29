namespace Nivara.Extensions.IO;

/// <summary>
/// Configuration options for Parquet writing operations
/// </summary>
public class ParquetWriteOptions
{
    /// <summary>
    /// Gets or sets the row group size for Parquet files
    /// </summary>
    /// <remarks>
    /// Specifies the number of rows to include in each row group.
    /// Larger row groups provide better compression but use more memory.
    /// Default is 10000.
    /// </remarks>
    public int RowGroupSize { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the compression algorithm to use
    /// </summary>
    /// <remarks>
    /// Supported compression algorithms: "none", "snappy", "gzip", "lz4", "brotli", "zstd".
    /// Default is "snappy" for good balance of compression ratio and speed.
    /// </remarks>
    public string Compression { get; set; } = "snappy";

    /// <summary>
    /// Gets or sets whether to validate the schema before writing
    /// </summary>
    /// <remarks>
    /// When enabled, validates the frame schema against Parquet requirements before writing.
    /// Can be disabled for performance-critical scenarios. Default is true.
    /// </remarks>
    public bool ValidateSchema { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to write metadata to the Parquet file
    /// </summary>
    /// <remarks>
    /// When enabled, includes additional metadata in the Parquet file such as
    /// creation time, software version, and custom properties.
    /// Default is true.
    /// </remarks>
    public bool WriteMetadata { get; set; } = true;
}