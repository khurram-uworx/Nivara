namespace Nivara.IO;

/// <summary>
/// Configuration options for Parquet reading operations
/// </summary>
public class ParquetReadOptions
{
    /// <summary>
    /// Gets or sets whether to stream row groups instead of loading the entire file
    /// </summary>
    /// <remarks>
    /// When enabled, processes Parquet files in streaming mode to reduce memory usage
    /// for large files. Default is false.
    /// </remarks>
    public bool StreamRowGroups { get; set; } = false;

    /// <summary>
    /// Gets or sets the batch size for streaming operations
    /// </summary>
    /// <remarks>
    /// Specifies the number of rows to process in each batch when streaming.
    /// Only used when StreamRowGroups is true. Default is 1000.
    /// </remarks>
    public int BatchSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to validate the schema before processing
    /// </summary>
    /// <remarks>
    /// When enabled, validates the Parquet schema against expected types before reading.
    /// Can be disabled for performance-critical scenarios. Default is true.
    /// </remarks>
    public bool ValidateSchema { get; set; } = true;
}
