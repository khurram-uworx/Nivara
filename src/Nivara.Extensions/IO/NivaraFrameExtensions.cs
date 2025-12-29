using Apache.Arrow;

namespace Nivara.IO;

/// <summary>
/// Extension methods for NivaraFrame providing fluent API for Arrow and Parquet operations
/// </summary>
/// <remarks>
/// These extension methods provide idiomatic C# integration with Arrow and Parquet I/O operations,
/// following .NET naming conventions and supporting both synchronous and asynchronous variants.
/// </remarks>
public static class NivaraFrameExtensions
{
    #region Parquet Extensions

    /// <summary>
    /// Writes the NivaraFrame to a Parquet file
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <exception cref="ArgumentNullException">Thrown when frame or filePath is null</exception>
    /// <exception cref="ArgumentException">Thrown when filePath is empty or whitespace</exception>
    /// <exception cref="NivaraIOException">Thrown when file writing fails</exception>
    public static void ToParquet(this NivaraFrame frame, string filePath, ParquetWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(filePath);

        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

        ParquetWriter.WriteParquet(frame, filePath, options);
    }

    /// <summary>
    /// Writes the NivaraFrame to a Parquet file asynchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="filePath">The path where the Parquet file will be created</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <returns>A task representing the asynchronous write operation</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or filePath is null</exception>
    /// <exception cref="ArgumentException">Thrown when filePath is empty or whitespace</exception>
    /// <exception cref="NivaraIOException">Thrown when file writing fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static Task ToParquetAsync(this NivaraFrame frame, string filePath, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(filePath);

        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

        return ParquetWriter.WriteParquetAsync(frame, filePath, options, cancellationToken);
    }

    /// <summary>
    /// Writes the NivaraFrame to a Parquet stream
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <exception cref="ArgumentNullException">Thrown when frame or stream is null</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not writable</exception>
    /// <exception cref="NivaraIOException">Thrown when stream writing fails</exception>
    public static void ToParquetStream(this NivaraFrame frame, Stream stream, ParquetWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable", nameof(stream));

        ParquetWriter.WriteParquet(frame, stream, options);
    }

    /// <summary>
    /// Writes the NivaraFrame to a Parquet stream asynchronously
    /// </summary>
    /// <param name="frame">The NivaraFrame to write</param>
    /// <param name="stream">The stream to write the Parquet data to</param>
    /// <param name="options">Optional Parquet writing options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <returns>A task representing the asynchronous write operation</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or stream is null</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not writable</exception>
    /// <exception cref="NivaraIOException">Thrown when stream writing fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static Task ToParquetStreamAsync(this NivaraFrame frame, Stream stream, ParquetWriteOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable", nameof(stream));

        return ParquetWriter.WriteParquetAsync(frame, stream, options, cancellationToken);
    }

    /// <summary>
    /// Loads a NivaraFrame from a Parquet file
    /// </summary>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <returns>A NivaraFrame containing the Parquet data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="ArgumentException">Thrown when filePath is empty or whitespace</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    /// <exception cref="NivaraIOException">Thrown when file reading fails</exception>
    public static NivaraFrame LoadParquet(string filePath, ParquetReadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(filePath);

        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

        return ParquetReader.ReadParquet(filePath, options);
    }

    /// <summary>
    /// Loads a NivaraFrame from a Parquet file asynchronously
    /// </summary>
    /// <param name="filePath">The path to the Parquet file</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <returns>A task containing a NivaraFrame with the Parquet data</returns>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null</exception>
    /// <exception cref="ArgumentException">Thrown when filePath is empty or whitespace</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    /// <exception cref="NivaraIOException">Thrown when file reading fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static Task<NivaraFrame> LoadParquetAsync(string filePath, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filePath);

        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

        return ParquetReader.ReadParquetAsync(filePath, options, cancellationToken);
    }

    /// <summary>
    /// Loads a NivaraFrame from a Parquet stream
    /// </summary>
    /// <param name="stream">The stream containing Parquet data</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <returns>A NivaraFrame containing the Parquet data</returns>
    /// <exception cref="ArgumentNullException">Thrown when stream is null</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not readable</exception>
    /// <exception cref="NivaraIOException">Thrown when stream reading fails</exception>
    public static NivaraFrame LoadParquetFromStream(Stream stream, ParquetReadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable", nameof(stream));

        return ParquetReader.ReadParquet(stream, options);
    }

    /// <summary>
    /// Loads a NivaraFrame from a Parquet stream asynchronously
    /// </summary>
    /// <param name="stream">The stream containing Parquet data</param>
    /// <param name="options">Optional Parquet reading options</param>
    /// <param name="cancellationToken">Token to cancel the operation</param>
    /// <returns>A task containing a NivaraFrame with the Parquet data</returns>
    /// <exception cref="ArgumentNullException">Thrown when stream is null</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not readable</exception>
    /// <exception cref="NivaraIOException">Thrown when stream reading fails</exception>
    /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled</exception>
    public static Task<NivaraFrame> LoadParquetFromStreamAsync(Stream stream, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable", nameof(stream));

        return ParquetReader.ReadParquetAsync(stream, options, cancellationToken);
    }

    #endregion

    #region Arrow Extensions

    /// <summary>
    /// Converts the NivaraFrame to an Apache Arrow Table
    /// </summary>
    /// <param name="frame">The NivaraFrame to convert</param>
    /// <param name="options">Optional Arrow conversion options</param>
    /// <returns>An Apache Arrow Table</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when a column type is not supported</exception>
    public static Table ToArrowTable(this NivaraFrame frame, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frame);

        return ArrowInterop.ToArrowTable(frame, options);
    }

    /// <summary>
    /// Creates a NivaraFrame from an Apache Arrow Table
    /// </summary>
    /// <param name="arrowTable">The Apache Arrow Table to convert</param>
    /// <param name="options">Optional Arrow conversion options</param>
    /// <returns>A NivaraFrame</returns>
    /// <exception cref="ArgumentNullException">Thrown when arrowTable is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when an Arrow type is not supported</exception>
    public static NivaraFrame FromArrowTable(this Table arrowTable, ArrowConversionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(arrowTable);

        return ArrowInterop.FromArrowTable(arrowTable, options);
    }

    #endregion
}