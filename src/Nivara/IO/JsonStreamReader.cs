using System.Buffers;
using System.Text.Json;

namespace Nivara.IO;

/// <summary>
/// Byte-range of a run of JSON records located by <see cref="JsonRecordStreamReader"/>.
/// </summary>
internal readonly record struct JsonRecordRange(long Start, long End, int Rows, bool Eof);

/// <summary>
/// Streaming JSON tokenizer that walks UTF-8 records inside a file and locates their
/// byte ranges, modeled on the partial-read / growable-buffer pattern documented at
/// https://learn.microsoft.com/dotnet/standard/serialization/system-text-json/use-utf8jsonreader.
/// A <see cref="Utf8JsonReader"/> is a ref struct, so the refill state (buffer, consumed
/// offset, absolute base offset, reader state) lives here across calls, allowing
/// <see cref="LocateRange(int, int, bool)"/> to resume sequentially without re-reading.
/// </summary>
internal sealed class JsonRecordStreamReader : IDisposable
{
    private const int InitialBufferSize = 64 * 1024;
    private const int MaxBufferSize = 256 * 1024 * 1024;

    /// <summary>
    /// Maps the reader-related settings from a <see cref="JsonSerializerOptions"/>
    /// instance onto <see cref="JsonReaderOptions"/>.
    /// </summary>
    public static JsonReaderOptions ToJsonReaderOptions(JsonSerializerOptions options)
    {
        return new JsonReaderOptions
        {
            CommentHandling = options.ReadCommentHandling,
            AllowTrailingCommas = options.AllowTrailingCommas,
            MaxDepth = options.MaxDepth
        };
    }

    /// <summary>
    /// Maps the document-related settings from a <see cref="JsonSerializerOptions"/>
    /// instance onto <see cref="JsonDocumentOptions"/>.
    /// </summary>
    public static JsonDocumentOptions ToJsonDocumentOptions(JsonSerializerOptions options)
    {
        return new JsonDocumentOptions
        {
            CommentHandling = options.ReadCommentHandling,
            AllowTrailingCommas = options.AllowTrailingCommas,
            MaxDepth = options.MaxDepth
        };
    }

    private readonly FileStream stream;
    private readonly JsonReaderOptions readerOptions;
    private readonly long dataStart;
    private readonly long dataEnd;

    private byte[] buffer;
    private int bytesInBuffer;
    private long baseOffset;
    private bool reachedEnd;
    private bool disposed;

    private int nextRecordIndex;
    private long nextRecordStart = -1;
    private bool atEnd;

    /// <summary>
    /// Initializes a new reader over the JSON file, positioned at the first JSON byte
    /// (any UTF-8 BOM is skipped).
    /// </summary>
    public JsonRecordStreamReader(string filePath, JsonReaderOptions readerOptions)
    {
        this.readerOptions = readerOptions;
        stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: false);
        dataEnd = stream.Length;

        int bomLength = DetectBomLength();
        dataStart = bomLength;
        stream.Position = bomLength;

        buffer = ArrayPool<byte>.Shared.Rent(InitialBufferSize);
    }

    /// <summary>
    /// Gets whether any token has been read by the most recent
    /// <see cref="LocateRange(int, int, bool)"/> call. False means the file contains no
    /// JSON tokens at all (empty or whitespace-only).
    /// </summary>
    public bool SawAnyToken { get; private set; }

    /// <summary>
    /// Gets the index of the next record that will be scanned, or the total record count
    /// when the previous walk reached the end of the document.
    /// </summary>
    public int NextRecordIndex { get; private set; }

    /// <summary>
    /// Locates the byte range covering records
    /// <c>[startRecord, startRecord + count)</c> by walking tokens from the current scan
    /// position. Sequential calls continue from where the previous walk stopped; backward
    /// access (or a request before the current position) resets to the top of the file
    /// and re-walks.
    /// </summary>
    /// <param name="startRecord">Zero-based index of the first record in the range.</param>
    /// <param name="count">Maximum number of records in the range.</param>
    /// <param name="isArray">True when the document is an array of records; false when it
    /// is a single top-level value.</param>
    /// <returns>The byte range (inclusive start, exclusive end), number of records found,
    /// and whether the walk reached the end of the document. Rows is 0 when fewer than
    /// <paramref name="startRecord"/> records exist.</returns>
    public JsonRecordRange LocateRange(int startRecord, int count, bool isArray)
    {
        if (startRecord < 0)
            throw new ArgumentOutOfRangeException(nameof(startRecord));
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count));

        SawAnyToken = false;

        if (atEnd && startRecord >= nextRecordIndex)
            return new JsonRecordRange(0, 0, 0, Eof: true);

        if (atEnd || startRecord < nextRecordIndex || nextRecordStart < 0)
            ResetToStart();

        // Begin scanning at the next record boundary (or at the top of the document).
        int recordBase = nextRecordStart >= 0 ? nextRecordIndex : 0;
        bool isArrayOpen = nextRecordStart >= 0;
        stream.Position = nextRecordStart >= 0 ? nextRecordStart : dataStart;
        baseOffset = nextRecordStart >= 0 ? nextRecordStart : dataStart;
        bytesInBuffer = 0;
        reachedEnd = false;

        int depth = isArrayOpen ? 1 : 0;
        int recordCount = 0;
        long pendingStart = -1;
        long rangeStart = -1;
        long rangeEnd = -1;
        int rows = 0;
        bool eof = false;
        bool exitedAtRecordStart = false;
        long boundaryStart = -1;

        int consumed = 0;

        while (true)
        {
            if (!reachedEnd)
            {
                if (bytesInBuffer > consumed)
                    Buffer.BlockCopy(buffer, consumed, buffer, 0, bytesInBuffer - consumed);
                bytesInBuffer -= consumed;
                baseOffset += consumed;
                consumed = 0;

                int toRead = buffer.Length - bytesInBuffer;
                if (toRead == 0)
                {
                    int newSize = buffer.Length * 2;
                    if (newSize > MaxBufferSize)
                        throw new JsonException($"JSON value exceeds the maximum supported size of {MaxBufferSize} bytes");
                    var newBuffer = ArrayPool<byte>.Shared.Rent(newSize);
                    Buffer.BlockCopy(buffer, 0, newBuffer, 0, bytesInBuffer);
                    ArrayPool<byte>.Shared.Return(buffer);
                    buffer = newBuffer;
                    toRead = buffer.Length - bytesInBuffer;
                }

                int read = stream.Read(buffer, bytesInBuffer, toRead);
                if (read == 0)
                    reachedEnd = true;
                else
                    bytesInBuffer += read;
            }

            var reader = new Utf8JsonReader(buffer.AsSpan(0, bytesInBuffer), reachedEnd, new JsonReaderState(readerOptions));

            while (reader.Read())
            {
                consumed = (int)reader.BytesConsumed;
                SawAnyToken = true;
                var token = reader.TokenType;
                long absStart = baseOffset + reader.TokenStartIndex;

                if (!isArrayOpen)
                {
                    if (isArray)
                    {
                        if (token == JsonTokenType.StartArray)
                        {
                            isArrayOpen = true;
                            depth = 1;
                            continue;
                        }
                        throw new JsonException("Expected a JSON array at the start of the document");
                    }

                    // Non-array mode: the first value token is the single record.
                    if (startRecord > 0)
                    {
                        eof = true;
                        goto done;
                    }
                    rangeStart = absStart;
                    rangeEnd = dataEnd;
                    rows = 1;
                    eof = true;
                    goto done;
                }

                bool isRecordStart;
                switch (token)
                {
                    case JsonTokenType.StartObject:
                        isRecordStart = depth == 1;
                        depth++;
                        break;
                    case JsonTokenType.StartArray:
                        isRecordStart = depth == 1;
                        depth++;
                        break;
                    case JsonTokenType.EndObject:
                        depth--;
                        isRecordStart = false;
                        break;
                    case JsonTokenType.EndArray:
                        depth--;
                        isRecordStart = false;
                        if (depth == 0)
                        {
                            // End of the top-level array (array mode only).
                            if (recordCount > 0)
                            {
                                int prevIndex = recordBase + recordCount - 1;
                                if (prevIndex >= startRecord)
                                {
                                    if (rangeStart < 0)
                                        rangeStart = pendingStart;
                                    rangeEnd = absStart;
                                    rows++;
                                }
                            }
                            eof = true;
                            goto done;
                        }
                        break;
                    case JsonTokenType.PropertyName:
                        isRecordStart = false;
                        break;
                    default:
                        isRecordStart = depth == 1;
                        break;
                }

                if (isRecordStart)
                {
                    if (recordCount > 0)
                    {
                        int prevIndex = recordBase + recordCount - 1;
                        if (prevIndex >= startRecord)
                        {
                            if (rangeStart < 0)
                                rangeStart = pendingStart;
                            rangeEnd = absStart;
                            rows++;
                            if (rows == count)
                            {
                                exitedAtRecordStart = true;
                                boundaryStart = absStart;
                                goto done;
                            }
                        }
                    }
                    pendingStart = absStart;
                    recordCount++;
                }
            }

            if (reachedEnd)
            {
                if (recordCount > 0)
                {
                    int prevIndex = recordBase + recordCount - 1;
                    if (prevIndex >= startRecord)
                    {
                        if (rangeStart < 0)
                            rangeStart = pendingStart;
                        rangeEnd = dataEnd;
                        rows++;
                    }
                }
                eof = true;
                break;
            }
        }

    done:
        if (exitedAtRecordStart)
        {
            nextRecordIndex = recordBase + recordCount;
            nextRecordStart = boundaryStart;
            atEnd = false;
        }
        else
        {
            nextRecordIndex = recordBase + recordCount;
            nextRecordStart = -1;
            atEnd = true;
        }

        return new JsonRecordRange(rangeStart, rangeEnd, rows, eof);
    }

    /// <summary>
    /// Reads the byte range <c>[start, end)</c> into <paramref name="destination"/>,
    /// starting at offset 2, leaving the first two bytes for the caller's array wrapper.
    /// </summary>
    /// <returns>The number of bytes read.</returns>
    public int ReadRange(long start, long end, byte[] destination)
    {
        int length = checked((int)(end - start));
        if (destination.Length < length + 2)
            throw new ArgumentOutOfRangeException(nameof(destination), "Destination buffer must hold the range plus 2 wrapper bytes");

        stream.Position = start;
        int total = 0;
        while (total < length)
        {
            int read = stream.Read(destination, 2 + total, length - total);
            if (read == 0)
                break;
            total += read;
        }
        return total;
    }

    /// <summary>
    /// Resets the walker to the top of the document (after any BOM).
    /// </summary>
    private void ResetToStart()
    {
        nextRecordIndex = 0;
        nextRecordStart = -1;
        atEnd = false;
    }

    private int DetectBomLength()
    {
        if (stream.Length >= 3)
        {
            int b0 = stream.ReadByte();
            int b1 = stream.ReadByte();
            int b2 = stream.ReadByte();
            if (b0 == 0xEF && b1 == 0xBB && b2 == 0xBF)
                return 3;
            stream.Position = 0;
        }
        return 0;
    }

    public void Dispose()
    {
        if (!disposed)
        {
            disposed = true;
            stream.Dispose();
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
