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
///
/// <see cref="Utf8JsonReader"/> is a ref struct, so the lexer state (buffer, consumed
/// offset, absolute base offset, <see cref="JsonReaderState"/>, container depth) lives
/// here as persistent fields and continues across <see cref="LocateRange(int, int, bool)"/>
/// calls. Sequential reads never seek the lexer's stream — backward access resets to the
/// top of the document and re-walks. A <see cref="Utf8JsonReader"/> reconstructed with a
/// captured <see cref="JsonReaderState"/> resumes at the parser context that state records,
/// so the refill and inter-call continuation both re-lex correctly mid-array.
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
    private readonly FileStream readStream;
    private readonly JsonReaderOptions readerOptions;
    private readonly long dataStart;
    private readonly long dataEnd;

    private byte[] buffer;
    private int bytesInBuffer;
    private long baseOffset;
    private bool reachedEnd;
    private bool disposed;

    // Persistent lexer state, continued across LocateRange calls without seeking.
    private JsonReaderState scanState;
    private int scanConsumed;
    private int scanDepth;
    private bool scanIsArrayOpen;
    private int scanRecordIndex;
    private long scanRecordStart = -1;
    private bool scanAtEnd;
    private bool scanSawAnyToken;

    /// <summary>
    /// Initializes a new reader over the JSON file, positioned at the first JSON byte
    /// (any UTF-8 BOM is skipped).
    /// </summary>
    public JsonRecordStreamReader(string filePath, JsonReaderOptions readerOptions)
    {
        this.readerOptions = readerOptions;
        stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true);
        readStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true);
        dataEnd = stream.Length;

        int bomLength = DetectBomLength();
        dataStart = bomLength;
        stream.Position = bomLength;

        buffer = ArrayPool<byte>.Shared.Rent(InitialBufferSize);
        baseOffset = dataStart;
        scanState = new JsonReaderState(readerOptions);
    }

    /// <summary>
    /// Gets whether any JSON token has been read by the walker since construction.
    /// False means the file contains no JSON tokens at all (empty or whitespace-only).
    /// </summary>
    public bool SawAnyToken => scanSawAnyToken;

    /// <summary>
    /// Gets the index of the next record that will be scanned. After a walk that stopped
    /// at a record boundary this is the boundary record's index (the first record of the
    /// next chunk); after reaching the end of the document it is the total record count.
    /// </summary>
    public int NextRecordIndex => scanRecordStart >= 0 ? scanRecordIndex - 1 : scanRecordIndex;

    /// <summary>
    /// Locates the byte range covering records
    /// <c>[startRecord, startRecord + count)</c> by continuing the token walk from the
    /// current scan position. Sequential calls continue without re-reading; backward
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
        => LocateRangeCore(startRecord, count, isArray, useAsync: false, CancellationToken.None);

    /// <summary>
    /// Asynchronous variant of <see cref="LocateRange"/>: the token walk itself stays
    /// synchronous (it is CPU-bound lexing over an already-buffered span), but each buffer
    /// refill is awaited via <see cref="FileStream.ReadAsync"/> so the lexer never blocks a
    /// thread while waiting on disk IO.
    /// </summary>
    public async ValueTask<JsonRecordRange> LocateRangeAsync(int startRecord, int count, bool isArray, CancellationToken cancellationToken = default)
    {
        if (startRecord < 0)
            throw new ArgumentOutOfRangeException(nameof(startRecord));
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count));

        cancellationToken.ThrowIfCancellationRequested();

        if (scanAtEnd && startRecord >= NextRecordIndex)
            return new JsonRecordRange(0, 0, 0, Eof: true);

        if (scanAtEnd || startRecord < scanRecordIndex - 1)
            ResetToStart();

        long rangeStart = -1;
        long rangeEnd = -1;
        int rows = 0;
        bool scanAtBoundary = false;
        long rangeStartAbs = -1;

        while (true)
        {
            var outcome = ScanBuffer(startRecord, count, isArray, ref rangeStart, ref rangeEnd, ref rows, ref scanAtBoundary, ref rangeStartAbs);
            if (outcome != ScanOutcome.NeedRefill)
            {
                var result = CompleteScan(outcome, startRecord, dataEnd, ref rangeStart, ref rangeEnd, ref rows, rangeStartAbs);
                return result;
            }

            if (!await RefillAsync(cancellationToken).ConfigureAwait(false))
            {
                if (scanConsumed > 0)
                    Compact();
            }
        }
    }

    /// <summary>
    /// Shared driver for <see cref="LocateRange"/> and <see cref="LocateRangeAsync"/>.
    /// The token walk is executed by the synchronous <see cref="ScanBuffer"/>; only the
    /// buffer refill differs between the two variants.
    /// </summary>
    private JsonRecordRange LocateRangeCore(int startRecord, int count, bool isArray, bool useAsync, CancellationToken cancellationToken)
    {
        if (startRecord < 0)
            throw new ArgumentOutOfRangeException(nameof(startRecord));
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count));

        if (scanAtEnd && startRecord >= NextRecordIndex)
            return new JsonRecordRange(0, 0, 0, Eof: true);

        if (scanAtEnd || startRecord < scanRecordIndex - 1)
            ResetToStart();

        long rangeStart = -1;
        long rangeEnd = -1;
        int rows = 0;
        bool scanAtBoundary = false;
        long rangeStartAbs = -1;

        while (true)
        {
            var outcome = ScanBuffer(startRecord, count, isArray, ref rangeStart, ref rangeEnd, ref rows, ref scanAtBoundary, ref rangeStartAbs);
            if (outcome != ScanOutcome.NeedRefill)
            {
                var result = CompleteScan(outcome, startRecord, dataEnd, ref rangeStart, ref rangeEnd, ref rows, rangeStartAbs);
                return result;
            }

            bool refilled = useAsync ? RefillAsync(cancellationToken).GetAwaiter().GetResult() : Refill();
            if (!refilled)
            {
                if (scanConsumed > 0)
                    Compact();
            }
        }
    }

    /// <summary>
    /// Converts a <see cref="ScanOutcome"/> into the final <see cref="JsonRecordRange"/>,
    /// finalizing the last open record when the end of the document was reached.
    /// </summary>
    private JsonRecordRange CompleteScan(ScanOutcome outcome, int startRecord, long endOffset, ref long rangeStart, ref long rangeEnd, ref int rows, long rangeStartAbs)
    {
        switch (outcome)
        {
            case ScanOutcome.CompletedEmpty:
                return new JsonRecordRange(0, 0, 0, Eof: true);
            case ScanOutcome.CompletedSingle:
                return new JsonRecordRange(rangeStartAbs, endOffset, 1, Eof: true);
            case ScanOutcome.CompletedAtBoundary:
                return new JsonRecordRange(rangeStart, rangeEnd, rows, Eof: false);
            default:
                var final = FinalizeRange(startRecord, endOffset, ref rangeStart, ref rangeEnd, ref rows);
                return final.Rows == 0 ? new JsonRecordRange(0, 0, 0, Eof: true) : final;
        }
    }

    /// <summary>
    /// Outcome of one synchronous pass over the current lexer buffer.
    /// </summary>
    private enum ScanOutcome
    {
        NeedRefill,
        CompletedEmpty,
        CompletedSingle,
        CompletedAtBoundary,
        CompletedAtEnd
    }

    /// <summary>
    /// Lexes the current buffer with a fresh <see cref="Utf8JsonReader"/> resumed from the
    /// captured <see cref="scanState"/>. The walk advances the persistent scan fields and
    /// returns when the requested range is complete, the end of the document is reached, or
    /// the buffer needs refilling. Synchronous only — <see cref="Utf8JsonReader"/> is a ref
    /// struct and cannot cross an await boundary.
    /// </summary>
    private ScanOutcome ScanBuffer(int startRecord, int count, bool isArray, ref long rangeStart, ref long rangeEnd, ref int rows, ref bool scanAtBoundary, ref long rangeStartAbs)
    {
        if (scanConsumed > 0)
            Compact();

        if (bytesInBuffer == 0 && reachedEnd)
        {
            // Nothing more to lex: empty or whitespace-only input, or the document
            // was fully consumed. The empty-buffer EOF case cannot yield more tokens.
            scanAtEnd = true;
            return ScanOutcome.CompletedEmpty;
        }

        var reader = new Utf8JsonReader(buffer.AsSpan(0, bytesInBuffer), reachedEnd, scanState);

        while (reader.Read())
        {
            scanConsumed = (int)reader.BytesConsumed;
            scanState = reader.CurrentState;
            scanSawAnyToken = true;

            var token = reader.TokenType;
            long absStart = baseOffset + reader.TokenStartIndex;

            if (!scanIsArrayOpen)
            {
                if (isArray)
                {
                    if (token == JsonTokenType.StartArray)
                    {
                        scanIsArrayOpen = true;
                        scanDepth = 1;
                        continue;
                    }
                    throw new JsonException("Expected a JSON array at the start of the document");
                }

                // Non-array mode: the first value token is the single record.
                if (startRecord > 0)
                {
                    scanAtEnd = true;
                    return ScanOutcome.CompletedEmpty;
                }

                scanRecordIndex = 1;
                scanRecordStart = -1;
                scanAtEnd = true;
                rangeStartAbs = absStart;
                return ScanOutcome.CompletedSingle;
            }

            bool isRecordStart = false;
            switch (token)
            {
                case JsonTokenType.StartObject:
                    isRecordStart = scanDepth == 1;
                    scanDepth++;
                    break;
                case JsonTokenType.StartArray:
                    isRecordStart = scanDepth == 1;
                    scanDepth++;
                    break;
                case JsonTokenType.EndObject:
                    scanDepth--;
                    break;
                case JsonTokenType.EndArray:
                    scanDepth--;
                    if (scanDepth == 0)
                    {
                        // End of the top-level array.
                        FinalizeRange(startRecord, absStart, ref rangeStart, ref rangeEnd, ref rows);
                        return ScanOutcome.CompletedAtEnd;
                    }
                    break;
                case JsonTokenType.PropertyName:
                    break;
                default:
                    isRecordStart = scanDepth == 1;
                    break;
            }

            if (isRecordStart)
            {
                if (scanRecordStart >= 0)
                {
                    int prevIndex = scanRecordIndex - 1;
                    if (prevIndex >= startRecord)
                    {
                        if (rangeStart < 0)
                            rangeStart = scanRecordStart;
                        rangeEnd = absStart;
                        rows++;
                    }
                }
                scanRecordStart = absStart;
                scanRecordIndex++;
                if (rows == count)
                {
                    // The record that just started is the first record of the next
                    // chunk; its start closes this range. Lex it to completion so the
                    // walker stops at a resumable position (array-element level), then
                    // return with that record left open for the next walk.
                    scanAtBoundary = true;
                }
            }

            if (scanAtBoundary && scanDepth == 1)
                return ScanOutcome.CompletedAtBoundary;
        }

        if (reachedEnd)
            return ScanOutcome.CompletedAtEnd;

        // The reader could not complete the next token: capture the resume point and
        // signal the caller to refill, then reconstruct the reader with the captured state.
        scanConsumed = (int)reader.BytesConsumed;
        scanState = reader.CurrentState;
        return ScanOutcome.NeedRefill;
    }

    /// <summary>
    /// Closes the last open record and records the end of the document. Returns true when
    /// at least one record was closed into the requested range.
    /// </summary>
    private JsonRecordRange FinalizeRange(int startRecord, long endOffset, ref long rangeStart, ref long rangeEnd, ref int rows)
    {
        if (scanRecordStart >= 0)
        {
            int prevIndex = scanRecordIndex - 1;
            if (prevIndex >= startRecord)
            {
                if (rangeStart < 0)
                    rangeStart = scanRecordStart;
                rangeEnd = endOffset;
                rows++;
            }
        }
        scanRecordStart = -1;
        scanAtEnd = true;
        return new JsonRecordRange(rangeStart, rangeEnd, rows, Eof: true);
    }

    /// <summary>
    /// Reads the byte range <c>[start, end)</c> into <paramref name="destination"/>,
    /// starting at offset 1, leaving the first byte for the caller's array wrapper.
    /// Uses a dedicated stream so the lexer's buffered position is never disturbed.
    /// </summary>
    /// <returns>The number of bytes read.</returns>
    public int ReadRange(long start, long end, byte[] destination)
    {
        int length = checked((int)(end - start));
        if (destination.Length < length + 2)
            throw new ArgumentOutOfRangeException(nameof(destination), "Destination buffer must hold the range plus 2 wrapper bytes");

        readStream.Position = start;
        int total = 0;
        while (total < length)
        {
            int read = readStream.Read(destination, 1 + total, length - total);
            if (read == 0)
                break;
            total += read;
        }
        return total;
    }

    /// <summary>
    /// Asynchronous variant of <see cref="ReadRange"/>.
    /// </summary>
    public async ValueTask<int> ReadRangeAsync(long start, long end, byte[] destination, CancellationToken cancellationToken = default)
    {
        int length = checked((int)(end - start));
        if (destination.Length < length + 2)
            throw new ArgumentOutOfRangeException(nameof(destination), "Destination buffer must hold the range plus 2 wrapper bytes");

        readStream.Position = start;
        int total = 0;
        while (total < length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int read = await readStream.ReadAsync(destination.AsMemory(1 + total, length - total), cancellationToken).ConfigureAwait(false);
            if (read == 0)
                break;
            total += read;
        }
        return total;
    }

    /// <summary>
    /// Moves any unread bytes to the front of the buffer and advances the absolute
    /// <see cref="baseOffset"/> by the number of consumed bytes.
    /// </summary>
    private void Compact()
    {
        if (bytesInBuffer > scanConsumed)
            Buffer.BlockCopy(buffer, scanConsumed, buffer, 0, bytesInBuffer - scanConsumed);
        bytesInBuffer -= scanConsumed;
        baseOffset += scanConsumed;
        scanConsumed = 0;
    }

    /// <summary>
    /// Compacts consumed bytes away and reads more data from the lexer's stream.
    /// </summary>
    /// <returns>False when the end of the stream was reached.</returns>
    private bool Refill()
    {
        Compact();

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
        {
            reachedEnd = true;
            return false;
        }
        bytesInBuffer += read;
        return true;
    }

    /// <summary>
    /// Asynchronous variant of <see cref="Refill"/>.
    /// </summary>
    private async ValueTask<bool> RefillAsync(CancellationToken cancellationToken)
    {
        Compact();

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

        int read = await stream.ReadAsync(buffer.AsMemory(bytesInBuffer, toRead), cancellationToken).ConfigureAwait(false);
        if (read == 0)
        {
            reachedEnd = true;
            return false;
        }
        bytesInBuffer += read;
        return true;
    }

    /// <summary>
    /// Resets the walker to the top of the document (after any BOM) and clears all scan
    /// state, forcing a full re-walk on the next <see cref="LocateRange(int, int, bool)"/>.
    /// </summary>
    private void ResetToStart()
    {
        stream.Position = dataStart;
        baseOffset = dataStart;
        bytesInBuffer = 0;
        reachedEnd = false;
        scanState = new JsonReaderState(readerOptions);
        scanConsumed = 0;
        scanDepth = 0;
        scanIsArrayOpen = false;
        scanRecordIndex = 0;
        scanRecordStart = -1;
        scanAtEnd = false;
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
            readStream.Dispose();
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
