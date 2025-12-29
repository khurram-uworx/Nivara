using System.Buffers;

namespace Nivara.IO;

/// <summary>
/// Manages streaming buffers for processing large datasets with bounded memory usage.
/// </summary>
/// <remarks>
/// This class provides memory-efficient streaming capabilities for Arrow/Parquet I/O operations,
/// ensuring that memory usage remains bounded regardless of dataset size.
/// </remarks>
internal sealed class StreamingBufferManager : IDisposable
{
    /// <summary>
    /// Default chunk size for streaming operations (1MB)
    /// </summary>
    private const int DefaultChunkSize = 1024 * 1024;

    /// <summary>
    /// Maximum memory budget for streaming operations (256MB)
    /// </summary>
    private const long DefaultMemoryBudget = 256L * 1024 * 1024;

    /// <summary>
    /// Current memory usage in bytes
    /// </summary>
    private long currentMemoryUsage = 0;

    /// <summary>
    /// Memory budget for streaming operations
    /// </summary>
    private readonly long memoryBudget;

    /// <summary>
    /// Chunk size for streaming operations
    /// </summary>
    private readonly int chunkSize;

    /// <summary>
    /// List of rented buffers that need to be returned
    /// </summary>
    private readonly List<IMemoryOwner<byte>> rentedBuffers = new();

    /// <summary>
    /// Lock for thread-safe operations
    /// </summary>
    private readonly object lockObject = new();

    /// <summary>
    /// Whether the manager has been disposed
    /// </summary>
    private bool disposed = false;

    /// <summary>
    /// Initializes a new instance of the StreamingBufferManager.
    /// </summary>
    /// <param name="memoryBudget">The maximum memory budget in bytes.</param>
    /// <param name="chunkSize">The chunk size for streaming operations.</param>
    public StreamingBufferManager(long memoryBudget = DefaultMemoryBudget, int chunkSize = DefaultChunkSize)
    {
        this.memoryBudget = memoryBudget;
        this.chunkSize = chunkSize;
    }

    /// <summary>
    /// Gets the current memory usage in bytes.
    /// </summary>
    public long CurrentMemoryUsage => Interlocked.Read(ref currentMemoryUsage);

    /// <summary>
    /// Gets the memory budget in bytes.
    /// </summary>
    public long MemoryBudget => memoryBudget;

    /// <summary>
    /// Gets the chunk size for streaming operations.
    /// </summary>
    public int ChunkSize => chunkSize;

    /// <summary>
    /// Gets whether the memory budget has been exceeded.
    /// </summary>
    public bool IsMemoryBudgetExceeded => CurrentMemoryUsage > memoryBudget;

    /// <summary>
    /// Rents a memory buffer for streaming operations.
    /// </summary>
    /// <param name="size">The size of the buffer to rent.</param>
    /// <returns>A memory owner for the rented buffer.</returns>
    /// <exception cref="OutOfMemoryException">Thrown when the memory budget would be exceeded.</exception>
    public IMemoryOwner<byte> RentBuffer(int size = DefaultChunkSize)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(StreamingBufferManager));

        lock (lockObject)
        {
            var newUsage = Interlocked.Read(ref currentMemoryUsage) + size;
            if (newUsage > memoryBudget)
            {
                throw new OutOfMemoryException($"Memory budget exceeded. Current: {CurrentMemoryUsage:N0} bytes, " +
                                             $"Requested: {size:N0} bytes, Budget: {memoryBudget:N0} bytes");
            }

            var buffer = MemoryPool<byte>.Shared.Rent(size);
            rentedBuffers.Add(buffer);
            Interlocked.Add(ref currentMemoryUsage, size);

            return buffer;
        }
    }

    /// <summary>
    /// Returns a memory buffer and updates memory usage tracking.
    /// </summary>
    /// <param name="buffer">The buffer to return.</param>
    public void ReturnBuffer(IMemoryOwner<byte> buffer)
    {
        if (buffer == null || disposed)
            return;

        lock (lockObject)
        {
            if (rentedBuffers.Remove(buffer))
            {
                var bufferSize = buffer.Memory.Length;
                Interlocked.Add(ref currentMemoryUsage, -bufferSize);
                buffer.Dispose();
            }
        }
    }

    /// <summary>
    /// Creates a streaming reader for processing data in chunks.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>An async enumerable of memory chunks.</returns>
    public async IAsyncEnumerable<ReadOnlyMemory<byte>> CreateStreamingReader(
        Stream stream,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(StreamingBufferManager));

        while (!cancellationToken.IsCancellationRequested)
        {
            var buffer = RentBuffer(chunkSize);
            try
            {
                var bytesRead = await stream.ReadAsync(buffer.Memory, cancellationToken);
                if (bytesRead == 0)
                    break;

                yield return buffer.Memory.Slice(0, bytesRead);
            }
            finally
            {
                ReturnBuffer(buffer);
            }
        }
    }

    /// <summary>
    /// Creates a streaming writer for writing data in chunks.
    /// </summary>
    /// <param name="stream">The stream to write to.</param>
    /// <returns>A streaming writer instance.</returns>
    public StreamingWriter CreateStreamingWriter(Stream stream)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(StreamingBufferManager));
        return new StreamingWriter(stream, this);
    }

    /// <summary>
    /// Forces garbage collection if memory usage is high.
    /// </summary>
    public void TryCollectGarbage()
    {
        if (CurrentMemoryUsage > memoryBudget * 0.8) // 80% threshold
        {
            GC.Collect(2, GCCollectionMode.Optimized);
            GC.WaitForPendingFinalizers();
        }
    }

    /// <summary>
    /// Disposes all rented buffers and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (disposed)
            return;

        lock (lockObject)
        {
            foreach (var buffer in rentedBuffers)
            {
                buffer.Dispose();
            }
            rentedBuffers.Clear();
            Interlocked.Exchange(ref currentMemoryUsage, 0);
            disposed = true;
        }
    }
}

/// <summary>
/// Provides streaming write capabilities with automatic buffer management.
/// </summary>
internal sealed class StreamingWriter : IDisposable
{
    private readonly Stream stream;
    private readonly StreamingBufferManager bufferManager;
    private IMemoryOwner<byte>? currentBuffer;
    private int currentPosition = 0;
    private bool disposed = false;

    internal StreamingWriter(Stream stream, StreamingBufferManager bufferManager)
    {
        this.stream = stream;
        this.bufferManager = bufferManager;
        this.currentBuffer = bufferManager.RentBuffer();
    }

    /// <summary>
    /// Writes data to the stream, automatically managing buffers.
    /// </summary>
    /// <param name="data">The data to write.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    public async Task WriteAsync(ReadOnlyMemory<byte> data, CancellationToken cancellationToken = default)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(StreamingWriter));

        if (currentBuffer == null)
            currentBuffer = bufferManager.RentBuffer();

        var remaining = data;
        while (!remaining.IsEmpty)
        {
            var availableSpace = currentBuffer.Memory.Length - currentPosition;
            var toCopy = Math.Min(remaining.Length, availableSpace);

            remaining.Slice(0, toCopy).CopyTo(currentBuffer.Memory.Slice(currentPosition));
            currentPosition += toCopy;
            remaining = remaining.Slice(toCopy);

            if (currentPosition >= currentBuffer.Memory.Length)
            {
                await FlushAsync(cancellationToken);
            }
        }
    }

    /// <summary>
    /// Flushes the current buffer to the stream.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(StreamingWriter));

        if (currentBuffer != null && currentPosition > 0)
        {
            await stream.WriteAsync(currentBuffer.Memory.Slice(0, currentPosition), cancellationToken);
            bufferManager.ReturnBuffer(currentBuffer);
            currentBuffer = null;
            currentPosition = 0;
        }
    }

    /// <summary>
    /// Disposes the writer and flushes any remaining data.
    /// </summary>
    public void Dispose()
    {
        if (disposed)
            return;

        try
        {
            FlushAsync().GetAwaiter().GetResult();
        }
        catch
        {
            // Ignore errors during disposal
        }

        if (currentBuffer != null)
        {
            bufferManager.ReturnBuffer(currentBuffer);
            currentBuffer = null;
        }

        disposed = true;
    }
}