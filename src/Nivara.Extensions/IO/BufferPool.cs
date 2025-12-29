using System.Buffers;
using System.Collections.Concurrent;

namespace Nivara.IO;

/// <summary>
/// Provides buffer pooling for I/O operations to reduce memory allocations and garbage collection pressure.
/// </summary>
/// <remarks>
/// This class manages reusable buffers for large dataset processing, helping to maintain bounded memory usage
/// and improve performance by reducing allocations during Arrow/Parquet I/O operations.
/// </remarks>
internal static class BufferPool
{
    /// <summary>
    /// Default buffer size for I/O operations (64KB)
    /// </summary>
    private const int DefaultBufferSize = 64 * 1024;

    /// <summary>
    /// Maximum number of buffers to keep in each pool
    /// </summary>
    private const int MaxBuffersPerPool = 16;

    /// <summary>
    /// Pool for byte arrays used in I/O operations
    /// </summary>
    private static readonly ConcurrentQueue<byte[]> ByteBufferPool = new();

    /// <summary>
    /// Pool for int arrays used in column processing
    /// </summary>
    private static readonly ConcurrentQueue<int[]> IntBufferPool = new();

    /// <summary>
    /// Pool for double arrays used in column processing
    /// </summary>
    private static readonly ConcurrentQueue<double[]> DoubleBufferPool = new();

    /// <summary>
    /// Current count of byte buffers in the pool
    /// </summary>
    private static int byteBufferCount = 0;

    /// <summary>
    /// Current count of int buffers in the pool
    /// </summary>
    private static int intBufferCount = 0;

    /// <summary>
    /// Current count of double buffers in the pool
    /// </summary>
    private static int doubleBufferCount = 0;

    /// <summary>
    /// Rents a byte buffer from the pool or creates a new one if none available.
    /// </summary>
    /// <param name="minimumSize">The minimum size required for the buffer.</param>
    /// <returns>A byte array that is at least the specified size.</returns>
    public static byte[] RentByteBuffer(int minimumSize = DefaultBufferSize)
    {
        if (ByteBufferPool.TryDequeue(out var buffer) && buffer.Length >= minimumSize)
        {
            Interlocked.Decrement(ref byteBufferCount);
            return buffer;
        }

        // Create new buffer with size rounded up to nearest power of 2
        var size = Math.Max(minimumSize, DefaultBufferSize);
        size = (int)Math.Pow(2, Math.Ceiling(Math.Log2(size)));
        return new byte[size];
    }

    /// <summary>
    /// Returns a byte buffer to the pool for reuse.
    /// </summary>
    /// <param name="buffer">The buffer to return to the pool.</param>
    public static void ReturnByteBuffer(byte[] buffer)
    {
        if (buffer == null || buffer.Length < DefaultBufferSize)
            return;

        if (byteBufferCount < MaxBuffersPerPool)
        {
            // Clear the buffer before returning to pool
            Array.Clear(buffer, 0, buffer.Length);
            ByteBufferPool.Enqueue(buffer);
            Interlocked.Increment(ref byteBufferCount);
        }
    }

    /// <summary>
    /// Rents an int buffer from the pool or creates a new one if none available.
    /// </summary>
    /// <param name="minimumSize">The minimum size required for the buffer.</param>
    /// <returns>An int array that is at least the specified size.</returns>
    public static int[] RentIntBuffer(int minimumSize)
    {
        if (IntBufferPool.TryDequeue(out var buffer) && buffer.Length >= minimumSize)
        {
            Interlocked.Decrement(ref intBufferCount);
            return buffer;
        }

        // Create new buffer with size rounded up to nearest power of 2
        var size = (int)Math.Pow(2, Math.Ceiling(Math.Log2(minimumSize)));
        return new int[size];
    }

    /// <summary>
    /// Returns an int buffer to the pool for reuse.
    /// </summary>
    /// <param name="buffer">The buffer to return to the pool.</param>
    public static void ReturnIntBuffer(int[] buffer)
    {
        if (buffer == null || buffer.Length < 1024)
            return;

        if (intBufferCount < MaxBuffersPerPool)
        {
            // Clear the buffer before returning to pool
            Array.Clear(buffer, 0, buffer.Length);
            IntBufferPool.Enqueue(buffer);
            Interlocked.Increment(ref intBufferCount);
        }
    }

    /// <summary>
    /// Rents a double buffer from the pool or creates a new one if none available.
    /// </summary>
    /// <param name="minimumSize">The minimum size required for the buffer.</param>
    /// <returns>A double array that is at least the specified size.</returns>
    public static double[] RentDoubleBuffer(int minimumSize)
    {
        if (DoubleBufferPool.TryDequeue(out var buffer) && buffer.Length >= minimumSize)
        {
            Interlocked.Decrement(ref doubleBufferCount);
            return buffer;
        }

        // Create new buffer with size rounded up to nearest power of 2
        var size = (int)Math.Pow(2, Math.Ceiling(Math.Log2(minimumSize)));
        return new double[size];
    }

    /// <summary>
    /// Returns a double buffer to the pool for reuse.
    /// </summary>
    /// <param name="buffer">The buffer to return to the pool.</param>
    public static void ReturnDoubleBuffer(double[] buffer)
    {
        if (buffer == null || buffer.Length < 1024)
            return;

        if (doubleBufferCount < MaxBuffersPerPool)
        {
            // Clear the buffer before returning to pool
            Array.Clear(buffer, 0, buffer.Length);
            DoubleBufferPool.Enqueue(buffer);
            Interlocked.Increment(ref doubleBufferCount);
        }
    }

    /// <summary>
    /// Gets the current statistics of the buffer pools.
    /// </summary>
    /// <returns>A tuple containing the count of buffers in each pool.</returns>
    public static (int ByteBuffers, int IntBuffers, int DoubleBuffers) GetPoolStatistics()
    {
        return (byteBufferCount, intBufferCount, doubleBufferCount);
    }

    /// <summary>
    /// Clears all buffer pools, releasing memory.
    /// </summary>
    public static void ClearPools()
    {
        while (ByteBufferPool.TryDequeue(out _))
        {
            Interlocked.Decrement(ref byteBufferCount);
        }

        while (IntBufferPool.TryDequeue(out _))
        {
            Interlocked.Decrement(ref intBufferCount);
        }

        while (DoubleBufferPool.TryDequeue(out _))
        {
            Interlocked.Decrement(ref doubleBufferCount);
        }
    }
}