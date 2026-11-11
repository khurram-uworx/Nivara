using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Extensions.Blas;

/// <summary>
/// <see cref="IGemmProvider" /> backed by OpenBLAS via the hand-written CBLAS binding in
/// <see cref="OpenBlasInterop" />. Uses the hybrid configuration: Nivara owns the row
/// parallelism. OpenBLAS's pool size is the constructor argument (default 1).
/// </summary>
public sealed class OpenBlasGemmProvider : IGemmProvider
{
    readonly int threadCount;
    readonly int openBlasThreads;

    public OpenBlasGemmProvider(int threadCount = 1, int openBlasThreads = 1)
    {
        this.threadCount = threadCount < 1 ? 1 : threadCount;
        this.openBlasThreads = openBlasThreads < 1 ? 1 : openBlasThreads;
        if (!OpenBlasInterop.TryInitialize(out string detail))
            throw new InvalidOperationException($"OpenBLAS is not available: {detail}");
    }

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n, bool bTransposed)
    {
        if (m <= 0 || n <= 0 || k <= 0)
            return;

        OpenBlasInterop.SetNumThreads(openBlasThreads);

        unsafe
        {
            fixed (float* paPtr = a, pbPtr = b, pcPtr = c)
            {
                int chunks = Math.Min(threadCount, m);
                int rowsPerChunk = (m + chunks - 1) / chunks;

                // Capture the pinned pointers in a heap object so the lambda does not
                // capture a fixed local (CS1764). The fixed block spans the Parallel.For,
                // so the memory stays pinned for the duration of the work.
                var ctx = new FloatChunkContext
                {
                    Pa = paPtr,
                    Pb = pbPtr,
                    Pc = pcPtr,
                    K = k,
                    N = n,
                    RowsPerChunk = rowsPerChunk,
                    M = m,
                    BTransposed = bTransposed,
                };

                Parallel.For(0, chunks, t =>
                {
                    int rowStart = t * ctx.RowsPerChunk;
                    int rows = Math.Min(ctx.RowsPerChunk, ctx.M - rowStart);
                    if (rows <= 0)
                        return;

                    var aSlice = new ReadOnlySpan<float>(ctx.Pa + rowStart * ctx.K, rows * ctx.K);
                    var bSlice = new ReadOnlySpan<float>(ctx.Pb, ctx.K * ctx.N);
                    var cSlice = new Span<float>(ctx.Pc + rowStart * ctx.N, rows * ctx.N);
                    CallSgemm(aSlice, bSlice, cSlice, rows, ctx.K, ctx.N, ctx.BTransposed);
                });
            }
        }
    }

    public void MatMul(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n, bool bTransposed)
    {
        if (m <= 0 || n <= 0 || k <= 0)
            return;

        OpenBlasInterop.SetNumThreads(openBlasThreads);

        unsafe
        {
            fixed (double* paPtr = a, pbPtr = b, pcPtr = c)
            {
                int chunks = Math.Min(threadCount, m);
                int rowsPerChunk = (m + chunks - 1) / chunks;

                var ctx = new DoubleChunkContext
                {
                    Pa = paPtr,
                    Pb = pbPtr,
                    Pc = pcPtr,
                    K = k,
                    N = n,
                    RowsPerChunk = rowsPerChunk,
                    M = m,
                    BTransposed = bTransposed,
                };

                Parallel.For(0, chunks, t =>
                {
                    int rowStart = t * ctx.RowsPerChunk;
                    int rows = Math.Min(ctx.RowsPerChunk, ctx.M - rowStart);
                    if (rows <= 0)
                        return;

                    var aSlice = new ReadOnlySpan<double>(ctx.Pa + rowStart * ctx.K, rows * ctx.K);
                    var bSlice = new ReadOnlySpan<double>(ctx.Pb, ctx.K * ctx.N);
                    var cSlice = new Span<double>(ctx.Pc + rowStart * ctx.N, rows * ctx.N);
                    CallDgemm(aSlice, bSlice, cSlice, rows, ctx.K, ctx.N, ctx.BTransposed);
                });
            }
        }
    }

    static void CallSgemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n, bool bTransposed)
    {
        int transB = bTransposed ? OpenBlasInterop.Trans : OpenBlasInterop.NoTrans;
        int ldb = bTransposed ? k : n;

        ref float aRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(a));
        ref float bRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(b));
        ref float cRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(c));

        OpenBlasInterop.Sgemm(
            OpenBlasInterop.RowMajor, OpenBlasInterop.NoTrans, transB,
            m, n, k, 1f,
            ref aRef, k,
            ref bRef, ldb,
            0f, ref cRef, n);
    }

    static void CallDgemm(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n, bool bTransposed)
    {
        int transB = bTransposed ? OpenBlasInterop.Trans : OpenBlasInterop.NoTrans;
        int ldb = bTransposed ? k : n;

        ref double aRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(a));
        ref double bRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(b));
        ref double cRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(c));

        OpenBlasInterop.Dgemm(
            OpenBlasInterop.RowMajor, OpenBlasInterop.NoTrans, transB,
            m, n, k, 1.0,
            ref aRef, k,
            ref bRef, ldb,
            0.0, ref cRef, n);
    }

    unsafe class FloatChunkContext
    {
        public float* Pa;
        public float* Pb;
        public float* Pc;
        public int K;
        public int N;
        public int RowsPerChunk;
        public int M;
        public bool BTransposed;
    }

    unsafe class DoubleChunkContext
    {
        public double* Pa;
        public double* Pb;
        public double* Pc;
        public int K;
        public int N;
        public int RowsPerChunk;
        public int M;
        public bool BTransposed;
    }
}
