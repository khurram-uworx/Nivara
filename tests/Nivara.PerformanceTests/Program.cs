using System.Diagnostics;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using Nivara.Tensors;

namespace Nivara.PerformanceTests;

static class Program
{
    static int Main()
    {
        Console.WriteLine("Nivara storage plan benchmark");
        Console.WriteLine($"  Runtime : {Environment.Version}");
        Console.WriteLine($"  Machine : {Environment.ProcessorCount} logical processors, {(Environment.Is64BitProcess ? "x64" : "x86")}");
        Console.WriteLine();
        Console.WriteLine($"{"Scenario",-46} {"ops/s",12} {"ns/op",8} {"B/op",12} {"gen0/op",7}");
        Console.WriteLine(new string('-', 92));

        Run("ColumnAdd 1M x float", 5, 200,
            () =>
            {
                var a = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var b = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                return () => a.Add(b);
            });

        Run("ColumnSigmoid 1M x float", 5, 200,
            () =>
            {
                var a = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                return () => a.Sigmoid();
            });

        Run("Linear forward [32x256] -> [32x256]", 5, 100,
            () =>
            {
                var linear = new Linear<float>(256, 256);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 256]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(32, 256);
                    linear.Forward(input);
                };
            });

        Run("Linear forward+backward [32x256]", 5, 20,
            () =>
            {
                var linear = new Linear<float>(256, 256);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 256]));
                var ones = Fill(new float[32 * 256]);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: true);
                        input.Reshape(32, 256);
                        var output = linear.Forward(input);
                        var gradient = new ReverseGradTensor<float>(NivaraColumn<float>.Create(ones), requiresGrad: false);
                        gradient.Reshape(32, 256);
                        output.Backward(gradient);
                    }
                };
            });

        Run("TransformerBlock forward [32x64, 4 heads]", 5, 30,
            () =>
            {
                var block = new TransformerBlock<float>(64, 4, dropout: 0.0, maxSeqLen: 32, normType: NormType.RMSNorm);
                var inputColumn = NivaraColumn<float>.Create(Fill(new float[32 * 64]));
                return () =>
                {
                    var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: false);
                    input.Reshape(32, 64);
                    block.Forward(input);
                };
            });

        return 0;
    }

    static void Run(string name, int warmup, int iterations, Func<Action> createOp)
    {
        var op = createOp();

        for (int i = 0; i < warmup; i++)
            op();

        long bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        int gen0Before = GC.CollectionCount(0);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            op();
        sw.Stop();
        long bytesAfter = GC.GetAllocatedBytesForCurrentThread();
        int gen0After = GC.CollectionCount(0);

        double nsPerOp = sw.Elapsed.TotalNanoseconds / iterations;
        double opsPerSec = 1e9 / nsPerOp;
        double bytesPerOp = (double)(bytesAfter - bytesBefore) / iterations;
        double gen0PerOp = (double)(gen0After - gen0Before) / iterations;

        Console.WriteLine($"{name,-46} {opsPerSec,12:N0} {nsPerOp,8:N0} {bytesPerOp,12:N0} {gen0PerOp,7:N2}");
    }

    static float[] Fill(float[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = i * 0.001f;
        return values;
    }
}
