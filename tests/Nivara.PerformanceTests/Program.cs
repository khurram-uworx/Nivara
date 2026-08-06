using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Numerics.Tensors;

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
                var a = Fill(new float[1_000_000]);
                var dest = new float[1_000_000];
                return () => TensorPrimitives.Sigmoid(a, dest);
            });

        Run("Span chain 1M x 3 ops (raw)", 5, 100,
            () =>
            {
                var a = Fill(new float[1_000_000]);
                var b = Fill(new float[1_000_000]);
                var c = Fill(new float[1_000_000]);
                var d = Fill(new float[1_000_000]);
                var t1 = new float[1_000_000];
                var t2 = new float[1_000_000];
                var result = new float[1_000_000];
                return () =>
                {
                    TensorPrimitives.Add(a, b, t1);
                    TensorPrimitives.Multiply(t1, c, t2);
                    TensorPrimitives.Subtract(t2, d, result);
                };
            });

        Run("Column chain 1M x 3 ops (wrapper)", 5, 100,
            () =>
            {
                var a = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var b = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var c = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                var d = NivaraColumn<float>.Create(Fill(new float[1_000_000]));
                return () =>
                {
                    var t1 = a.Add(b);
                    var t2 = t1.Multiply(c);
                    _ = t2.Subtract(d);
                };
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

        RunBatchedAttentionScenarios();

        return 0;
    }

    static void RunBatchedAttentionScenarios()
    {
        const int B = 16, L = 128, D = 64, H = 4;
        float scale = 1f / MathF.Sqrt(D / H);

        var qData = Fill(new float[B * L * D]);
        var kData = Fill(new float[B * L * D]);
        var vData = Fill(new float[B * L * D]);
        var dOut = Fill(new float[B * L * D]);
        var causalPerSeq = BuildCausalMask(L);
        var causalBatched = BuildCausalMask(B, L);

        Run($"Attn per-seq forward [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var mask = ReverseGradTensor<float>.FromMatrix(causalPerSeq, L, L, requiresGrad: false);
                return () =>
                {
                    for (int b = 0; b < B; b++)
                    {
                        var q = Mat2D(Slice(qData, b, L, D), L, D, false);
                        var k = Mat2D(Slice(kData, b, L, D), L, D, false);
                        var v = Mat2D(Slice(vData, b, L, D), L, D, false);
                        ReverseGradOperations.MultiHeadAttention(q, k, v, H, scale, mask);
                    }
                };
            });

        Run($"Attn batched forward [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var q = Mat3D(qData, B, L, D, false);
                var k = Mat3D(kData, B, L, D, false);
                var v = Mat3D(vData, B, L, D, false);
                var mask = Mat3D(causalBatched, B, L, L, false);
                return () => { ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask); };
            });

        Run($"Attn per-seq fwd+bwd [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var mask = ReverseGradTensor<float>.FromMatrix(causalPerSeq, L, L, requiresGrad: false);
                var ones = Fill(new float[L * D]);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        for (int b = 0; b < B; b++)
                        {
                            var q = Mat2D(Slice(qData, b, L, D), L, D, true);
                            var k = Mat2D(Slice(kData, b, L, D), L, D, true);
                            var v = Mat2D(Slice(vData, b, L, D), L, D, true);
                            var output = ReverseGradOperations.MultiHeadAttention(q, k, v, H, scale, mask);
                            output.Backward(Mat2D(ones, L, D, false));
                        }
                    }
                };
            });

        Run($"Attn batched fwd+bwd [B{B} L{L} D{D} H{H}]", 3, 12,
            () =>
            {
                var q = Mat3D(qData, B, L, D, true);
                var k = Mat3D(kData, B, L, D, true);
                var v = Mat3D(vData, B, L, D, true);
                var mask = Mat3D(causalBatched, B, L, L, false);
                var dout = Mat3D(dOut, B, L, D, false);
                return () =>
                {
                    using (GradientUtils.Grad())
                    {
                        var output = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, H, scale, mask);
                        output.Backward(dout);
                    }
                };
            });
    }

    static float[] BuildCausalMask(int l)
    {
        var mask = new float[l * l];
        for (int i = 0; i < l; i++)
            for (int j = i + 1; j < l; j++)
                mask[i * l + j] = float.NegativeInfinity;
        return mask;
    }

    static float[] BuildCausalMask(int b, int l)
    {
        var mask = new float[b * l * l];
        var perSeq = BuildCausalMask(l);
        for (int i = 0; i < b * l * l; i++)
            mask[i] = perSeq[i % (l * l)];
        return mask;
    }

    static float[] Slice(float[] data, int b, int rows, int cols)
    {
        var slice = new float[rows * cols];
        Array.Copy(data, b * rows * cols, slice, 0, rows * cols);
        return slice;
    }

    static ReverseGradTensor<float> Mat2D(float[] data, int rows, int cols, bool requiresGrad)
        => ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);

    static ReverseGradTensor<float> Mat3D(float[] data, int b, int l, int d, bool requiresGrad)
    {
        var tensor = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad);
        tensor.Reshape(b, l, d);
        return tensor;
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

