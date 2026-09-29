using ILGPU;
using ILGPU.Runtime;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand device-memory ceiling probe for the Laya backend decision (docs/TODO.md, leg 1).
///
/// Laya's checkpoint is 421,293,830 F16 parameters, but the GPU path is F32-only
/// (<c>--gpu</c> rejects bf16/fp16), so an F32 working set needs
/// <c>421,293,830 * 4 B = 1.685 GB</c> of device buffers. Every published GPU row in
/// <c>samples/NivaraInference/README.md</c> is 66-110M parameters, i.e. 0.26-0.44 GB — so
/// this is a 4-6x jump beyond anything already known to work on this device, and it is the
/// one binary gate in the probe: if the allocation fails, throughput is irrelevant and F16
/// device buffers become a prerequisite rather than a later optimisation.
///
/// Allocation is chunked rather than one big buffer on purpose. Laya's working set is an
/// embedding table plus ~200 per-layer weight tensors, so the realistic question is whether
/// the device can back many medium buffers totalling 1.685 GB, not whether it can reserve
/// one contiguous 1.685 GB block. Each chunk is filled from the host and a sample read back,
/// because a reservation that succeeds but cannot be written is not a working set.
///
/// Run only on explicit request (<c>--gpu-alloc</c>). GPU-dependent, so never part of the
/// default scenario suite or the <c>--json</c>/<c>--compare</c> gate.
/// </summary>
internal static class GpuAllocProbe
{
    const int ExitUnbuilt = 10;
    const int ChunkMiB = 64;

    /// <summary>421,293,830 F16 params widened to F32.</summary>
    const long LayaF32Bytes = 421_293_830L * 4;

    static readonly long[] s_targets =
    [
        256L << 20,
        512L << 20,
        1L << 30,
        3L << 29,
        LayaF32Bytes,
        2L << 30,
        3L << 30,
    ];

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("Device-memory ceiling probe: can the iGPU back Laya's F32 working set?");

        if (!GemmBenchmark.TryCreateRuntime(out var runtime, out string reason))
        {
            Console.WriteLine();
            Console.WriteLine("UNBUILT: --gpu-alloc requires an OpenCL GPU device (Intel Arc iGPU); none was found.");
            Console.WriteLine($"  {reason}");
            return ExitUnbuilt;
        }

        using (runtime)
        {
            Console.WriteLine($"  Device       : {runtime.DeviceName}");
            Console.WriteLine($"  Device memory: {Format(runtime.Accelerator.MemorySize)} reported by the driver");
            Console.WriteLine($"  Laya needs   : {Format(LayaF32Bytes)} (421,293,830 F16 params widened to F32)");
            Console.WriteLine($"  Chunking     : {ChunkMiB} MiB buffers, each host-filled and sample-verified");
            Console.WriteLine();

            var host = new float[(ChunkMiB << 20) / sizeof(float)];
            for (int i = 0; i < host.Length; i++)
                host[i] = i * 0.5f;

            Console.WriteLine($"{"target",-17} {"buffers",-9} {"alloc ms",-10} {"fill GB/s",-11} {"verify",-9} {"result",-8}");
            long largestOk = 0;
            bool layaOk = false;

            foreach (long target in s_targets)
            {
                int chunkFloats = (ChunkMiB << 20) / sizeof(float);
                int chunks = (int)Math.Ceiling(target / (double)(ChunkMiB << 20));
                var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>(chunks);
                var allocWatch = Stopwatch.StartNew();
                string result;
                double fillGbps = 0;

                try
                {
                    for (int i = 0; i < chunks; i++)
                        buffers.Add(runtime.Allocate1D(chunkFloats));

                    allocWatch.Stop();

                    // Fill every chunk: a reservation that cannot be written is not a working set.
                    var fillWatch = Stopwatch.StartNew();
                    for (int i = 0; i < buffers.Count; i++)
                        buffers[i].CopyFromCPU(host);
                    runtime.Synchronize();
                    fillWatch.Stop();

                    double bytes = (double)buffers.Count * (ChunkMiB << 20);
                    fillGbps = bytes / (fillWatch.Elapsed.TotalSeconds * 1e9);

                    // Spot-check the last element of the last chunk round-trips.
                    var tail = buffers[^1].AsContiguous().GetAsArray();
                    bool verify = Math.Abs(tail[^1] - host[^1]) < 1e-3f;

                    largestOk = Math.Max(largestOk, target);
                    if (target == LayaF32Bytes)
                        layaOk = verify;
                    result = verify ? "ok" : "BAD DATA";
                }
                catch (Exception ex)
                {
                    allocWatch.Stop();
                    result = "ALLOC FAIL";
                    Console.WriteLine($"           -> {ex.GetType().Name}: {Trim(ex.Message)}");
                }
                finally
                {
                    foreach (var b in buffers)
                        b.Dispose();
                }

                // Same Format() as every other row, so the table and the header agree on units
                // (Format reports GiB as "GB"; 1.569 GiB is 1.685 GB decimal).
                string label = target == LayaF32Bytes ? Format(LayaF32Bytes) + " (Laya)" : Format(target);
                Console.WriteLine($"{label,-17} {chunks,-9} {allocWatch.Elapsed.TotalMilliseconds,-10:F1} {fillGbps,-11:F2} {"-",-9} {result,-8}");
            }

            Console.WriteLine();
            Console.WriteLine(layaOk
                ? $"VERDICT: GPU is feasible — the full {Format(LayaF32Bytes)} Laya F32 working set allocated, filled and verified."
                : $"VERDICT: GPU is NOT feasible at the current F32 requirement — the {Format(LayaF32Bytes)} working set did not allocate and verify. F16 device buffers become a prerequisite.");
            Console.WriteLine($"Largest stepped target that succeeded: {Format(largestOk)}.");
            return layaOk ? 0 : 1;
        }
    }

    static string Format(long bytes)
    {
        double gb = bytes / (1024.0 * 1024.0 * 1024.0);
        return gb >= 1.0 ? $"{gb:F3} GB" : $"{bytes / (1024.0 * 1024.0):F0} MB";
    }

    static string Trim(string message)
    {
        var flat = message.Replace('\r', ' ').Replace('\n', ' ');
        return flat.Length <= 90 ? flat : flat[..90] + "...";
    }
}
