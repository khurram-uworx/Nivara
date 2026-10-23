using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// Shared shape table and helpers for the Silk/CPU GEMM probes. The ILGPU-specific
/// tiled-GEMM regression gate (<c>--gemm</c>) / ILGPU allocation probe (<c>--gpu-alloc</c>)
/// were removed with issue #814; what remains here is the neutral data and timing utilities
/// the surviving probes (<c>--silk-gemm</c>, <c>--cpu-gemm</c>, the kernel gate, etc.) build on.
/// </summary>
internal static class GemmBenchmark
{
    [StructLayout(LayoutKind.Sequential)]
    struct SystemPowerStatus
    {
        public byte AclLineStatus;
        public byte BatteryFlag;
        public byte BatteryLifePercent;
        public byte SystemStatusFlag;
        public uint BatteryLifeTime;
        public uint BatteryFullLifeTime;
    }

    [DllImport("kernel32.dll")]
    static extern bool GetSystemPowerStatus(out SystemPowerStatus status);

    internal readonly record struct GemmShape(string Name, int Arows, int Acols, int Bcols)
    {
        public long Macs => (long)Arows * Acols * Bcols;
    }

    internal static readonly GemmShape[] s_shapes =
    [
        new("distilbert qkv/o", 128, 768, 768),
        new("distilbert fc1", 128, 768, 3072),
        new("distilbert fc2", 128, 3072, 768),
        new("distilbert head", 128, 768, 2),
        new("minilm qkv/o", 128, 384, 384),
        new("minilm fc1", 128, 384, 1536),
        new("minilm fc2", 128, 1536, 384),
        new("minilm head", 128, 384, 2),
        new("edge padded rows", 100, 770, 70),
        new("edge padded K", 64, 1032, 130),

        new("laya qkv", 512, 1024, 3072),
        new("laya attn out", 512, 1024, 1024),
        new("laya fc1 (Wi)", 512, 1024, 5248),
        new("laya fc2 (Wo)", 512, 2624, 1024),
        new("laya head ff1", 512, 1024, 4096),
        new("laya head ff2", 512, 4096, 1024),
        new("laya qkv@128", 128, 1024, 3072),
    ];

    internal static void PrintPowerState()
    {
        if (!GetSystemPowerStatus(out var status))
        {
            Console.WriteLine("Power  : GetSystemPowerStatus unavailable — AC state unknown");
            return;
        }

        if (status.AclLineStatus == 0)
        {
            Console.WriteLine("WARNING: on battery — the iGPU throttles flat, GMAC/s numbers are INVALID (docs/ACCELERATION.md §1 battery lesson).");
            Console.WriteLine("         The correctness gate is battery-safe and still runs.");
        }
        else if (status.AclLineStatus == 1)
        {
            Console.WriteLine("Power  : AC line — GMAC/s valid");
        }
        else
        {
            Console.WriteLine("Power  : unknown AC status (255)");
        }
    }

    internal static float NextUniformF(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state / 4294967295.0f) * 2.0f - 1.0f;
    }

    internal static double TimeBest(Action launch, int warmups, int rounds)
    {
        for (int w = 0; w < warmups; w++)
            launch();

        double bestUs = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            var sw = Stopwatch.StartNew();
            launch();
            sw.Stop();
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMilliseconds * 1000.0);
        }

        return bestUs;
    }
}