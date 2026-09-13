using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// Deterministic SmolLM-shaped BF16 fixtures shared byte-identically by every leg
/// (CPU, L0/SPIR-V, DX12). Shapes are the verified SmolLM-135M constants — hidden
/// 576, heads 9, kv 3, intermediate 1536, silu, vocab 49152 — embedded so the probe
/// never depends on the gitignored model download; reading
/// samples/data/smollm-135m/config.json is an optional override only. Values mirror
/// the Qwen synthetic generator (samples/NivaraInference/Qwen.cs): xorshift32 with
/// seed 0x9E3779B9, (rng / uint.Max - 0.5f) * 0.1f, narrowed via BFloat16.CreateChecked.
/// </summary>
internal sealed class KernelFixtures
{
    public const int HiddenSize = 576;
    public const int IntermediateSize = 1536;
    public const int NumHiddenLayers = 30;
    public const int NumAttentionHeads = 9;
    public const int NumKeyValueHeads = 3;
    public const int VocabSize = 49152;
    public const int Dot16Length = 16;
    public const uint RngSeed = 0x9E3779B9;

    public BFloat16[] Dot16A { get; }
    public BFloat16[] Dot16B { get; }
    public BFloat16[] Dot576A { get; }
    public BFloat16[] Dot576B { get; }
    public BFloat16[] GemvX { get; }
    public BFloat16[] GemvW { get; }
    public BFloat16[] SiluX { get; }

    private KernelFixtures(
        BFloat16[] dot16A,
        BFloat16[] dot16B,
        BFloat16[] dot576A,
        BFloat16[] dot576B,
        BFloat16[] gemvX,
        BFloat16[] gemvW,
        BFloat16[] siluX)
    {
        Dot16A = dot16A;
        Dot16B = dot16B;
        Dot576A = dot576A;
        Dot576B = dot576B;
        GemvX = gemvX;
        GemvW = gemvW;
        SiluX = siluX;
    }

    /// <summary>
    /// Generates every fixture from one deterministic RNG stream (fixed draw order:
    /// dot16 A/B, dot576 A/B, gemv X, gemv W, silu X), so the buffers are stable
    /// across runs and identical across all legs within a run.
    /// </summary>
    public static KernelFixtures Generate()
    {
        uint state = RngSeed;
        BFloat16 Next() => BFloat16.CreateChecked((NextRng(ref state) / (float)uint.MaxValue - 0.5f) * 0.1f);
        BFloat16[] Fill(int count)
        {
            var values = new BFloat16[count];
            for (int i = 0; i < count; i++)
                values[i] = Next();
            return values;
        }

        return new KernelFixtures(
            dot16A: Fill(Dot16Length),
            dot16B: Fill(Dot16Length),
            dot576A: Fill(HiddenSize),
            dot576B: Fill(HiddenSize),
            gemvX: Fill(HiddenSize),
            gemvW: Fill(IntermediateSize * HiddenSize),
            siluX: Fill(HiddenSize));
    }

    /// <summary>Writes raw BF16 patterns (the .NET <see cref="BFloat16"/> memory layout)
    /// into a native buffer — the GPU wire format is the type itself, no host widening.</summary>
    public static void WriteBf16(IntPtr destination, ReadOnlySpan<BFloat16> values)
    {
        byte[] bytes = MemoryMarshal.AsBytes(values).ToArray();
        Marshal.Copy(bytes, 0, destination, bytes.Length);
    }

    /// <summary>Reads f32 kernel results back from a native buffer.</summary>
    public static float[] ReadF32(IntPtr source, int count)
    {
        byte[] bytes = new byte[count * 4];
        Marshal.Copy(source, bytes, 0, bytes.Length);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToSingle(bytes, i * 4);
        return result;
    }

    private static uint NextRng(ref uint state)
    {
        uint x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
}