using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.D3d12;

/// <summary>
/// HLSL kernel source (compiled to <c>sm_5_1</c> DXBC) for the three production SmolLM
/// kernels, plus the BF16→packed-uint32 transport layout the DX12 leg uploads. All
/// inputs are concatenated into one <c>StructuredBuffer&lt;uint&gt;</c>; BF16 stays the
/// wire format (2 bytes/element, byte-identical to the CPU and SYCL legs), packed
/// two-per-uint32 (element 2k in the HIGH half, 2k+1 in the LOW) so every element is a
/// 4-byte-aligned load, and widened to f32 in-shader with the standard
/// <c>asfloat(bits &lt;&lt; 16)</c> trick. Output is one <c>RWStructuredBuffer&lt;float&gt;</c>.
/// </summary>
internal static class GemvKernels
{
    public const int ThreadsPerGroup = 256;

    /// <summary>Packs contiguous BF16 elements two-per-uint32 (2k → high half, 2k+1 →
    /// low half) — the exact layout the HLSL <see cref="Widen"/> reads back.</summary>
    public static uint[] PackBf16(ReadOnlySpan<BFloat16> values)
    {
        var packed = new uint[(values.Length + 1) / 2];
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(values);
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = (uint)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
            if ((i & 1) == 0)
                packed[i >> 1] |= bits << 16;
            else
                packed[i >> 1] |= bits;
        }
        return packed;
    }

    private const string Widen = @"
float Widen(uint packed, uint element)
{
    return (element & 1u) == 0u ? asfloat(packed & 0xFFFF0000u) : asfloat((packed & 0xFFFFu) << 16);
}";

    /// <summary>dot16: input = [a(16) | b(16)] → one f32.</summary>
    public static string Dot16(int k) => $@"
StructuredBuffer<uint> gInput : register(t0);
RWStructuredBuffer<float> gOutput : register(u0);
{Widen}
[numthreads(1, 1, 1)]
void CsDot16(uint3 dt : SV_DispatchThreadID)
{{
    float acc = 0.0f;
    for (uint i = 0; i < {k}; ++i)
        acc += Widen(gInput[i >> 1], i) * Widen(gInput[({k} + i) >> 1], {k} + i);
    gOutput[0] = acc;
}}";

    /// <summary>silu: input = [x(n)] → n f32, sigmoid-then-multiply like <c>Activation.Silu</c>.</summary>
    public static string Silu(int n) => $@"
StructuredBuffer<uint> gInput : register(t0);
RWStructuredBuffer<float> gOutput : register(u0);
{Widen}
[numthreads({ThreadsPerGroup}, 1, 1)]
void CsSilu(uint3 dt : SV_DispatchThreadID)
{{
    if (dt.x >= {n}) return;
    float x = Widen(gInput[dt.x >> 1], dt.x);
    gOutput[dt.x] = x / (1.0f + exp(-x));
}}";

    /// <summary>gemv: input = [w(rows·cols) | x(cols)] → rows f32, one row of y = W·x per thread.</summary>
    public static string Gemv(int rows, int cols) => $@"
StructuredBuffer<uint> gInput : register(t0);
RWStructuredBuffer<float> gOutput : register(u0);
static const uint Rows = {rows};
static const uint Cols = {cols};
static const uint WeightElems = {rows * cols};
{Widen}
[numthreads({ThreadsPerGroup}, 1, 1)]
void CsGemv(uint3 dt : SV_DispatchThreadID)
{{
    if (dt.x >= Rows) return;
    float acc = 0.0f;
    for (uint k = 0; k < Cols; ++k)
    {{
        uint w = dt.x * Cols + k;
        acc += Widen(gInput[w >> 1], w) * Widen(gInput[(WeightElems + k) >> 1], WeightElems + k);
    }}
    gOutput[dt.x] = acc;
}}";
}