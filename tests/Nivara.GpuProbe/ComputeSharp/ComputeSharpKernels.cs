#if WINDOWS
using System.Runtime.Versioning;
using ComputeSharp;

namespace Nivara.GpuProbe.ComputeSharp;

/// <summary>
/// The three production SmolLM kernels as ComputeSharp shaders: ordinary C# structs that
/// the library's source generator lowers to HLSL (compute shader model 6.x), compiles to
/// DXIL via the bundled DXC (ComputeSharp.Dxc), and runs on the Arc 140T iGPU through
/// D3D12. Kernels feed the same byte-identical BF16 fixtures as every other leg over the
/// packed-2-per-uint transport shared with the D3D12 leg (<c>D3d12.GemvKernels.PackBf16</c> /
/// its in-shader <c>Widen</c>). BF16→f32 widening happens in-shader via
/// <see cref="Hlsl.AsFloat"/> (the HLSL <c>asfloat</c> bit-cast, the ComputeSharp mirror of
/// <c>Interop.IntAsFloat</c>); silu uses <see cref="Hlsl.Exp"/> (HLSL <c>exp</c>). Thread
/// group geometry mirrors <c>D3d12.GemvKernels</c> exactly (dot16 1×1×1; silu/gemv 256×1×1),
/// and every kernel bounds-checks its X index because D3D12 pads the grid to group-size
/// multiples (same reason the HLSL leg guards <c>dt.x &gt;= n</c>).
/// </summary>
/// <remarks>Compiled only on Windows hosts (<c>WINDOWS</c> symbol): the source generator that
/// emits <see cref="ComputeSharp.Descriptors.IComputeShaderDescriptor{T}"/> requires the
/// Windows SDK host (DllNotFoundException on Linux). See the csproj gate.</remarks>
[SupportedOSPlatform("windows6.2")] // D3D12 — the hand-rolled DX12 leg is Windows-only too
internal static partial class ComputeSharpKernels
{
    /// <summary>BF16→f32 in-shader widen for the packed layout (element 2k in the HIGH half,
    /// 2k+1 in the LOW) — the ComputeSharp mirror of <c>D3d12.GemvKernels.Widen</c>.</summary>
    internal static float Widen(uint packed, uint element) =>
        (element & 1u) == 0u
            ? Hlsl.AsFloat(packed & 0xFFFF0000u)
            : Hlsl.AsFloat((packed & 0xFFFFu) << 16);

    /// <summary>dot16: single thread, serial K=16 f32 accumulate over packed [a(K) | b(K)].</summary>
    [ThreadGroupSize(1, 1, 1)]
    [GeneratedComputeShaderDescriptor]
    internal readonly partial struct Dot16Shader(
        ReadOnlyBuffer<uint> input,
        ReadWriteBuffer<float> output,
        int k) : IComputeShader
    {
        public void Execute()
        {
            float acc = 0f;
            for (int j = 0; j < k; j++)
                acc += Widen(input[j >> 1], (uint)j) * Widen(input[(k + j) >> 1], (uint)(k + j));
            output[0] = acc;
        }
    }

    /// <summary>silu: one thread per element, x / (1 + exp(−x)) (the production
    /// sigmoid-then-multiply SiLU shape, <c>Activation.Silu</c>).</summary>
    [ThreadGroupSize(256, 1, 1)]
    [GeneratedComputeShaderDescriptor]
    internal readonly partial struct SiluShader(
        ReadOnlyBuffer<uint> input,
        ReadWriteBuffer<float> output,
        int n) : IComputeShader
    {
        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= n)
                return;
            float x = Widen(input[i >> 1], (uint)i);
            output[i] = x / (1f + Hlsl.Exp(-x));
        }
    }

    /// <summary>gemv: one thread per output row of y = W·x over packed [w(rows·cols) | x(cols)],
    /// serial K=cols f32 accumulate — the production one-row-per-thread BLAS2 shape.</summary>
    [ThreadGroupSize(256, 1, 1)]
    [GeneratedComputeShaderDescriptor]
    internal readonly partial struct GemvShader(
        ReadOnlyBuffer<uint> input,
        ReadWriteBuffer<float> output,
        int rows,
        int cols,
        int weightElems) : IComputeShader
    {
        public void Execute()
        {
            int row = ThreadIds.X;
            if (row >= rows)
                return;
            float acc = 0f;
            for (int k = 0; k < cols; k++)
            {
                int w = row * cols + k;
                acc += Widen(input[w >> 1], (uint)w) * Widen(input[(weightElems + k) >> 1], (uint)(weightElems + k));
            }
            output[row] = acc;
        }
    }
}
#endif