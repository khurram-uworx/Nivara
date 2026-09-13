namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// Results of one leg over the fixed SmolLM-shaped fixture set: the three production
/// kernels on the same byte-identical BF16 inputs. Every leg (CPU, SYCL, DX12 later)
/// produces this; <see cref="KernelGate"/> compares each against the CPU leg.
/// </summary>
internal sealed record LegResults(float Dot16, float[] Silu, float[] Gemv);