namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// Results of one leg over the fixed SmolLM-shaped fixture set: the three production
/// kernels on the same byte-identical BF16 inputs, plus per-kernel wall time in
/// microseconds. Every leg (CPU, SYCL, DX12 later) produces this; <see cref="KernelGate"/>
/// compares each against the CPU leg and prints the timing table.
/// </summary>
internal sealed record LegResults(
    float Dot16,
    float[] Silu,
    float[] Gemv,
    double Dot16Us,
    double SiluUs,
    double GemvUs)
{
    public static LegResults Untimed(float dot16, float[] silu, float[] gemv)
        => new(dot16, silu, gemv, 0, 0, 0);

    /// <summary>Per-kernel wall time (µs) by kernel name, used by the multi-leg timing table.</summary>
    public double? UsFor(string kernel) => kernel switch
    {
        "dot16" => Dot16Us,
        "silu" => SiluUs,
        "gemv" => GemvUs,
        _ => null
    };
}