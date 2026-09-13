namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// The multi-leg correctness gate. Every leg runs the same byte-identical BF16 fixtures and
/// produces a <see cref="LegResults"/>; the CPU leg (the production Nivara kernels, see
/// <see cref="CpuLeg"/>) is the gold target, and every GPU leg is gated against it with
/// <c>|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|</c>. Per-kernel rows report the worst ULP
/// at the reference's magnitude (a *diagnostic*, not the gate) and the pass count. Exit code
/// = number of failed kernels (0 = all gates pass).
/// </summary>
internal static class KernelGate
{
    /// <summary>Runs the CPU reference + one GPU leg and prints the gate table.</summary>
    public static int Run(KernelFixtures fixtures, string gpuLegName, Func<KernelFixtures, LegResults?> gpuLeg)
    {
        Console.WriteLine();
        Console.WriteLine("--- kernel gates vs CPU (production Nivara kernels) ---");

        LegResults cpu = CpuLeg.ComputeLeg(fixtures);

        int failures = 0;
        failures += Gate(cpu, cpu, "CPU (production Nivara)");
        failures += GateLeg(gpuLegName, gpuLeg(fixtures), cpu);

        Console.WriteLine($"  kernels mode exit: {failures} failed kernel(s), 0 unexpected");
        return failures;
    }

    private static int GateLeg(string name, LegResults? results, LegResults cpu)
    {
        if (results is null)
        {
            Console.WriteLine($"  {name.PadRight(28)} UNBUILT / UNAVAILABLE (3 kernels failed)");
            return 3;
        }

        Console.WriteLine();
        return Gate(cpu, results, name);
    }

    private static int Gate(LegResults reference, LegResults leg, string name)
    {
        if (ReferenceEquals(reference, leg))
        {
            // Identity row: the CPU leg against itself (0 ULP sanity check of the harness).
            Console.WriteLine($"  {name.PadRight(28)} dot16 0.0 ULP | silu 0/0 failed | gemv 0/0 failed");
            return 0;
        }

        int failedDots = GateDot16(reference, leg, name);
        int failedSilu = GateVector(reference.Silu, leg.Silu, "silu", name);
        int failedGemv = GateVector(reference.Gemv, leg.Gemv, "gemv", name);
        return failedDots + failedSilu + failedGemv;
    }

    private static int GateDot16(LegResults reference, LegResults leg, string name)
    {
        bool pass = CpuLeg.WithinTolerance(leg.Dot16, reference.Dot16);
        Console.WriteLine($"  {name.PadRight(28)} dot16 {leg.Dot16:G9} vs CPU {reference.Dot16:G9} -> {(pass ? "PASS" : "FAIL")} ({CpuLeg.UlpDistance(leg.Dot16, reference.Dot16):F1} ULP)");
        return pass ? 0 : 1;
    }

    private static int GateVector(float[] reference, float[] leg, string kernel, string name)
    {
        int passed = 0, failed = 0;
        double worstUlp = 0;
        int worstIndex = -1;
        float worstLeg = 0f, worstRef = 0f;
        for (int i = 0; i < leg.Length; i++)
        {
            double ulp = CpuLeg.UlpDistance(leg[i], reference[i]);
            if (ulp > worstUlp)
            {
                worstUlp = ulp;
                worstIndex = i;
                worstLeg = leg[i];
                worstRef = reference[i];
            }
            if (CpuLeg.WithinTolerance(leg[i], reference[i]))
                passed++;
            else
                failed++;
        }
        if (worstIndex >= 0)
        {
            double worstGate = CpuLeg.GateAbs + CpuLeg.GateRel * Math.Abs(worstRef);
            Console.WriteLine($"  {name.PadRight(28)} {kernel} {passed}/{leg.Length} within gate ({failed} failed), worst {worstUlp:F1} ULP @ index {worstIndex} (leg {worstLeg:G9} vs CPU {worstRef:G9}, gate {worstGate:G3}, |diff| {Math.Abs(worstLeg - worstRef):G3})");
        }
        else
        {
            Console.WriteLine($"  {name.PadRight(28)} {kernel} {passed}/{leg.Length} within gate ({failed} failed)");
        }
        return failed;
    }
}