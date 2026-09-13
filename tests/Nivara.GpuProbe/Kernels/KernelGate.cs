namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// The multi-leg correctness gate. Every leg runs the same byte-identical BF16 fixtures and
/// produces a <see cref="LegResults"/>; the CPU leg (the production Nivara kernels, see
/// <see cref="CpuLeg"/>) is the gold target, and every GPU leg is gated against it with
/// <c>|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|</c>. Per-kernel rows report the worst ULP
/// at the reference's magnitude (a *diagnostic*, not the gate) and the pass count. Exit code
/// = number of failed kernels across every leg (0 = all gates pass).
/// </summary>
internal static class KernelGate
{
    /// <summary>Single-leg convenience overload.</summary>
    public static int Run(KernelFixtures fixtures, string gpuLegName, Func<KernelFixtures, LegResults?> gpuLeg)
        => Run(fixtures, (gpuLegName, gpuLeg));

    /// <summary>Runs the CPU reference + every GPU leg and prints the gate table.</summary>
    public static int Run(KernelFixtures fixtures, params (string Name, Func<KernelFixtures, LegResults?> Leg)[] legs)
    {
        Console.WriteLine();
        Console.WriteLine("--- kernel gates vs CPU (production Nivara kernels) ---");

        // Warm up the production kernels once: the first call pays JIT compilation
        // cost and would dominate the timing rows. Discard it; time the second pass.
        _ = CpuLeg.ComputeLeg(fixtures);
        LegResults cpu = CpuLeg.ComputeLeg(fixtures);

        int failures = Gate(cpu, cpu, "CPU (production Nivara)");

        var gpuResults = new List<(string Name, LegResults? Results)>();
        foreach ((string name, Func<KernelFixtures, LegResults?> leg) in legs)
        {
            LegResults? results = leg(fixtures);
            gpuResults.Add((name, results));
            failures += GateLeg(name, results, cpu);
        }

        Console.WriteLine();
        PrintTimingTable(cpu, gpuResults);

        Console.WriteLine($"  kernels mode exit: {failures} failed kernel(s), 0 unexpected");
        return failures;
    }

    /// <summary>Prints CPU vs each GPU leg's per-kernel wall time (µs). GPU times come from
    /// the legs themselves; the CPU times are captured by <see cref="CpuLeg.ComputeLeg"/>.</summary>
    private static void PrintTimingTable(LegResults cpu, List<(string Name, LegResults? Results)> gpus)
    {
        Console.WriteLine("  kernel timings (µs per invocation, excluding process/device setup):");
        PrintTimingRow("dot16", cpu.Dot16Us, gpus);
        PrintTimingRow("silu", cpu.SiluUs, gpus);
        PrintTimingRow("gemv", cpu.GemvUs, gpus);
    }

    private static void PrintTimingRow(string kernel, double cpuUs, List<(string Name, LegResults? Results)> gpus)
    {
        string cpuCell = $"CPU {cpuUs,8:F1} µs";
        var gpuCells = new List<string>();
        foreach ((string name, LegResults? results) in gpus)
        {
            string shortName = name.Split(' ', '(')[0];
            double? gpuUs = results?.UsFor(kernel);
            gpuCells.Add(gpuUs is null or <= 0
                ? $"{shortName,7} unavailable"
                : $"{shortName,7} {gpuUs.Value,8:F1} µs");
        }
        Console.WriteLine($"    {kernel.PadRight(7)} {cpuCell}   {string.Join("   ", gpuCells)}");
    }

    private static int GateLeg(string name, LegResults? results, LegResults cpu)
    {
        if (results is null)
        {
            Console.WriteLine($"\n  {name.PadRight(28)} UNBUILT / UNAVAILABLE (3 kernels failed)");
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