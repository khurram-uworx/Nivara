namespace Nivara.GpuProbe.Kernels;

/// <summary>
/// The multi-leg correctness gate. Every leg runs the same byte-identical BF16 fixtures and
/// produces a <see cref="LegResults"/>; the CPU leg (the production Nivara kernels, see
/// <see cref="CpuLeg"/>) is the gold target, and every GPU leg is gated against it with
/// <c>|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|</c>. Per-kernel rows report the worst ULP
/// at the reference's magnitude (a *diagnostic*, not the gate) and the pass count. Exit code
/// = number of failed kernels across every leg (0 = all gates pass).
///
/// <para>
/// Timing is the repeated same-host profile (#552 Arc 140T, #553 Iris Xe): each leg's own
/// best-of-<c>N</c> steady-state sample is taken, but the whole leg set is swept
/// <see cref="Rounds"/> times with a rotating leg order, so the decision table reports the
/// run-to-run <b>median</b> and <b>min–max band</b> per cell rather than a single sample.
/// Rotating the start point keeps any one leg from being systematically first or last. The
/// per-leg internal best-of-N is still the gauge for a single sample; the median and band are
/// the cross-run statistic the two host issues exist to record.
/// </para>
/// <para>
/// A leg may declare an <see cref="KernelDiagnosis"/> for one of its kernels: a
/// structural gate failure that is an expected, evidenced precision finding rather
/// than a miscode. Failing elements are then judged against a reference derived from
/// the same fixtures as the gold (never from the leg's own output), and any element
/// that matches it within the diagnosis's half-ULP tolerance is counted as
/// <em>diagnosed</em> rather than <em>unexpected</em>. The exit code is the count of
/// unexpected failures; diagnosed-honest ones are reported, not counted (the
/// OpenVINO BF16-declared silu, resolved to F16 execution by the GPU plugin, is the
/// first such row — #606).
/// </para>
/// </summary>
internal static class KernelGate
{
    /// <summary>Sweeps every leg this many times (<c>#552</c>/<c>#553</c> require ≥5) with a
    /// rotating leg order, so per-cell timing can be reported as a median + min–max band.</summary>
    private const int Rounds = 5;

    /// <summary>Single-leg convenience overload (unchanged single-sample behaviour).</summary>
    public static int Run(KernelFixtures fixtures, string gpuLegName, Func<KernelFixtures, LegResults?> gpuLeg)
        => Run(fixtures, (gpuLegName, gpuLeg, (KernelDiagnosis?)null));

    /// <summary>
    /// Runs the CPU reference + every GPU leg <see cref="Rounds"/> times with a rotating order,
    /// gates correctness once (round 0), then prints the correctness table and the median +
    /// min–max timing table.
    /// </summary>
    public static int Run(KernelFixtures fixtures,
        params (string Name, Func<KernelFixtures, LegResults?> Leg, KernelDiagnosis? Diagnosis)[] legs)
    {
        Console.WriteLine();
        Console.WriteLine("--- kernel gates vs CPU (production Nivara kernels) ---");

        // Warm up the production kernels once: the first call pays JIT compilation
        // cost and would dominate the timing rows. Discard it; time every round after.
        _ = CpuLeg.ComputeLeg(fixtures);

        // samples[kernel][leg] = every per-round steady-state sample for that cell.
        var samples = new Dictionary<string, Dictionary<string, List<double>>>(StringComparer.Ordinal)
        {
            ["dot16"] = new(StringComparer.Ordinal) { ["CPU (production Nivara)"] = [] },
            ["silu"] = new(StringComparer.Ordinal) { ["CPU (production Nivara)"] = [] },
            ["gemv"] = new(StringComparer.Ordinal) { ["CPU (production Nivara)"] = [] },
        };
        foreach ((string name, _, _) in legs)
            foreach (Dictionary<string, List<double>> byKernel in samples.Values)
                byKernel[name] = [];

        int failures = 0, diagnosed = 0;
        LegResults? cpuReference = null;

        for (int round = 0; round < Rounds; round++)
        {
            Console.WriteLine();
            Console.WriteLine($"--- round {round + 1}/{Rounds} (leg order rotates so no leg is systematically first/last) ---");

            LegResults cpu = CpuLeg.ComputeLeg(fixtures);
            cpuReference ??= cpu;
            if (round == 0)
                failures += Gate(cpu, cpu, "CPU (production Nivara)", null, null, out _);
            RecordSample(samples, cpu, "CPU (production Nivara)");

            foreach ((string name, Func<KernelFixtures, LegResults?> leg, KernelDiagnosis? diagnosis) in Rotate(legs, round))
            {
                LegResults? results = leg(fixtures);
                if (round == 0) // correctness is deterministic; gate once, not every sweep
                {
                    ushort[]? diagRef = diagnosis?.ReferenceFactory(fixtures);
                    failures += GateLeg(name, results, cpuReference, diagnosis, diagRef, out int diagnosedHere);
                    diagnosed += diagnosedHere;
                }
                if (results is not null)
                    RecordSample(samples, results, name);
            }
        }

        PrintTimingTable(samples, legs);

        Console.WriteLine();
        Console.WriteLine($"  kernels mode exit: {failures} unexpected failed kernel(s), {diagnosed} diagnosed honest, {Rounds} sweeps");
        return failures;
    }

    private static void RecordSample(
        Dictionary<string, Dictionary<string, List<double>>> samples, LegResults results, string leg)
    {
        samples["dot16"][leg].Add(results.Dot16Us);
        samples["silu"][leg].Add(results.SiluUs);
        samples["gemv"][leg].Add(results.GemvUs);
    }

    /// <summary>Rotates the leg list by <paramref name="offset"/> so no leg keeps the same position.</summary>
    private static IEnumerable<(string Name, Func<KernelFixtures, LegResults?> Leg, KernelDiagnosis? Diagnosis)> Rotate(
        (string Name, Func<KernelFixtures, LegResults?> Leg, KernelDiagnosis? Diagnosis)[] legs, int offset)
    {
        int n = legs.Length;
        for (int i = 0; i < n; i++)
            yield return legs[(i + offset) % n];
    }

    /// <summary>
    /// Prints the per-kernel decision table: each leg's median steady-state wall time and its
    /// min–max band over all rounds. Cells that never produced a sample (unavailable on this
    /// host) are marked so — absent is a different fact from slow.
    /// </summary>
    private static void PrintTimingTable(
        Dictionary<string, Dictionary<string, List<double>>> samples,
        (string Name, Func<KernelFixtures, LegResults?> Leg, KernelDiagnosis? Diagnosis)[] legs)
    {
        Console.WriteLine();
        Console.WriteLine("  kernel timings over all sweeps (µs per invocation, kernel-only; median[min–max]):");
        foreach (string kernel in new[] { "dot16", "silu", "gemv" })
        {
            Console.WriteLine();
            Console.WriteLine($"    [{kernel}]");
            Console.WriteLine($"      {"CPU (production Nivara)",-26} {Cell(samples[kernel]["CPU (production Nivara)"])}");
            foreach ((string name, _, _) in legs)
                Console.WriteLine($"      {name,-26} {Cell(samples[kernel][name])}");
        }
    }

    private static string Cell(List<double> values)
    {
        if (values.Count == 0)
            return "unavailable on this host";
        var ordered = values.OrderBy(v => v).ToArray();
        double median = ordered.Length % 2 == 1
            ? ordered[ordered.Length / 2]
            : 0.5 * (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]);
        return $"{median,9:F1} [{ordered[0]:F1}–{ordered[^1]:F1}]  (n={values.Count})";
    }

    private static int GateLeg(string name, LegResults? results, LegResults cpu,
        KernelDiagnosis? diagnosis, ushort[]? diagRef, out int diagnosed)
    {
        if (results is null)
        {
            diagnosed = 0;
            Console.WriteLine($"\n  {name.PadRight(28)} UNBUILT / UNAVAILABLE (3 kernels failed)");
            return 3;
        }

        Console.WriteLine();
        return Gate(cpu, results, name, diagnosis, diagRef, out diagnosed);
    }

    private static int Gate(LegResults reference, LegResults leg, string name,
        KernelDiagnosis? diagnosis, ushort[]? diagRef, out int diagnosed)
    {
        if (ReferenceEquals(reference, leg))
        {
            // Identity row: the CPU leg against itself (0 ULP sanity check of the harness).
            diagnosed = 0;
            Console.WriteLine($"  {name.PadRight(28)} dot16 0.0 ULP | silu 0/0 failed | gemv 0/0 failed");
            return 0;
        }

        int failedDots = GateDot16(reference, leg, name);
        int failedSilu = GateVector(reference.Silu, leg.Silu, "silu", name, diagnosis, diagRef, out diagnosed);
        int failedGemv = GateVector(reference.Gemv, leg.Gemv, "gemv", name, null, null, out _);
        return failedDots + failedSilu + failedGemv;
    }

    private static int GateDot16(LegResults reference, LegResults leg, string name)
    {
        bool pass = CpuLeg.WithinTolerance(leg.Dot16, reference.Dot16);
        Console.WriteLine($"  {name.PadRight(28)} dot16 {leg.Dot16:G9} vs CPU {reference.Dot16:G9} -> {(pass ? "PASS" : "FAIL")} ({CpuLeg.UlpDistance(leg.Dot16, reference.Dot16):F1} ULP)");
        return pass ? 0 : 1;
    }

    /// <summary>
    /// Gates one kernel row and returns the count of <em>unexpected</em> failures
    /// (failures minus those diagnosed by <paramref name="diagnosis"/>). When a
    /// diagnosis applies, every element is also matched against the diagnosis's
    /// reference in the half lattice, and the worst half-ULP is reported so the
    /// "honest" classification is checkable in the log, not asserted in prose.
    /// </summary>
    private static int GateVector(float[] reference, float[] leg, string kernel, string name,
        KernelDiagnosis? diagnosis, ushort[]? diagRef, out int diagnosedOut)
    {
        int passed = 0, failed = 0, diagnosed = 0;
        double worstUlp = 0;
        int worstIndex = -1;
        float worstLeg = 0f, worstRef = 0f;
        int worstHalfUlp = 0, worstHalfIndex = -1;
        bool applies = diagnosis is not null && diagRef is not null
            && string.Equals(diagnosis.Kernel, kernel, StringComparison.Ordinal);

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
            {
                passed++;
            }
            else
            {
                failed++;
                if (applies)
                {
                    int halfUlp = CpuLeg.HalfUlpDistance(
                        BitConverter.HalfToUInt16Bits((Half)leg[i]), diagRef![i]);
                    if (halfUlp > worstHalfUlp)
                    {
                        worstHalfUlp = halfUlp;
                        worstHalfIndex = i;
                    }
                    if (halfUlp <= diagnosis!.MaxHalfUlp)
                        diagnosed++;
                }
            }
        }
        if (worstIndex >= 0)
        {
            double worstGate = CpuLeg.GateAbs + CpuLeg.GateRel * Math.Abs(worstRef);
            string diagnosisNote = diagnosed > 0 ? $", {diagnosed} diagnosed-honest" : "";
            Console.WriteLine($"  {name.PadRight(28)} {kernel} {passed}/{leg.Length} within gate ({failed} failed{diagnosisNote}), worst {worstUlp:F1} ULP @ index {worstIndex} (leg {worstLeg:G9} vs CPU {worstRef:G9}, gate {worstGate:G3}, |diff| {Math.Abs(worstLeg - worstRef):G3})");
            if (diagnosed > 0)
                Console.WriteLine($"  {name.PadRight(28)} {kernel} F16-mirror: worst {worstHalfUlp} half-ULP @ index {worstHalfIndex} ({diagnosis?.Reason})");
        }
        else
        {
            Console.WriteLine($"  {name.PadRight(28)} {kernel} {passed}/{leg.Length} within gate ({failed} failed)");
        }

        diagnosedOut = diagnosed;
        return failed - diagnosed;
    }
}

/// <summary>
/// A declared honest gate failure: <see cref="Kernel"/>'s failures are counted as
/// <em>diagnosed</em> (not unexpected) when every failing element also matches the
/// reference produced by <see cref="ReferenceFactory"/> within
/// <see cref="MaxHalfUlp"/> half-ULPs. The reference is derived from the same
/// fixtures as the gold — never from the leg's own output — so a leg cannot
/// self-certify: the classification is re-measured every gate run.
/// </summary>
internal sealed record KernelDiagnosis(
    string Kernel,
    string Reason,
    Func<KernelFixtures, ushort[]> ReferenceFactory,
    int MaxHalfUlp)
{
    /// <summary>
    /// The OpenVINO BF16-declared silu: the IR declares Sigmoid+Multiply in BF16,
    /// the GPU plugin resolves execution to F16 (the read-back
    /// <c>INFERENCE_PRECISION_HINT</c> is <c>f16</c>; BF16 is a CPU-only token on
    /// this release), so every elementwise failure against the f32 gold is an
    /// F16-rounded value of the same formula. See issue #606.
    /// </summary>
    public static KernelDiagnosis OvBf16Silu() => new(
        Kernel: "silu",
        Reason: "F16 elementwise sigmoid+multiply on a BF16-declared model (GPU plugin resolves BF16 to F16)",
        ReferenceFactory: fixtures => CpuLeg.SiluF16ReferenceBits(fixtures.SiluX),
        MaxHalfUlp: 4);
}
