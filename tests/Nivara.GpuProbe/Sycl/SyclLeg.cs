using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.Sycl;

/// <summary>
/// oneAPI/SYCL leg of the kernel probe (commit-5 sandbox): spawns the DPC++-compiled
/// <c>sycl_runner.exe</c> (built with <c>Sycl/build.cmd</c>) over the same byte-identical
/// BF16 fixtures and gates the f32 results against the production CPU leg (<see cref="CpuLeg"/>).
///
/// Runs the real production toolchain shape llama.cpp's SYCL backend uses (SYCL runtime on
/// Level Zero underneath), so it answers the decisive question the L0 hand-authored leg could
/// not: does IGC compute BF16 FP mul/div correctly on *compiler-produced* bytecode?
/// </summary>
internal static class SyclLeg
{
    /// <summary>Repo-relative path to the SYCL sources folder.</summary>
    private static readonly string SyclSrcDir = Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tests", "Nivara.GpuProbe", "Sycl");

    private static readonly string RunnerExe = Path.Combine(SyclSrcDir, "sycl_runner.exe");
    private static readonly string BuildCmd = Path.Combine(SyclSrcDir, "build.cmd");
    private static readonly string RunCmd = Path.Combine(SyclSrcDir, "run.cmd");

    public static int Run(KernelFixtures fixtures)
    {
        if (!File.Exists(RunnerExe))
        {
            Console.WriteLine("[sycl] sycl_runner.exe not found. Build it first:");
            Console.WriteLine($"       {BuildCmd}");
            return 1;
        }

        Console.WriteLine();
        Console.WriteLine("--- oneAPI / SYCL leg (sycl_runner.exe over production BF16 fixtures) ---");
        Console.WriteLine($"  runner: {RunnerExe}");
        Console.WriteLine($"  launcher: {RunCmd} (sources Intel oneAPI setvars so the SYCL runtime DLLs resolve)");

        int failures = 0;
        failures += GateDot16(fixtures);
        failures += GateSilu(fixtures);
        failures += GateGemv(fixtures);
        return failures;
    }

    private static int GateDot16(KernelFixtures fixtures)
    {
        string dir = SyclTempDir();
        string inFile = Path.Combine(dir, "dot16_in.bin");
        string outFile = Path.Combine(dir, "dot16_out.bin");
        File.WriteAllBytes(inFile, WriteBf16Concat(fixtures.Dot16A, fixtures.Dot16B));

        var (rc, stdout, stderr, results) = RunRunner("dot16", inFile, outFile, 0, 0);
        if (rc != 0)
        {
            Console.WriteLine($"  [dot16] runner failed (exit {rc}): {stderr.Trim()}");
            return 1;
        }

        float cpu = CpuLeg.Dot(fixtures.Dot16A, fixtures.Dot16B);
        float gpu = results[0];
        bool pass = CpuLeg.WithinTolerance(gpu, cpu);
        Console.WriteLine($"  [dot16] {gpu:G9} vs CPU {cpu:G9} -> {(pass ? "PASS" : "FAIL")} ({CpuLeg.UlpDistance(gpu, cpu):F1} ULP)");
        Console.WriteLine($"          {stdout.Trim()}");
        return pass ? 0 : 1;
    }

    private static int GateSilu(KernelFixtures fixtures)
    {
        string dir = SyclTempDir();
        string inFile = Path.Combine(dir, "silu_in.bin");
        string outFile = Path.Combine(dir, "silu_out.bin");
        File.WriteAllBytes(inFile, WriteBf16Concat(fixtures.SiluX));

        var (rc, stdout, stderr, results) = RunRunner("silu", inFile, outFile, KernelFixtures.HiddenSize, 0);
        if (rc != 0)
        {
            Console.WriteLine($"  [silu] runner failed (exit {rc}): {stderr.Trim()}");
            return 1;
        }

        float[] cpu = CpuLeg.Silu(fixtures.SiluX);
        int passed = 0, failed = 0;
        double worstUlp = 0;
        for (int i = 0; i < results.Length; i++)
        {
            double ulp = CpuLeg.UlpDistance(results[i], cpu[i]);
            if (ulp > worstUlp)
                worstUlp = ulp;
            if (CpuLeg.WithinTolerance(results[i], cpu[i]))
                passed++;
            else
                failed++;
        }
        Console.WriteLine($"  [silu] {passed}/{results.Length} within the gate vs CPU ({failed} failed), worst {worstUlp:F1} ULP");
        Console.WriteLine($"        {stdout.Trim()}");
        return failed == 0 ? 0 : 1;
    }

    private static int GateGemv(KernelFixtures fixtures)
    {
        string dir = SyclTempDir();
        string inFile = Path.Combine(dir, "gemv_in.bin");
        string outFile = Path.Combine(dir, "gemv_out.bin");
        File.WriteAllBytes(inFile, WriteBf16Concat(fixtures.GemvW, fixtures.GemvX));

        // KernelFixtures.GemvW is [intermediate x hidden] row-major; the runner expects
        // W[rows x cols] then x[cols] with rows=m, cols=k.
        var (rc, stdout, stderr, results) = RunRunner("gemv", inFile, outFile, KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize);
        if (rc != 0)
        {
            Console.WriteLine($"  [gemv] runner failed (exit {rc}): {stderr.Trim()}");
            return 1;
        }

        float[] cpu = CpuLeg.Gemv(fixtures.GemvX, fixtures.GemvW, KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize);
        int passed = 0, failed = 0;
        double worstUlp = 0;
        int worstRow = -1;
        float worstLeg = 0f, worstCpu = 0f;
        for (int i = 0; i < results.Length; i++)
        {
            double ulp = CpuLeg.UlpDistance(results[i], cpu[i]);
            if (ulp > worstUlp)
            {
                worstUlp = ulp;
                worstRow = i;
                worstLeg = results[i];
                worstCpu = cpu[i];
            }
            if (CpuLeg.WithinTolerance(results[i], cpu[i]))
                passed++;
            else
                failed++;
        }
        double worstGate = CpuLeg.GateAbs + CpuLeg.GateRel * Math.Abs(worstCpu);
        Console.WriteLine($"  [gemv] {passed}/{results.Length} rows within the gate vs CPU ({failed} failed), worst {worstUlp:F1} ULP");
        Console.WriteLine($"         worst row {worstRow}: leg {worstLeg:G9} vs CPU {worstCpu:G9} (gate {worstGate:G3}, |diff| {Math.Abs(worstLeg - worstCpu):G3})");
        Console.WriteLine($"        {stdout.Trim()}");
        return failed == 0 ? 0 : 1;
    }

    private static (int exitCode, string stdout, string stderr, float[] results) RunRunner(string mode, string inFile, string outFile, int dim0, int dim1)
    {
        // Launch through run.cmd: the SYCL runtime DLLs are only on PATH after oneAPI setvars,
        // and a child spawned directly from the .NET process would fail with STATUS_DLL_NOT_FOUND.
        var psi = new ProcessStartInfo("cmd.exe")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        string args = $" /d /c \"\"{RunCmd}\" {mode} \"{inFile}\" \"{outFile}\"";
        if (dim0 > 0) args += $" {dim0}";
        if (dim1 > 0) args += $" {dim1}";
        psi.Arguments = args + "\"";

        using var process = Process.Start(psi)!;
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode == 0)
        {
            byte[] bytes = File.ReadAllBytes(outFile);
            var results = new float[bytes.Length / 4];
            for (int i = 0; i < results.Length; i++)
                results[i] = BitConverter.ToSingle(bytes, i * 4);
            return (0, stdout, stderr, results);
        }

        return ((int)process.ExitCode, stdout, stderr, Array.Empty<float>());
    }

    private static string SyclTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "opencode", "sycl");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static byte[] WriteBf16Concat(params BFloat16[][] arrays)
    {
        var bytes = new List<byte>();
        foreach (var array in arrays)
            bytes.AddRange(MemoryMarshal.AsBytes(array.AsSpan()).ToArray());
        return bytes.ToArray();
    }
}