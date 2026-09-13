using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.Sycl;

/// <summary>
/// oneAPI/SYCL leg of the kernel probe: spawns the DPC++-compiled <c>sycl_runner.exe</c>
/// (built with <c>Sycl/build.cmd</c>) over the same byte-identical BF16 fixtures and returns
/// the f32 kernel results. Gating happens in <see cref="KernelGate"/> against the production
/// CPU leg — this type is transport only.
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

    /// <summary>
    /// Runs all three kernels via the SYCL runner and returns the results, or <c>null</c>
    /// when the runner is unavailable or a launch fails (exit code reported on stderr).
    /// </summary>
    public static LegResults? RunLeg(KernelFixtures fixtures)
    {
        if (!File.Exists(RunnerExe))
        {
            Console.WriteLine("[sycl] sycl_runner.exe not found. Build it first:");
            Console.WriteLine($"       {BuildCmd}");
            return null;
        }

        Console.WriteLine($"  launcher: {RunCmd} (sources Intel oneAPI setvars so the SYCL runtime DLLs resolve)");

        var dot16 = RunKernel("dot16", WriteBf16Concat(fixtures.Dot16A, fixtures.Dot16B), out var dot16Out, KernelFixtures.Dot16Length);
        var silu = RunKernel("silu", WriteBf16Concat(fixtures.SiluX), out var siluOut, KernelFixtures.HiddenSize);
        var gemv = RunKernel("gemv", WriteBf16Concat(fixtures.GemvW, fixtures.GemvX), out var gemvOut,
            KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize);
        if (dot16 is null || silu is null || gemv is null)
            return null;

        Console.WriteLine(dot16Out.Trim());  // device line + dot16 value from the runner

        return new LegResults(dot16[0], silu, gemv);
    }

    /// <summary>Runs one kernel through the runner; returns the f32 results or null on failure.</summary>
    private static float[]? RunKernel(string mode, byte[] input, out string stdout, int dim0, int dim1 = 0)
    {
        string dir = Path.Combine(Path.GetTempPath(), "opencode", "sycl");
        Directory.CreateDirectory(dir);
        string inFile = Path.Combine(dir, $"{mode}_in.bin");
        string outFile = Path.Combine(dir, $"{mode}_out.bin");
        File.WriteAllBytes(inFile, input);

        var (rc, so, se, results) = Spawn(mode, inFile, outFile, dim0, dim1);
        stdout = so;
        if (rc != 0)
        {
            Console.WriteLine($"  [{mode}] runner failed (exit {rc}): {se.Trim()}");
            return null;
        }
        return results;
    }

    private static (int exitCode, string stdout, string stderr, float[] results) Spawn(string mode, string inFile, string outFile, int dim0, int dim1)
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

    private static byte[] WriteBf16Concat(params BFloat16[][] arrays)
    {
        var bytes = new List<byte>();
        foreach (var array in arrays)
            bytes.AddRange(MemoryMarshal.AsBytes(array.AsSpan()).ToArray());
        return bytes.ToArray();
    }
}