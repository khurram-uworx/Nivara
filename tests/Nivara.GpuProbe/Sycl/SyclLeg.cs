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

        var dot16 = RunKernel("dot16", WriteBf16Concat(fixtures.Dot16A, fixtures.Dot16B), KernelFixtures.Dot16Length);
        var silu = RunKernel("silu", WriteBf16Concat(fixtures.SiluX), KernelFixtures.HiddenSize);
        var gemv = RunKernel("gemv", WriteBf16Concat(fixtures.GemvW, fixtures.GemvX), KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize);
        if (dot16 is null || silu is null || gemv is null)
            return null;

        Console.WriteLine(dot16.Value.stdout.Trim());  // device line + dot16 value from the runner

        return new LegResults(dot16.Value.results[0], silu.Value.results, gemv.Value.results,
            dot16.Value.timeUs, silu.Value.timeUs, gemv.Value.timeUs);
    }

    /// <summary>Runs one kernel through the runner; returns results + kernel-only time (µs), or null on failure.</summary>
    private static (float[] results, double timeUs, string stdout)? RunKernel(string mode, byte[] input, int dim0, int dim1 = 0)
    {
        string dir = Path.Combine(Path.GetTempPath(), "opencode", "sycl");
        Directory.CreateDirectory(dir);
        string inFile = Path.Combine(dir, $"{mode}_in.bin");
        string outFile = Path.Combine(dir, $"{mode}_out.bin");
        File.WriteAllBytes(inFile, input);

        var (rc, so, se, results) = Spawn(mode, inFile, outFile, dim0, dim1);
        if (rc != 0)
        {
            Console.WriteLine($"  [{mode}] runner failed (exit {rc}): {se.Trim()}");
            return null;
        }
        return (results, ParseTimeUs(so, mode), so);
    }

    /// <summary>Extracts the kernel-only time from the runner's "TIME &lt;mode&gt; = X us" stdout line.</summary>
    private static double ParseTimeUs(string stdout, string mode)
    {
        foreach (string line in stdout.Split('\n'))
        {
            string t = line.Trim();
            if (!t.StartsWith($"TIME {mode} =", StringComparison.Ordinal))
                continue;
            int eq = t.IndexOf('=', StringComparison.Ordinal);
            if (eq > 0 && double.TryParse(t[(eq + 1)..].Trim().Split(' ')[0],
                    System.Globalization.CultureInfo.InvariantCulture, out double us))
                return us;
        }
        return 0;
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