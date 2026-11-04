using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.Samples;

/// <summary>
/// Gate for the argument surface of <c>NivaraClassifier --benchmark</c> (#901). Spawns the built
/// sample once per rejected arg set and asserts the guard's observable contract: exit code 2 and the
/// message on stderr. Asserting the message — not only the exit code — is what makes the gate real.
/// No wall-clock assertions, so no <c>Performance</c> category.
/// </summary>
[TestFixture]
public class ClassifierCliArgumentTests
{
    readonly record struct CliResult(int ExitCode, string StdOut, string StdErr);

    static int ProcessTimeoutMs = 30_000;
    static readonly string DllPath = Path.Combine(AppContext.BaseDirectory, "NivaraClassifier.dll");

    static readonly string CliWorkingDirectory = CreateEmptyWorkingDirectory();

    [OneTimeSetUp]
    public void TheSampleMustBeBuilt()
    {
        Assert.That(File.Exists(DllPath), Is.True,
            $"NivaraClassifier.dll was not found at {DllPath}. Build the solution before testing; "
            + "this project references samples/NivaraClassifier so 'dotnet build' produces it.");

        int canaryMs = MeasureCanary();
        int calibrated = Math.Max(30_000, canaryMs * 10);
        if (int.TryParse(Environment.GetEnvironmentVariable("NIVARA_CLI_TIMEOUT_MS"), out int overrideMs)
            && overrideMs > 0)
            calibrated = Math.Max(calibrated, overrideMs);
        ProcessTimeoutMs = calibrated;
    }

    [OneTimeTearDown]
    public void RemoveCliWorkingDirectory() => Directory.Delete(CliWorkingDirectory, recursive: true);

    [Test]
    public void Silk_Generate_Rejected()
    {
        var result = Run("--silk", "--command", "generate");
        Assert.That(result.ExitCode, Is.EqualTo(2),
            $"expected exit code 2, got {result.ExitCode}. stderr:{Environment.NewLine}{result.StdErr}");
        Assert.That(result.StdErr, Does.Contain("--silk is supported only with --command train, predict or benchmark."),
            $"stderr did not contain the guard message. stderr:{Environment.NewLine}{result.StdErr}");
    }

    [Test]
    public void Benchmark_LoadOnnx_Rejected()
    {
        var result = Run("--benchmark", "--load-onnx", "x");
        Assert.That(result.ExitCode, Is.EqualTo(2),
            $"expected exit code 2, got {result.ExitCode}. stderr:{Environment.NewLine}{result.StdErr}");
        Assert.That(result.StdErr, Does.Contain("--benchmark does not use --load-onnx"),
            $"stderr did not contain the guard message. stderr:{Environment.NewLine}{result.StdErr}");
    }

    static int MeasureCanary()
    {
        var startInfo = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = CliWorkingDirectory,
        };
        startInfo.ArgumentList.Add(DllPath);
        startInfo.ArgumentList.Add("--help");

        var stopwatch = Stopwatch.StartNew();
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start 'dotnet' for the canary.");
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(60_000))
        {
            process.Kill(entireProcessTree: true);
            throw new InvalidOperationException("NivaraClassifier canary did not exit within 60 s.");
        }
        stopwatch.Stop();
        _ = stdout.GetAwaiter().GetResult();
        _ = stderr.GetAwaiter().GetResult();
        return (int)stopwatch.ElapsedMilliseconds;
    }

    static CliResult Run(params string[] args)
    {
        var startInfo = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = CliWorkingDirectory,
        };
        startInfo.ArgumentList.Add(DllPath);
        foreach (string arg in args)
            startInfo.ArgumentList.Add(arg);

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start 'dotnet'.");

        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(ProcessTimeoutMs))
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail($"NivaraClassifier {string.Join(' ', args)} did not exit within {ProcessTimeoutMs / 1000} s.");
        }

        string stdoutText = stdout.GetAwaiter().GetResult();
        string stderrText = stderr.GetAwaiter().GetResult();
        return new CliResult(process.ExitCode, stdoutText, stderrText);
    }

    static string CreateEmptyWorkingDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "nivara-classifier-cli-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}