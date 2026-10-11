using System.Diagnostics;
using NUnit.Framework;

namespace Nivara.Tests.Inference;

/// <summary>
/// Gate for the argument surface of <c>NivaraInference</c> (#581). It spawns the built sample
/// (<c>dotnet NivaraInference.dll</c>) once per rejected arg set and asserts the guard's observable
/// contract: exit code <c>1</c> and the message on stderr. Asserting the message — not only the exit
/// code — is what makes the gate real: a removed guard falls through to a different exit-1 path
/// (unknown model, or the model-file check), so the code alone would still be <c>1</c>.
/// <para>
/// <b>Coverage.</b> The <c>--seq</c>/<c>--warmup</c>/<c>--iters</c> scope rejects (#474), the
/// <c>--precision</c>/<c>--seed</c>/<c>--teacher-examples</c> value rejects (#479), the <c>--gpu</c>
/// model-scope reject (#568), the <c>--teacher-examples</c>/<c>--seed</c>/<c>--force</c> scope rejects
/// (#571), the guard-before-disk ordering, and the <c>qwen distill</c> happy path where all three
/// flags are accepted. The ordering and happy-path cases use <c>qwen</c> because its checkpoint is
/// never present (<c>samples/data/qwen2.5-0.5b-instruct/</c> is gitignored and no <c>*.safetensors</c>
/// is tracked), so the guard runs before the first disk touch in every environment.
/// </para>
/// </summary>
[TestFixture]
public class InferenceCliArgumentTests
{
    readonly record struct CliResult(int ExitCode, string StdErr);

    const int ProcessTimeoutMs = 30_000;
    const string ModelLookupMessage = "Model file not found";

    static readonly string DllPath = Path.Combine(AppContext.BaseDirectory, "NivaraInference.dll");
    static readonly string RepoRoot = FindRepoRoot();

    [OneTimeSetUp]
    public void TheSampleMustBeBuilt()
    {
        Assert.That(File.Exists(DllPath), Is.True,
            $"NivaraInference.dll was not found at {DllPath}. Build the solution before testing; "
            + "this project references samples/NivaraInference so 'dotnet build' produces it.");
    }

    [Test]
    public void SeqWarmupIters_OnNonModernBertModel_Rejected()
    {
        const string expected = "are options of 'modernbert benchmark' only";
        AssertRejected(Run("minilm", "benchmark", "--seq", "128"), expected);
        AssertRejected(Run("minilm", "benchmark", "--warmup", "2"), expected);
        AssertRejected(Run("minilm", "benchmark", "--iters", "2"), expected);
    }

    [Test]
    public void SeqWarmupIters_OnModernBertWithoutBenchmarkMode_Rejected()
    {
        const string expected = "need the 'benchmark' mode; got 'modernbert'";
        AssertRejected(Run("modernbert", "--seq", "128"), expected);
        AssertRejected(Run("modernbert", "--warmup", "2"), expected);
        AssertRejected(Run("modernbert", "--iters", "2"), expected);
    }

    [Test]
    public void Seq_NonPositiveLength_Rejected() =>
        AssertRejected(Run("modernbert", "benchmark", "--seq", "0"), "--seq expects positive integers");

    [Test]
    public void Seq_OverCpuDenseMaskLimit_Rejected() =>
        AssertRejected(Run("modernbert", "benchmark", "--seq", "4096"),
            "exceeds the CPU path's dense-mask limit");

    [Test]
    public void Seq_MissingValue_Rejected() =>
        AssertRejected(Run("modernbert", "benchmark", "--seq"), "--seq needs a value.");

    [Test]
    public void Warmup_InvalidValue_Rejected()
    {
        const string expected = "--warmup expects a positive integer";
        AssertRejected(Run("modernbert", "benchmark", "--warmup", "0"), expected);
        AssertRejected(Run("modernbert", "benchmark", "--warmup", "x"), expected);
    }

    [Test]
    public void Iters_InvalidValue_Rejected()
    {
        const string expected = "--iters expects a positive integer";
        AssertRejected(Run("modernbert", "benchmark", "--iters", "0"), expected);
        AssertRejected(Run("modernbert", "benchmark", "--iters", "x"), expected);
    }

    [Test]
    public void Gpu_OnModelWithoutGpuPath_Rejected() =>
        AssertRejected(Run("smollm", "--gpu"), "--gpu is not supported for 'smollm'");

    [Test]
    public void Gpu_RejectedBeforeModelFileLookup()
    {
        CliResult result = Run("qwen", "--gpu");
        Assert.That(result.StdErr, Does.Not.Contain(ModelLookupMessage),
            "the --gpu guard must run before the model-file check; 'qwen' weights are never present");
        AssertRejected(result, "--gpu is not supported for 'qwen'");
    }

    [Test]
    public void Precision_UnparseableValue_Rejected() =>
        AssertRejected(Run("minilm", "--precision", "foo"),
            "--precision expects f32, bf16 or fp16; got 'foo'.");

    [Test]
    public void Precision_MissingValue_Rejected() =>
        AssertRejected(Run("minilm", "--precision"), "--precision needs a value");

    [Test]
    public void Seed_UnparseableValue_Rejected() =>
        AssertRejected(Run("qwen", "distill", "--seed", "abc"), "--seed expects an integer; got 'abc'.");

    [Test]
    public void TeacherExamples_UnparseableValue_Rejected() =>
        AssertRejected(Run("qwen", "distill", "--teacher-examples", "abc"),
            "--teacher-examples expects an integer; got 'abc'.");

    [Test]
    public void Seed_MissingValue_Rejected() =>
        AssertRejected(Run("qwen", "distill", "--seed"), "--seed needs a value.");

    [Test]
    public void TeacherExamplesSeedForce_OnNonQwenModel_Rejected()
    {
        const string expected = "are options of 'qwen distill' only; no other model takes them.";
        AssertRejected(Run("minilm", "--seed", "5"), expected);
        AssertRejected(Run("minilm", "--teacher-examples", "3"), expected);
        AssertRejected(Run("minilm", "--force"), expected);
    }

    [Test]
    public void TeacherExamplesSeedForce_OnQwenWithoutDistillMode_Rejected()
    {
        const string expected = "need the 'distill' mode; got 'qwen'";
        AssertRejected(Run("qwen", "--seed", "5"), expected);
        AssertRejected(Run("qwen", "--teacher-examples", "3"), expected);
        AssertRejected(Run("qwen", "--force"), expected);
    }

    [Test]
    public void QwenDistill_WithAllThreeFlags_PassesValidation()
    {
        CliResult result = Run("qwen", "distill", "--seed", "7", "--teacher-examples", "3", "--force");

        // Validation passed: the run reached the weight load, which is the deepest point observable
        // without a real Qwen checkpoint (distill determinism is #577).
        Assert.That(result.StdErr, Does.Contain(ModelLookupMessage));
        Assert.That(result.StdErr, Does.Not.Contain("are options of 'qwen distill' only"));
        Assert.That(result.StdErr, Does.Not.Contain("need the 'distill' mode"));
        Assert.That(result.StdErr, Does.Not.Contain("expects an integer"));
        Assert.That(result.StdErr, Does.Not.Contain("needs a value"));
    }

    static void AssertRejected(CliResult result, string message)
    {
        Assert.That(result.ExitCode, Is.EqualTo(1),
            $"expected exit code 1, got {result.ExitCode}. stderr:{Environment.NewLine}{result.StdErr}");
        Assert.That(result.StdErr, Does.Contain(message),
            $"stderr did not contain the guard message. stderr:{Environment.NewLine}{result.StdErr}");
    }

    static CliResult Run(params string[] args)
    {
        var startInfo = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = RepoRoot,
        };
        startInfo.ArgumentList.Add(DllPath);
        foreach (string arg in args)
            startInfo.ArgumentList.Add(arg);

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start 'dotnet'.");

        // Drain both pipes asynchronously: the process-API docs warn that reading the two streams
        // synchronously can deadlock. Only stderr is asserted; stdout is drained so it cannot block.
        Task stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(ProcessTimeoutMs))
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail($"NivaraInference {string.Join(' ', args)} did not exit within {ProcessTimeoutMs / 1000} s.");
        }

        stdout.GetAwaiter().GetResult();
        return new CliResult(process.ExitCode, stderr.GetAwaiter().GetResult());
    }

    static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Nivara.slnx")))
                return dir.FullName;
        }

        throw new InvalidOperationException($"Nivara.slnx not found above {AppContext.BaseDirectory}.");
    }
}
