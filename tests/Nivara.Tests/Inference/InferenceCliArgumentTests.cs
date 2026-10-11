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
/// <b>Coverage.</b> Every reject the CLI owns is asserted here: the <c>--seq</c>/<c>--warmup</c>/
/// <c>--iters</c> scope (#474), the <c>--precision</c>/<c>--seed</c>/<c>--teacher-examples</c> values
/// (#479), the <c>--ilgpu</c> model scope (#568), the <c>--teacher-examples</c>/<c>--seed</c>/
/// <c>--force</c> scope (#571), the unknown model and the per-model mode allow-list (#587), the
/// narrow <c>--precision</c> consumed by the selected run (#586), the strict <c>--simd-widen</c>
/// scope (#591), and the <c>--plain</c>/<c>--text</c>/<c>--no-kv-cache</c>/<c>--synthetic-weights</c>
/// scopes (#580). #589 requires each reject to be observable before the first disk touch, so the
/// ordering cases and the <c>qwen</c> happy paths use <c>qwen</c> (its checkpoint is never present:
/// <c>samples/data/qwen2.5-0.5b-instruct/</c> is gitignored and no <c>*.safetensors</c> is tracked)
/// and the mode cases pin the message, not the exit code — a guard that fell through to the
/// model-file check would exit <c>1</c> with a different message.
/// </para>
/// </summary>
[TestFixture]
public class InferenceCliArgumentTests
{
    readonly record struct CliResult(int ExitCode, string StdOut, string StdErr);

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
        AssertRejected(Run("smollm", "--ilgpu"), "--ilgpu is not supported for 'smollm'");

    [Test]
    public void Gpu_RejectedBeforeModelFileLookup()
    {
        CliResult result = Run("qwen", "--ilgpu");
        Assert.That(result.StdErr, Does.Not.Contain(ModelLookupMessage),
            "the --ilgpu guard must run before the model-file check; 'qwen' weights are never present");
        AssertRejected(result, "--ilgpu is not supported for 'qwen'");
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

    [Test]
    public void UnknownModel_Rejected() =>
        AssertRejected(Run("bogus"), "Unknown model type: bogus");

    /// <summary>
    /// The usage path: no model, or <c>-h</c>/<c>--help</c>, prints the usage block to stdout and
    /// exits non-zero. stdout, not stderr — the usage is not an error message.
    /// </summary>
    [Test]
    public void Help_AndEmptyModel_PrintUsageAndExitNonZero()
    {
        Assert.Multiple(() =>
        {
            foreach (string[] args in HelpCases)
            {
                CliResult result = Run(args);
                Assert.That(result.ExitCode, Is.EqualTo(1),
                    $"'{string.Join(' ', args)}' should exit 1");
                Assert.That(result.StdOut, Does.Contain("Usage: NivaraInference"),
                    $"'{string.Join(' ', args)}' should print the usage block");
            }
        });
    }

    /// <summary>
    /// #587, generalised to every model: an unrecognised mode used to fall through to the model's
    /// default runner (the vision models printed "Image not found" and exited 0). The vision message
    /// names the image-path escape hatch; every other model lists its keywords.
    /// </summary>
    [Test]
    public void UnknownMode_RejectedOnEveryModel()
    {
        Assert.Multiple(() =>
        {
            foreach ((string[] args, string expected) in UnknownModeCases)
                AssertRejected(Run(args), expected);
        });
    }

    [Test]
    public void UnknownMode_RejectedBeforeModelFileLookup()
    {
        // 'minilm' weights may be present in a developer workspace but not on CI; the mode guard must
        // fire before the model-file check either way, or the message would differ between the two.
        CliResult result = Run("minilm", "garbag");
        Assert.That(result.StdErr, Does.Not.Contain(ModelLookupMessage));
        AssertRejected(result, "Unsupported mode 'garbag' for 'minilm'");
    }

    /// <summary>
    /// #586: a narrow <c>--precision</c> is only meaningful where the dispatch selects a narrow
    /// runner. The vision models have no bf16/fp16 path, and laya/modernbert <c>compare</c> are
    /// F32-only, so the flag there loaded BF16/Half weights and then ran F32.
    /// </summary>
    [Test]
    public void NarrowPrecision_OnRunThatDoesNotConsumeIt_Rejected()
    {
        Assert.Multiple(() =>
        {
            foreach ((string[] args, string expected) in NarrowPrecisionCases)
                AssertRejected(Run(args), expected);
        });
    }

    [Test]
    public void QwenPrecision_Rejected()
    {
        AssertRejected(Run("qwen", "--precision", "bf16"),
            "bf16 supports only the 'benchmark' and '--synthetic-weights' modes");
        AssertRejected(Run("qwen", "--precision", "fp16"),
            "fp16 is not supported for the qwen mode");
    }

    [Test]
    public void GpuPrecisionCombination_Rejected() =>
        AssertRejected(Run("minilm", "--ilgpu", "--precision", "bf16"), "--ilgpu is F32-only in this phase");

    [Test]
    public void GpuUnsupportedMode_Rejected()
    {
        Assert.Multiple(() =>
        {
            foreach ((string[] args, string expected) in GpuModeCases)
                AssertRejected(Run(args), expected);
        });
    }

    /// <summary>
    /// The flag's migration contract (#616): the pre-Silk name is rejected with the new name named,
    /// not silently aliased, so an old invocation cannot quietly keep running the ILGPU path.
    /// </summary>
    [Test]
    public void GpuRenamedFlag_RejectedWithNewName() =>
        AssertRejected(Run("minilm", "--gpu"), "--gpu was renamed --ilgpu");

    /// <summary>
    /// #591: <c>--simd-widen</c> is an A/B lever only where the run reads the toggle. smollm and qwen
    /// enable widen for their narrow runs themselves, so the flag is inert there; an F32 run is
    /// unchanged; and the vision models have no narrow path at all.
    /// </summary>
    [Test]
    public void SimdWiden_OnRunThatDoesNotReadIt_Rejected()
    {
        const string expected = "does not read the widen kernels";
        AssertRejected(Run("mobilenet_v2", "--simd-widen"), expected);
        AssertRejected(Run("minilm", "--simd-widen"), expected);
        AssertRejected(Run("laya", "--simd-widen"), expected);
        AssertRejected(Run("smollm", "--precision", "bf16", "--simd-widen"), expected);
        AssertRejected(Run("smollm", "ab", "--simd-widen"), expected);
        AssertRejected(Run("qwen", "benchmark", "--precision", "bf16", "--simd-widen"), expected);
    }

    [Test]
    public void SyntheticWeights_OutsideScope_Rejected()
    {
        AssertRejected(Run("minilm", "--synthetic-weights"),
            "is only supported for the qwen benchmark mode");
        AssertRejected(Run("qwen", "distill", "--synthetic-weights"),
            "supports only the 'benchmark' mode");
    }

    [Test]
    public void Text_MissingValue_Rejected() =>
        AssertRejected(Run("qwen", "--text"), "--text needs a value.");

    [Test]
    public void PlainAndPromptFlags_OutsideScope_Rejected()
    {
        AssertRejected(Run("minilm", "--plain"), "is a 'qwen' option only");
        AssertRejected(Run("qwen", "tools", "--plain"), "only valid with the default single-shot run or 'benchmark'");
        AssertRejected(Run("minilm", "--text", "hi"), "is a 'qwen' / 'qwen tools' prompt option only");
        AssertRejected(Run("minilm", "--no-kv-cache"), "is the 'qwen tools' path only");
    }

    /// <summary>
    /// The counterpart to the rejects above: every accepted qwen mode still reaches the weight load.
    /// <c>qwen</c>'s checkpoint is never present, so "Model file not found" is the deepest point this
    /// gate can observe; the point is that no guard fires before it.
    /// </summary>
    [Test]
    public void QwenAcceptedModes_ReachTheModelFileLookup()
    {
        Assert.Multiple(() =>
        {
            foreach (string[] args in AcceptedQwenCases)
            {
                CliResult result = Run(args);
                string label = string.Join(' ', args);
                Assert.That(result.StdErr, Does.Contain(ModelLookupMessage),
                    $"'{label}' should reach the weight load; stderr:{Environment.NewLine}{result.StdErr}");
                Assert.That(result.StdErr, Does.Not.Contain("Unsupported mode"));
                Assert.That(result.StdErr, Does.Not.Contain("not supported for"));
                Assert.That(result.StdErr, Does.Not.Contain("precision error"));
            }
        });
    }

    static readonly (string[] Args, string Expected)[] UnknownModeCases =
    [
        (["qwen", "garbag"], "Unsupported mode 'garbag' for 'qwen'"),
        (["minilm", "garbag"], "Unsupported mode 'garbag' for 'minilm'"),
        (["distilbert", "predict"], "Unsupported mode 'predict' for 'distilbert'"),
        (["distilbert", "compare_diag"], "Unsupported mode 'compare_diag' for 'distilbert'"),
        (["distilbert_sst", "compare_diag"], "Unsupported mode 'compare_diag' for 'distilbert_sst'"),
        (["modernbert", "garbag"], "Unsupported mode 'garbag' for 'modernbert'"),
        (["smollm", "garbag"], "Unsupported mode 'garbag' for 'smollm'"),
        (["laya", "compare_diag"], "Unsupported mode 'compare_diag' for 'laya'"),
        (["laya", "garbag"], "Unsupported mode 'garbag' for 'laya'"),
        (["mobilenet_v2", "garbag"], "Unsupported mode or image 'garbag' for 'mobilenet_v2'"),
        (["resnet18", "garbag"], "Unsupported mode or image 'garbag' for 'resnet18'"),
    ];

    static readonly string[][] HelpCases = [[], ["--help"], ["-h"]];

    static readonly (string[] Args, string Expected)[] NarrowPrecisionCases =
    [
        (["mobilenet_v2", "--precision", "bf16"], "--precision bf16 is not supported for 'mobilenet_v2'"),
        (["mobilenet_v2", "--precision", "fp16", "benchmark"], "--precision fp16 is not supported for 'mobilenet_v2 benchmark'"),
        (["resnet18", "--precision", "bf16", "compare"], "--precision bf16 is not supported for 'resnet18 compare'"),
        (["laya", "compare", "--precision", "bf16"], "--precision bf16 is not supported for 'laya compare'"),
        (["modernbert", "compare", "--precision", "fp16"], "--precision fp16 is not supported for 'modernbert compare'"),
    ];

    static readonly (string[] Args, string Expected)[] GpuModeCases =
    [
        (["minilm", "--ilgpu", "similarity"], "--ilgpu does not support mode 'similarity' for 'minilm'"),
        (["distilbert_sst", "--ilgpu", "predict"], "--ilgpu does not support mode 'predict' for 'distilbert_sst'"),
        (["modernbert", "--ilgpu", "compare_diag"], "--ilgpu does not support mode 'compare_diag' for 'modernbert'"),
    ];

    static readonly string[][] AcceptedQwenCases =
    [
        ["qwen"],
        ["qwen", "tools"],
        ["qwen", "distill"],
        ["qwen", "benchmark"],
        ["qwen", "--precision", "bf16", "benchmark"],
    ];

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
        // synchronously can deadlock. Only stdout/stderr are asserted once the process has exited.
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(ProcessTimeoutMs))
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail($"NivaraInference {string.Join(' ', args)} did not exit within {ProcessTimeoutMs / 1000} s.");
        }

        string stdoutText = stdout.GetAwaiter().GetResult();
        string stderrText = stderr.GetAwaiter().GetResult();
        return new CliResult(process.ExitCode, stdoutText, stderrText);
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
