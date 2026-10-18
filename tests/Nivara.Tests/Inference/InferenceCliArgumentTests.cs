using NUnit.Framework;
using System.Diagnostics;

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
/// scopes (#580), plus the <c>typesafe</c> surface (mode allow-list, GPU scope, F32-only precision,
/// and the <c>--url</c>, <c>--model</c>, engine-positional and benchmark scope). The <c>typesafe</c>
/// happy path is
/// deliberately not gated: it needs a live decision-model server. #589 requires each reject to be
/// observable before the first disk touch, so the ordering cases and the <c>qwen</c> happy paths use
/// <c>qwen</c> and the mode cases pin the message, not the exit code — a guard that fell through to
/// the model-file check would exit <c>1</c> with a different message.
/// </para>
/// <para>
/// <b>No checkpoint in reach.</b> The sample resolves a model directory against its working directory
/// (<c>samples/data/&lt;model&gt;/model.safetensors</c>). The qwen checkpoint is gitignored, which is
/// exactly what lets a developer keep the one the benchmark workflow asks for, so spawning the sample
/// from the repository root made the qwen happy paths load the weights and run real generation
/// instead of reaching the model-file check. Every invocation therefore runs in an empty temporary
/// directory, which makes "no checkpoint is present" a property of this gate rather than of the
/// machine it runs on.
/// </para>
/// </summary>
[TestFixture]
public class InferenceCliArgumentTests
{
    readonly record struct CliResult(int ExitCode, string StdOut, string StdErr);

    // Host-adaptive: cold `dotnet` startup for this heavy sample can exceed a fixed 30 s on a slow
    // machine even though the CLI itself is correct. Calibrated in TheSampleMustBeBuilt (see below).
    // NIVARA_CLI_TIMEOUT_MS overrides the bound absolutely; NIVARA_CLI_TIMEOUT_MULTIPLIER scales it.
    static int ProcessTimeoutMs = 30_000;
    const string ModelLookupMessage = "Model file not found";

    static readonly string DllPath = Path.Combine(AppContext.BaseDirectory, "NivaraInference.dll");

    // The sample resolves the model directory against its working directory
    // (samples/data/<model>/model.safetensors), so spawning it with the repository as the working
    // directory made this gate depend on which checkpoints the developer keeps: the qwen happy paths
    // below observe "Model file not found" as the deepest reachable point, and a real checkpoint turns
    // that one-second exit into a full generation run. Every invocation runs in an empty directory, so
    // "no checkpoint is present" is a property of the gate rather than of the machine it runs on.
    static readonly string CliWorkingDirectory = CreateEmptyWorkingDirectory();

    [OneTimeSetUp]
    public void TheSampleMustBeBuilt()
    {
        Assert.That(File.Exists(DllPath), Is.True,
            $"NivaraInference.dll was not found at {DllPath}. Build the solution before testing; "
            + "this project references samples/NivaraInference so 'dotnet build' produces it.");

        // Calibrate the per-process timeout to this host's cold-start speed: run the usage path once
        // (no model load) and give the real invocations a 10x margin over that, floored at 30 s. This
        // keeps the gate on CI's 30 s bound while not reporting a false failure on a slow machine.
        // The gate's assertions are functional (exit code + stderr message), not timing, so it is
        // deliberately NOT marked [Category("Performance")] — categorising it would drop the
        // argument-surface gate from CI's filtered suite.
        int canaryMs = MeasureCanary();
        int calibrated = Math.Max(30_000, canaryMs * 10);
        if (int.TryParse(Environment.GetEnvironmentVariable("NIVARA_CLI_TIMEOUT_MS"), out int overrideMs)
            && overrideMs > 0)
            calibrated = Math.Max(calibrated, overrideMs);
        if (double.TryParse(Environment.GetEnvironmentVariable("NIVARA_CLI_TIMEOUT_MULTIPLIER"),
                            out double multiplier) && multiplier > 0)
            calibrated = (int)(calibrated * multiplier);
        ProcessTimeoutMs = calibrated;
    }

    [OneTimeTearDown]
    public void RemoveCliWorkingDirectory() => Directory.Delete(CliWorkingDirectory, recursive: true);

    [Test]
    public void SeqWarmupIters_OnNonModernBertModel_Rejected()
    {
        // --seq names a shape, so it stays modernbert-only; --warmup / --iters parameterize the
        // sampling protocol, which #676 opened to 'typesafe benchmark' as well.
        AssertRejected(Run("minilm", "benchmark", "--seq", "128"),
            "--seq is an option of 'modernbert benchmark' only");
        AssertRejected(Run("minilm", "benchmark", "--warmup", "2"),
            "--warmup / --iters are options of 'modernbert benchmark' and 'typesafe benchmark' only");
        AssertRejected(Run("minilm", "benchmark", "--iters", "2"),
            "--warmup / --iters are options of 'modernbert benchmark' and 'typesafe benchmark' only");
        AssertRejected(Run("typesafe", "benchmark", "--seq", "128"),
            "--seq is an option of 'modernbert benchmark' only");
    }

    /// <summary>
    /// <c>typesafe benchmark</c> takes <c>--warmup</c>/<c>--iters</c> since #676. The proof that
    /// they clear the scope guard is a rejection at the *next* guard — value validation, or the
    /// benchmark-mode gate — which runs before any HTTP dispatch, so the gate needs no live
    /// decision engine to observe the flags reaching their consumer.
    /// </summary>
    [Test]
    public void WarmupIters_OnTypesafeBeyondItsScope_Rejected()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("typesafe", "benchmark", "--warmup", "0"), "--warmup expects a positive integer");
            AssertRejected(Run("typesafe", "benchmark", "--iters", "0"), "--iters expects a positive integer");
            AssertRejected(Run("typesafe", "--warmup", "2"), "--warmup / --iters need the 'benchmark' mode; got 'typesafe'");
            AssertRejected(Run("typesafe", "--iters", "2"), "--warmup / --iters need the 'benchmark' mode; got 'typesafe'");
        });
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
    public void Seq_LargeLengthOnCpu_NoLongerHitsDenseMaskCap() =>
        AssertRejected(Run("modernbert", "benchmark", "--seq", "4096", "--warmup", "0"),
            "--warmup expects a positive integer");

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
            "the --ilgpu guard must run before the model-file check; no checkpoint is in reach");
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
    /// The flag's migration contract: the pre-Silk name is rejected with the new name named,
    /// not silently aliased, so an old invocation cannot quietly keep running the ILGPU path.
    /// </summary>
    [Test]
    public void GpuRenamedFlag_RejectedWithNewName() =>
        AssertRejected(Run("minilm", "--gpu"), "--gpu was renamed --ilgpu");

    /// <summary>
    /// The Silk phase: <c>--silk</c> is a second GPU backend whose model scope (the hand-authored
    /// kernels cover the BERT-family encoders, MiniLM and DistilBERT, and DistilBERT's classifier
    /// head) and mode scope are narrower than <c>--ilgpu</c>'s, and it is F32-only. Each
    /// rejection is pinned by message for the same reason the ILGPU scopes are: a removed guard would
    /// fall through to a different exit-1 path and the exit code alone would still be <c>1</c>.
    /// </summary>
    [Test]
    public void SilkSmolLM_Precision_Rejected() =>
        AssertRejected(Run("smollm", "--silk", "--precision", "bf16"), "--silk is F32-only");

    [Test]
    public void SilkQwen_DistillMode_Rejected() =>
        AssertRejected(Run("qwen", "--silk", "distill"),
            "--silk does not support mode 'distill' for 'qwen'");

    [Test]
    public void Silk_RejectedBeforeModelFileLookup()
    {
        CliResult result = Run("qwen", "--silk", "distill");
        Assert.That(result.StdErr, Does.Not.Contain(ModelLookupMessage),
            "the --silk guard must run before the model-file check; no checkpoint is in reach");
        AssertRejected(result, "--silk does not support mode 'distill'");
    }

    [Test]
    public void SilkAndIlgpu_CompetingBackends_Rejected() =>
        AssertRejected(Run("minilm", "--ilgpu", "--silk"),
            "--ilgpu and --silk select competing GPU backends");

    [Test]
    public void SilkPrecisionCombination_Rejected() =>
        AssertRejected(Run("minilm", "--silk", "--precision", "bf16"), "--silk is F32-only");

    [Test]
    public void SilkUnsupportedMode_Rejected() =>
        AssertRejected(Run("minilm", "--silk", "similarity"),
            "--silk does not support mode 'similarity' for 'minilm'");

    /// <summary>
    /// <c>compare_diag</c> is not a Laya mode at all, so the general allow-list rejects it before
    /// the Silk mode guard. The Silk-specific message is what <c>modernbert --silk compare_diag</c>
    /// hits, because that mode exists on the CPU path.
    /// </summary>
    [Test]
    public void SilkLaya_CompareDiag_Rejected() =>
        AssertRejected(Run("laya", "--silk", "compare_diag"),
            "Unsupported mode 'compare_diag' for 'laya'");

    [Test]
    public void SilkLaya_Precision_Rejected() =>
        AssertRejected(Run("laya", "--silk", "--precision", "bf16"), "--silk is F32-only");

    [Test]
    public void SilkModernBert_CompareDiag_Rejected() =>
        AssertRejected(Run("modernbert", "--silk", "compare_diag"),
            "--silk does not support mode 'compare_diag' for 'modernbert'");

    /// <summary>
    /// The F32 message, rather than the model-scope reject, is what proves <c>modernbert</c> is in
    /// <c>SilkModes</c>. A model outside the allow-list never reaches this guard. Unlike the qwen
    /// lookup pin, this one does not depend on the checkpoint being absent.
    /// </summary>
    [Test]
    public void SilkModernBert_Precision_Rejected() =>
        AssertRejected(Run("modernbert", "--silk", "--precision", "bf16"), "--silk is F32-only");

    /// <summary>
    /// 4096 is past the CPU dense-mask cap. The precision reject runs after <c>--seq</c> is parsed
    /// and before any weight load, so a dense-mask message here would mean Silk was treated as CPU.
    /// </summary>
    [Test]
    public void SilkModernBert_Seq4096_IsNotTheCpuCap()
    {
        CliResult result = Run("modernbert", "--silk", "benchmark", "--seq", "4096", "--precision", "bf16");
        AssertRejected(result, "--silk is F32-only");
        Assert.That(result.StdErr, Does.Not.Contain("dense-mask"));
    }

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

    /// <summary>
    /// The TypeSafe demo is a model key whose run is an HTTP client, not a checkpoint, so its
    /// dispatch sits before the model-file lookup. Each reject therefore has to name its own guard
    /// rather than fall through to "Model file not found". The happy path is deliberately absent:
    /// it needs a live decision engine, which a CI runner does not have — and neither does it
    /// assert <c>benchmark</c> as unsupported here, because it is a supported mode (the scope of
    /// <c>--warmup</c>/<c>--iters</c> on it, opened by #676, is pinned in
    /// <see cref="WarmupIters_OnTypesafeBeyondItsScope_Rejected"/>).
    /// </summary>
    [Test]
    public void Typesafe_BeyondItsCliSurface_Rejected()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("typesafe", "predict"), "Unsupported mode 'predict' for 'typesafe'");
            AssertRejected(Run("typesafe", "--ilgpu"), "--ilgpu is not supported for 'typesafe'");
            AssertRejected(Run("typesafe", "--precision", "bf16"), "--precision bf16 is not supported for 'typesafe'");
        });
    }

    /// <summary>
    /// <c>--url</c> points the <c>typesafe</c> demo at an engine and is read by no other run. Its
    /// value is required — unlike an optional-value flag it can never be confused with the mode
    /// that follows it, which is how a well-formed <c>typesafe ollama benchmark</c> used to be
    /// rejected as a bad URL.
    /// </summary>
    [Test]
    public void Url_BeyondItsScopeOrWithoutAValue_Rejected()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("minilm", "--url", "http://localhost:11434"), "--url is a 'typesafe' option only");
            AssertRejected(Run("typesafe", "--url"), "--url needs a value.");
            AssertRejected(Run("typesafe", "--url", "not-a-url"), "--url expects an absolute http(s) URL");
        });
    }

    /// <summary>
    /// The first positional of a <c>typesafe</c> run is the engine when it names one, and the mode
    /// otherwise; the two token sets are disjoint, which is what lets one slot hold both. Asserted
    /// through the allow-list rather than a connection attempt, so the gate needs no engine: an
    /// engine token that is not a mode still parses (no URL error), a second engine token is read
    /// as a mode and rejected, and the same token is an unsupported mode for any other model.
    /// </summary>
    [Test]
    public void EnginePositional_IsReadAsEngineOrModeByNameOnly()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("typesafe", "bogus"), "Unsupported mode 'bogus' for 'typesafe'");
            AssertRejected(Run("typesafe", "ollama", "bogus"), "Unsupported mode 'bogus' for 'typesafe'");
            AssertRejected(Run("typesafe", "ollama", "ollaya"), "Unsupported mode 'ollaya' for 'typesafe'");
            AssertRejected(Run("minilm", "ollama"), "Unsupported mode 'ollama' for 'minilm'");
        });
    }

    /// <summary>
    /// <c>--model</c> names the model to load on the selected decision-model server and is read only
    /// by the <c>typesafe</c> run, so it is rejected elsewhere; the value is consumed even when it is
    /// the last token, so it cannot fall through to the positional-mode branch and be read as a mode
    /// name. The happy path (an override that the live server accepts) is deliberately not gated: it
    /// needs a decision-model server, which a CI runner does not have.
    /// </summary>
    [Test]
    public void Model_BeyondItsScopeOrWithoutAValue_Rejected()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("minilm", "--model", "laya:en"), "--model is a 'typesafe' option only");
            AssertRejected(Run("typesafe", "--model"), "--model needs a value.");
            AssertRejected(Run("typesafe", "--model", "   "), "--model expects a non-empty model name");
        });
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
    /// The sample runs outside the repository, so no checkpoint is in reach and "Model file not found"
    /// is the deepest point this gate can observe; the point is that no guard fires before it.
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

        var stopwatch = Stopwatch.StartNew();
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start 'dotnet' for the canary.");
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(60_000))
        {
            process.Kill(entireProcessTree: true);
            throw new InvalidOperationException("NivaraInference canary did not exit within 60 s.");
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

    static string CreateEmptyWorkingDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "nivara-cli-args-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    [Test]
    public void Typesafe_Nivara_Rejects()
    {
        Assert.Multiple(() =>
        {
            AssertRejected(Run("typesafe", "ollama", "verify"), "'verify' requires the 'nivara' engine.");
            AssertRejected(Run("typesafe", "nivara", "--url", "http://localhost:11435"),
                "--url is not supported for the in-process 'nivara' engine.");
            AssertRejected(Run("typesafe", "nivara", "--model", "tev1:0.8b"),
                "--model must name 'convaiinnovations/laya' for the 'nivara' engine.");
            AssertRejected(Run("typesafe", "nivara", "verify", "--model", "laya"),
                "--model is not supported for 'typesafe nivara verify'.");
            AssertRejected(Run("typesafe", "nivara", "bogus"), "Unsupported mode 'bogus' for 'typesafe'");
        });
    }

}
