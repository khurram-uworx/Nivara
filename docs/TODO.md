# Plan — #581: automated gate over the `NivaraInference` CLI argument surface

## Problem

Five consecutive issues have changed `NivaraInference`'s command-line parsing and
validation (`--seq`/`--warmup`/`--iters` scope #474, unparseable `--seed`/
`--teacher-examples`/`--precision` #479, `--gpu` for models with no GPU path #568,
`qwen distill` init from `--seed` #570, `--teacher-examples`/`--seed`/`--force`
outside `qwen distill` #571). Each was verified by hand — direct invocation with
exit-status capture — because `Program.Main` is a private method in the sample exe and
`tests/Nivara.Tests` does not reference `samples/NivaraInference`.

The messages and exit codes are the entire observable contract of these guards, and
nothing fails if a future edit breaks one. Evidence that this is latent rather than
theoretical: the only built binary on disk (`samples/NivaraInference/bin/Debug/net10.0/
NivaraInference.exe`) predates all five changes — it accepts `minilm benchmark --seq 128`
(exit 0), `qwen --seed 5`, and `minilm --seed 5`, and for `modernbert benchmark --warmup x`
it loaded a 1.5 GB safetensors before failing on an unrelated path.

## Proposed change

Shape chosen: **invoke the built exe** (issue option 1), launched via `dotnet <dll>`.
It gates the exact observable contract (exit code + message + ordering relative to the
first disk touch), needs **zero change to `Main`** (so it cannot perturb the guards it
protects), and is cheap: reject paths return before the weight load, measured at
~45–55 ms per invocation.

### 1. `tests/Nivara.Tests/Nivara.Tests.csproj`

Add a project reference so the sample is built before `dotnet test --no-build` (CI builds
the whole `.slnx`, which already contains `NivaraInference`) and its `dll` +
`runtimeconfig.json` + `deps.json` are copied into the test output:

```xml
<ProjectReference Include="..\..\samples\NivaraInference\NivaraInference.csproj" />
```

Precedent: the test project already references `NivaraChat` (also `OutputType=Exe`), and the
`NivaraChat` apphost/`runtimeconfig`/`deps` already appear in the test output directory.

### 2. New file `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`

- `[TestFixture]`; NUnit `[Test]` only (repo forbids `[TestCase]`); no `Stopwatch` and no
  allocation measurement, so `SlowTestCategorisationTests` stays green and the fixture stays
  in the `Category!=Performance` suite.
- `DllPath = Path.Combine(AppContext.BaseDirectory, "NivaraInference.dll")`.
- `RepoRoot` = walk up from `AppContext.BaseDirectory` for `Nivara.slnx` (same helper shape as
  `tests/Nivara.Tests/Docs/DocSnippet.FindRepoRoot`).
- `[OneTimeSetUp]`: fail loudly if `NivaraInference.dll` is absent, so an unbuilt sample is a
  failure rather than a vacuous pass (a gate must state its own coverage).
- Runner helper:

```csharp
readonly record struct CliResult(int ExitCode, string StdOut, string StdErr);

static CliResult Run(params string[] args)
{
    var psi = new ProcessStartInfo("dotnet")
    {
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        WorkingDirectory = RepoRoot,
    };
    psi.ArgumentList.Add(DllPath);
    foreach (var a in args) psi.ArgumentList.Add(a);

    using var process = Process.Start(psi)!;
    string stdout = process.StandardOutput.ReadToEnd();
    string stderr = process.StandardError.ReadToEnd();
    if (!process.WaitForExit(30_000))
    {
        process.Kill(entireProcessTree: true);
        Assert.Fail($"NivaraInference {string.Join(' ', args)} did not exit within 30 s");
    }
    return new CliResult(process.ExitCode, stdout, stderr);
}

static void AssertRejected(CliResult result, string message)
{
    Assert.That(result.ExitCode, Is.EqualTo(1));
    Assert.That(result.StdErr, Does.Contain(message));
}
```

Message assertions are mandatory: exit code alone is insufficient, because a removed guard
falls through to a different exit-1 path (unknown model / model file not found).

### Test matrix (exact arg sets → asserted stderr substring)

All substrings are literal strings in `Program.Main` (`samples/NivaraInference/Program.cs`).

| Group | Args | Asserted stderr substring |
|---|---|---|
| #474 scope (non-model) | `minilm benchmark --seq 128` / `--warmup 2` / `--iters 2` | `are options of 'modernbert benchmark' only` |
| #474 scope (no mode) | `modernbert --seq 128` / `--warmup 2` / `--iters 2` | `need the 'benchmark' mode; got 'modernbert'` |
| #474 parse | `modernbert benchmark --seq 0` | `--seq expects positive integers` |
| #474 parse | `modernbert benchmark --seq 4096` | `exceeds the CPU path's dense-mask limit` |
| #474 parse | `modernbert benchmark --seq` | `--seq needs a value.` |
| #474 parse | `modernbert benchmark --warmup x` / `--iters x` | `--warmup expects a positive integer` / `--iters expects a positive integer` |
| #568 | `smollm --gpu` | `--gpu is not supported for 'smollm'` |
| #568 + ordering | `qwen --gpu` | `--gpu is not supported for 'qwen'` and **not** `Model file not found` |
| #479 | `minilm --precision foo` | `--precision expects f32, bf16 or fp16; got 'foo'.` |
| #479 | `minilm --precision` | `--precision needs a value` |
| #479 | `qwen distill --seed abc` / `--teacher-examples abc` | `expects an integer; got 'abc'.` |
| #479 | `qwen distill --seed` | `--seed needs a value.` |
| #571 scope (model) | `minilm --seed 5` / `--force` / `--teacher-examples 3` | `are options of 'qwen distill' only; no other model takes them.` |
| #571 scope (mode) | `qwen --seed 5` / `--force` / `--teacher-examples 3` | `need the 'distill' mode; got 'qwen'.` |
| Happy path | `qwen distill --seed 7 --teacher-examples 3 --force` | `Model file not found` and none of the three reject strings |

`qwen` is used for the ordering and happy-path cases because its weights are absent in every
environment (`samples/data/qwen2.5-0.5b-instruct/` is gitignored and no `*.safetensors` is
tracked), so a guard reordered after the `File.Exists` check shows up as
`Model file not found` instead of the guard message. The happy path is "accepted → reached
the weight load"; deeper `qwen distill` determinism is #577, out of scope.

## Blast radius

- `tests/Nivara.Tests/Nivara.Tests.csproj` — adds one `ProjectReference`. Downstream: the test
  project's build graph now includes `NivaraInference`, `System.Drawing.Common` (never loaded;
  the gate runs a separate child process and touches no image path) and the already-present
  `Nivara.Samples`.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — new; no existing test touched.
- `src/Nivara/**` — untouched. `samples/NivaraInference/Program.cs` — untouched.
- CI (`.github/workflows/ci.yml`) builds the whole solution and runs `dotnet test --no-build
  --configuration Release --filter "Category!=Performance"`; the new fixture is picked up and
  runs on ubuntu-latest. Estimated added runtime < 2 s total (reject paths ~50 ms each).

## Verification

1. `dotnet build Nivara.slnx -c Release` — must succeed with no new warnings
   (`TreatWarningsAsErrors=true`).
2. `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` — new
   fixture green.
3. Negative control: temporarily remove one guard (or move a block below the `File.Exists`
   check), confirm the matching test fails, then restore. Do not report the gate as working
   until a check has been seen fail.
4. Full suite (`dotnet test -c Release --filter "Category!=Performance"`) — only after asking
   the human.

## Planned commits

1. `docs: plan #581 CLI argument gate in TODO.md`
2. `docs: record #581 G1 grounding in TODO.md` (only if grounding adds content)
3. `test(inference): gate NivaraInference CLI argument validation (#581)`
4. `test(inference): verify the gate catches a removed guard (#581)` — only if a change is needed
5. `docs: remove TODO.md - plan executed`

## GitHub issues log

- (none yet)
