# Plan: fix #601 — InferenceCliArgumentTests 30 s subprocess timeout on slow hosts

## Problem

`InferenceCliArgumentTests` spawns the built `NivaraInference.dll` via `dotnet` and guards each
subprocess with a hard-coded `ProcessTimeoutMs = 30_000`. On a slow host the cold `dotnet`
startup/JIT for that heavy sample can exceed 30 s for a single invocation, so
`QwenAcceptedModes_ReachTheModelFileLookup` and `QwenDistill_WithAllThreeFlags_PassesValidation`
report a timeout failure even though the CLI behaved correctly. The timeout is a hang-safety net,
not the thing under test.

## Proposed changes

Test-only change in `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`:

- Replace `const int ProcessTimeoutMs = 30_000;` with `static int ProcessTimeoutMs = 30_000;`.
- In `TheSampleMustBeBuilt` (OneTimeSetUp), after the DLL-existence assertion, calibrate
  `ProcessTimeoutMs`:
  - Run a canary invocation of the same DLL with no args (usage path: parses args, prints usage,
    exits 1 — no model load, so it measures host cold-start/JIT, not work).
  - Set `ProcessTimeoutMs = max(30_000, canaryElapsedMs * 10)`.
  - Honor optional overrides `NIVARA_CLI_TIMEOUT_MS` (absolute) and
    `NIVARA_CLI_TIMEOUT_MULTIPLIER` (factor), taking the larger of the calibrated value and any
    override.
- Add a small `MeasureCanary()` helper that starts the same `dotnet DllPath` process, drains both
  pipes asynchronously, waits with a 60 s floor, kills on timeout, and returns elapsed milliseconds.
- Keep `Run` logic and the 30 s floor/assertion message unchanged; only the numeric bound becomes
  host-derived.
- Do **not** add `[Category("Performance")]`: the gate's assertions are functional (exit code +
  stderr message), and the subprocess duration on CI is well under the 2 s rule. Categorising it
  would drop the argument-surface gate from CI's filtered suite, which is the wrong trade-off for a
  correctness gate. Document that rationale in a comment.

## Blast radius

- Single test file: `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`.
- No production code, sample, or storage changes.
- `ProcessTimeoutMs` changes from `const` to `static int`; only referenced inside this class.
- `SlowTestCategorisationTests` should remain green: `MeasureCanary` contains `Stopwatch` but is
  only invoked from `OneTimeSetUp` (not a `[Test]`), and `Run` uses no measurement APIs, so no
  test method becomes tainted.

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
  on this host: expect 41 passed (canary should land near the 30 s floor).
- Confirm `SlowTestCategorisationTests` still passes.
- Optional override check: set `NIVARA_CLI_TIMEOUT_MS=5000` to confirm the knob is honored (expect
  the two qwen tests to time out), then unset.

## Planned commits

1. `docs: plan #601 in TODO.md`
2. `test(inference): make CLI subprocess timeout host-adaptive for #601`

## GitHub issues log

- [ ] #601 — InferenceCliArgumentTests: 30 s subprocess timeout fails on slow hosts (in progress)