# Plan: Allow --warmup/--iters on 'typesafe benchmark' (issue #676)

## Problem statement recap

Issue #676: `NivaraInference typesafe benchmark` uses a fixed protocol (1 warmup + 3 timed passes) and does not accept `--warmup`/`--iters`. The guard in Program.cs rejects them with a message saying they are options of 'modernbert benchmark' only. For typesafe (HTTP round-trip), users may want to adjust sample count. `--seq` must remain modernbert-only. Update Program.cs, TypeSafe.cs, and the pinned test.

## Grounding (Resistance G1) - pre-implementation

We need to ground via microsoft-learn MCP (tensors/CLI patterns if relevant) and code-memory MCP to understand the exact flow. Key symbols to inspect:
- `samples/NivaraInference/Program.cs` - seq/warmup/iters parsing guard (~lines 405-436), typesafe dispatch (~593-596)
- `samples/NivaraInference/TypeSafe.cs` - `Benchmark()` method and constants `WarmupPasses`, `TimedPasses`
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` - `SeqWarmupIters_OnNonModernBertModel_Rejected()`

Blast radius:
- samples/NivaraInference/Program.cs: parsing/validation + dispatch (small, localized)
- samples/NivaraInference/TypeSafe.cs: benchmark implementation (small)
- tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs: one test method (and possibly comment updates)

No src/ changes. Behavior change is additive for typesafe benchmark (now accepts warmup/iters). Other models unchanged except the error message text for warmup/iters rejection.

## Proposed changes

### 1. samples/NivaraInference/TypeSafe.cs
- Replace hardcoded `WarmupPasses`/`TimedPasses` constants with configurable parameters.
- Change `Benchmark` signature to: `public static async Task<int> Benchmark(string engine, string? urlOverride, string? modelOverride, int warmup = 1, int iters = 3)`
- Update implementation to use `warmup`/`iters` instead of constants.
- Update console output line ~138 to show the actual values: `$"{warmup} warmup + {iters} timed passes per news item..."`

### 2. samples/NivaraInference/Program.cs
- Restructure seq/warmup/iters guard (lines ~405-436):
  - If `--seq` set: must be modernbert + benchmark. Validate seq lengths.
  - If `--warmup` or `--iters` set: allowed if (modernbert AND benchmark) OR (typesafe AND benchmark). Otherwise reject with: `"--warmup / --iters are options of 'modernbert benchmark' and 'typesafe benchmark' only; no other model's benchmark takes them."`
  - Keep existing validation rules (positive ints).
- Update typesafe dispatch (lines ~593-596): pass warmup/iters. Note: when not specified, defaults are ModernBert.DefaultWarmup (1) and ModernBert.DefaultIterations (3) - same as current behavior, so compatible.
  Change to: `return mode == "benchmark" ? TypeSafe.Benchmark(engine, urlArg, modelArg, warmup, iterations).GetAwaiter().GetResult() : ...`

### 3. tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs
- Update `SeqWarmupIters_OnNonModernBertModel_Rejected()` (lines ~69-80):
  - New message: for warmup/iters rejections going forward, the message will be the new combined message above. But the test currently asserts the same expected string for all cases. After change: minilm with seq gets rejected by seq-specific check? Wait - look: if seq is set on non-modernbert, our new code will hit the seq branch and reject with `"--seq is an option of 'modernbert benchmark' only..."` (different message than warmup/iters). Also the test name is general.
  - But the issue says "update the pinned gate message and its three assertions accordingly". The test currently has 5 assertions (minilm seq/warmup/iters + typesafe warmup/iters). After change, typesafe warmup/iters will be allowed (not rejected) - so remove those 2. Keep the 3 minilm cases. The expected message in the test needs to match what the code now produces. For minilm + warmup/iters: our new code will reject in warmup/iters branch with the combined message. For minilm + seq: rejects in seq branch with seq-specific message. So the test as written uses the same expected substring for all three - but the actual messages differ. That means either (a) split into separate assertions with correct messages, or (b) adjust.
- But looking at how the test is structured: it uses `AssertRejected` with the same `expected` parameter. The issue says "pinned gate message" - maybe the key part is that the flags are not accepted for non-modernbert in the old specific way? Or perhaps the issue's suggested direction is: "Loosen the scope to \"modernbert benchmark and typesafe benchmark\", keep --seq modernbert-only, and update the pinned gate message and its three assertions accordingly." So the "pinned gate message" refers to the error message text. The three assertions - probably the minilm ones remain; typesafe ones are no longer rejected. So the test needs to change: expected message becomes more general? Or we need to see the exact test more.

Also looking up the test again: it asserts `const string expected = "are options of 'modernbert benchmark' only";` for all cases including typesafe warmup/iters. After our fix, typesafe warmup/iters won't be rejected - so those 2 assertions go away. The remaining 3 (minilm seq/warmup/iters) - minilm+warmup/iters will now produce the new combined message containing "modernbert benchmark and typesafe benchmark" (so the old expected substring won't match). minilm+seq will produce seq-specific message "modernbert benchmark only" (different). So the test needs to be restructured.

But the issue says "update the pinned gate message and its three assertions accordingly" - this is a bit ambiguous. But the core change is clear: allow warmup/iters for typesafe benchmark.

## Verification steps

1. Build: `dotnet build Nivara.slnx` (or build specific projects)
2. Run inference CLI tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj --filter "FullyQualifiedName~InferenceCliArgumentTests" --no-restore`
3. Also run any related CLI arg tests to be safe.
4. Manual smoke test of parsing (will fail to connect but should parse): `dotnet run --project samples/NivaraInference -- typesafe benchmark --warmup 2 --iters 5` - should not reject due to flags.

## Planned commits

1. `feat: allow --warmup/--iters for typesafe benchmark` - modify TypeSafe.Benchmark to accept configurable warmup/iters and update Program.cs validation/dispatch. Commit after making both changes together? Or split? The changes are related - TypeSafe needs to accept params, Program needs to pass them and allow them. Logical unit is "enable warmup/iters for typesafe benchmark end-to-end".
2. `test: update inference CLI arg tests for typesafe warmup/iters scope` - update the test to match new behavior.

## GitHub issues log

- [ ] (to be filled if any follow-ups discovered during implementation)

## Blast radius (stated)

- samples/NivaraInference/TypeSafe.cs: +2 params, behavior unchanged if defaults used. Console message now shows actual counts.
- samples/NivaraInference/Program.cs: validation logic split; error messages more specific. typesafe dispatch passes through params.
- tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs: test expectations change per new allowed scope.
- No public API changes in src/. No behavior change for existing valid invocations.
