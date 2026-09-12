# Plan — #420: perf gate ops/s leg not representative for bandwidth-bound Qwen rows

## Problem

While running the 16-row `--compare` gate (`qwen-decode-block-baseline.json`) for
#413, every Qwen scenario failed on the **ops/s leg** — including scenarios
unrelated to the PR — at ~2.5–3× slower than the recorded baseline. A
same-scope decode-only A/B pair (khurram/411 vs khurram/413, same session)
showed the change ~10% *faster*, confirming the ops/s difference is
machine-state drift, not code.

Root cause: the Qwen rows are DRAM-bandwidth-bound single-row GEMV /
memory-streaming kernels (docs/QWEN.md: "F32 decode ceiling at ~30 GB/s
effective bandwidth"). ops/s swings ~2.5–3× with machine state, while **B/op is
byte-identical across same-day baselines** (`qwen-prefill-baseline.json` 05:42Z
vs `qwen-decode-block-baseline.json` 08:53Z: e.g. LM head matmul 24.1 → 43.3
ops/s, 1.80×; attn proj 2,121 → 4,812, 2.27×). The README already states "ops/s
are load-sensitive; B/op and gen0/op are the reliable signals" — the gate's hard
90% ops/s floor contradicts that policy.

Secondary: the committed baseline is **pre-fusion** — decode block 131,985 B/op
and decode fwd 3,620,919 B/op, while current code runs ~1 B/op and ~619,631 B/op
(after #404/#413) — so those two rows' B/op legs are vacuous until re-baselined.

## Proposed changes

1. **Gate logic — `tests/Nivara.PerformanceTests/Program.cs`**
   - Add `const double BandwidthBoundMinOpsFraction = 0.25`.
   - Name classifier `static bool IsBandwidthBound(string name)` →
     `name.StartsWith("Qwen ", StringComparison.Ordinal)` (exactly the 16 gate
     rows, nothing else).
   - **New `GateEvaluator` (internal, new file `GateEvaluator.cs`)** — pure
     per-row decision, extracted from `Compare()` so it is unit-testable:
     - strict rows: ops/s ≥ `minOpsFraction` (default 0.90), B/op ≤ baseline ×
       1.01, gen0 ≤ baseline + 0.05;
     - bandwidth-bound rows: ops/s ≥ `BandwidthBoundMinOpsFraction` (0.25),
       B/op and gen0 strict as above (rigor: `--tolerance` still configures
       only the stable-row floor).
   - `Compare()` keeps only console rendering; header prints the split floor
     ("minOps 90% (25% bandwidth-bound)").
   - `InternalsVisibleTo("Nivara.Tests")` via csproj ItemGroup.

2. **Unit tests — `tests/Nivara.Tests/`**
   - Add `<ProjectReference Include="..\Nivara.PerformanceTests\Nivara.PerformanceTests.csproj" />`
     to `Nivara.Tests.csproj` (referencing an Exe project is supported by the
     SDK; compile-time only, no harness execution).
   - New `PerfGateEvaluatorTests`: bandwidth-bound row passes at 26% ops/s but
     fails at 20%; fails on B/op > ×1.01 or gen0 > +0.05 regardless of ops/s;
     stable row still fails at 89% and passes at 91%; boundary at exactly 25%.

3. **Docs**
   - `tests/Nivara.PerformanceTests/README.md` — gate-criteria table gains the
     bandwidth-bound rule + the why (#420, observed ~2.5–3× noise vs stable B/op).
     `--tolerance` row gains "stable rows only".
   - `docs/QWEN.md` — harness-reference paragraph (~line 376) notes the 25%
     bandwidth-bound ops/s floor and #420.

4. **Re-record `tests/Nivara.PerformanceTests/qwen-decode-block-baseline.json`**
   - After build: `dotnet run --project tests/Nivara.PerformanceTests -c Release
     -- --json tests/Nivara.PerformanceTests/qwen-decode-block-baseline.json
     --only Qwen --runs 3` from an **idle machine** (~1–2 min), commit the
     refreshed file. Grounds ops/s to a representative state and re-arms the
     decode-block / decode-fwd B/op legs (~1 B/op, ~619,631 B/op).
   - Verify: `--compare qwen-decode-block-baseline.json --only Qwen --runs 3`
     → 16/16 PASS regardless of machine load.

## Verification steps

- `dotnet build Nivara.slnx` (after each code change).
- `dotnet test tests/Nivara.Tests` filtered to the new gate tests (ask human
  first per AGENTS.md).
- Baseline re-record + `--compare` run (ask human first; needs idle machine).
- Full `dotnet test` optional; changes are harness/docs-only, additive.

## Planned commits

1. `docs: plan perf gate ops/s tolerance for bandwidth-bound rows (#420) in TODO.md`
2. `perf: gate bandwidth-bound Qwen ops/s at 25% floor in --compare (#420)`
3. `test: cover bandwidth-bound gate decisions in per-row compare evaluator (#420)`
4. `docs: document 25% ops/s floor for bandwidth-bound rows in perf gate (#420)`
5. `perf: re-record qwen-decode-block-baseline.json (fused decode state)`
6. `docs: remove TODO.md — issue #420 executed`

## Blast radius

- `tests/Nivara.PerformanceTests/Program.cs` — `Compare()` internals only; no
  harness CLI/API change, no public library change. Downstream: none
  (`src/Nivara/*` untouched). Existing tensors/columns/query surfaces untouched.
- `tests/Nivara.PerformanceTests/GateEvaluator.cs` (new) — consumed only by
  `Program.Compare` and the new tests.
- `tests/Nivara.Tests` — one added ProjectReference + one new test file; pulls
  the harness (and transitively `Nivara.Samples`, already referenced) into the
  test build at compile time only. No existing tests touched.
- Baseline JSON re-record — committed artifact; B/op/gen0 values for shared rows
  are unchanged except the code-driven decode-block/decode-fwd drops.
- Coverage: existing perf-gate behavior has no unit tests today; the new
  `GateEvaluatorTests` pin the new semantics and the stable-row behavior.

## GitHub issues log

- [ ] #420 — perf gate ops/s leg not representative for bandwidth-bound Qwen rows
      (being implemented on this branch). Any new deferred work/concern found
      during execution → `gh issue create --repo khurram-uworx/Nivara`
      immediately and record here — don't hold it in memory.