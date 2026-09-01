# TODO — Phase 3: BFloat16 Widen-SIMD A/B + correctness + docs

**Branch:** `khurram/smollm-3`
**Source plan:** `docs/BFLOAT16-TRANSFORMER.md` §6 Phase 3
**Status:** In progress

---

## Problem

Phase 1 delivered the widen-compute-narrow SIMD kernels for `BFloat16`/`Half`
(`WidenPrimitives` / `NarrowFloatKernels`, gated behind `NivaraPrimitives.UseWidenSimd`,
default off) and Phase 2 delivered the SmolLM-135M causal-LM sample that auto-enables the
toggle for its narrow runs. Phase 3's job is to make the **scalar vs widen comparison
first-class** and produce the correctness/perf numbers that close the A/B story:

1. **No `--simd-widen` CLI flag** — `samples/NivaraInference/Program.cs` `RunSmolLMCore`
   (lines ~1193–1196) hardcodes `NivaraPrimitives.UseWidenSimd = true` for narrow
   (BFloat16/Half) SmolLM runs. There is no way for a user to A/B scalar (off) vs widen
   (on) on the same model, and no way to opt other narrow models (MiniLM, DistilBERT,
   SST-2) into the widen path from the CLI.
2. **No SmolLM `benchmark` mode** — the `smollm` switch case (Program.cs lines 136–139)
   ignores `mode == "benchmark"` and always runs generation. Every other transformer model
   wires `benchmark ? BenchmarkX(...)`. SmolLM needs a benchmark that reports
   median ms / ms-per-token over full 32-token generations (the established median-of-3
   methodology in `samples/NivaraInference/README.md` and `RELEASING.md`).
3. **No A/B runner** — nothing compares scalar ma-tmul vs widen SIMD timing side-by-side,
   which is the whole point of Phase 3 ("run the same model twice — off = scalar, on =
   widen — and diff correctness + timings").

The correctness fixtures (Python reference generator, argmax/logit diff) already exist from
Phase 2; they are reused, not re-implemented.

## Facts verified during planning

- `WidenPrimitives.ShouldWiden<T>(len)` gates on: `UseWidenSimd` + `Vector.IsHardwareAccelerated`
  + `IsNarrowFloat` + `len >= Vector<byte>.Count * 4`.
- `NivaraPrimitives.UseWidenSimd` is a static property, default **off**; reads an
  `AppContext` switch `Nivara.Primitives.WidenSimd` OR a runtime field.
- `RunSmolLMCore<T>` saves the prior `UseWidenSimd`, sets it for the narrow run, and restores
  it in `finally`.
- SmolLM is **BF16-native on disk**; **Half is unusable for SmolLM** (NaN logits, 0/32 —
  documented in README). So the SmolLM A/B is **BF16-only**. The `--simd-widen` flag is
  still useful for Half on the other 4 models (which pass 8/8).
- Other transformer models already route `benchmark` (MiniLM, DistilBERT, SST-2). Only
  SmolLM lacks it.
- `docs/BFLOAT16.md`, `docs/BFLOAT16-TRANSFORMER.md`, and `samples/NivaraInference/README.md`
  are already current for Phases 0–2; they receive **results** after measurement, not
  code-plan edits.

## Proposed changes

### 1. `--simd-widen` CLI flag — `samples/NivaraInference/Program.cs`

- Parse `--simd-widen` in the args loop (next to `--precision`, around lines 28–46) into a
  top-level `bool simdWiden` variable (default `false`). Note: `--simd-widen` is a boolean
  flag, not a value-taking flag.
- Update the `--help` usage text and the precision section to document it.
- In `RunSmolLMCore<T>`: instead of the current unconditional narrow auto-enable, use the
  flag:
  - If `--simd-widen` passed → `UseWidenSimd = true` for the run (all precisions).
  - If not passed and narrow → keep the existing auto-enable (backward compat: bare
    `smollm --precision bf16` still "just works").
  - If not passed and F32 → leave off (WidenPrimitives is transparent for float; no-op).
- For the other narrow models (MiniLM, DistilBERT, SST-2 BF16/Half), thread the flag through
  their run entry points so `--simd-widen` opts them into the widen path (currently they
  never enable it). This is additive and gated (default off), so existing behavior is
  unchanged.

Threading concern: to keep the change minimal and avoid signature churn across many methods,
pass `simdWiden` into `RunSmolLMCore<T>` and the four narrow run/benchmark entry points that
need it, or set `UseWidenSimd` once early in `Main` under the same save/restore pattern used by
`RunSmolLMCore<T>`. Prefer the save/restore-in-Main approach to keep per-method signatures
stable, matching the existing `RunSmolLMCore<T>` pattern.

### 2. SmolLM `benchmark` mode — `samples/NivaraInference/Program.cs`

- Wire `benchmark` into the `smollm` switch case: currently all three precisions call
  `RunSmolLM(...)`; when `mode == "benchmark"`, route to a new `BenchmarkSmolLM<T>`.
- Add `BenchmarkSmolLM<T>` (private static) following the established median-of-3
  methodology:
  - Load model + tokenizer (same as `RunSmolLMGenerate`).
  - One warmup generation (discarded).
  - Three timed full 32-token generations; report median ms, ms/token, generated count.
  - Header with precision name, weight MB, and `UseWidenSimd` state.
- `mode == "benchmark"` handling must occur inside `RunSmolLMCore<T>` (which owns the
  save/restore of `UseWidenSimd`), so the benchmark respects `--precision` and `--simd-widen`.

### 3. A/B (scalar vs widen) comparison — `samples/NivaraInference/Program.cs`

- Add an `ab` sub-mode for SmolLM BF16 that runs the same generation twice: once with
  `UseWidenSimd = false` (scalar), once with `UseWidenSimd = true` (widen), and prints a
  side-by-side table (median ms, ms/token, generated-token argmax match vs PyTorch fixture,
  final-logits cosine).
- Implementation: in `RunSmolLMCore<T>`, when `mode == "ab"`, run the benchmark/generation
  under each toggle state and report both. Guard: for SmolLM only run A/B for BF16 (Half is
  unusable; F32 is not affected by the widen path). If `--precision fp16` or `f32` is passed
  with `ab`, print a note and fall back to a single benchmark run rather than produce a
  meaningless Half comparison.

Design decision (verified in README): **SmolLM A/B is BF16-only.** The doc explicitly says
"pick BF16 (native) or F32 — never Half — for SmolLM generation."

## Verification steps

1. `dotnet build Nivara.slnx` — must pass.
2. `dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 --simd-widen`
   — full 32-token generation with widen enabled; should match existing 22/32 argmax, 0.94 cosine.
3. `dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 benchmark`
   — prints median ms + ms/token over 3 generations.
4. `dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 ab`
   — side-by-side scalar vs widen table, both produce valid (non-NaN) tokens; widen should be
   faster per pass (the whole point of the work).
5. `dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision bf16 compare`
   (with and without `--simd-widen`) — argmax stays 8/8 both ways (regression).
6. `dotnet test tests/Nivara.Tests --filter WidenPrimitivesPhase3Tests` (pending human OK to
   run tests — ask first).

Note: steps 2–5 require the SmolLM-135M weights (~269 MB) to be present under
`samples/data/smollm-135m/`. If weights/Python fixtures are missing, note it explicitly and
do not fabricate numbers.

## Planned commits

1. `feat: add --simd-widen CLI flag for narrow-float SIMD in NivaraInference`
2. `feat: add SmolLM benchmark mode (median-of-3 generation timing)`
3. `feat: add SmolLM BF16 A/B (scalar vs widen) comparison mode`
4. `test: add WidenPrimitivesPhase3Tests for flag/benchmark/A-B`
5. `docs: record Phase 3 results in BFLOAT16-TRANSFORMER.md, BFLOAT16.md, README.md (+ CHANGELOG)`

## Blast radius

- **`samples/NivaraInference/Program.cs`** — the only core file edited. New methods
  (`BenchmarkSmolLM<T>`, A/B runner) + args parsing + runtime `UseWidenSimd` handling. Public
  model-loading code and the Python scripts are untouched.
- **`tests/Nivara.Tests/Primitives/WidenPrimitivesPhase3Tests.cs`** — new test file.
- **Docs** — results-only updates post-measurement.
- **No change to `src/Nivara`.** The widen kernels, `KernelSelector`, and `NivaraPrimitives`
  toggle are all Phase 1–2 work and are intentionally left as-is. The runtime toggle is read
  (not modified) by the sample.
- **Downstream callers:** the sample is a standalone console project; nothing in `src/Nivara`
  depends on these changes. No library API changes.
- **Tests covering affected behavior:** existing `WidenPrimitivesPhase1Tests` (toggle on/off,
  matmul). These must stay green — Phase 3 only adds CLI plumbing, it must not change kernel
  dispatch or toggle semantics.

## GitHub issues log

- (none yet — created at discovery time during execution)

---

*Remember: as each task executes, if you find deferred work or a concern (known limitations,
follow-ups, refactors) outside the current plan, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log above — don't
rely on memory or wait until the plan finishes.*
