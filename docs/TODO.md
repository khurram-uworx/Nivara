# #480 — Measure `ApplyMask`'s scalar cost against the `TensorPrimitives.Add` it replaced

Branch: `khurram/480`

## Problem

`AttentionKernels<T>.ApplyMask` (added by #448) replaced a vectorized
`TensorPrimitives.Add` over the `[qLen, kvLen]` score buffer with a scalar
compare-and-select loop, because BCL `TensorPrimitives` has no select/blend
primitive. The justification on record is an **inference from a MAC count**, not
a stopwatch reading:

- `docs/LAYA.md` §3 item 3 records attention at **0.13%** of Laya's MACs, and
  excludes attention from the backend decision on those grounds.
- The plan for #448 deliberately added no perf gate, on the grounds that the
  pass is only worth measuring if it ever shows up.

`GUIDELINES.md` (*Verification, Evidence & Claim Discipline* → *Superseded
Measurements Move Together*) asks that an unverified measurement be moved, not
left as a hunch. The hunch currently lives in the kernel's own XML doc
(`AttentionKernels.cs:122-124`): *"A scalar loop rather than a `TensorPrimitives`
call, because BCL has no select/blend primitive … Tracked as #480."*

This issue is **measurement only**. #480 explicitly says: *"Do not pre-commit to
(b). The point of the measurement is to find out whether there is a problem, not
to justify a change already chosen."* No `src/` behaviour change in this branch.

## What the code actually does (grounded before planning)

- `ApplyMask` has two overloads sharing one `MaskedCell` helper
  (`AttentionKernels.cs:213`): `if (mask == T.NegativeInfinity) return -inf;`
  else `score + mask`, plus `dead = mask != 0 && masked == mask`.
- **All production call sites use the 5-arg tracking overload**:
  `ReverseGradOperations.cs:584, :837`, `ForwardGradOperations.cs:1622, :1834`.
  The 2-arg overload (`:132`) is test-only. Both branches of the kernel
  (`scoreIndependentRows.IsEmpty` at `:178`, per-row `dead` at `:186`) are live:
  inference passes empty flags, training passes populated ones.
- **The decode paths never mask this way.** `DecodeAttention` (`:250`) and
  `BatchedAttention` (`:411`) use a `Keep` predicate and never call `ApplyMask`.
  So a `[1, 512]` row measures *the mask API at decode shape*, not the decode
  path's cost — a scoping caveat the output must carry.
- On a causal mask ~50% of cells are `-inf`, so `MaskedCell`'s
  `mask == NegInf` branch is maximally unpredictable. That is a plausible large
  share of any gap versus a branchless vectorized add, and it is exactly the
  kind of thing a MAC-count inference cannot see.

## Proposed changes

### 1. New opt-in mode: `--mask` in `tests/Nivara.PerformanceTests`

New file `tests/Nivara.PerformanceTests/ApplyMaskProbe.cs`, mirroring the
`--gemm-legs` / `--cpu-gemm` precedent (`Program.cs:56-65` dispatch,
`ParseArgs` at `:1017`, usage string at `:1072`).

It calls the **real** `internal AttentionKernels<T>.ApplyMask` and
`GradKernels` — `src/Nivara/Nivara.csproj:24-26` already grants
`InternalsVisibleTo("Nivara.PerformanceTests")`. **No kernel copy, no `src`
edit, no new project reference.** Deliberately *not* a `Nivara.SimdProbe` mode:
SimdProbe has no Nivara reference and would need a verbatim copy of the kernel,
which cannot measure the real generic-over-`T` `ApplyMask`.

Profiling, not a gate: exits 0, asserts only parity. Not added to the scenario
table, so no baseline-JSON churn and no permanent rows in the rolling Results
table.

### 2. Leg 1 — kernel A/B: `ApplyMask` vs `TensorPrimitives.Add`

Route A: in-place `TensorPrimitives.Add(scores, mask, scores)` (literally what
#448 replaced). Route B: `ApplyMask(scores, mask, rowFlags, rows, cols)`.

Methodology copied from `TransposeKernelProbe`
(`tests/Nivara.SimdProbe/TransposeKernelProbe.cs`) because **#482 is the
precedent for getting this wrong** — its doc comment (`:16-37`) names the three
structural defects: best-of-N per route compared once, sequential routes with
one always first, and no warmup.

| Aspect | Choice | Why |
|---|---|---|
| Statistics | interleaved A/B, `Rounds = 30`, alternate first-measured route each round | cancels first-measured systematic drift |
| Warmup | 5 untimed passes **per route per `T`** | `ApplyMask<T>` is generic-specialized; tiered JIT settles per instantiation |
| Estimate | median ratio + win rate + range | distribution, not a point estimate |
| Reported **per cell** | median **ns/op**, **ns/element**, **GB/s**, *and* the ratio | see G1 finding 3 below — a bare ratio cannot separate a bandwidth-bound parity from a compute-bound one |
| Verdict band | ±3% = noise (reuse the transpose probe's band) | do not invent a threshold per probe |
| Build | `IsDebugBuild()` guard + `BuildConfigurationWarning` | Debug numbers are void, not a result (#482) |

Grid: shapes `[512,512]`, `[2048,2048]`, `[1,512]` (the issue's three) × types
`float`, `double`, `Half`, `BFloat16` (ADR-001's `IFloatingPointIeee754`
domain) × mask regimes `all-zero` / `causal (~50% -inf)` / `all -inf` (isolates
branch-prediction cost from loop overhead) × `rowFlags` empty vs populated.

### 3. Parity gate before any timing

`TransposeKernelProbe.cs:164-166` refuses to time routes that disagree. These
routes **legitimately** disagree — that is #448's fix (`NaN + (-inf) = NaN`).
So instead:

- **Assert exactness where it is available**: over all-finite scores with mask ∈
  `{0, -inf}`, assert bit-identical output vs `TensorPrimitives.Add` for every
  shape × type. Exact, so asserted **exactly** — no tolerance band (per *Assert
  Exactness When Exactness Is Available*). Matches the zero-delta contract the
  suite already encodes (`AttentionKernelsTests.cs:146`).
- **Document the two intentional divergences** (NaN score, `+inf` score under a
  `-inf` cell) rather than pretend they do not exist.
- Note open **#489** (disputes `MaskedCell`'s `dead` rule) so a future change
  under this probe is not mistaken for probe rot.

Parity failure aborts before timing, on a **separate exit path** from a numeric
verdict (per *One Failure Reason Per Catch, Per Counter, Per Message*).

### 4. Leg 2 — CPU attention leg profile (the "end to end" clause)

A kernel ratio cannot answer *"is it within noise end to end"*, so attribute one
Laya-shaped forward across its legs — the CPU analogue of `--gemm-legs`.

Shape **B=1, S=512, d=1024, nhead=16, headDim=64** (`docs/LAYA.md:13,16`),
28 encoder layers. Legs per `ReverseGradOperations.cs:570-595`:
`PackHeads` ×3 → per head `MatMulTransposedB` (QK^T) →
`TensorPrimitives.Multiply` (scale) → **`ApplyMask`** → `SoftmaxRows` →
`MatMul` (PV) → `ScatterHead`. Forward-only (empty `rowFlags`, no saved-weights
copy). Laya's CPU path is `ModernBertModel.cs:307` →
`ReverseGradOperations.MultiHeadAttention`.

Output `ApplyMask` as a **percent of attention** and a **percent of the Laya
forward**, so the Leg-1 ratio converts into a materiality verdict.

> **Executed with one deliberate deviation.** The percent-of-attention is
> reported (1.79%). The **percent-of-forward is not**, because the denominators
> do not survive contact with the measurement: this run's own leg profile puts
> attention *alone* at 316–389 ms/layer = 8.9–10.9 s per forward, which cannot fit
> inside `docs/LAYA.md`'s recorded ~4.1 s whole forward, and the README's 2414 ms
> figure is a **GEMM-only projection that excludes attention** — mis-cited as a
> forward total in the first implementation of this probe. The probe now prints
> the reconciliation conflict instead of a percentage. Filed as **#492**.
>
> The plan's premise that a ratio plus a denominator yields a materiality verdict
> assumed the denominator was sound. It was not, and finding that out is part of
> what the measurement was for.

### 5. What gets recorded — one change, all sites

- `AttentionKernels.cs:122-124` — replace the *inferred* justification with the
  measured ratio and its conditions. This is the site the supersession
  discipline actually targets.
- `tests/Nivara.PerformanceTests/README.md` — new `--mask` section: shapes,
  grid, methodology, recorded baseline, noise band, parity contract, `[1,512]`
  scoping caveat.
- `docs/LAYA.md` item 3 (line 183) — **only if** the measurement contradicts it.
  The 0.13% is a *MAC* share and does not move from this measurement; if the leg
  profile shows the mask is material in *time* despite 0.13% of MACs, that is the
  same MAC-share-≠-time-share lesson as `--gemm-legs` (attention at 57% of time)
  and the doc must say so.
- Issue #480 — the verdict, coverage stated as *N of M cells*, reduced coverage
  counted as a failure rather than a neutral event.

### 6. Decision point (measured, then decided)

- Ratio within noise, or mask share of Laya below the measurable floor → close
  with the measurement recorded; the scalar loop settles.
- Material → open a **separate** issue carrying the number. That issue decides
  the "no hand-rolled SIMD" convention (`TensorsHelper.cs:107,132`) explicitly
  for this one kernel, and covers the type-dispatch fallback the issue itself
  identifies (`ConditionalSelect` is not uniform across `Half`/`BFloat16`).
  Do **not** implement the blend here.

## Probe-existence check (skill step 0 — why a new mode, not an extension)

Confirmed no existing mode covers the CPU `ApplyMask`:

- `tests/Nivara.SimdProbe/Program.cs` modes: `cpu`, `support`, `correctness`,
  `benchmark`, `scalar`, `transpose`. `transpose` is the A/B-methodology
  template but measures `TensorsHelper.Transpose`; `scalar` measures
  GEMV/attention-V/rotary at Laya layers, not the score mask.
- `tests/Nivara.PerformanceTests` flags: `--dataset-test`, `--safetensors-mmap`,
  `--gemm`, `--gemm-legs`, `--gpu-alloc`, `--cpu-gemm`.
- **Nearest neighbour rejected:** `GemmLegBenchmark.cs:134-148` allocates a
  `maskBuf`, but that is a **GPU** mask for the ILGPU `BatchedAttention`
  (`samples/Nivara.Samples/Gpu/AttentionKernels.cs`) — a different kernel that
  never calls `AttentionKernels<T>.ApplyMask`. `--gemm-legs` attributes GPU time
  and cannot see the CPU mask at all.

So a new `--mask` mode in `Nivara.PerformanceTests` is justified rather than an
extension, and the temp-harness path is skipped entirely (nothing novel to
stage: the harness is a known shape and lands directly in its permanent home).

## Grounding (G1) — microsoft-learn + code-memory

1. **In-place `TensorPrimitives.Add` is legal.** The `Add<T>(x, y, destination)`
   overload documents its overlap exception as *"reference overlapping memory
   locations **and do not begin at the same location**"* — exact aliasing is
   explicitly permitted. Route A is valid as written. Same page corroborates the
   divergence: *"If either of the element-wise input values is equal to NaN, the
   resulting element-wise value is also NaN."*
2. **The issue's reason #2 for not using a blend is confirmed.** The documented
   element set for `Vector128<T>`/`Vector256<T>`/`Vector512<T>` is
   `byte, sbyte, short, ushort, int, uint, long, ulong, float, double, nint,
   nuint`; `ConditionalSelect<T>` throws `NotSupportedException` otherwise.
   **`Half` and `BFloat16` are not supported**, so a `ConditionalSelect` fast
   path cannot cover ADR-001's domain without a type-dispatch branch.
3. **Red flag → probe must report absolute throughput, not just a ratio.** The
   SIMD guidance states *"Speedups are rarely perfect… memory throughput,
   alignment, and instruction latency all factor in"* and that a 256-bit vector
   on 32-bit elements will not reliably be 8× faster. Both routes move identical
   bytes over identical spans, so two possibilities must be distinguished:
   - the pass may be **bandwidth-bound**, in which case scalar and vectorized can
     reach parity and the ratio means little without GB/s to prove saturation;
   - `scores[i] = MaskedCell(...)` is a select the **JIT may already
     auto-vectorize**, making "scalar" a misnomer and part of the issue's premise
     wrong. GB/s plus ns/element makes this falsifiable.

   The direction of the result is therefore **not** pre-declared; the measurement
   decides. Human confirmed the amended output shape.
4. Type domain confirmed against the suite: `float`, `double`, `Half`,
   `BFloat16` (`AttentionKernelsTests.cs:258-265`).

## Blast radius

- **New**: `tests/Nivara.PerformanceTests/ApplyMaskProbe.cs`.
- **Modified, harness-only**: `tests/Nivara.PerformanceTests/Program.cs` —
  `Main` dispatch (one `if (mask) return ApplyMaskProbe.Run();`), `ParseArgs`
  signature + switch, usage string. The flag is opt-in and returns before
  `RegisterScenarios()`, so **no existing scenario row, baseline JSON, or
  `--compare` gate changes behaviour**.
- **Documentation only**: `tests/Nivara.PerformanceTests/README.md`,
  `AttentionKernels.cs` XML comment, `docs/LAYA.md` (conditional).
- **No `src/` behaviour change.** No public API change. No runtime
  characteristic touched. Nothing in CI's path: no NUnit timing test is added,
  so `--filter "Category!=Performance"` is unaffected.
- **Callers of `ApplyMask`** (unchanged, listed for regression awareness):
  `ReverseGradOperations.cs:584, :837`, `ForwardGradOperations.cs:1622, :1834`.
  Covered today by `tests/Nivara.Tests/AutoDiff/AttentionKernelsTests.cs`
  (20 `ApplyMask_*` tests) and `AttentionMaskTests.cs`.
- **Risk**: measurement quality, not correctness. The failure mode to avoid is
  reporting a single best-of-N ratio as fact — the #482 trap.

## Execution record

All steps run; results as measured, not as hoped.

| step | result |
|---|---|
| 1. `dotnet build Nivara.slnx -c Release` | clean, 0 warnings |
| 2. `--mask` in Release | ran, exit 0 |
| 3. parity N of N | **72 of 72** bit-identical |
| 4. `dotnet test -c Release` | **Passed: 3654, Failed: 0, Skipped: 14** (3 m 6 s). The 14 skips are checkpoint/fixture-dependent and unrelated. |
| 5. re-run to confirm the ratio is not a coin flip | ran 3× total. Absolute delta reproduces to **0.02 ms** (129.54 / 129.52); `float` wide-prefill holds at 0–1 of 30 rounds won. But the attention leg total swung 389→316 ms/layer and one cell straddled the noise band — **recorded as run-to-run variance rather than smoothed over.** |

### Coverage

- **Parity: 72 of 72** cells bit-identical.
- **Timing: 48 of 72** measurable. **24 excluded** — every `[1,512]` cell, below clock resolution. The first implementation reported those as 5–10× "regressions" that were pure quantization; that was a defect in the probe, found and fixed, not a finding.

### Defects found in this probe and fixed before recording

1. Published below-timer-resolution ratios as if they were results.
2. Divided by zero-tick samples, which could poison the median via `Array.Sort` NaN ordering.
3. Computed a "% of forward" from a denominator this same run contradicts, and mis-labelled a GEMM-only projection as a forward total.
4. Printed `NoiseBand` but never applied it, and counted `ratio > 1.0` as a regression — so a 1.02 cell would have been reported as one.
5. Warmed the leg profile on different buffers than it timed.
6. Derived a per-layer multiplier by string-comparing a label.
7. Said "cells below disagree" for a per-cell property, over-claiming the damage.

Each was a way this probe could have made a false claim. None reached the record.

## Verification steps

1. `dotnet build Nivara.slnx` — compiles clean.
2. `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --mask` —
   on an idle machine, AC line. **Release only**: in Debug the handwritten kernel
   runs unoptimized while `System.Numerics.Tensors` ships ReadyToRun and stays
   optimized, which is the exact configuration that made #482 fail 3/5 on a
   clean tree.
3. Parity leg must report **N of N** cells bit-identical; any shortfall is a
   failure, not a neutral event.
4. `dotnet test -c Release` — full suite green (confirms no behaviour change).
   **Ask the human before running.**
5. Re-run `--mask` twice to confirm the recorded ratio sits inside the reported
   noise band and is not a coin-flip ordering.

## Planned commits

1. `docs: plan #480 ApplyMask measurement in TODO.md`
2. `test: add --mask probe measuring ApplyMask against TensorPrimitives.Add`
3. `docs: record the measured #480 ratio and update the ApplyMask justification`

## GitHub issues log

- [x] **#491** — decide the `ApplyMask` fast path. Carries the measurement. Created while
      working on the decision step, because the measurement is material on `float` (1.8-3.3x,
      0 of 30 rounds lost) but **actively constrains** the obvious fix: `ApplyMask` is currently
      *faster* than the BCL add on `Half`/`BFloat16`, and `Vector256.ConditionalSelect<T>` throws
      `NotSupportedException` for those types, so the fast path cannot be spelled for ADR-001's
      narrow domain without a type dispatch that must preserve the win.
- [x] **#492** — Laya CPU forward time does not reconcile with the attention leg profile
      (10 900 ms attention alone vs a 4 100 ms recorded whole forward; the 2414 ms README figure
      is GEMM-only and excludes attention). Created while working on Leg 2. Until resolved, no
      percentage-of-forward is defensible, and `docs/LAYA.md` §3's MAC-share exclusion of
      attention rests on an unverified time-share inference.
- [ ] #489 (pre-existing, open) — disputes `MaskedCell`'s `dead` rule. It does **not** affect
      the probe's parity contract (which is stated over mask in `{0,-inf}` with all-finite
      scores), but the tracking-flag rows are the ones that would move if it lands.

> As each task executes, if you find deferred work or a concern (known
> limitations, follow-ups, refactors) outside this plan, create a tracked issue
> immediately via `gh issue create --repo khurram-uworx/Nivara` and record its
> number in the log above. Do not rely on memory or wait until the plan finishes —
> compaction during execution can lose it.