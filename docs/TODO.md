# Plan: issue #1067 — probe whether classification counting can beat scalar hash pass

## Problem
`Metrics.ConfusionMatrix` and `Metrics.PerClassMetrics` record `KernelType.Scalar`. Docs state why: class
counting is a hash lookup per row with no SIMD histogram primitive. This is an honest description, not a
proven ceiling. Two hypotheses require measurement before anyone writes a kernel:

- **H1 — partial vectorisation.** `Correct` and `Support` do not need a histogram: for numeric labels
  they are an element-wise compare plus a count. Only off-diagonal cells need the matrix.
- **H2 — dense histogram.** For small `k` with dense integer labels `0..k-1`, a `long[k]` indexed by
  label should beat `Dictionary<T, int>` hashing; for strings or sparse `k`, hashing should win.

See `src/Nivara/Metrics/ClassificationMetrics.cs::count` for the algorithm being measured.

## Proposed changes
1. Add `tests/Nivara.SimdProbe/MetricsCountingProbe.cs` (`ClassificationCountingProbe`) with a `"metrics"`
   mode entry point `Run()` (returns int failures, prints tables).
2. Wire `"metrics"` into `tests/Nivara.SimdProbe/Program.cs` (early dispatch before the BFloat16 banner,
   matching `chunked-arrow`).
3. Document the mode in `tests/Nivara.SimdProbe/README.md`.

### Scope details
- Scenarios: `n ∈ {1e4, 1e5, 1e6}`, `k ∈ {2,10,100,1000}`, labels `int` and `string`.
- Mismatched predicted/actual arrays so the matrix is not trivially diagonal.
- Routes:
  - Baseline scalar hash pass (faithful reproduction of `ClassificationMetrics.count`).
  - H2 dense (int only): direct flat-matrix indexing under dense `0..k-1` labels.
  - H1 marginal (both types): only `Tp` + marginals, no full matrix.
- Methodology: interleaved A/B, alternating first route, warmups, median ratio; Release-build guard.

### Blast radius
- `tests/Nivara.SimdProbe/Program.cs`, new `MetricsCountingProbe.cs`, README.
- No `src/` changes; no docs-coverage pin changes; no new NuGet deps.

## Verification steps
1. `dotnet build -c Release tests/Nivara.SimdProbe/Nivara.SimdProbe.csproj`
2. `dotnet run -c Release --project tests/Nivara.SimdProbe -- metrics` (human-run, ~minutes) — capture conclusion on
   issue #1067.
3. Update README counts? N/A.
4. Run `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` after edits.

## Planned commits
1. `docs: plan issue 1067 in TODO.md`
2. `feat(probe): add metrics classification-counting mode for issue 1067` (Program + probe + README)

## GitHub issues log
- (none yet; deferred concerns captured as found)
