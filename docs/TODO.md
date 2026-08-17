# Issue #292 — Incident sample: Polars cross-validation fixtures for incident data

## Problem

`samples/NivaraIncident/Python/gen_reference.py` only contains generic rank/rolling/quantile
test cases on small fixed arrays. Phase 3a step 3a.6 called for incident-specific fixtures that
exercise the same APIs on realistic telemetry distributions.

## Required fixtures

1. **Latency percentiles (P50/P95/P99) per service** — compare with `NivaraSeries<double>.Quantile(q)`
2. **Error-rate rolling windows per service** — compare with `PartitionedWindowEngine` → `RollingMean`
3. **Rank/PercentRank per service by error delta** — compare with `RankKernel.Compute` → `Rank`/`PercentRank`
4. **StdDev per service** — compare with `NivaraSeries<double>.StdDev(ddof)`

## Data approach

Hand-authored fixed arrays (no RNG), matching the existing generic fixture pattern. Values are
chosen to be plausible telemetry (latency in ms, error rates as percentages, etc.). 4-5 services,
10-20 values each, with nulls interspersed.

## Changes

### Step 1: Extend `gen_reference.py`

Add `emit_incident_fixtures(pl)` function that:
- Creates 4-5 services with hand-authored data arrays
- Computes expected values via Polars (same semantics as existing fixtures)
- Writes `samples/data/polars-incident/manifest.json`

Fixture JSON shape per case:
```json
{
  "name": "latency_p99_per_service",
  "kind": "quantile_per_service",
  "services": { "gateway": [12.3, 45.1, null, ...], ... },
  "q": 0.99,
  "expected": { "gateway": 89.2, ... }
}
```

### Step 2: Run `python gen_reference.py`

Generate the manifest. Verify all 4 categories produce expected values.

### Step 3: Create `PolarsIncidentCrossValidationTests.cs`

NUnit test class in `tests/Nivara.Tests/Query/`:
- Reads `samples/data/polars-incident/manifest.json`
- For each `kind`, creates Nivara columns/series and asserts parity with Polars expected values
- Tolerance: 1e-9 (same as existing cross-validation tests)

### Step 4: Build verification

Run `dotnet build Nivara.slnx` to verify clean compilation.

## Blast radius

- **Modified:** `samples/NivaraIncident/Python/gen_reference.py` (add function, call from `run()`)
- **Created:** `samples/data/polars-incident/manifest.json`, `tests/Nivara.Tests/Query/PolarsIncidentCrossValidationTests.cs`
- **No core library changes** — test-only + fixture generation
- **Downstream:** existing `Polars*CrossValidationTests` classes are unaffected (different fixture directory)

## Verification

1. `python gen_reference.py` produces `samples/data/polars-incident/manifest.json` with all 4 kinds
2. `dotnet build Nivara.slnx` clean
3. `dotnet test --filter PolarsIncidentCrossValidation` passes (after human confirmation)

## Planned commits

1. `docs: plan incident Polars cross-validation fixtures in TODO.md`
2. `feat(incident): add incident-specific Polars fixture generation to gen_reference.py`
3. `test: add PolarsIncidentCrossValidationTests`

## GitHub issues log

- (none yet)
