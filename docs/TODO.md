# Plan: Issue #317 — Nivara × Streamix Integration Samples

## Problem

Issue #317 asks for 3 streaming scenarios in the NivaraIncident sample demonstrating the Nivara × Streamix bridge, plus a string-based `ToFluxWithTimestamp(columnName)` convenience overload (from the "richer bridge" feature ideas in #317's comment).

## Scope

### Phase A: String-based `ToFluxWithTimestamp` overload
- Add `ToFluxWithTimestamp(this QueryFrame, string timestampColumn, ...)` and `ToFluxWithTimestamp(this NivaraFrame, string timestampColumn, ...)` to `NivaraFlux.cs`
- Implementation: thin wrapper that builds the lambda internally: `row => row.GetValue<DateTimeOffset>(timestampColumn)`
- Unit tests in `StreamixBridgeTests.cs`

### Phase B: Streamix scenarios for NivaraIncident
- Create `samples/Nivara.Samples/Incident/StreamixScenarios.cs` with 3 static methods:
  1. `RunFaultTolerantStreaming(datasetPath, scenario)` — `ToFlux` + Retry + Checkpoint + ForEachAsync
  2. `RunWindowedAnalytics(datasetPath, scenario)` — `ToFluxWithTimestamp("Timestamp")` + WindowByTime + RollingMean per window
  3. `RunOnlineAutoDiffLearning(datasetPath, scenario)` — `ToFluxRows` + BufferFrames + Grad() + simple Linear model training
- Add Streamix package reference to `Nivara.Samples.csproj` (for `IFlux<T>`, operators)
- Wire into CLI: new `streamix` command in `Program.cs`

### Phase C: Tests
- Add `StreamixScenarioTests.cs` to `tests/Nivara.Tests/Incident/` covering all 3 scenarios
- Each test generates a small dataset, runs the scenario, asserts no exceptions and correct output shape

### Phase D: Documentation
- Update `samples/NivaraIncident/README.md` with Streamix scenarios section

## Files to modify/create

| File | Action |
|------|--------|
| `src/Nivara.Extensions/Streamix/NivaraFlux.cs` | Add string-based ToFluxWithTimestamp overloads |
| `tests/Nivara.Tests/Streamix/StreamixBridgeTests.cs` | Add tests for new overloads |
| `samples/Nivara.Samples/Nivara.Samples.csproj` | Add Streamix package reference |
| `samples/Nivara.Samples/Incident/StreamixScenarios.cs` | **NEW** — 3 streaming scenarios |
| `samples/NivaraIncident/NivaraIncident.Cli/Program.cs` | Add `streamix` command |
| `tests/Nivara.Tests/Incident/StreamixScenarioTests.cs` | **NEW** — tests for scenarios |
| `samples/NivaraIncident/README.md` | Update with Streamix section |

## Blast radius

- NivaraFlux.cs: public API addition only (new overloads), no changes to existing methods
- Nivara.Samples.csproj: new package reference (Streamix)
- CLI Program.cs: new switch case, no changes to existing commands
- Tests: new files only, no existing test modifications

## Verification

- `dotnet build Nivara.slnx` must pass
- Focused tests: `dotnet test --filter "FullyQualifiedName~StreamixBridgeTests.ToFluxWithTimestamp_String"` and `dotnet test --filter "FullyQualifiedName~StreamixScenarioTests"`
- No regressions in existing Streamix bridge tests

## GitHub issues log

- (none yet)
