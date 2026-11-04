# Plan: CI stabilize device-dependent tests and OpenCL classification runs (#902)

## Problem
Some CI runs for classifier/Silk PRs were canceled or queued, and OpenCL GPU devices are
not available on CI runners. Tests that require a device cannot reliably run there, and
a few tests crash on stale model files instead of skipping.

## Proposed changes
1. Introduce a `Gpu` category for device-dependent tests (all `tests/Nivara.Tests/Gpu/*`,
   `Incident/GpuIncidentTests`, `Incident/GpuOffloadDecisionTests`, and the four
   `Qwen/SilkQwen*` parity fixtures). Keep existing `Assert.Ignore` guards.
2. Exclude `Gpu` in CI filters (ci.yml, cd.yml) and document in AGENTS.md.
3. Make Qwen sentiment model loading resilient (wrap `ModeHelpers.LoadSentimentModel`).
4. Add job timeout and (optionally) concurrency hygiene to ci.yml.
5. Document local execution of device-dependent tests in docs/SILK.md.
6. Review DocsCoveragePins (no doc-count change expected; SILK.md already counted).

## Blast radius
- Test fixtures: ~20 files gain `[Category("Gpu")]`.
- Workflows: ci.yml, cd.yml filters.
- AGENTS.md: filter strings and guidance.
- docs/SILK.md: new section.
- Qwen test static constructors.

## Verification
- dotnet format whitespace verify
- targeted: `dotnet test ... --filter "Category=Gpu"`
- SlowTestCategorisationTests green

## Planned commits
1. docs: plan #902 in TODO.md
2. test: categorize device-dependent tests as Gpu
3. ci: exclude Gpu category from CI filters
4. test: make Qwen sentiment model loading resilient
5. docs: document running device-dependent tests
6. (fixes as needed)

## GitHub issues log
(none yet)
