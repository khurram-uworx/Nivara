# v1.5.0 Release Retrospective

Observations and improvement ideas from the v1.5.0 release process. Not
action items yet — internal discussion first.

## What went well

- **Coverage gates caught stale documentation during prep.** Compressing
  `CHANGELOG.md` dropped two tolerated historical mentions and several
  citations; `DocReferenceTests.ToleratedHistoricalMentions_*` and
  `DocsCoverageTests` failed immediately and named the dropped entries,
  so the allowlist and pins were reconciled in the same commit rather
  than discovered in CI.
- **The flaky Streamix backpressure bridge test is now deterministic.**
  Replacing the `Task.Yield`-paced producer with a completed-task await
  lets the capacity-1 boundary overflow on the second write regardless
  of scheduler timing; five consecutive runs passed.
- **Release-prep fixes from the 1.4 review landed.** `--quick` on
  `NivaraIncident` gives a single-pass benchmark for large datasets;
  Python vision benchmarks pin `torch.set_num_threads(os.cpu_count())`;
  `RELEASING.md` now requires a green CI check before merge and documents
  the cross-machine Prev/Current rule.
- **Build+pack succeeds on a stable 1.5.0 version** over the .NET 11 RC
  dependency set by overriding the `NU5104` warning for the prerelease
  dependencies already accepted by the repo's .NET 11 upgrade.

## Bugs / Tech Debt

### 1. Benchmark run was not fully captured

The core perf harness run exceeded the shell timeout during prep, and the
model-weight-dependent benchmarks (NivaraInference, NivaraFineTuning)
were not runnable in this environment. Per RELEASING.md step 5 those
numbers are left as-is and the reason is recorded here rather than being
carried forward as fresh readings.

**Fix options:**
- Split the harness into a machine-runnable quick mode, or document
  expected runtimes in each `## Release Benchmark` section.
- Cache and commit benchmark JSON from a contributor machine where the
  weights are available.

### 2. `docs/` coverage accounting is mechanical

The six pins in `DocsCoveragePins` move whenever citations are removed or
added, so a changelog compression necessarily looks like "reduced
coverage". The pin bundle obscures whether the shrinkage is intentional
(typography cleanup) or accidental (a doc stopped citing the code it
describes). Manual reasoning required; the test cannot tell.

**Suggestion:** Split the pin into "live docs" and "CHANGELOG" scopes so
that release bookkeeping does not penalise source documentation.

### 3. Stable package depends on prerelease dependencies

`System.Numerics.Tensors 11.0.0-rc.*` (and ML/Parquet previews) forced
`NU5104` to be suppressed via `WarningsNotAsErrors`. NuGet.org's stable-
versus-prerelease guidance is not a build-time gate but a consumer-facing
contract; the override is undocumented in the csproj.

**Suggestion:** Add an ADR or csproj comment explaining that the .NET 11
RC framework temporarily forces prerelease dependencies, and revert the
override once .NET 11 ships stable.

## Release Process

### 4. Docs refresh step was applied unevenly

Step 2 of RELEASING.md asks to review README, GETTING-STARTED,
ARCHITECTURE, AUTODIFF, EXAMPLES, samples README, AGENTS and ADRs.
Only AGENTS gained a one-line v1.5 summary and the changelog was
condensed; deep per-file review was not performed. That defers work the
release doc explicitly calls for.

**Suggestion:** Schedule the doc refresh as a separate, pre-merge
checklist item with named owners, or split README/GETTING-STARTED updates
out of the release branch.

### 5. Version arithmetic across the preview line

Moving from a `2.0.0-preview` development version back to a stable
`1.5.0` left the.csproj working-tree version behind the dev line; the
next dev bump must reset to `2.0.0-preview` (or `1.6.0-preview`) after
the tag is pushed, otherwise local packs report 1.5.0 on main.

**Suggestion:** Document the post-tag reset in RELEASING.md step 9.

---

*Recorded during the v1.5.0 release prep on 2026-11-07.*
