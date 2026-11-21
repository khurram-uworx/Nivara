# Plan: Issue #1038 — percent-rank Silk kernel (Option A: stays on CPU)

## Problem

Issue #1038 tracks a Silk/OpenCL percent-rank kernel for NivaraIncident's regional
partitioning analysis (`AnalyzeRegionalPartitioning` → `PercentRank("DurationPercentRank", …, "Region")`).
The CPU reference is `RankKernel.Compute` with `RankKind.PercentRank`.

Percent rank is not a per-element pass: it requires partition-local ordering. Silk has no
sorting primitive, and a naive counted-ranking kernel is O(n²) per partition. Sibling
incident kernels on the shared iGPU are all slower than CPU (rolling mean 16.4×, rolling max 9.3×,
multi-stat 4.3×). The issue's maintainer comment already recommends staying on CPU unless a
discrete-GPU speedup or a clean OpenCL sort primitive becomes available.

## Decision (Option A)

Do **not** implement a Silk percent-rank kernel. Keep `RankKernel` as the reference and
correctness baseline. Document the rationale and the decision in the incident Silk availability
table and in the issue.

## Proposed changes

1. `samples/NivaraIncident/README.md` — add a `percent_rank` row to the "Silk (OpenCL)
   acceleration → Availability and status" table, marked "No — stays CPU; see #1038", with
   measured-status placeholder "n/a (not prototyped)" and notes citing the rationale.
2. Add a short "Percent rank" note under the Silk section explaining why it stays CPU
   (partition-local sort requirement, no OpenCL sort primitive, sibling-kernel cost evidence,
   CPU `RankKernel` remains reference).
3. Comment on / close issue #1038 recording the decision.

No kernel, host driver, tests, or `--silk` wiring changes.

## Blast radius

- Documentation only (`samples/NivaraIncident/README.md`). No source, tests, or public API.
- No `DocsCoveragePins` impact: only `samples/` README, not `docs/*.md`.
- No build/test surface affected beyond doc snippet gate (README not gated).

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (no code changes, should pass).
- Targeted docs tests if any README-adjacent; none expected.
- Review diff for accuracy.

## Planned commits

1. `docs(incident): document percent-rank stays on CPU (issue #1038)` — README update.
2. `docs: remove TODO.md — plan executed`.

## GitHub issues log

- [ ] #1038 — percent-rank Silk kernel (documented rationale; stays CPU) (this issue)
