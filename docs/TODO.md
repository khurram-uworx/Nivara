# TODO: #1506 - Refresh benchmark README tables from v1.5.4 artefacts

## Problem

GitHub issue [#1506](https://github.com/khurram-uworx/Nivara/issues/1506): benchmark README tables were partially refreshed or deferred in prior releases. For this pass, the scope is **one-time**: refresh benchmark README tables once using last release data, i.e. **`release/v1.5.4/`** committed artefacts.

Before implementing, the human requested an additional process artifact: save the implementation plan as `release/TASK-TEMPLATE-BENCHMARKS.md` and update `RELEASING.md` so future 1506-like benchmark-table-refresh tasks use that template for a consistent/predictable/deterministic release process.

## Scope / decisions

- **In scope:** one-time table refresh from `release/v1.5.4/` artefacts; add reusable task template; wire RELEASING.md to template; fix clear `release/run-benchmarks.cmd` gaps found while auditing missing artefacts.
- **Out of scope:** rerunning benchmarks on this machine (different/older machine, no Intel Arc iGPU; readings would be non-comparable). Human will run remaining benchmarks from the original benchmark machine later.
- **Rule:** no fabricated numbers. Where v1.5.4 has no capture for a table/row, **skip the update** and leave existing values as carried/not refreshed with an explicit reason. Record missing artefacts in an issue #1506 comment so they can be captured later on the correct machine.
- **Confirmed by human:** skip unavailable data; comment on issue #1506; do not take readings here; check/fix runner gaps.

## v1.5.4 artefacts (authoritative)

`release/v1.5.4/status.txt` present. Sources:
- `release/v1.5.4/nivara-perf-v1.5.4.json` (2026-11-25T05:07:03Z, 16 logical processors, .NET 11.0.0, runs=3)
- `release/v1.5.4/nivaraincident-v1.5.4.txt`
- `release/v1.5.4/nivarainference-cpu-v1.5.4.txt`
- `release/v1.5.4/nivarainference-silk-v1.5.4.txt`
- `release/v1.5.4/nivarainference-blas-v1.5.4.txt`
- `release/v1.5.4/nivarafinetuning-v1.5.4.txt`
- `release/v1.5.4/pytorch-minilm-v1.5.4.txt`
- `release/v1.5.4/pytorch-distilbert-v1.5.4.txt`
- `release/v1.5.4/pytorch-modernbert-v1.5.4.txt`
- `release/v1.5.4/pytorch-smollm-f32-v1.5.4.txt`
- `release/v1.5.4/pytorch-smollm-bf16-v1.5.4.txt`

## Proposed changes

### A. Add reusable template + wire RELEASING.md (before table refresh)

1. **`release/TASK-TEMPLATE-BENCHMARKS.md` (new)** — deterministic task template for "refresh benchmark README tables from release/vX.Y.Z artefacts". Include:
   - Purpose/scope, inputs (version, artefact folder, artefact files), non-goals.
   - Table inventory mapping each benchmark table to source artefact(s), update rule, and carried/blocked handling.
   - Step-by-step execution checklist (verify artefacts, update tables, recompute ratios/derived estimates, update recording/footnote lines, verify, PR checklist).
   - PR body/test-plan skeleton and "no fabricated numbers / no deferral as normal path" rules.
   - Explicit note that release retros go to `release/vX.Y.Z/RELEASE-REVIEW.md`, not `docs/`.

2. **`RELEASING.md`** — update Step 5 benchmark refresh issue instruction to reference `release/TASK-TEMPLATE-BENCHMARKS.md` as the canonical template/body/checklist source for 1506-like tasks. Keep existing Step 5 rules; add clear pointer so issue creation is consistent/predictable.

### B. Refresh benchmark README tables from v1.5.4 (available data only)

3. **`tests/Nivara.PerformanceTests/README.md`**
   - Source: `nivara-perf-v1.5.4.json`.
   - `## Results` table: roll Current -> Prev; set Current from v1.5.4 `opsPerSec`; recompute Δ%; update B/op/gen0 from artefact where changed; update recording/machine lines to 2026-11-25 v1.5.4 capture.
   - Verify scenario name mapping exactly (incl. `Streaming cancel mid-stream 200k rows x 10k chunk`).

4. **`samples/NivaraIncident/README.md`**
   - Source: `nivaraincident-v1.5.4.txt`.
   - Nivara-only table: roll Current -> Prev, set Current from v1.5.4 medians (Degradation 15790.8, Deployment 28572.9, Saturation 142.2, Regional 28585.1, Grouped 12533.2, Typed LINQ 27022.1), recompute Ratio/Δ%, update recording line.
   - Nivara vs Polars: **no v1.5.4 Polars artefact**. Skip refresh; add explicit carried/not-refreshed reason. No fabricated rerun, no cross-basis ratio changes.

5. **`samples/NivaraInference/README.md`**
   - Sources: cpu/silk/blas/pytorch v1.5.4 txt artefacts.
   - Update OpenBLAS settings/end-to-end tables and main Performance benchmarks table from v1.5.4 captures where available. Update footnote to reference `release/v1.5.4/*` artefacts and 2026-11-25 capture details.
   - Skip/carry rows with no v1.5.4 capture (e.g. MobileNet/ResNet PyTorch, DistilBERT SST-2 PyTorch, SmolLM F32 CPU/PyTorch if missing from cpu artefact, any other missing capture) — explicit reason, no cross-session ratio recomputation. Preserve Qwen prompt caveat (CPU 206-token tool prompt vs Silk 36-token synthetic prompt).
   - Keep table formatting/readability; avoid unrelated edits.

6. **`samples/NivaraFineTuning/BERT-TUNING.md`**
   - Source: `nivarafinetuning-v1.5.4.txt`.
   - Main benchmark table: PyTorch 410.2 ms/batch, Nivara CPU 1858.7 ms/batch, Silk 253.1 ms/batch; recompute slowdowns; update recording line to v1.5.4 capture.
   - Estimated full-run table: recompute from refreshed measured ms/batch; keep disclaimer.
   - Paired CPU vs Silk #1003 B=2/4/8 table: not in v1.5.4 artefact — keep carried with explicit reason/date; do not fabricate.

### C. Runner gap fixes and issue comment

7. **`release/run-benchmarks.cmd`** — audit and fix clear gaps causing missing artefacts:
   - Add missing `smollm --precision f32 benchmark` CPU capture (README lists it; v1.5.4 cpu artefact lacked it).
   - Align NivaraIncident Nivara-only capture with README Release Benchmark command (`--benchmark` not `--quick`) where appropriate.
   - Add Polars comparison command/capture path (or clearly documented manual step) so future releases can produce the Nivara vs Polars artefact.
   - Keep .cmd CRLF/escaping rules; keep changes minimal and deterministic.

8. **Issue #1506 comment** — record: different/older machine, no Intel Arc iGPU, no readings taken here; v1.5.4 artefacts used where available; tables/rows skipped due to missing data; runner gaps found/fixed; remaining benchmarks to be run from original benchmark machine later.

## Blast radius

| File | Change | Risk | Downstream |
|---|---|---|---|
| `release/TASK-TEMPLATE-BENCHMARKS.md` | New template doc | Low; release tooling/doc only | Used by RELEASING.md / future release issues |
| `RELEASING.md` | Add template reference in Step 5 | Low; doc/process | Release maintainers/agents |
| `tests/Nivara.PerformanceTests/README.md` | Benchmark table refresh | Low; data-only doc | Release readers, perf comparison |
| `samples/NivaraIncident/README.md` | Benchmark tables refresh | Low-medium; two tables, carried Polars caveat | Sample readers |
| `samples/NivaraInference/README.md` | Multiple benchmark tables refresh | Medium; many rows/columns/caveats | Sample readers, release review |
| `samples/NivaraFineTuning/BERT-TUNING.md` | Benchmark tables refresh | Low-medium; derived estimates | Sample readers |

No production code changes. No docs/ coverage pin expected (new file is under `release/`, not `docs/`). Will confirm DocsCoveragePins before PR.

## Verification

- [ ] `git diff --check`
- [ ] Manual spot-check: every refreshed number traces to `release/v1.5.4/*` artefact; carried/skipped cells have explicit reason.
- [ ] Arithmetic spot-check: Δ%, ratios, slowdowns, estimated full-run values recomputed.
- [ ] Confirm no fabricated reruns (Polars, BERT-TUNING #1003 matrix, missing v1.5.4 rows).
- [ ] Confirm runner fixes are minimal and .cmd-safe.
- [ ] Check DocsCoveragePins/doc-coverage tests impact; revise only if required.
- [ ] Ask human before running tests; if run, at minimum `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` and targeted docs-related tests only if impacted (expect none). No code tests required.

## Planned commits

1. `docs: plan #1506 benchmark table refresh in TODO`
2. `docs: add benchmark refresh task template and wire RELEASING`
3. `chore: fix release benchmark runner capture gaps`
4. `docs: refresh core perf benchmark table from v1.5.4 artefacts`
5. `docs: refresh NivaraIncident benchmark tables from v1.5.4 artefacts`
6. `docs: refresh NivaraInference benchmark tables from v1.5.4 artefacts`
7. `docs: refresh NivaraFineTuning benchmark tables from v1.5.4 artefacts`
8. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1506 — Refresh all benchmark README tables from committed artefacts in every release (no deferrals) / clarified by human as one-time refresh using v1.5.4 data; this branch implements that refresh + adds reusable template.
