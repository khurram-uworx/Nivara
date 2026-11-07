# Plan: GitHub issue #990 — Validate apparent Linear/attention regressions in 1.5 core-harness numbers

## Problem

The 1.5 core-harness table claims surprising regressions on four rows:
- `Linear forward [32x256] -> [32x256]` −52.4%
- `Linear forward+backward [32x256]` −0.8%
- `Attn per-seq forward [B16 L128 D64 H4]` −4.4%
- `Attn batched forward [B16 L128 D64 H4]` −10.1%

Inspection shows the printed `Prev` for these rows is the prior release’s **Current**
(.NET 10.0.11 → .NET 11.0.0 transition), while their `Δ%` is still computed against
the even-older baseline. The four `Prev`/`Current`/`Δ%` triples are internally
inconsistent (e.g. Linear `Prev=1,363`, `Current=457`, but `Δ=-52.4%` only matches
`457` vs `960`). This is a cross-machine/cross-runtime baseline mismatch, not a
validated code regression; per `RELEASING.md`’s cross-machine rule the defensible
action is to drop `Prev` (machine-difference note) on those rows.

Cross-check of allocation columns (`B/op`, `gen0/op`) shows no allocation change on
these rows, which supports “no structural regression” but ops/s cannot be attributed
cleanly across runtimes.

## Proposed changes

Doc-only change in `tests/Nivara.PerformanceTests/README.md`:
- Set `Prev` and `Δ%` to `—` on the four affected rows, keeping measured
  `Current` and the allocation columns.
- Add a short Notes bullet documenting the #990 machine-drift decision.

Rows to change:
- `Linear forward [32x256] -> [32x256]`: `— | 457 | — | 68,072 | 0.00`
- `Linear forward+backward [32x256]`: `— | 123 | — | 929,327 | 0.10`
- `Attn per-seq forward [B16 L128 D64 H4]`: `— | 87 | — | 2,130,051 | 0.17`
- `Attn batched forward [B16 L128 D64 H4]`: `— | 304 | — | 528,637 | 0.00`

No source, test, ADR, or citation-count (`DocsCoveragePins`) changes.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (format gate).
- Confirm README table still renders (no column-count drift); snippet gate unaffected.
- Optional (human-run, not agent): re-run the four scenarios with
  `--runs 3 --only "..."` and compare medians to the retained `Current` values.

## Blast radius

- `tests/Nivara.PerformanceTests/README.md` only.
- No symbols or assemblies affected; no tests reference perf-table numbers.
- Symbols under measurement (reference only): `Nivara.AutoDiff.Nn.Linear` (src/Nivara/AutoDiff/Nn/Linear.cs)
  and `ReverseGradOperations.MultiHeadAttention` / `BatchedMultiHeadAttention`.

## Planned commits

1. `docs: plan issue 990 in TODO.md`
2. `docs: mark Linear/attention perf rows as machine-drift (issue #990)`

## GitHub issues log

- #990 — this issue (validating apparent Linear/attention regressions) — in progress
" 