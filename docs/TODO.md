# Plan: Clean up stale `` / `--gpu` CLI references (issue #987)

Branch: `khurram/987` off `khurram/988`. Target PR base: `khurram/988`.

## Problem

NivaraInference dropped `--gpu`/`` in 1.5.0 and exposes the only GPU backend as
`--silk`. Several documents and console hints still show ` benchmark` and related
commands, which no longer parse (`Unsupported mode ''`). The fix is a documentation,
sample, and test-comment audit: replace live command invocations with `--silk`, or annotate
them as historical; leave the NivaraIncident.Cli `--gpu` flag and `--gpu-alloc` untouched
(they are a separate, live feature).

## Proposed changes

1. `samples/NivaraInference/README.md` — historical-annotate the four remaining
   `` command lines instead of converting them to `--silk` (old baselines are not
   Silk defaults).
2. `samples/NivaraInference/ModernBert.cs`, `Laya.cs` — update user-facing console
   hints and XML doc summaries to the live `--silk` spelling.
3. `docs/DISTELBERT.md` — remove present-tense "two backends" claim; mark ILGPU section
   historical.
4. `docs/MODERNBERT.md` — annotate `modernbert ` / ` benchmark` as
   historical; note live equivalent.
5. `docs/LAYA.md` — mark `` references and the historical `laya `
   command lines as historical; reword "remains F32-only" as historical.
6. `docs/SILK.md` — replace the stale mutual-exclusion bullet with the current reality;
   annotate the back-to-back `--silk`/` benchmark` measurement as historical.
7. `docs/QWEN.md` — clarify `` was removed, not merely unsupported.
8. `docs/ROADMAP-SUGGESTION.md` — annotate PR #436 `` lines as historical.
9. `tests/.../InferenceCliArgumentTests.cs` — annotate comments that imply `` is
   still a live surface.

No CLI behavior changes, no new rejection guards, no backwards-compatible alias. ``
continues to fail loudly as an unsupported mode.

## Blast radius

- Docs only plus two console-hint strings in `samples/NivaraInference`. No public runtime
  API, no storage/query/AutoDiff code, no tests assert the changed strings.
- `DocsCoveragePins` expected counts unchanged: no new/removed docs files, no new
  `File.cs:NN` citations.
- `InferenceCliArgumentTests` still valid; console text changes do not affect assertions
  (it asserts on stderr guard messages, not these hints).

## Verification

- `grep -R -- "" docs samples tests` → every hit is historical/removed.
- `grep -R -- "--gpu" docs samples tests` → only legitimate NivaraIncident / `--gpu-alloc`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Targeted test: `InferenceCliArgumentTests` (sample must be built).

## Planned commits

1. `docs: plan issue 987 in TODO.md`
2. `samples: update stale  hints to --silk spelling`
3. `docs: mark ILGPU backend references as historical in DISTILBERT/MODERNBERT/LAYA`
4. `docs: mark stale  commands as historical (SILK/QWEN/ROADMAP/SILK-readme)`
5. `tests: annotate InferenceCliArgumentTests  references as historical`
6. `docs: remove TODO.md - plan executed`

## GitHub issues log

- (none captured during this plan)