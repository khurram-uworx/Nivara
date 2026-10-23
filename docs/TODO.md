# Plan: tidy ILGPU user-facing surface after #824 (#825)

Base: `khurram/824` (`72d688d0`)
Branch: `khurram/825`

## Problem

`khurram/824` already removed the ILGPU backends for minilm / distilbert / distilbert_sst and
dropped `OpenIlgpuEncoder`, plus its test changes. What remains untidy is the user-facing
help text and a consistency pass:

- `samples/NivaraInference/Program.cs` still advertises
  `GPU (distilbert / distilbert_sst / modernbert / laya):` for `--ilgpu`, but only
  `modernbert` and `laya` are ILGPU-capable now.
- Only `modernbert` is in `GpuModeCases` (correct), and `Gpu_OnModelWithoutGpuPath_Rejected` pins
  minilm/distilbert/distilbert_sst. Verify this test coverage still matches the surface.

## Proposed changes

- Update the `--help` header to `GPU (modernbert / laya):`.
- No source/dispatch change needed (already done in #824).
- Keep docs/help consistent with `GpuModes` = { modernbert, laya }.

## Blast radius

- Strings only in CLI help. No execution path changes.

## Verification

1. `dotnet build samples/NivaraInference`.
2. Targeted `InferenceCliArgumentTests` (must stay green: 47 tests).
3. Manual `--help` output check.

## Planned commits

1. `docs: plan #825 in TODO.md` (already committed)
2. `docs(inference): correct --ilgpu model list in help text`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- #825 — this issue.
- #824 — predecessor (minilm + DistilBERT GPU drops), already landed.
