# Plan: Issue #1043 — Silk fp16 CLI routing and parity validation (follow-up #1019)

Branch: `khurram/1043` (off `khurram/1019`)
Issue: https://github.com/khurram-uworx/Nivara/issues/1043

## Decision

Reject `--silk --precision fp16` from the sample CLI. The kernel/launcher `fp16`
(`Half`) paths and their GPU parity tests stay; they are evidence, not a CLI
surface. Measured cause: on the target Intel Arc 140T, OpenCL fp16 GEMM ties f32
(0.97–1.07×) so fp16 yields no speedup at current shapes, while bf16 keeps the
same half-size memory benefit and is the preferred narrow transport (see
`docs/SILK.md` §Precision on this iGPU). Prefer bf16 over fp16 in Silk docs/CLI.

## Problem

`samples/NivaraInference/Program.cs` allows `--silk --precision fp16` for the
`modernbert` and `laya` models and dispatches through `.RunGpu(tensorsHalf, …)`
while benchmark/compare use the F32 runner via `SilkModernBertSession.Open`, so
the fp16 path is inconsistent and is not a winning precision. The rejection
message still claims “no translation unit”, which is false: fp16 kernels exist.
Docs (`docs/SILK.md`, `docs/HALF-F16.md`, `docs/DISTILBERT.md`,
`fine-tuning README`) are stale and contradictory (some say all Silk narrow is
rejected, others say translation unit missing).

## Proposed changes

1. `samples/NivaraInference/Program.cs`
   - Guard `if (useSilk && fp16)` (all models), not just `!silkNarrowModel`.
   - New message: `--silk --precision fp16 is not supported: measured on Intel
     Arc 140T the fp16 kernels tie f32 (0.97–1.07×), so they show no speedup at
     current shapes; use --precision bf16 or drop --silk.`
   - Remove the now-dead `silkNarrowModel` local.
   - Delete the `fp16`-only Silk dispatch blocks in `laya` and `modernbert`.
   - Keep bf16 Silk paths and CPU fp16 paths.
2. `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
   - Update expected rejection strings.
   - Replace the stale “modernbert/laya support --silk fp16 (run mode)” comment
     with tests pinning rejection for `modernbert`/`laya` (run/benchmark/compare).
3. Docs wording only (no new `docs/*.md` files; no citation additions).

## Blast radius

- Source: `Program.cs` (CLI argument handling + model dispatch).
- Tests: `InferenceCliArgumentTests` (message pins).
- Docs: four files, wording only.
- GPU parity fixtures and Silk fp16 kernels untouched (retain evidence).
- `DocsCoveragePins` unchanged unless a doc file is added/removed (none).

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted: `InferenceCliArgumentTests` (Release, `Category!=Performance` already excluded by default filter? run explicitly).
- Docs subset if doc wording changes: `Nivara.Tests.Docs`.
- Build `NivaraInference` sample in Release.

## GitHub issues log

- (none created yet; add follow-ups if discovered during execution)
