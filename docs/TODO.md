# Plan: Issue #1020 — Qwen: support or explicitly reject fp16 (Half) precision

Decision (human-approved): explicitly reject `--precision fp16` for the Qwen mode.
Rationale: `Half` has a 5-bit exponent (max ~65504) and 10-bit mantissa, which is insufficient
for the Qwen2.5-0.5B activations/RoPE/logits and risks overflow/NaN; no Qwen fp16 parity path
exists (CPU or Silk/OpenCL). Qwen already has a measured BF16-widening path (`--precision bf16`,
benchmark/`--synthetic-weights` only). Alternatives: `f32` (default) and `bf16`.

## Problem
- `samples/NivaraInference/Program.cs` rejects Qwen fp16 but its message does not state the reason.
- `docs/*` say fp16 is rejected but not why; the issue asks the rejection text and docs to state
  the reason and alternatives.

## Proposed changes
1. `samples/NivaraInference/Program.cs` guard (~line 566): keep rejection, extend message with
   reason (`range/precision`) and alternatives (`f32`, `bf16`). Preserve prefix
   `fp16 is not supported for the qwen mode` for assertion compatibility. Update usage text
   (~line 369) to match.
2. `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: `QwenPrecision_Rejected` asserts
   the reason-containing message.
3. Docs: `docs/HALF-F16.md`, `docs/QWEN.md`, `docs/ACCELERATION.md`,
   `samples/NivaraInference/README.md`: align wording.

## Blast radius
- Sample CLI argument surface and its argument gate test only.
- No change to Qwen weight loading, model graph, or execution: no CPU/Silk fp16 dispatch added.
- No `DocsCoveragePins` citation count change expected (no ADR/source citation edits).

## Planned commits
1. docs: plan issue #1020 fp16 decision in TODO.md
2. fix(sample): state Qwen fp16 rejection reason and alternatives
3. test(inference): pin Qwen fp16 rejection reason
4. docs: align Qwen fp16 rejection wording across HALF-F16/QWEN/ACCELERATION/README
5. docs: remove TODO.md — plan executed (after G2)

## Verification
- `dotnet build samples/NivaraInference/NivaraInference.csproj -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` (ask human first)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## GitHub issues log
- (none created; deferred work none identified)
