# Plan: fix stale narrow-float SIMD wording in docs

## Problem

The widen-compute-narrow path is now on by default (commit `e7b21eac`), gated by
`NivaraPrimitives.DisableWidenSimd`, with `ShouldWiden` using `Vector.IsHardwareAccelerated`
and a `Vector<byte>.Count * 4` length gate. Several docs still describe the old toggle
(`NivaraPrimitives.UseWidenSimd`), an old branch, and `Vector128.IsHardwareAccelerated`, and say
the path is opt-in. This makes the narrow-float docs (BF16, F16) internally inconsistent and
misleading. QWEN/SMOLLM docs have the same stale toggle wording.

## Proposed changes

1. `docs/BFLOAT16.md`
   - Remove branch `khurram/smollm-1` and "as of the next release" wording.
   - `Vector128.IsHardwareAccelerated` → `Vector.IsHardwareAccelerated`.
   - Qualify "scalar MultiplyRowScalar" and AutoDiff MatMul note with widen path.
   - `(scalar for BF16)` example comments → auto-widened for large buffers.
   - Heading "Opt-in..." → "Auto-widened SIMD path (default on)".

2. `docs/HALF-F16.md`
   - Replace `UseWidenSimd` with `DisableWidenSimd` (inverted semantics).
   - `Vector128.IsHardwareAccelerated` → `Vector.IsHardwareAccelerated`.
   - Add length-gate note.
   - Headings/toggle wording to default-on.

3. `docs/QWEN.md`, `docs/SMOLLM.md`
   - `NivaraPrimitives.UseWidenSimd` wording → `DisableWidenSimd` (default-on, opt-out).

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests ... --filter "FullyQualifiedName~Nivara.Tests.Docs"` (66 tests).
- If editing code samples in docs, run snippet gate after edits.

## Planned commits
- docs: plan narrow-float SIMD wording cleanup
- docs: align BFLOAT16 narrow-float SIMD wording with shipped default-on path
- docs: align HALF-F16 narrow-float SIMD wording with shipped default-on path
- docs: align QWEN/SMOLLM widen toggle wording

## Blast radius
- Docs only (`docs/*.md`). No core code, no tests. No citation/top-level document count changes unless no new fenced blocks or citations. No snippet-gate impact expected.

## GitHub issues log
- (none created yet)
