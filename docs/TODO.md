# Plan: #1017 — Half/f16 capability matrix documentation

## Problem

`BFloat16` has a deep capability document (`docs/BFLOAT16.md`) and it is
cross-referenced widely, while `Half` (IEEE-754 half precision, f16) is only
mentioned alongside it. The Half surface is therefore unevenly documented:
readers cannot tell which column/query paths, AutoDiff paths, kernels, I/O paths
and sample paths are genuinely first-class for `Half` vs. inherited from generic
code, nor where it differs from BFloat16 (range, precision, SIMD cost).

## Proposed changes

1. Add `docs/HALF-F16.md` mirroring `docs/BFLOAT16.md`:
   - Why Half (memory, precision/range trade-offs vs BFloat16).
   - Vectorization note (scalar BCL fallback, widen-compute-narrow kernels,
     toggle `NivaraPrimitives.UseWidenSimd`, Half's portable-conversion cost).
   - Column/query layer: typed columns, arithmetic, null masks, window/cumulative,
     sorting, aggregation (promote to double), fused expressions, numeric
     promotion.
   - AutoDiff domain: constraint, TypeValidator, non-nullable boundary,
     modules/optimizers/losses/serialization.
   - SafeTensors F16 reads + widening.
   - Sample narrow-precision runs and CLI flags.
   - Precision & limitations section.
2. Add cross-references: `README.md` docs list, `GETTING-STARTED.md`,
   `docs/TENSORS.md`, and a one-line pointer from `docs/BFLOAT16.md`.
3. Update `DocsCoveragePins.ExpectedDocumentCount` (and reference count if links
   added) to reflect the new top-level document and any new markdown links.
4. Run docs coverage/snippet gates and targeted docs tests.

## Blast radius

- New document only; no source code changes expected for this issue.
- Docs pins in `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` must move with the
  new document count/link count.
- If gated ```csharp blocks are added, the snippet gate must compile them.

## Planned commits

1. `docs: plan #1017 half/f16 capability matrix in TODO.md`
2. `docs: add HALF-F16 capability matrix`
3. `docs: cross-reference HALF-F16 from README/GETTING-STARTED/TENSORS/BFLOAT16`
4. `test: update DocsCoveragePins for HALF-F16.md`
5. `docs: remove TODO.md — plan executed` (after G2)

## GitHub issues log

- [x] #1017 — Half/f16 capability matrix documentation (created while planning this issue)
- Follow-ups (already tracked): #1018 (widen default), #1019 (Silk narrow), #1020 (Qwen fp16), #1021 (tests)

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (66 tests, prior baseline) after updating pins.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.