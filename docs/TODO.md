# TODO — #136: BCL tensor swap verification (net11)

GitHub issue: https://github.com/khurram-uworx/Nivara/issues/136
Branch: `khurram/136` (off `feature/net11`)

## Problem

Issue #136 asked to replace the handwritten MatMul/Transpose kernels in
`TensorsHelper` with BCL `Tensor.Transpose<T>` / `Tensor.MatrixMultiply<T>` once the
repo migrated to net11. Research findings (grounded in the 11.0.0-preview.7 ref API):

- ✅ net11 migration is done (`feature/net11`); SNT package bumped to
  `11.0.0-preview.7.26381.103` (commit `314b9ff`).
- ⚠️ `Tensor.Transpose<T>` **ships** but is a zero-copy stride-swap *view* over the
  same backing array — not a physical materialization. All Nivara callers need
  contiguous row-major output spans for `TensorPrimitives.Dot` SIMD kernels.
- ❌ `Tensor.MatrixMultiply` **does not exist** in preview.7 or main. Only an open
  api-suggestion: dotnet/runtime#95863 (BLAS epic dotnet/runtime#93286). Element-wise
  `Tensor.Multiply` is NOT matmul.
- `TensorsHelper.Multiply(Tensor<T>,…)` overloads have zero production callers.
- Public-surface audit is clean: helpers are `internal`; public surface limited to
  sanctioned scoring interop + AutoDiff domain ops (per docs/TENSORS.md).

## Planned changes

### Phase B — Transpose swap verification (permanent regression tests)
`tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`:
- Add `[Category("Performance")]` benchmark tests using existing `MeasureBestOfFive`:
  tiled kernel vs BCL route (`Tensor.Create(...)` → `Tensor.Transpose(t)` → `FlattenTo(dst)`)
  across shapes 2×3 / 128×256 / 512×512 / 1024×1024.
- Add correctness-parity test: both paths produce identical element layout.
- Decision gate: adopt BCL only where it matches/beats; expected outcome per research
  is to keep the tiled physical kernel (callers need contiguous rows).

### Phase C — dead code removal
`src/Nivara/Tensors/TensorsHelper.cs`:
- Delete `Multiply<T>(Tensor<T>, Tensor<T>, int, int, int)` and
  `Multiply<T>(Tensor<T>, Tensor<T>, T[], int, int, int)` (no production callers).
- Update the direct call in `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`.

### Phase D — annotations & docs
- `TensorsHelper.cs`: rewrite header banner + XML docs. "Swap target" wording now
  cites dotnet/runtime#95863 instead of claiming a shipping net11 API; document that
  `Tensor.Transpose<T>` shipped as a logical view and this kernel is its materializer.
- `docs/TENSORS.md`: correct the "BCL swap targets" section the same way.
- `CHANGELOG.md`: entry under the net11 migration.

### Phase E — verify & issue hygiene
- `dotnet build Nivara.slnx`; ask before running `dotnet test`.
- Comment findings on #136 (temp-file body → `--body-file`) and close as partially
  executed; upstream tracking stays with dotnet/runtime#95863.
- Remove this TODO.md once every item above is done.

## Blast radius

- `TensorsHelper` callers: `GradKernels.MatMul/MatMulTransposedB/Transpose`
  (`src/Nivara/AutoDiff/Operations/GradKernels.cs:631-641`) → consumed by
  `ReverseGradOperations` matmul/attention backward paths. Deleting the two dead
  `Tensor<T>`-level overloads touches nothing on these paths.
- Test coverage: `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs` (shape-matrix
  correctness via `ReferenceMatMul`, null-mask tests, perf harness),
  `tests/Nivara.Tests/AutoDiff/GradKernelsTests.cs` (facade parity, Half guard).
- Docs only otherwise: no runtime behavior change expected from Phase B–D.

## Verification

1. `dotnet build Nivara.slnx` after each phase.
2. `dotnet test tests/Nivara.Tests` (ask first) — full suite must stay green;
   new benchmark/parity tests pin the acceptance criteria.

## GitHub issues log

- [x] #136 — swap TensorsHelper MatMul/Transpose kernels for BCL APIs (existing issue being executed/closed by this plan)
