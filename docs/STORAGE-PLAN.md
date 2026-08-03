# Storage Consolidation Plan — Single `ColumnStorage<T>`

## Purpose

This document breaks the column-storage consolidation into concrete, assignable
tasks for coding agents. It is the **authoritative** design for the storage
layer and supersedes `NEXT-REFACTORING.md` §1 ("Column storage: nullable vs
non-nullable"). `NEXT-REFACTORING.md` now points here.

**Headline decision:** replace the two-storage split (`TensorStorage<T>` /
`MemoryStorage<T>`) with **one** `ColumnStorage<T>` class that owns a single
contiguous `T[]` plus an optional `bool[]` null mask. Kernel selection
(vectorized vs scalar) is a **runtime, per-op** decision owned by
`KernelSelector` — it is never encoded as a storage class.

### Why this exists (the two wrong assumptions)

1. **Storage class ≠ speed.** `TensorPrimitives` operates on any
   `ReadOnlySpan<T>`; an array-backed column vectorizes identically to a
   `Tensor<T>`-backed one. In practice the split was inverted:
   `TensorStorage<T>`'s span access required a cached `FlattenTo` **copy**, and
   `NivaraColumn`'s `StorageType == Tensor` branches copied elements one-by-one
   into pooled buffers (slower than the memory path they were meant to beat),
   while `MemoryStorage<T>.Data.Span` was the actual zero-copy path.
2. **Nulls never justified a second class.** A `bool[]` mask flag handles
   nullability. ADR-001 already locates nulls at the storage layer; it never
   required two storage classes.

The BCL made the split obsolete once `TensorPrimitives` went generic + span-based
in .NET 10. The single-storage design is the correct simplification **now**.

### Locked decisions

- Single class `ColumnStorage<T>` replaces `MemoryStorage<T>` and deletes `TensorStorage<T>`.
- Backing store: `T[] data` (sole owner, contiguous). `Data` property = `data.AsMemory(start, length)`.
- Optional `bool[]? nullMask`; `null` ⇒ non-nullable column.
- Lazy cached `Tensor<T>` view via `Tensor.Create(data, [length])` (zero-copy wrap of the array); slices via `Tensor.Create(array, start, lengths, strides)`.
- `AsTensor()` guard: **unmanaged `T` only** (throws otherwise). `Half`/`BFloat16` are unmanaged and must pass, so the guard is *not* `IsVectorizable<T>()`.
- Drop `StorageType` and `ProvidesZeroCopySpanAccess` from `IColumnStorage<T>`.
- `ColumnDiagnostics` uses constants, not a `StorageType` switch.
- AutoDiff boundary (ADR-001) becomes a **runtime throw** at `FromColumn`/`FromSeries`/`FromArray`/`FromMatrix` on `HasNulls`, not just `Debug.Assert`.
- Kernel selection stays in `KernelSelector.DetermineKernelType()`; storage type is never consulted.

### Relationship to other docs

- `NEXT-REFACTORING.md` §1/§7/§8/§9 are updated to reference this plan.
- `docs/adr/001-autodiff-nonnullable-domain.md` storage-layer names
  (`TensorStorage`, `MemoryStorage`) become stale; addendum needed (Task 6).

## Suggested Execution Order

1. Task 1: `ColumnStorage<T>` — unified storage class (rename + redesign)
2. Task 2: `ColumnStorageFactory` simplification
3. Task 3: `IColumnStorage<T>` surface cleanup + diagnostics
4. Task 4: `NivaraColumn<T>` dual-path branch collapse
5. Task 5: AutoDiff boundary — runtime enforcement + zero-copy `FromColumn`
6. Task 7: Benchmark / performance gate
7. Task 6: Docs alignment (`NEXT-REFACTORING.md`, ADR-001) — parallel-safe

## Coordination Notes

- **Decision gates:** Task 1 (`AsTensor()` unmanaged guard, mask semantics),
  Task 2 (`IsVectorizable` scope for `Half`/`BFloat16`).
- **Merge conflict risk:** Tasks 1–3 all touch `src/Nivara/Storage/`
  (`ColumnStorage.cs`, `ColumnStorageFactory.cs`, `Interfaces.cs`) — must be
  done in order by one agent pass or coordinated handoffs.
- **Parallel-safe:** Task 6 (docs) can run concurrently with Tasks 1–5.
  Task 7 (benchmark) should not start until Task 4–5 land.
- **Shared files:** `src/Nivara/NivaraColumn.cs` is the largest blast radius
  (~20 `MemoryStorage` references); do not split it across agents.

## Task 1: `ColumnStorage<T>` — unified storage class

### Priority

High

### Goal

One storage class owning `T[] data` + `bool[]? nullMask`, replacing both
`MemoryStorage<T>` and `TensorStorage<T>`, with a lazy zero-copy `Tensor<T>`
view for AutoDiff interop.

### Why this exists

The two-storage split encoded a runtime kernel heuristic as a type-level fork,
duplicated `NivaraColumn` paths, and made the "fast" tensor path pay a cached
`FlattenTo` copy. A single contiguous `T[]` gives every column the true
zero-copy span path that `MemoryStorage` already had, plus an optional
`Tensor<T>` projection for consumers that need a real tensor.

### Decision required

- **`AsTensor()` guard = unmanaged-only.** `Tensor<T>` requires `unmanaged`;
  `Half`/`BFloat16` satisfy it and AutoDiff (ADR-001, `IFloatingPointIeee754<T>`)
  must keep working. Do **not** gate on `IsVectorizable<T>()`.
- Mask semantics: `null` mask ⇒ non-nullable column (`HasNulls == false`,
  `TryGetSpan` always succeeds). Present mask ⇒ nullable; `TryGetSpan` returns
  `false` and `AsTensor()` callers must check `HasNulls` first.

### Scope

- Rename `src/Nivara/Storage/MemoryStorage.cs` → `ColumnStorage.cs`, class
  `MemoryStorage<T>` → `ColumnStorage<T>`.
- Replace `ReadOnlyMemory<T> data` backing with sole-owner `T[] data`;
  keep `internal ReadOnlyMemory<T> Data => data.AsMemory()` for transport.
- Keep `bool[]? nullMask` semantics (non-null `bool[]` iff nullable).
- Constructors:
  - `ColumnStorage(ReadOnlySpan<T> values, bool detectNulls = false)` (ref-type null detection)
  - `ColumnStorage(ReadOnlyMemory<T> data, ReadOnlyMemory<bool>? nullMask = null)`
  - `ColumnStorage(T[] ownedData, bool[]? nullMask = null)` (sole-owner, no copy)
- Implement all `IColumnStorage<T>` members; `Slice` stays a shared-buffer view
  (`data.AsMemory(start, length)` + mask slice) since columns are immutable.
- Add `internal Tensor<T> AsTensor()` — lazy cached
  `Tensor.Create(data, [data.Length])`; guard `typeof(T)` for unmanaged.
- Delete `TensorStorage.cs`; port no behavior from it beyond the `Tensor<T>`
  view idea (no flattened caches, no mask tensors).

### Constraints

- ADR-001: null semantics remain at the storage layer; AutoDiff never sees nulls.
- `.editorconfig` at repo root is authoritative.
- Columns are immutable; do not reintroduce mutable span exposure.
- All existing constructors of `NivaraColumn<T>` that `new MemoryStorage<T>(...)`
  (lines 346, 421, 452, 479, 512) must be updated in the same commit.

### Suggested implementation path

1. Copy `MemoryStorage.cs` → `ColumnStorage.cs`; switch backing field to `T[]`.
2. Add `AsTensor()` behind an unmanaged guard; lazily create once.
3. Keep the public/internal member surface identical to today's `MemoryStorage`
   plus `AsTensor()`.
4. Grep-clean all `MemoryStorage<T>` references (factory, `NivaraColumn`, XML docs).
5. Delete `TensorStorage.cs` and its factory helpers after Task 2.

### Acceptance criteria

- No `MemoryStorage<T>` or `TensorStorage<T>` types exist anywhere in `src/`.
- `dotnet build Nivara.slnx` succeeds with no new warnings.
- Existing column/storage/null-propagation tests pass unchanged.
- `AsTensor()` returns a zero-copy view (`ReferenceEquals` on the backing array
  holds via `Tensor.TryGetSpan`), and throws for reference/non-unmanaged `T`.
- `ColumnStorage<string>` (mask present) and `ColumnStorage<float>` (no mask)
  both construct correctly; `TryGetSpan` returns `false`/`true` respectively.

### Files likely involved

- `src/Nivara/Storage/MemoryStorage.cs` (rename → `ColumnStorage.cs`)
- `src/Nivara/Storage/TensorStorage.cs` (delete)
- `src/Nivara/Storage/ColumnStorageFactory.cs`
- `src/Nivara/NivaraColumn.cs` (constructor sites 346, 421, 452, 479, 512)
- `src/Nivara/Interfaces.cs` (XML docs)

## Task 2: `ColumnStorageFactory` simplification

### Priority

High

### Goal

Collapse the factory to direct `ColumnStorage<T>` construction; remove the
11-way runtime type switches and all `TensorStorage` helpers.

### Why this exists

`ColumnStorageFactory.cs` routes every `Create` through `IsVectorizable<T>() &&
IsUnmanagedType<T>()` and then a 11-way `type switch` (lines 254–268, 285–299).
With one storage class there is nothing to dispatch on — the type switch is pure
boilerplate and a maintenance hazard.

### Decision required

- `IsVectorizable<T>()` stays (used by `KernelSelector` heuristics and
  `IColumnStorage.IsVectorizable`). Whether to extend it to `Half`/`BFloat16`
  is a separate kernel-selection change — do **not** bundle it here unless the
  agent can validate `TensorPrimitives` overloads for those types.
- Replace the duplicate `IsUnmanagedType<T>()` list with
  `RuntimeHelpers.IsReferenceOrContainsReferences<T>() == false`.

### Scope

- Rewrite `Create<T>(ReadOnlySpan<T>)`, `Create<T>(ReadOnlySpan<T>, ReadOnlyMemory<bool>?)`,
  `CreateFromOwnedArray<T>`, and `Create<T>(ReadOnlySpan<T?>)` to build
  `ColumnStorage<T>` directly (only nullable-value unboxing remains).
- Delete `createTensorStorage`, `createMemoryStorage`,
  `CreateTensorStorageForType`, `CreateTensorStorageForOwnedArray`,
  `CreateTensorStorageForNullableType`, `IsUnmanagedType`.
- Keep `CreateFromOwnedArray<T>(T[], ReadOnlyMemory<bool>?)` as the zero-copy
  wrap used by `NivaraColumn` hot paths.

### Constraints

- `Create<T>(ReadOnlySpan<T?>)` must preserve `default(T)` + mask at null positions.
- No behavior change observable from `NivaraColumn` callers.

### Suggested implementation path

1. Implement `ColumnStorage<T>` constructors first (Task 1).
2. Rewrite each `Create` to one `return new ColumnStorage<T>(...)`.
3. Delete the 11-way switches; compile to catch stragglers.
4. Update `IsVectorizable` callers (property on `ColumnStorage<T>`) to the shared helper.

### Acceptance criteria

- Factory file drops from ~301 to <100 lines.
- `IsVectorizable<T>()` and `IsUnmanagedType<T>()` are no longer duplicated.
- All `NivaraColumn`/storage tests pass; nullable-value creation preserves masks.
- `CreateFromOwnedArray` performs a zero-copy wrap (no `.ToArray()` on the input).

### Files likely involved

- `src/Nivara/Storage/ColumnStorageFactory.cs`
- `src/Nivara/Storage/ColumnStorage.cs`
- `src/Nivara/NivaraColumn.cs`

## Task 3: `IColumnStorage<T>` surface cleanup + diagnostics

### Priority

High

### Goal

Drop `StorageType` and `ProvidesZeroCopySpanAccess` from the storage contract;
replace `ColumnDiagnostics`' storage-type switch with constants.

### Why this exists

`StorageType` and `ProvidesZeroCopySpanAccess` exist only to work around the
two-storage split (tensor = copy, memory = view). With one zero-copy storage,
the distinction is meaningless and the property invites incorrect branching.

### Decision required

None — mechanical. Decide in review whether `StorageType` enum (line 183) is
deleted outright or repurposed as a kernel-kind (Tensor/Simd vs Scalar) label
for diagnostics. Prefer **delete** unless `ColumnDiagnostics` consumers need it.

### Scope

- Remove `StorageType StorageType { get; }` (`Interfaces.cs:117`) and
  `ProvidesZeroCopySpanAccess` (`Interfaces.cs:127`).
- `ColumnDiagnostics`: replace the `StorageType switch` (lines 126–151) with
  constants (single overhead/efficiency value) and drop the property from the
  `ToString` (line 171).
- Update tests asserting `Diagnostics.StorageType`
  (`ComprehensiveIntegrationTest.cs:32–36`, `ComplexScenarioIntegrationTests.cs:131–133`)
  to assert `IsVectorizable` or a kernel-kind label instead.
- Fix `IColumnStorage` XML docs referencing `MemoryStorage<T>`/`TensorStorage<T>`.

### Constraints

- `OperationDiagnostics`/`ColumnDiagnostics` population flow must not change shape.

### Acceptance criteria

- Zero compile-time references to `StorageType` or `ProvidesZeroCopySpanAccess` in `src/`.
- Diagnostics still record kernel selection; tests assert on `IsVectorizable` or kernel kind.
- Docs in `Interfaces.cs` reference only `ColumnStorage<T>`.

### Files likely involved

- `src/Nivara/Interfaces.cs`
- `src/Nivara/Diagnostics/ColumnDiagnostics.cs`
- `src/Nivara/NivaraColumn.cs` (line 1600 usage)
- `tests/Nivara.Tests/ComprehensiveIntegrationTest.cs`
- `tests/Nivara.Tests/ComplexScenarioIntegrationTests.cs`

## Task 4: `NivaraColumn<T>` dual-path branch collapse

### Priority

High

### Goal

Remove all `StorageType == Tensor` pooled-copy branches and `MemoryStorage<T>`
casts; every column operation uses one span path.

### Why this exists

`NivaraColumn.cs` has ~15 dual-path blocks (lines 550–649, 750–870, 945–1036,
1050–1231, 1245–1426) that branch on storage type. The `StorageType == Tensor`
branches rent a pooled buffer and copy element-by-element via the indexer —
strictly slower than the `MemoryStorage` zero-copy path. AutoDiff transitively
pays this cost on every columnar op it calls (Softmax, MatMul, ApplyRMSNorm,
ApplyDropout, `AccumulateGradient` Add/Scale).

### Decision required

None — the target shape is locked (Task 1 decisions).

### Scope

- Replace `if (storage.StorageType == StorageType.Tensor)` / `else if (storage is MemoryStorage<T> ...)` splits with a single path using `storage.TryGetSpan(out var span)` (or `AsSpan()` when the mask is handled separately) + `ColumnStorageFactory.CreateFromOwnedArray` for results.
- Preserve null-mask OR propagation exactly (data + mask produced together).
- Update constructor sites 346/421/452/479/512 and the `storage.StorageType` read at 1600.
- Keep pooled-buffer `BufferPool` usage only where a temporary output buffer is genuinely needed (>1024 elements), not as an input-copy workaround.

### Constraints

- Null-mask semantics: `resultNullMask = leftNullMask OR rightNullMask`; boolean results keep the mask with `false` at null positions.
- No behavior change — this is a performance/structural refactor; tests must pass unmodified where they assert semantics.

### Suggested implementation path

1. Do Tasks 1–3 first so `storage` is always `ColumnStorage<T>`.
2. For each arithmetic/comparison op, take the `MemoryStorage` zero-copy path as the single canonical implementation, deleting the tensor-pooled branch.
3. Build result via `CreateFromOwnedArray(resultArr, resultNullMask)`.
4. Run the null-propagation + arithmetic test suites.

### Acceptance criteria

- No `StorageType == Tensor` branches or `is MemoryStorage<T>` casts remain.
- Null-propagation tests (`NullMaskMaintenance_*`) pass unchanged.
- Vectorized kernels are reached via `KernelSelector` heuristics, not storage type.
- `dotnet build` clean; full column test suite green.

### Files likely involved

- `src/Nivara/NivaraColumn.cs`
- `src/Nivara/Tensors/TensorsHelper.cs` (only if a helper assumed tensor storage)
- `src/Nivara/Tensors/NivaraTensorExtensions.cs`

## Task 5: AutoDiff boundary — runtime enforcement + zero-copy `FromColumn`

### Priority

High (depends on Tasks 1–4)

### Goal

Upgrade ADR-001's `Debug.Assert` boundary to a **runtime throw** on `HasNulls`,
and make `FromColumn`/`FromSeries`/`FromArray`/`FromMatrix` take the zero-copy
`ColumnStorage<T>.AsTensor()` view.

### Why this exists

Today the non-null boundary is `Debug.Assert(data is null || !data.HasNulls)`
(`ReverseGradTensor.cs:47,58`; `ForwardGradTensor.cs:38,53`) — stripped in
Release. NEXT-REFACTORING §1 already specified "`FromColumn` throws if
`HasNulls`". Enforcing at runtime is the intended behavior; `AsTensor()` makes
the enter path a true zero-copy view for non-nullable columns.

### Decision required

- Exception type: prefer `AutoGradException` (consistent with `TypeValidator`)
  or `InvalidOperationException` with a message containing "ADR-001".
- `FromColumn` semantics on nullable input: **throw** (per NEXT-REFACTORING).
  Callers must `DropNulls()` first, per ADR-001 ("strip nulls before entering").

### Scope

- `ReverseGradTensor<T>`/`ForwardGradTensor<T>` constructors: replace
  `Debug.Assert(!data.HasNulls)` with a runtime throw (keep the debug assert as
  defense-in-depth if desired).
- `ReverseGradTensor.FromColumn` (line 70): if `column.HasNulls` throw; else
  back the tensor with `storage.AsTensor()` (zero-copy) + shape, instead of
  copying. `FromSeries` (85), `FromArray` (100), `FromMatrix` (121) likewise.
- Ensure `GradTensor.Data` still exposes what `ReverseGradOperations.AsSpan`
  (line 2351) needs; remove the `ModuleHelpers.GetSpan` fallback copy if
  `TryGetSpan` now always succeeds for AutoDiff tensors.
- Preserve `TypeValidator` gatekeeping (`TypeValidator.cs` stays as-is).

### Constraints

- ADR-001: AutoDiff is non-nullable; nulls never enter the graph.
- Inference-default `GradientUtils.Grad()` behavior unchanged.
- `ReverseGradTensor<T>.Data` remains accessible to ops and optimizer code.

### Acceptance criteria

- `FromColumn(nullableCol)` throws at runtime (Release build) with a message containing "ADR-001".
- `FromColumn(nonNullableCol)` produces a tensor whose data span is the same backing array (zero-copy) — verifiable via reference identity or a copy-count assertion.
- `NullHandlingTests`, `TypeSafetyTests`, and the full AutoDiff suite pass.
- No debug-only guard is the sole enforcement of the boundary.

### Files likely involved

- `src/Nivara/AutoDiff/ReverseGradTensor.cs`
- `src/Nivara/AutoDiff/ForwardGradTensor.cs`
- `src/Nivara/AutoDiff/GradTensor.cs`
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` (`AsSpan`)
- `src/Nivara/AutoDiff/Nn/ModuleHelpers.cs` (`GetSpan`)
- `docs/adr/001-autodiff-nonnullable-domain.md`

## Task 6: Docs alignment — `NEXT-REFACTORING.md` + ADR-001

### Priority

Low (parallel-safe)

### Goal

Make `NEXT-REFACTORING.md` consume this plan and fix stale ADR-001 storage names.

### Scope

- `NEXT-REFACTORING.md` §1 (lines 274–374): replace the two-storage design with
  a pointer to `STORAGE-PLAN.md` plus a 3–5 line summary of the change.
- "What moves where" rows for `ColumnStorageFactory`/`TensorStorage<T>`
  (lines 269–270): factory = simplify (not "add nullable dispatch"); drop the
  obsolete nullable-storage row.
- §7 manifest (lines 681–764): remove the obsolete nullable-storage new-file row;
  change `TensorStorage.cs` "rewritten" → deleted; add `ColumnStorage.cs`
  (rename from `MemoryStorage.cs`); fix line-count deltas table.
- §8 risks #1–3 (lines 770–782): reword — no separate nullable storage class;
  risk #1 (`Tensor.TryGetSpan` contiguity) still applies to the lazy
  `AsTensor()` view; nullable path = `data[]` + `mask[]` in the same storage.
- §9 sequence steps 1–4 (lines 823–827): replace the obsolete storage-class
  steps / `TensorStorage clean` with the consolidation tasks, keeping the
  VALIDATE gate.
- ADR-001 (`docs/adr/001-autodiff-nonnullable-domain.md`): update the
  storage-layer names (`NivaraColumn`, `TensorStorage`, `MemoryStorage`) to
  `ColumnStorage<T>`; note enforcement is a runtime throw at `FromColumn`.

### Acceptance criteria

- No docs/ file references the obsolete nullable-storage class by name; `rg` over `docs/` finds nothing.
- `rg -n "TensorStorage|MemoryStorage" docs/NEXT-REFACTORING.md docs/adr/001*` returns only intended historical/adjusted references.
- Both docs reference `STORAGE-PLAN.md` by relative link.

### Files likely involved

- `docs/NEXT-REFACTORING.md`
- `docs/adr/001-autodiff-nonnullable-domain.md`
- `docs/STORAGE-PLAN.md`

## Task 7: Benchmark / performance gate

### Priority

Medium (after Tasks 4–5)

### Goal

Quantify the indirect AutoDiff gains and the copy-elimination at the columnar layer.

### Why this exists

Expectations are mixed by design: elementwise span ops in AutoDiff
(`TensorPrimitives.X(a.AsSpan(), ...)`) are already zero-copy via
`CreateFromOwnedArray`; the real wins are (a) `FromColumn` entry tensors
(no `FlattenTo` copy), (b) removal of the `StorageType == Tensor` pooled-copy
branches in the columnar ops AutoDiff calls (Softmax, MatMul, ApplyRMSNorm,
`AccumulateGradient` Add/Scale), and (c) allocation reduction (no `Tensor<T>`
wrapper + flattened cache per column). Measure before claiming.

### Scope

- Micro-benchmarks (or a small perf test in `tests/Nivara.PerformanceTests` if
  it exists): `NivaraColumn<float>.Add`, `Sigmoid` column op, `Linear<T>` forward,
  `Linear<T>` forward+backward, one `TransformerBlock` forward.
- Record before/after: ops/sec, allocations/op, GC gen0/op.
- Assert/plot the flatten-copy removal at entry (`FromColumn` zero-copy).

### Constraints

- Do not force GC in the measurements; use steady-state warmup.
- Compare against the same hardware/config.

### Acceptance criteria

- A before/after table exists in the task notes or `docs/STORAGE-PLAN.md`.
- Documented: entry zero-copy, columnar-op branch removal, and AutoDiff indirect gains (or a clear explanation if a measurement is flat).

### Results

The before/after benchmark results and documented findings were captured in
`tests/Nivara.PerformanceTests/README.md` (the canonical home for this data).

Highlights (median of 3 Release runs each, baseline = git worktree at `549c6cc`,
same harness/machine): ColumnAdd +10.0x (columnar-op branch removal), AutoDiff
allocations −33%/−38%/−42% (Linear forward, forward+backward, TransformerBlock),
ColumnSigmoid flat within noise (issue #109).

### Files likely involved

- `tests/Nivara.PerformanceTests/` (if present) or a new benchmark sample
- `samples/Nivara.SampleApp/` (if a perf harness is reused)

## Additional Tasks

- **ADR-001 addendum** (fold into Task 6): record that the storage layer is now
  a single `ColumnStorage<T>` and that boundary enforcement is a runtime throw.
- **`TENSORS.md` cross-ref** (optional, low): note the lazy `AsTensor()` view as
  the intended zero-copy interop surface; keep `FlattenTo` only for
  non-contiguous/multi-dim cases.

## Suggested Agent Handout Batches

### Batch A: decision-critical (one agent, in order)

- Task 1 (`ColumnStorage<T>`)
- Task 2 (`ColumnStorageFactory`)
- Task 3 (`IColumnStorage<T>` + diagnostics)

### Batch B: implementation (after Batch A)

- Task 4 (`NivaraColumn` collapse)
- Task 5 (AutoDiff boundary)

### Batch C: tests and docs

- Task 6 (docs alignment)
- Task 7 (benchmark gate)

## Final Checklist

- every task has a clear owner-sized scope
- every task has acceptance criteria
- decision-gate tasks are clearly marked (Task 1, Task 2, Task 5)
- likely files are listed to reduce agent search time
- execution order reflects real dependencies (1→5 sequential; 6 parallel; 7 last)
