# Plan — #536: Assert the `ColumnStorage` tensor-view aliasing invariant

## Problem

`ColumnStorage<T>.AsTensor()` (`src/Nivara/Storage/ColumnStorage.cs:219`) returns a
**cached, zero-copy** `Tensor<T>` view over the sole-owner `data` array. `Tensor.Create`
does not copy (already verified by `TensorApiProbe`, `tests/Nivara.SimdProbe`), and the
view is memoised in `tensorView` (field at line 24) for the lifetime of the storage.

Today this is safe only because no read path writes `data`. Nothing asserts that, so any
future in-place kernel (sort, compaction/dedup, in-place fill, `ArrayPool` reuse) would
silently corrupt every tensor view already handed out. Issue #536 asks for the invariant
to be pinned — either by a test, by documentation, or both.

Two things in the issue body are worth correcting rather than reproducing:

1. The stated blast radius names `NivaraFrame.ToTensors<T>()` (`NivaraFrame.cs:422`) and
   `TensorInteropExtensions.ToTensor<T>` (`:130`). Both **copy** — they route through
   `NivaraColumn.ToTensor()` (`NivaraColumn.cs:1423`). The real aliasing paths are
   `AsTensor()` → `NivaraColumn/NivaraSeries.AsTensorView()` and `GradTensor.AsTensor()`.
2. "`ColumnStorage<T>` never writes to `data` in place" is true *for reads*, but the type
   exposes a **writable** span: `IColumnStorage<T>.AsWritableSpan()` (`Interfaces.cs:153`,
   impl `ColumnStorage.cs:260`) and `NivaraColumn<T>.AsWritableSpan()` (internal,
   `NivaraColumn.cs:1262`), which the AutoDiff optimizers (Adam/AdamW/SGD) use for in-place
   parameter updates through `ReverseGradTensor.Data` (`GradTensor.cs:26`, a
   `NivaraColumn<T>`). A `Parameter` wraps its array zero-copy
   (`Parameter.cs:82` → `ReverseGradTensor.FromArray` → `NivaraColumn.CreateFromOwnedArray`),
   so a `column.AsTensorView()` held over a training parameter **observes optimizer steps**.
   The invariant must therefore be stated honestly: *no read path writes `data`;
   `AsWritableSpan()` is the sole (internal) write path, and it cannot invalidate the cache.*

Sources re-checked: `docs/TENSORS.md` §"The zero-copy `AsTensorView()` surface"
(107–122); `docs/plan/AISTACK-ROADMAP.md:125` ("immutability is convention, not a
`ReadOnlyTensorSpan<T>` type").

## Proposed changes

### 1. Pin the invariant with a test (`tests/Nivara.Tests/Storage/ColumnStorageTests.cs`)

Add to the `#region Span access & slice view semantics` region (near the existing
`ColumnStorage_AsTensor_ReturnsZeroCopyView` / `..._IsLazyCached`):

`ColumnStorage_AsTensor_ViewAliasesSoleOwnerArray_AndSurvivesReadPaths`

- Build from a **sole-owner array** via the `T[]` ctor (owned-array overload).
- Assert **reference identity** of the tensor span against `array[0]`
  (`Unsafe.AreSame(ref MemoryMarshal.GetReference(span), ref array[0])`) — identity, so a
  future `AsTensor` that copies fails, unlike today's span `==` (start + length only).
- Exercise the read surface (`this[i]`, `Data`, `AsSpan()` / `TryGetSpan`, `NullMask`,
  `Slice(...)` read through the slice), then assert `AsTensor()` still returns the same
  cached instance (`ReferenceEquals`) and contents equal a clone snapshot taken before.
- **Negative control**: write through the tensor span and assert the source array /
  `storage[i]` change, then restore — this is what makes the identity assertion
  non-tautological (mirrors `TensorApiProbe.ReferenceOverlapsTensor`).

No timing/allocation assertion → no `[Category("Performance")]`.

### 2. Document the invariant

- **XML `<remarks>` on `ColumnStorage<T>`** (class doc, `ColumnStorage.cs:7–14`): state
  `data` is sole-owner, no member of the type writes it after construction on a read path,
  `AsTensor()` aliases it and caches the view, and any future in-place mutation must copy
  first or invalidate `tensorView` (`tensorView = null`). Name `AsWritableSpan()` as the
  deliberate, internal write path that cannot invalidate hosted views.
- **`docs/TENSORS.md`** §"The zero-copy `AsTensorView()` surface" (107–122): same contract
  in prose, including the future-kernel rule.
- **`NivaraColumn<T>.AsTensorView()` XML doc** (`NivaraColumn.cs:1232–1238`): the current
  "The column remains immutable" overstates it, given the optimizer's in-place path. Reword
  to "no read path mutates the backing array; AutoDiff optimizer steps write in place
  through the internal writable span, so a live view over a training parameter observes
  those updates" and keep the "mutating the returned tensor corrupts the column" warning.

### 3. CHANGELOG

Add an `Added`/`Fixed` bullet under `## [Unreleased]` in `CHANGELOG.md` recording the
pinned invariant + doc correction (no behavior change).

## Blast radius

- **Production code**: `src/Nivara/Storage/ColumnStorage.cs` (XML comments only),
  `src/Nivara/NivaraColumn.cs` (XML comment on `AsTensorView()` only). No logic change.
- **Docs**: `docs/TENSORS.md`, `CHANGELOG.md`, `docs/TODO.md`.
- **Tests**: `tests/Nivara.Tests/Storage/ColumnStorageTests.cs` (one new test).
- **Aliasing consumers re-verified**: `NivaraColumn.AsTensorView()` /
  `NivaraSeries.AsTensorView()` / `GradTensor.AsTensor()` → `ColumnStorage.AsTensor()`;
  `AsWritableSpan()` consumers are the AutoDiff optimizers only (grep: Adam, AdamW, SGD).
- **Existing coverage** that must stay green: `ColumnStorage_AsTensor_ReturnsZeroCopyView`,
  `ColumnStorage_AsTensor_IsLazyCached`, `ColumnStorage_AsTensor_ThrowsForReferenceContainingTypes`,
  `NivaraColumn/NivaraSeries/GradTensor_AsTensorView/AsTensor_IsZeroCopyView`
  (`TypeSafetyTests.cs`).

## Verification

1. `dotnet build Nivara.slnx` (or the tests project) — compiles.
2. **Prove the test can fail**: temporarily make `AsTensor()` copy into a fresh array,
   observe the new test FAIL, then restore (never report a check not seen fail).
3. Targeted: `dotnet test -c Release --filter "FullyQualifiedName~ColumnStorageTests"`
   (also `~TypeSafetyTests`).
4. Full suite: **ask the human first**; always with `--filter "Category!=Performance"`.

## Planned commits

1. `docs: plan #536 AsTensor aliasing invariant in TODO.md`
2. `test(storage): pin the AsTensor aliasing + cached-view invariant (#536)`
3. `docs(storage): state the AsTensor aliasing invariant (#536)`
4. `docs: record #536 in CHANGELOG`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] No follow-ups captured yet. Create via `gh issue create --repo khurram-uworx/Nivara`
      at discovery time and record the number here.

## Reminder

As each task executes, if you find deferred work or a concern that is out of scope, create
a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
number in the log above. Do not rely on memory; compaction during execution can lose it.
