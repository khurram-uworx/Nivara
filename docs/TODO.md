# Plan: Issue #1290 — TryGetReadOnlyTensorSpan fallible zero-copy view contract

## Problem
Phase 5 delivered `AsTensorView()` (throwing), but the fallible `TryGet` read-only span/view contract remains open. Need a conservative `TryGetReadOnlyTensorSpan` that returns a `ReadOnlyTensorSpan<T>` view (no copy) for flat, contiguous, null-free, unmanaged columns/series and `false` otherwise, so callers can fall back to the copy path.

## Proposed changes

### Storage interface
`src/Nivara/Interfaces.cs` — add to `IColumnStorage<T>`:
```csharp
internal bool TryGetReadOnlyTensorSpan(out ReadOnlyTensorSpan<T> span);
```

### ColumnStorage<T>
`src/Nivara/Storage/ColumnStorage.cs` — implement:
- disposed → throw ObjectDisposedException
- nulls → false
- reference/non-unmanaged T → false
- empty → true, span = default
- otherwise reuse cached `AsTensor()` view and return `view.AsReadOnlyTensorSpan()`

### ChunkedColumn<T>
`src/Nivara/Storage/ChunkedColumn.cs` — implement returning false, span = default.

### NivaraColumn<T>
`src/Nivara/NivaraColumn.cs` — public:
```csharp
public bool TryGetReadOnlyTensorSpan(out ReadOnlyTensorSpan<T> span)
{
    ObjectDisposedException.ThrowIf(disposed, this);
    return storage.TryGetReadOnlyTensorSpan(out span);
}
```

### NivaraSeries<T>
`src/Nivara/NivaraSeries.cs` — public, delegates to `values`.

### Optional extension
`src/Nivara/Tensors/TensorInteropExtensions.cs` — add thin static extension methods delegating to instance methods for call-site flexibility.

## Verification
- Build: `dotnet build Nivara.slnx`
- Targeted tests: TypeSafetyTests, NivaraColumnTests TryGetSpan tests, Storage tests.
- New tests: null/non-compatible false; flat null-free true and parity with copy path; chunked false; reference type false; empty true.

## Planned commits
1. docs: plan issue #1290 in TODO.md
2. feat(storage): add TryGetReadOnlyTensorSpan to IColumnStorage + impls
3. feat(column/series): expose TryGetReadOnlyTensorSpan publicly
4. feat(interop): add extension-method shim (optional)
5. test: parity/edge-case tests
6. docs: update TENSORS.md
7. docs: remove TODO.md

## Blast radius
- Internal interface addition: all IColumnStorage<T> impls (ColumnStorage, ChunkedColumn). Only two.
- Public API addition on NivaraColumn<T>/NivaraSeries<T>: additive, no breaking.
- Storage tests may need interface impl updates; ChunkedColumn updated.

## GitHub issues log
- (none yet)
