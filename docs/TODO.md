# Plan: Issue #132 — string column hash caching (+ join alignment)

Base branch: khurram/766. Target PR: khurram/766.

## Problem

IDEA.md §5.4 lists string-column optimizations that are not implemented:
- Hash caching
- Dictionary encoding (future)
- Batched scalar execution

Current state:
- No hash caching for string comparisons/grouping.
- No dictionary-encoded storage (single plain `ColumnStorage<T>` for everything).
- Joins allocate `object?[]` per row and recompute string hashes.

## Scope decision

This slice implements **persistent string hash caching** plus **join hash reuse**.
Dictionary encoding remains ARROW-ROADMAP Phase E (depends on Phases A–C).
Batched scalar execution is already partially covered by fused `EvaluateChunked` and scalar span loops.

## Proposed changes

### 1. Lazy storage-level hash cache (strings only)

Files:
- `src/Nivara/Interfaces.cs` (`IColumnStorage<T>`)
- `src/Nivara/Storage/ColumnStorage.cs`

Add:
```csharp
internal bool TryGetHashCodes(out ReadOnlySpan<int> hashes);
```
Returns false for non-string. For strings, lazily allocate `int[dataLength]`, fill 0 at null positions and `string.GetHashCode()` otherwise, cache for storage lifetime.

### 2. Expose via NivaraColumn

File: `src/Nivara/NivaraColumn.cs`

```csharp
internal bool TryGetStringHashCodes(out ReadOnlySpan<int> hashes)
    => storage.TryGetHashCodes(out hashes);
```

### 3. Use cache in grouping/distinct

File: `src/Nivara/Operations/GroupKeyReaders.cs`

`GroupKeyReader<T>.GetHashCode` uses cached hashes when `T == string`.

### 4. Join hash reuse

File: `src/Nivara/Operations/JoinOperation.cs`

Add `CompositeKey(object?[] values, int hashCode)`. Compute combined hash via `TypedGroupHash.ComputeRowHash` (which now uses cached string hashes) and pass it in. Keep existing equality logic.

## Blast radius

- Storage interface + implementation (internal only).
- `NivaraColumn` (internal method).
- `GroupKeyReader` (internal).
- `JoinOperation` (internal).
- Tests: storage, groupby, join, distinct.

No public API changes.

## Verification

- Targeted tests in Release for affected classes.
- Perf scenarios: GroupBy string, Join.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. docs: plan #132 in TODO.md
2. feat(storage): add lazy string hash cache to ColumnStorage
3. feat(columns): expose string hash cache on NivaraColumn
4. perf(groupby): use cached string hashes in GroupKeyReader
5. perf(join): reuse cached hashes in CompositeKey
6. test: add hash-cache coverage
7. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet; deferred work will be tracked here)
```

Then commit plan.<|end|><|start|><|start|>assistant to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="path">
docs/TODO.md