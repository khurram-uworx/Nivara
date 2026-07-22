# ADR-001: AutoDiff is a Non-Nullable Domain

**Status:** Accepted  
**Date:** 2026-07-23

## Context

The AutoDiff subsystem (`src/Nivara/AutoDiff/`) has pervasive null-handling code throughout its operations, backward functions, and optimizer steps. Nearly every operation in `ReverseGradOperations.cs` contains:

- `HasNulls` checks on inputs
- `WithoutNulls()` calls to strip null masks before computation
- `TryGetNullMask` + `TryGetSpan` dual paths
- Null mask propagation logic (OR-ing masks across operands)
- `IsNull(index)` checks in backward loops
- Conditional `CreateFromSpans` vs `Create` based on null presence

This adds ~40% code volume to every operation and creates two execution paths that must be tested and maintained. The null handling belongs to the **storage layer** (`NivaraColumn`, `TensorStorage`, `MemoryStorage`), not to the mathematical computation graph.

## Decision

**AutoDiff operations operate on non-nullable data.** The null boundary is enforced at domain entry points:

1. **`NivaraColumn<T>` → `ReverseGradTensor<T>` conversion** (Program.cs, DataLoader, etc.): strip nulls before entering the autograd graph. Replace null positions with `default(T)` and discard the mask.

2. **`ReverseGradOperations.*` methods**: assume all inputs have no nulls. Remove all `HasNulls`, `WithoutNulls`, `TryGetNullMask`, `IsNull` checks from the hot path. Every operation produces non-null output.

3. **`AccumulateGradient`**: simplifies to pure `TensorPrimitives.Add` — no null-merge logic.

4. **Adam/SGD optimizer step**: no null-aware branches. Straight `data[i] - lr * ...` loops.

5. **`CrossEntropyLoss`**: targets with nulls are invalid input; validate at the call site.

### Enforcement

- `ReverseGradTensor<T>` constructor should assert `!data.HasNulls` (debug-only, fast-path).
- `AccumulateGradient` strips nulls defensively once at the boundary, never per-operation.
- `NivaraColumn<T>.Create(T[])` and `Create(ReadOnlySpan<T>)` are already null-free by construction — these are the natural entry points.

### What stays nullable

- `NivaraColumn<T>` storage layer — null masks remain for SQL-like semantics in the DataFrame/Frame API.
- `IColumn.IsNull(index)` — external consumers may still query nullability.
- `TensorStorage` / `MemoryStorage` — null masks are a storage concern.

## Consequences

**Positive:**
- ~40% less code per AutoDiff operation (single path instead of dual)
- Fewer allocations (no `WithoutNulls()` copy, no merged null-mask arrays)
- Easier to add new operations (write one path, not two)
- Faster execution (no per-element null checks in hot backward loops)
- `AccumulateGradient` becomes a simple `TensorPrimitives.Add` + overwrite
- **SIMD/Tensor-ready by default:** null-free data means `TryGetSpan(out var span)` always succeeds, `TensorStorage` always backs the column, and every operation can use `TensorPrimitives` / `Vector<T>` / `Span<T>` fast paths without fallback branches. The null-checking overhead eliminated is not just branch prediction — it's the *entire scalar fallback path* that can now be deleted.

**Negative:**
- Callers must strip nulls before entering AutoDiff (one-time cost at boundary)
- Debug assertion in `ReverseGradTensor` constructor (negligible overhead)

**Migration:**
- New operations: follow the non-nullable contract from the start
- Existing operations: remove null paths opportunistically (high-traffic ops first: MatMul, Softmax, Concat, Gather, Slice, Add, Multiply)
- No API break — `NivaraColumn` stays nullable; only the internal AutoDiff path changes
