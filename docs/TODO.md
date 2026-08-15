## Problem
The ColumnFilterHelper's filter/concat kernels box every element via `IColumn.GetValue`, causing memory allocations. This affects filter/concat operations by doubling-copy through CreateFromSpans -> Create.

## Proposed Fix
1. Add typed fast path for NivaraColumn<T> (value-type only):
   - Use typed indexer/IsNull instead of GetValue
   - Call CreateFromOwnedArray when data is null-free
2. Keep boxed path as fallback for non-NivaraColumn<T> columns

## Verification
- Add allocation regression budgets
- Test with null patterns and non-null patterns
- Verify with OProfile or dotnet-tcpdump to confirm reduced allocations

## GitHub issues log
- [ ] #259 - perf: boxed GetValue in ColumnFilterHelper filter/concat kernels
