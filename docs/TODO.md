# Plan: Arrow interop real zero-copy for flat, null-free columns (issue #835)

## Problem
Arrow interop has no real zero-copy path; all conversion copies. Need one honest, narrow path for flat, null-free columns.

## Proposed changes
- Add `ArrowConversionOptions.UseZeroCopy` (default false).
- In `ArrowInterop.ConvertColumnToArrowArrayTyped<T>`, when `UseZeroCopy`, call `CreateZeroCopyArrowArray<T>`.
- Implement `CreateZeroCopyArrowArray<T>` + per-type helpers + `UnmanagedMemoryManager<T>` in ArrowInterop.
- Throw `InvalidOperationException` for null-bearing columns, `UnsupportedTypeException` for unsupported types/layouts.
- Remove guardrail tests that asserted absence of zero-copy members.

## Blast radius
- `src/Nivara.Extensions/IO/ArrowConversionOptions.cs`
- `src/Nivara.Extensions/IO/ArrowInterop.cs`
- `tests/Nivara.Tests/IO/ArrowInteropTests.cs` (remove 2 guardrail tests)

## GitHub issues log
- (none yet)
