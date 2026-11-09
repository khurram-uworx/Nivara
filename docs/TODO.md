# Plan: perf: chunked-vs-flat baseline mode (#1076)

## Problem
The chunk layer buys structural fidelity but no compute: arithmetic on a chunked column materialises one contiguous copy per operand (`ChunkedColumn<T>.EnsureMaterialized`), then runs the existing flat kernel. `tests/Nivara.Tests/Storage/ChunkedColumnPerformanceTests.cs` records one flat-vs-8-chunk ratio for scalar add with no timing assertion, leaving the kernel decision without a baseline. We do not know how the one-copy cost scales with chunk count, operand count, or operation.

## Proposed changes
1. Add `tests/Nivara.PerformanceTests/ChunkedColumnProbe.cs` implementing a `--chunked-baseline` mode (aliases `--chunked`, `--chunked-vs-flat`).
2. Wire aliases into `tests/Nivara.PerformanceTests/Program.cs`.
3. Document in `tests/Nivara.PerformanceTests/README.md`.
4. Mode prints interleaved A/B distributions (30 rounds, 5 warmups, alternating first-measured route) for flat vs 2 / 8 / 64 chunks: scalar add, column add, equality, greater-than, less-than; null-bearing chunked operand; cached second-call path; slice-then-op path.
5. No `src/` or core test changes. No kernels implemented (non-goal).

## Verification
- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Run probe manually in Release and inspect output.

## Planned commits
- docs: plan #1076 baseline mode in TODO.md
- feat(perf): add ChunkedColumnProbe baseline mode
- chore(perf): wire --chunked aliases and document mode
- docs: remove TODO.md

## Blast radius
- Touches only PerformanceTests project + README + this plan. No core symbols, public API, or NUnit tests. `DocsCoveragePins` likely unaffected (no docs/*.md code citations added).

## GitHub issues log
- None captured yet.