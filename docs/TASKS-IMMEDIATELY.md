# Immediate Claims-Integrity Tasks

## Purpose

This document turns the false/overstated claims found during the three lens reviews
(`docs/POLARS-REVIEW.md`, `docs/ARROW-REVIEW.md`, `docs/AISTACK-REVIEW.md`) into
execution-ready tasks that fix the *claims themselves* — making the code and public
docs honest **now**, without waiting for the larger roadmap work those reviews propose.

The six review/roadmap documents are the *release-planning* reference. They are left
intact. This doc is the *immediate* triage list: small, owner-sized, testable fixes
whose acceptance is "no public claim contradicts the code."

## How To Use

- Each task is sized so one coding agent can own it end-to-end.
- Task 1 applies the recorded decision — remove the `UseZeroCopy` API option entirely
  rather than keep a placeholder that advertises unsupported capability. Nothing below it
  touches Arrow interop zero-copy.
- Tasks 3–4 are docs/API-honesty fixes that must land even if the underlying
  zero-copy physics are deferred to `docs/ARROW-ROADMAP.md`.
- Task 7 (docs alignment) is intentionally **last**: it should reflect the post-fix
  reality, not the current one.
- Acceptance criteria are observable; run `dotnet build Nivara.slnx` and the listed
  test suites before marking a task done.

## Suggested Execution Order

1. Task 1: remove the `UseZeroCopy` property from `ArrowConversionOptions` (decision recorded)
2. Task 2: remove zero-copy placeholder code and tests in `ArrowInterop` (same files as Task 1)
3. Task 3: internal span access — view vs copy honesty
4. Task 4: `TensorStorage.Slice` — document/rename copy semantics (after Task 3)
5. Task 5: boxed expression evaluator — typed fast paths (parallel-safe with 3–4)
6. Task 6: `OrderBy` complex-expression support (parallel-safe with 3–5)
7. Task 8: regression guardrails (after 2, 3, 5 land)
8. Task 7: align README/GETTING-STARTED/ARCHITECTURE/CHANGELOG/AGENTS claims (last)

## Coordination Notes

- **Decision gate (settled):** we do not advertise capability we do not support. Task 1
  removes the `UseZeroCopy` option entirely; real zero-copy returns only with
  `ARROW-ROADMAP` Phase D, which will add real APIs and advertising then. Task 2 is the
  same cleanup in the same files — run them together.
- Tasks 3–6 are independent of 1–2 and of each other and can run in parallel.
- **Shared files / merge risk:** Task 3 (`Interfaces.cs`, `NivaraColumn.cs`,
  `TensorStorage.cs`) and Task 4 (`TensorStorage.cs`) touch overlapping code — run
  Task 4 after Task 3, or have the same agent do both.
- Task 7 edits the same files the reviews already quote (`README.md:147`,
  `GETTING-STARTED.md:1330`, `ARCHITECTURE.md:871`); update the line references in the
  review docs only if a task changes what those lines say.
- **Do not touch** the six docs (`docs/POLARS-REVIEW.md`, `docs/POLARS-ROADMAP.md`,
  `docs/ARROW-REVIEW.md`, `docs/ARROW-ROADMAP.md`, `docs/AISTACK-REVIEW.md`,
  `docs/AISTACK-ROADMAP.md`). They are the release plan, not the triage list. Their
  `UseZeroCopy` references remain valid as evidence and as the Phase D plan.
- **Not in scope here:** chunked columns, layout/bitmap validity, ONNX, dataset/eval
  layer — those are roadmap phases, tracked in the roadmap docs.

## Task 1: Remove `UseZeroCopy` from the public API

### Priority

High

### Goal

`ArrowConversionOptions` no longer exposes a `UseZeroCopy` option that advertises
zero-copy interop, because none is implemented.

### Why this exists

`UseZeroCopy` defaults to `true` (`src/Nivara.Extensions/IO/ArrowConversionOptions.cs:18`)
yet every `TryCreateZeroCopy*Array` method is a placeholder that unconditionally returns
`null` (`src/Nivara.Extensions/IO/ArrowInterop.cs:977-1065`), so all interop silently
falls back to element-by-element copying while the option claims otherwise. We do not
advertise capability we do not support.

### Decision required

**Settled (recorded here):** remove the option entirely. Breaking change accepted — the
library is pre-1.0 and honesty wins. Real zero-copy returns only with `ARROW-ROADMAP`
Phase D, which will add real APIs and advertising then.

### Scope

- Delete the `UseZeroCopy` property and its XML doc from `ArrowConversionOptions`.
- Remove `UseZeroCopy` from the `GETTING-STARTED.md:1330` sample.
- Reword `README.md:147` ("...with zero-copy optimization support" → "bidirectional
  conversion").
- Add a CHANGELOG entry noting the removal and the Phase D return.

### Constraints

- Do not touch the six review/roadmap docs (their `UseZeroCopy` references are evidence
  and plan, not current behavior).

### Suggested implementation path

- Delete the property, then fix every compile error surfaced by `dotnet build Nivara.slnx`
  (all in tests + the three `ArrowInterop` call sites, handled by Task 2).

### Acceptance criteria

- `rg -n "UseZeroCopy"` across `src/`, `tests/`, `README.md`, `GETTING-STARTED.md`,
  `CHANGELOG.md` returns no matches.
- `ArrowConversionOptions` has no zero-copy-related member or doc.
- Full `dotnet build Nivara.slnx` green.

### Files likely involved

- `src/Nivara.Extensions/IO/ArrowConversionOptions.cs`
- `GETTING-STARTED.md`
- `README.md`
- `CHANGELOG.md`

## Task 2: Remove zero-copy placeholder code and tests in `ArrowInterop`

### Priority

High

### Goal

No zero-copy code path, placeholder, comment, or test remains in the Arrow interop layer.

### Why this exists

The three `if (options.UseZeroCopy)` branches (`ArrowInterop.cs:359-364`, `386-391`,
`448-454`) and the three `TryCreateZeroCopy*Array` placeholders (`ArrowInterop.cs:977-1065`)
are dead scaffolding that promised zero-copy. Deleting them removes the lie at the code level.

### Scope

- Delete the `TryCreateZeroCopyBooleanArray`/`Int32Array`/`DoubleArray` methods.
- Delete the `if (options.UseZeroCopy)` branches in `CreateBooleanArray`/`CreateInt32Array`/
  `CreateDoubleArray`; conversion always uses the builder+copy path.
- Keep the `options` parameters on all creators for uniform signatures (existing
  `CreateInt64Array`/`CreateFloatArray`/`CreateStringArray` already pass-but-don't-use
  `options`); only the zero-copy branches are removed.
- Reword the file-header remark "with support for zero-copy operations" (`ArrowInterop.cs:11`).
- Update tests: remove `ToArrowTable_WithUseZeroCopyEnabled_AttemptsZeroCopy`, simplify
  `ToArrowTable_WithUseZeroCopyDisabled_UsesCopying` to a plain copy test, drop the
  `UseZeroCopy` default assertion in `ArrowConversionOptions_DefaultValues_AreCorrect`,
  and reduce the round-trip options matrix to `ValidateTypes`/`TimeZone`/`StringEncoding`
  only. Remove `UseZeroCopy = false` from `ArrowParquetIntegrationTests.cs:336`.

### Constraints

- Preserve all non-zero-copy behavior and round-trip semantics.

### Acceptance criteria

- `rg -n "ZeroCopy|zero-copy" src/Nivara.Extensions/IO/ArrowInterop.cs` returns no matches.
- No test references `UseZeroCopy`; all Arrow round-trip/data-preservation tests pass.

### Files likely involved

- `src/Nivara.Extensions/IO/ArrowInterop.cs`
- `tests/Nivara.Tests/IO/ArrowInteropTests.cs`
- `tests/Nivara.Tests/IO/ArrowParquetIntegrationTests.cs`

## Task 3: Internal span access — view vs copy honesty

### Priority

High

### Goal

Storage APIs stop documenting copy paths as "zero-copy."

### Why this exists

`IColumnStorage<T>.AsSpan()`/`TryGetSpan`/`AsWritableSpan` are documented as "zero-copy
access" (`src/Nivara/Interfaces.cs:119-141`), and `NivaraColumn.TryGetSpan`
(`src/Nivara/NivaraColumn.cs:2282`) says "Prefer ... for zero-copy access"
(`:2186`). But for `TensorStorage`, `GetFlattenedSpan()` allocates `new T[...]` +
`FlattenTo` — a cached **copy** (`TensorStorage.cs:200-211`). Only `MemoryStorage`
provides a true view (`MemoryStorage.cs:130-139`, `:170`). The claim is true for half the
backends and false for the fast-path one.

### Scope

- Correct the XML docs on `Interfaces.cs:119-141` to distinguish **view** (memory
  storage) from **copy** (tensor storage).
- Correct the XML docs on `NivaraColumn.cs:2186` and `:2282` (`TryGetSpan`).
- Add an `internal` capability flag (e.g. `bool ProvidesZeroCopySpanAccess`) on
  `IColumnStorage<T>` so callers can ask, rather than assume.

### Constraints

- Do not change the public `TryGetSpan` signature or `NivaraColumn` immutability contract
  (`AGENTS.md`: "Nivara columns are immutable — divergence from BCL `Tensor<T>.TryGetSpan`
  is deliberate").

### Suggested implementation path

- Keep `GetFlattenedSpan()` but rename its intent in docs: "cached flattened copy."
- Expose the memory-storage view path (already real) as the zero-copy option.

### Acceptance criteria

- No XML doc on `IColumnStorage<T>`/`NivaraColumn<T>.TryGetSpan` claims zero-copy for the
  tensor-backed path.
- A unit test asserts `TensorStorage.GetFlattenedSpan()` is a copy (mutating the returned
  span does not change storage data) and `MemoryStorage` slice shares the buffer.

### Files likely involved

- `src/Nivara/Interfaces.cs`
- `src/Nivara/Storage/TensorStorage.cs`
- `src/Nivara/Storage/MemoryStorage.cs`
- `src/Nivara/NivaraColumn.cs`
- `tests/Nivara.Tests/Tensors/TensorInteropTests.cs`

## Task 4: `TensorStorage.Slice` — document/rename copy semantics

### Priority

Medium

### Goal

The difference between `TensorStorage.Slice` (copy) and `MemoryStorage.Slice` (view) is
explicit and documented.

### Why this exists

`TensorStorage.Slice` materializes via `GetFlattenedSpan().Slice(...).ToArray()` +
`Tensor.Create` (`TensorStorage.cs:145-156`) while `MemoryStorage.Slice` is a true
`ReadOnlyMemory.Slice` view (`MemoryStorage.cs:130-139`). Same method name, opposite
semantics — a correctness trap for callers who assume slicing is cheap.

### Scope

- Document the copy semantics on `TensorStorage.Slice`.
- If Task 3 added the capability flag, have `Slice` callers branch on it instead of
  assuming a view.

### Constraints

- Do not make `TensorStorage.Slice` return a view without the layout work in
  `ARROW-ROADMAP` (tensor slicing is not contiguous).

### Acceptance criteria

- XML doc on `TensorStorage.Slice` states it returns an independent copy.
- Existing slice tests still pass (`NivaraColumn` slice/index tests).

### Files likely involved

- `src/Nivara/Storage/TensorStorage.cs`
- `src/Nivara/Storage/MemoryStorage.cs`
- `tests/Nivara.Tests/Storage/` (slice tests)

## Task 5: Boxed expression evaluator — typed fast paths

### Priority

High

### Goal

Filter/Select over same-typed numeric columns no longer boxes every element to `object?`.

### Why this exists

`ExpressionEvaluator.ApplyBinaryOperation`/`ApplyComparisonOperation`
(`src/Nivara/Helpers/ExpressionEvaluator.cs:220-259`) loop element-wise through
`GetValue(i)` + `Func<object?, object?, ...>`, boxing each value. This is the path used by
`FilterOperation.cs:62` and `SelectOperation.cs:78`, and it is the "fatal flaw" named in
`POLARS-REVIEW.md` — a columnar query engine whose hot path is row-at-a-time boxing
contradicts the README's "tensor-accelerated" claim.

### Scope

- Add typed fast paths in `ExpressionEvaluator` for the common same-type cases
  (`int`, `double`, `float`, `long`, `decimal`) that operate on spans when both operands
  share a type and the storage exposes `TryGetSpan`.
- Keep the existing `object?` path as the heterogeneous/nullable fallback.
- Route `FilterOperation`/`SelectOperation` through the typed path when eligible.

### Constraints

- Preserve null-mask semantics exactly: result null mask is left-OR-right; comparison
  results at null positions are `false` with the mask set (per AGENTS.md).
- Do not change the public `ExpressionEvaluator` API shape.

### Suggested implementation path

- For binary ops, dispatch on the resolved operand types once (not per element), then run
  a span loop with `TensorPrimitives` where type-compatible, scalar `Span<T>` loop
  otherwise.
- Record `OperationDiagnostics` for kernel choice (typed vs boxed).

### Acceptance criteria

- Filter/Select over same-typed numeric columns produce byte-identical results to the
  boxed path, including null positions.
- New test asserts the typed path is selected for same-type numeric columns (via
  diagnostics or a hook) and falls back for mixed types.
- Full `dotnet test` suite green (existing query tests are the compatibility net).

### Files likely involved

- `src/Nivara/Helpers/ExpressionEvaluator.cs`
- `src/Nivara/Operations/FilterOperation.cs`
- `src/Nivara/Operations/SelectOperation.cs`
- `tests/Nivara.Tests/Query/`

## Task 6: `OrderBy` complex-expression support

### Priority

Medium

### Goal

`OrderBy` no longer throws `NotSupportedException` for computed sort keys.

### Why this exists

`NivaraLinqExtensions.OrderBy` only accepts direct `ColumnReference` or simple named
expressions; anything else throws (`src/Nivara/Linq/NivaraLinqExtensions.cs:64-88`).
This is a documented limitation but a frequent first-run trap and an argument against the
"LINQ-like" claim.

### Scope

- Implement computed-key sorting by materializing the key expression into a column, then
  sorting on it (project-then-sort), reusing existing `SortOperation`.
- Keep the direct `ColumnReference` fast path.

### Constraints

- Match existing `Sort` semantics for null ordering and direction.
- Do not change the `OrderBy(Func<RowExpressionBuilder, ColumnExpression>, bool)` signature.

### Acceptance criteria

- `OrderBy(e => e["A"] + e["B"])` returns a sorted frame with correct order and null
  placement, no exception.
- Existing `Sort`/`OrderBy` tests pass unchanged.

### Files likely involved

- `src/Nivara/Linq/NivaraLinqExtensions.cs`
- `src/Nivara/Operations/SortOperation.cs`
- `tests/Nivara.Tests/Query/`

## Task 7: Align public claims with code (docs pass — run last)

### Priority

Medium

### Goal

Every repo doc that claims zero-copy, lazy, or tensor-accelerated behavior reflects the
post-fix code.

### Why this exists

- `README.md:147` / `GETTING-STARTED.md:1330` — zero-copy interop claim and sample; removed
  by Tasks 1–2, verify no stray mention remains.
- `README.md:79` — "lazy, eager, streaming, parallel — all fully implemented" (true for
  strategies; the expression path caveat is Task 5).
- `ARCHITECTURE.md:871` — "Zero-copy via `TryGetSpan`" (overstated for tensor storage;
  Tasks 3–4).
- `AGENTS.md:155, 306` — update the "Zero-copy Arrow arrays: placeholder implementation"
  known-issue note to "removed from public API; revisit in `ARROW-ROADMAP` Phase D".

### Scope

- Reword the claims above to match post-fix behavior.
- Reference `docs/TASKS-IMMEDIATELY.md` and the roadmap docs instead of asserting
  unimplemented capability.
- Keep the CHANGELOG entry from Task 1 consistent with the final state.

### Constraints

- Do not edit the six review/roadmap docs.
- Keep wording truthful even where it means advertising less.

### Acceptance criteria

- `rg -n "zero-copy|ZeroCopy" README.md GETTING-STARTED.md ARCHITECTURE.md CHANGELOG.md
  AGENTS.md` returns no claim contradicted by the code at HEAD.

### Files likely involved

- `README.md`
- `GETTING-STARTED.md`
- `ARCHITECTURE.md`
- `CHANGELOG.md`
- `AGENTS.md`

## Task 8: Regression guardrails for claim honesty

### Priority

High

### Goal

Future changes cannot silently reintroduce a claim/code mismatch.

### Why this exists

The three review lenses found mismatches that had survived for months because nothing
tested the *contracts* behind the claims (`UseZeroCopy` presence, span copy-vs-view,
typed vs boxed evaluator).

### Scope

- Guardrail: `rg -n "UseZeroCopy"` across `src/`, `tests/`, `README.md`,
  `GETTING-STARTED.md` returns no matches.
- Test: the Arrow conversion API exposes no zero-copy option; copying paths preserve
  round-trip data (existing tests cover this once Tasks 1–2 land).
- Test: `TensorStorage.GetFlattenedSpan()` is a copy; `MemoryStorage` slice shares buffer.
- Test: typed evaluator output equals boxed evaluator output for same-type numeric
  columns, including nulls.
- Test: `OrderBy` on a computed key does not throw.

### Constraints

- Follow AGENTS.md testing conventions: NUnit 4, `Method_Scenario_ExpectedBehavior`
  naming, no `[TestCase]` with null arrays, no GC forcing.

### Acceptance criteria

- The guardrail checks exist, are green, and fail if the corresponding honesty fix is
  reverted.

### Files likely involved

- `tests/Nivara.Tests/IO/ArrowInteropTests.cs`
- `tests/Nivara.Tests/Tensors/TensorInteropTests.cs`
- `tests/Nivara.Tests/Query/`
- new `tests/Nivara.Tests/Query/ExpressionEvaluatorTests.cs` if none exists

## Additional Tasks

- **Perf microbenchmark for interop copy paths** (Low): benchmark the current copy paths on
  `int`/`double` columns so the Phase D zero-copy payoff is measurable when it lands.
  Reference: `docs/ARROW-ROADMAP.md` Phase D.
- **`ARCHITECTURE.md` storage diagram** (Low): refresh the storage/layout diagram once
  Tasks 3–4 settle view-vs-copy terminology.

## Suggested Agent Handout Batches

### Batch A: decision-critical

- Task 1 (decision recorded: remove `UseZeroCopy`; breaking change accepted)

### Batch B: implementation

- Task 2 (same files as Task 1; run together)
- Task 3, then Task 4 (same agent, overlapping files)
- Task 5
- Task 6

### Batch C: tests and docs

- Task 8 (after Tasks 2, 3, 5 land)
- Task 7 (last, reflects post-fix reality)

## Final Checklist

- every task has a clear owner-sized scope
- every task has acceptance criteria
- decision task (Task 1) records the settled outcome and its dependent (Task 2) is gated on it
- likely files are listed to reduce agent search time
- execution order reflects real dependencies (7 last; 4 after 3; 2 after 1)
- the six review/roadmap docs remain untouched
