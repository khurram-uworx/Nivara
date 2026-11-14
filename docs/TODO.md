# Plan: #1179 — Make `NivaraFrame` row-enumerable and expose `NivaraRow`

> Tracking: https://github.com/khurram-uworx/Nivara/issues/1179
> Branch: `khurram/1179`
>
> Reminder (iterative-work): as each task executes, if you find deferred work or a concern outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the **GitHub issues log** below. Don’t rely on memory or wait until the plan finishes.

## Problem

`NivaraFrame` is not enumerable and has no row-access API (`IEnumerable<NivaraRow>`, `GetEnumerator`, `GetRow`, `Rows` absent). `NivaraRow` is `public readonly struct` but its constructors are `internal`, and the only place a consumer can observe a row today is inside `NivaraFrameExtensions.Where(NivaraFrame, Func<NivaraRow,bool>)` (`src/Nivara/NivaraFrameExtensions.cs:842-858`), which materializes a mask and returns a filtered frame (not rows). This blocks straightforward row iteration/interop (exporters, loggers, debug printing) and makes `NivaraRow` effectively unreachable despite being public API.

**Current grounded facts (code-memory + source):**
- `IFrame` (`src/Nivara/Interfaces.cs:191-229`): `RowCount`, `ColumnCount`, `ColumnNames`, `Schema`, `GetColumn<T>(string)`, `HasColumn(string)` only.
- `NivaraFrame` (`src/Nivara/NivaraFrame.cs:18`): `public sealed class NivaraFrame : IFrame`, no row enumeration/members.
- `NivaraRow` (`src/Nivara/NivaraRow.cs:16-314`): thin view over `IColumn[] columns`, optional `string[] columnNames`, `IReadOnlyDictionary<string,int>? map`, `int rowIndex`; internal ctors (23, 33); rich accessors (`GetValue<T>`, `TryGetValue`, `IsNull`, `GetColumn*`, indexer, `AsReader()`), nullable-element fast path via cached delegates; `default(NivaraRow)` throws on access.
- `Where` builds `new NivaraRow(columns, columnNames, map, i)` per predicate call; columns/name array/map prepared once per call.
- `FromRows<T>` input exists (`NivaraFrame.FromRows<T>(IEnumerable<(string Label, T[] Vector)> rows, ...)`, `src/Nivara/NivaraFrame.cs:137-194`) — labeled-vector ingestion shape, not a general typed-row record API.
- Typed query materialization today: `NivaraQuery<T>.Collect()/ToList()/ToListAsync()/ToObjectsAsync(...)` (`src/Nivara/Linq/NivaraQuery.cs`); docs mention `ToObjects()`/`ToRows()` in places but source methods with those exact names were not found (doc/API drift to reconcile).

## Proposed changes (draft — subject to G1/human decisions)

### 1) Frame row access surface
- Add row iteration to `NivaraFrame` with a non-allocating `struct` enumerator:
  - `public Enumerator GetEnumerator()` (and explicit `IEnumerable<NivaraRow>` / `IEnumerator<NivaraRow>` as needed for `foreach`/LINQ compatibility) returning `NivaraRow` values in row order.
  - `public NivaraRow GetRow(int rowIndex)` returning a row usable for the lifetime of the frame (bounds/disposed checks).
- **Decision point (escalate in G1):** implement on `NivaraFrame` only vs also extend `IFrame` (`src/Nivara/Interfaces.cs`). Prefer minimal contract change unless polymorphism is required; confirm.
- Enumerator design sketch (allocation-free per row): struct enumerator holds frame/column refs + prebuilt name/index structures + current index; `MoveNext` advances index; `Current` returns/copies a `NivaraRow` view for that index without per-row allocations. Document iteration-lifetime/validity semantics precisely (and test chosen behavior).
- All entry points (`GetEnumerator`, enumerator `MoveNext`/`Current` as appropriate, `GetRow`) must throw `ObjectDisposedException` when frame is disposed.

### 2) `NivaraRow` public constructibility & semantics
- Satisfy “constructible from public API so a predicate can be unit-tested without a frame” while keeping `NivaraRow` a thin view (no per-cell allocation / no second data model).
- **Decision point (escalate in G1):** API shape for public construction/factory (e.g. factory over `NivaraFrame`/columns+names, or other safe surface) that allows predicate unit tests without requiring an internal ctor, without breaking invariants or encouraging expensive misuse. Preserve existing internal hot-path construction.
- Keep/document validity contract: row is a view tied to underlying columns/index; clarify behavior when used after enumerator advances and after frame disposal (pick one, document, test).

### 3) Interop projection (naming-sensitive)
- Issue asks for `ToRecords()`/`ToRows<T>()` for interop binding columns once; maintainer comment requires **not** resurrecting `ToRows`/`ToRows<T>` and prefers one canonical path (enumeration/GetRow) plus at most one convenience projection with a non-colliding name.
- **Decision point (escalate in G1):** whether to add any convenience projection now and, if so, its exact name/shape (`ToRecords()` / `AsRows()` / `Rows` / none) to avoid M11-style duplication and align with existing `NivaraQuery<T>` materialization verbs. Default proposal: no new typed `ToRows<T>` verb; defer/choose a single untyped convenience only if clearly justified.

### 4) Documentation & cost warnings
- XML docs on enumerator/`GetRow`/any row projection: row-major access is much slower than column access; prefer `GetColumn<T>`/column ops in hot paths; cross-reference `NivaraRowReader` fast path and M12 context.
- Update `docs/LINQ.md` and `docs/AGENT-CODE-EXAMPLES.md` (affected per issue) to show supported row iteration/interop patterns and correct any stale `ToObjects()`/`ToRows()` references if they diverge from actual API (verify before editing; docs are gated).

### 5) Tests (good-enough, acceptance-focused)
- Enumeration over 0/1/N rows yields `RowCount` rows in order with correct `RowIndex`/values.
- `GetRow(i)` returns usable row outside loop for frame lifetime; bounds behavior.
- Enumerating/`GetRow` on disposed frame throws `ObjectDisposedException`.
- No per-row allocation assertion for enumeration (careful, release-appropriate measurement).
- `NivaraRow` constructible from public API and predicate unit-testable without a frame (per chosen design).
- Round-trip: define and test `FromRows<T>` ↔ output path per resolved semantics (flagged — existing `FromRows<T>` signature is vector/label form).
- Extend existing suites where natural (e.g. `NivaraRowTests`, frame tests) rather than redundant new files.

## Verification steps
- Before each commit: targeted build/format checks as applicable (`dotnet build`, whitespace format where relevant).
- **Ask before running tests** (skill/AGENTS). Planned targeted runs (after human confirmation): row/frame-related tests in `tests/Nivara.Tests` (e.g. `NivaraRowTests`, frame/ownership tests), then broader docs/tests as needed.
- If `docs/*` changes: run Docs namespace tests (`dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`) and snippet gate considerations; update `DocsCoveragePins` only if counts/citations actually change (review in G2/pre-PR).
- Pre-push (per AGENTS/user request): `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`, relevant targeted tests, and doc gates if docs changed.

## Planned commit list
1. `docs: plan #1179 row-enumerable NivaraFrame in TODO.md` (this plan)
2. `feat: add NivaraFrame row enumeration and GetRow` (core API + enumerator + disposed checks)
3. `feat: expose public NivaraRow construction/testing surface` (per decided design) — may merge with (2) if small/cohesive
4. `feat/docs: add interop row projection API (if decided) and cost documentation`
5. `test: cover #1179 enumeration, GetRow, disposal, construction, and round-trip`
6. `docs: update LINQ/AGENT-CODE-EXAMPLES and pins if required`
7. `docs: remove TODO.md — plan executed` (only after G2 clears)

(Commits may be split/merged to keep one logical change per commit; follow iterative-work rules.)

## Blast radius
- **Core API/types:** `src/Nivara/Interfaces.cs` (`IFrame` — if extended), `src/Nivara/NivaraFrame.cs` (new members/enumerator, disposal behavior), `src/Nivara/NivaraRow.cs` (public construction/lifetime semantics), `src/Nivara/NivaraFrameExtensions.cs` (possible reuse of row-creation helpers / interaction with `Where`).
- **Callers/dependents:** broad (many consumers of `NivaraFrame`/`IFrame` per code-memory impact); additive API is low-risk for behavior but interface changes affect implementers/consumers and binary/source compatibility considerations.
- **Docs:** `docs/LINQ.md`, `docs/AGENT-CODE-EXAMPLES.md`; possible `DocsCoveragePins`/citation counts only if fenced blocks/citations change.
- **Tests:** `tests/Nivara.Tests/NivaraRowTests.cs` (existing row-predicate coverage), frame/ownership/query tests; new/extended tests for enumeration and disposal.

## G1 grounding notes (to complete after plan commit, before implementation)
- Use `microsoft-learn` MCP to ground: .NET patterns for allocation-free struct enumerators (`IEnumerable<T>`/`IEnumerator<T>`/`GetEnumerator`), `foreach` behavior, `IDisposable` enumerators, and `ObjectDisposedException` guidance.
- Use `code-memory` to confirm symbol relationships/usages as implementation proceeds.
- **Red flags / decisions requiring human input before implementation:**
  1. **`IFrame` contract change?** Add enumeration/GetRow to `IFrame` or only to `NivaraFrame`.
  2. **Interop projection naming/scope:** add any convenience projection now? if yes, choose one name (`ToRecords`/`AsRows`/`Rows`/other) — must not add `ToRows`/`ToRows<T>`; keep single canonical path.
  3. **Public `NivaraRow` construction API:** concrete shape that enables predicate unit tests without a frame while preserving thin-view invariants and lifetime clarity.
  4. **Round-trip acceptance criterion:** `FromRows<T>` currently ingests `(string Label, T[] Vector)` rows; define what “`FromRows<T>` and the new output path round-trip” means given chosen output API.
  5. **Enumerator/row validity contract:** exact semantics for “valid only for current iteration” vs “usable outside loop for lifetime of frame”, and behavior after disposal/advance (must be documented and tested).

## GitHub issues log

- [ ] (none yet)
