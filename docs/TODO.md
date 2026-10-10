# Plan: #557 — `NivaraQuery<T>.AsStream` context overloads (LINQ DSL parity)

## Problem

#514 exposed the streaming memory budget through `QueryFrame.AsStream(NivaraExecutionContext)`.
The LINQ-DSL wrapper `NivaraQuery<T>.AsStream(int chunkSize = 10000, CancellationToken ct = default)`
(`src/Nivara/Linq/NivaraQuery.cs:327`) is a passthrough with no context overload, so a caller
working through the documented LINQ entry point cannot set `NivaraExecutionContext.MemoryBudget`
without first converting (`query.AsQueryFrame().AsStream(context)`). The two entry points disagree
in surface: `QueryFrame` exposes three `AsStream` overloads, `NivaraQuery<T>` exposes one.

No new behavior — purely surface parity, delegating to `frame.AsStream(...)`, mirroring
`QueryFrame`'s XML docs.

## Proposed changes

### 1. `src/Nivara/Linq/NivaraQuery.cs`

Add `using Nivara.Execution;` (for `NivaraExecutionContext`), then two overloads directly after
the existing `AsStream` (line 328), mirroring `QueryFrame.AsStream` (:455/:475/:491) docs:

```csharp
public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, CancellationToken ct = default)
    => frame.AsStream(context, ct);

public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, int chunkSize, CancellationToken ct = default)
    => frame.AsStream(context, chunkSize, ct);
```

- Pure delegation; `ArgumentNullException.ThrowIfNull(context)` and context-cloning stay in
  `QueryFrame` (single source of truth).
- Same overload shape as `QueryFrame`, so no new overload ambiguity beyond what `QueryFrame`
  already has.

### 2. Tests — `tests/Nivara.Tests/Query/AsStreamBudgetTests.cs`

Wrapper-level mirrors of the existing `QueryFrame` budget tests, via the internal
`FromFrame<T>(QueryFrame)` extension (`src/Nivara/Linq/TypedLinqExtensions.cs:34`,
`InternalsVisibleTo` covers `Nivara.Tests`):

- `AsStream_ContextWithMemoryBudget_DerivesChunkSizeFromBudget` — `new QueryFrame(chunkedSource)
  .FromFrame<IntRow>()` then `query.AsStream(context)` → 2,000-row chunks (budget reached the
  strategy through the typed wrapper).
- `AsStream_ContextWithExplicitChunkSize_UsesChunkSizeNotBudget` — `query.AsStream(context, 3_000)`.
- `AsStream_NullContext_ThrowsArgumentNullException` — `query.AsStream((NivaraExecutionContext)null!)`.
- Add `sealed class IntRow { public int A { get; set; } }` (column `A` is `int`); add
  `using Nivara.Linq;`.

Existing `AsyncStreamingTests.NivaraQuery_T_AsStream_Passthrough` (:944) keeps covering the bare
`int` overload.

### 3. Docs + CHANGELOG

- `docs/STREAMING.md` :10 — expand the `NivaraQuery<T>.AsStream` passthrough table row to list all
  three overloads; one sentence in §"Setting the budget" (:181) noting the typed wrapper mirrors
  `QueryFrame`.
- `docs/adr/006-streaming-memory-budget-exposure.md` — short addendum (2026-10-11) recording the
  LINQ-wrapper surface parity (#557).
- `CHANGELOG.md` `[Unreleased] → Added` — entry for #557 in the #514/#556 style.

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet test -c Release --filter "FullyQualifiedName~AsStreamBudgetTests"` (targeted)
- `dotnet test -c Release --filter "FullyQualifiedName~AsyncStreamingTests"` (targeted, regression)
- Docs gates touched by the doc edits: `DocumentationSnippetTests` / `AdrIndexTests` / doc
  citation gate (`File.cs:NN` refs) — run those fixtures.
- No full `dotnet test` suite run without asking.

## Planned commits

1. `docs: plan #557 AsStream context passthroughs in TODO.md`
2. `feat(linq): add context-taking AsStream overloads to NivaraQuery<T>` — src change only
3. `test(query): pin NivaraQuery<T> AsStream context passthrough parity` — tests only
4. `docs: record #557 NivaraQuery<T> AsStream parity in STREAMING/ADR-006/CHANGELOG`
5. (G2 passes) `docs: remove TODO.md — plan executed`
6. Offer push + PR (human confirms; never push without confirmation)

## Blast radius

- **`NivaraQuery<T>`** (public, `src/Nivara/Linq/NivaraQuery.cs`): additive overloads only — no
  existing call site changes. Downstream callers of the existing passthrough (`ToObjectsAsync`,
  Streamix `NivaraFlux`, samples, tests) are unaffected.
- **`QueryFrame.AsStream`** (unchanged): receives the delegated calls; null-guard, cloning, and
  budget derivation behavior identical to today.
- **Tests**: only new parity tests in `AsStreamBudgetTests`; `AsyncStreamingTests` untouched.
- **Docs**: `STREAMING.md`, `adr/006`, `CHANGELOG.md` — under the existing snippet/citation/ADR
  gates. ADR-006 keeps its heading (ADR index test reads the `docs/adr/` listing, not the body).
- No public API removal or signature change; NuGet surface is purely additive.

## Reminder

As each task executes, if deferred work or a concern surfaces that is outside this plan, create a
GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number
below — don't rely on memory; compaction during execution can lose it.

## GitHub issues log

- (none yet — #514/#556/#557 are the tracked parent/follow-up issues)