# Plan — #507: `NivaraQuery<T>.Frame` is dead internal code, and now an ambiguous disposal path

Branch: `khurram/507`, based on `khurram/502` (which is stacked on `khurram/501`).

## Problem

`src/Nivara/Linq/NivaraQuery.cs:34` declares:

```csharp
internal QueryFrame Frame => frame;
```

It has **no reader anywhere in the repository**, and it has never had one. Three
independent confirmations:

1. `rg '\.Frame\b'` across the repo returns nothing. The only hits are `docs/TODO.md`
   prose and an unrelated `result.Frame` in `docs/STREAMING.md:247`.
2. code-memory: symbol `ebe2d5862ca4416ab87baa192ef74062`
   (`Nivara.Linq.NivaraQuery<T>.Frame`) has **zero rows** in `RelationshipRecord`.
3. `git grep '\.Frame\b'` returns nothing at `fddee926` — the commit that *introduced*
   the property — and nothing at `1072a420~1`.

So it is dead from birth, not merely unused. It is also a redundant duplicate:
`public QueryFrame AsQueryFrame()` (`NivaraQuery.cs:54`) returns the identical object and
has done since #264 (`1072a420`), which is when `AsQueryFrame` was made public.

The hazard #501 created is real. `QueryFrame.Dispose()` is `handle.Release()` on the
shared `QuerySourceHandle` (`QueryFrame.cs:1080`) — one handle per source for the whole
chain — and `NivaraQuery<T>.Dispose()` (`NivaraQuery.cs:57`) forwards to the same thing.
So `query.Frame.Dispose()` and `query.Dispose()` are the *identical action*, and the
`Frame` spelling carried none of the one-source-one-release documentation that `Dispose`
inherits from the type-level `<remarks>`.

### Why no compiler caught it

Grounded against Microsoft Learn rather than assumed:

- **IDE0051** "flags unused *private* methods, fields, properties, and events" — private
  only, so an `internal` property is out of scope.
- **CS0169** "The *private* field ... is never used" — private only, and fields only.
- **CS0414** covers private/internal *fields* that are *assigned but never read*. A
  get-only property is not in scope at any accessibility. Its docs also state the
  underlying reason: *"If the variable is declared as protected or public, no error will
  be generated because the compiler cannot know whether a derived class might use it."*
- **CA1823** is private-fields-only and **not enabled by default in .NET 10**.

The project has `TreatWarningsAsErrors` and `EnforceCodeStyleInBuild` both on, and
neither would ever have flagged this. That is why it survived 10 commits and ~7 weeks.

## Proposed changes

### 1. Delete the property

Remove `NivaraQuery.cs:34` and its trailing blank line. Nothing else in the file moves.

### 2. Document the hazard on the accessor that survives

Deleting `Frame` *relocates* the hazard rather than removing it. `AsQueryFrame()` is
public, returns the same object, and its doc comment is only "Returns the underlying
lazy query frame for advanced composition" — carrying none of #501's disposal
contract. A reader holding the frame is one line away from the undocumented release.

Add a `<remarks>` to `NivaraQuery.cs:54` stating that the returned frame shares the
query's source, so disposing it releases the source for the whole chain, and that
`Dispose` on the query is the same action.

### 3. Gate the shape of the release surface

One test in `tests/Nivara.Tests/Query/NivaraQueryDisposalTests.cs`, beside the existing
type-shape gate at line 58. Requires two added usings (`System.Reflection`,
`Nivara.Query`); `System.Reflection` is not an implicit using.

It enumerates declared members of `NivaraQuery<T>` whose type is `typeof(QueryFrame)` —
public **and** non-public, instance, `DeclaredOnly` — and asserts the set is exactly
`{ AsQueryFrame() }`. A second verdict asserts `NivaraGroupedQuery<TKey, T>` declares
**none**: it deliberately has no `AsQueryFrame()`, and #501's CHANGELOG frames that
absence as the defect, so growing one back would regress the fix's intent.

Two verdicts with two distinct messages, so a structural failure never reports as a
numeric one.

**The gate's doc comment must state what it proves and what it cannot:**

- *Proves* — the release surface's **shape**, so the next name (`RawFrame`,
  `UnderlyingFrame`, a second accessor) has to be re-argued rather than added silently.
  It generalises past the single spelling `Frame`.
- *Does not prove* — that any member is unused. That claim rests on the search.
- *Cannot prove* — that `AsQueryFrame()` is documented. `GenerateDocumentationFile` is
  not set and no test reads XML docs, so the strongest form of this gate is not
  buildable in this repo. This limit is the honest reason the gate is shape-only.

Precedent: `tests/Nivara.Tests/AutoDiff/WeightAccessConsistencyTests.cs` pins removed
accessors with `GetProperty(...) Is.Null`.

## Blast radius

**Commit 1 (delete).** `Nivara.Linq.NivaraQuery<T>` only.

- **The compiler is a complete proof here, not a heuristic.** `Nivara.csproj` grants
  `InternalsVisibleTo` to exactly three assemblies — `Nivara.Tests`, `Nivara.Extensions`,
  `Nivara.PerformanceTests` — with no strong-name or public-key variants and no external
  grant, and all three are in `Nivara.slnx`. So a clean `dotnet build Nivara.slnx`
  *proves* no in-repo reader existed: any reader would have been a compile error. That is
  a stronger statement than ripgrep can make.
- `InternalsVisibleTo` does **not** extend to `Nivara.Samples` or `NivaraChat`, so the
  samples could not have read it either.
- Tests covering the surrounding contract, all expected unaffected:
  `NivaraQueryDisposalTests`, `QuerySourceHandleTests`, `ScanAsQueryFrameHandleTests`.

**Commit 2 (doc comment).** Zero. A `<remarks>` cannot change behaviour, and no caller
of `AsQueryFrame()` is touched.

**Commit 3 (gate).** `tests/Nivara.Tests/Query/NivaraQueryDisposalTests.cs` only. New
test, no existing assertion modified.

**Deliberately not touched:** `NivaraGroupedQuery<TKey, T>` holds `baseFrame` as a field
with no exposing property — it has no duplicate of this defect and nothing to fix.

## Verification

1. `dotnet build Nivara.slnx -c Release` — expect zero new warnings. Note there is *no*
   analyzer safety net for this class of change (see above), so the clean build is the
   finding, not a formality.
2. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~NivaraQueryDisposalTests"`
   — the new gate must run and pass. As cheap insurance this also covers
   `QuerySourceHandleTests` and `ScanAsQueryFrameHandleTests`.

### Results

| Step | Result |
|---|---|
| `dotnet build Nivara.slnx -c Release` | succeeded, **0 warnings**, 0 errors — after each of commits 2, 3, 4, 5 |
| Targeted fixtures (disposal / handle / streaming) | **58/58 pass** |
| Full suite, `--filter "Category!=Performance"` | **3795 passed, 14 skipped, 0 failed** (2 m 43 s) |
| `rg 'AsQueryFrame\(\)\.Dispose'` over `src/` and `tests/` | no code hits; the two remaining hits are the CHANGELOG's #501 entry and this file |
| `rg 'owning frame'` repo-wide | no hits — the three stale comments are gone |

### The stale-build trap, recorded because it nearly produced a false pass

The gate's negative control was run by adding a temporary `internal QueryFrame RawFrame`
and confirming the test went red. Removing it and rebuilding left `RawFrame` **compiled into
`bin/`, and a subsequent `--no-build` run reported the gate failing on a property that no
longer existed in source** — an incremental build that did not notice the reverted file.

`dotnet build --no-incremental` cleared it and the fixture returned to 9/9. Worth knowing
for anyone re-running that negative control: a red result immediately after an edit is not
trustworthy until the build is forced. Had the stale copy been *green* rather than red, the
control would have looked like it had never fired.

### G2 review findings

Both pre-deletion reviews cleared with no code changes required.

- **Against #507** — the issue offered two options, delete or document, and recommended
  deletion. Both landed: the property is gone, and the surviving accessor carries the
  warning. Root cause addressed rather than symptom.
- **Dead code left behind** — checked. The `frame` field is still read by 20+ members, and
  `AsQueryFrame()` is now the sole `QueryFrame`-returning member on the type.
- **Gate proven in both directions** — not assumed. See the negative control above.
- **Plan drift** — one addition, recorded at G1 and at commit `581685f9`: commit 5 (#512),
  confirmed by the human rather than folded in silently.
- **Commit count** — the plan listed 6; 6 landed, in order, with no unrelated commits.

### Coverage this work does not claim

The full-suite result is a **regression check**, not a verification of the fix. Nothing in
the suite can observe the absence of a member that had no readers; the deletion is proved by
the clean build plus the search, and the doc comment by nothing at all — no XML doc file is
produced, so no test can assert on it. 3795 green means "nothing that was passing broke",
which is a weaker claim than it looks and is the only claim the suite supports here.

### Claims this work does NOT make

- Commits 1 and 2 have **no behavioural verification**. A green suite proves only that
  nothing changed, which is the point rather than a result. The evidence that `Frame`
  was dead is the search, not a test.
- The gate is a **decision** gate. It can fail on reintroduction, but it cannot prove
  deadness and must not be described as doing so.

## No CHANGELOG entry

`Frame` is `internal`, had no reader, and produces no observable behaviour change;
commit 2 is a doc comment on an existing member. Stated explicitly in the PR rather
than recorded as a hollow entry.

## Planned commits

1. `docs: plan #507 in TODO.md`
2. `refactor: delete the dead internal NivaraQuery<T>.Frame`
3. `docs: state the chain-release consequence on AsQueryFrame()`
4. `test: gate the shape of the query release surface`
5. `test: repoint the remaining AsQueryFrame().Dispose() call sites` — **added at G1**, see below
6. `docs: remove TODO.md — plan executed`

> Commit 5 was not in the original plan. G1 grounding found #512's four call sites, and
> commit 3 documents "dispose the query, not the frame you got from `AsQueryFrame()`".
> Landing that while four repo tests do the opposite would make the branch
> self-contradictory — and `NivaraQueryToObjectsAsyncTests.cs:89-100` already does the
> harmful version, disposing a **derived** query. Confirmed by the human at G1 to be fixed
> here rather than left open on its own branch. It stays a separate commit, so each commit
> still carries one reason.

> As each task executes, if you find deferred work or a concern outside this plan, create a
> tracked issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
> number below — do not rely on memory, as compaction during execution can lose items.

## GitHub issues log

- [x] #507 — the subject of this plan.
- [x] #512 — four tests still release a lazy source via `query.AsQueryFrame().Dispose()`
  instead of `query.Dispose()` (`AsyncStreamingTests.cs:485`,
  `NivaraQueryToObjectsAsyncTests.cs:66,100,155`). Pre-#501 leftovers that commit
  9c9e10fd missed when it repointed `ScanAsQueryFrameHandleTests`. One of them
  (`NivaraQueryToObjectsAsyncTests.cs:89-100`) disposes a **derived** query
  (`ScanQuery<Person>(file).Where(...)`), so it releases the source for the whole chain —
  the precise hazard #507 accuses `Frame` of, already realised through the surviving
  accessor. Mechanically fixable and behaviour-preserving. Filed during grounding, then
  **confirmed at G1 to be fixed in this branch** as commit 5 — see the note under
  Planned commits. Closes with that commit.
