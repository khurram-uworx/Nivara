# Plan — #520: fix the stale factory shape, and stage the doc gate onto GETTING-STARTED.md and EXAMPLES.md

Branch: `khurram/520`, branched off `khurram/518` at `fb984f3c` ("cleanup").

`khurram/518` contains `khurram/515` (the snippet compile gate) and `khurram/518`'s own
markdown citation gate, so the extractor, compiler and preamble stubs are all available here.

## Problem

Two problems, one small and one large.

**Small (#520).** 15 documentation call sites use `NivaraColumn<T>.CreateFromNullable(...)`, which
does not compile. The factory is `NivaraColumn.CreateFromNullable<T>(T?[])` on the non-generic
class (`src/Nivara/NivaraColumn.Factory.cs:19`), where `T` is inferred.

**Large (stage 2 of #515).** `GETTING-STARTED.md` (63 blocks) and `EXAMPLES.md` (15 blocks) have
never been compiled by anything. #515 staged the gate onto `docs/STREAMING.md` and
`docs/AGENT-CODE-EXAMPLES.md` only. Those 78 blocks are the primary onboarding material, and one
stale API shape already drifted across 15 sites in them, which is strong evidence the rest is not
clean.

## Premise correction: #520 as filed was wrong about the cause

The issue I filed said the API was the outlier and hinted at restoring symmetry. That is incorrect,
and the evidence is unambiguous:

- `CHANGELOG.md:757` records **"Removed `NivaraColumn<T>.CreateFromNullable(Array)` (breaking,
  #222)"** — the generic-class overload was deleted deliberately, because it boxed via
  `Array.GetValue`. `NivaraColumn.CreateFromNullable<T>(T?[])` is documented as **"the single entry
  point"**, and the entry records that **"all internal dispatch and every call site now use it"**.
- #222's own body frames the boxing-free factory as the one that "supersedes" the generic-class
  overload, with the follow-up being to migrate callers and mark it obsolete.
- `where T : struct` (line 20) is why the method had to move: on the non-generic class it is a
  generic method that can constrain `T`, and it makes reference-type arguments a compile-time error
  rather than a runtime `InvalidOperationException`.

So the code migration was completed and **the documentation was never migrated**. These are stale
pre-#222 docs. Fix the docs; do not touch the API.

Two further corrections to the issue as filed:

- It counted 16 sites. `CHANGELOG.md:757` is the *before* side of a migration instruction and is
  correct in context — **15** real sites.
- `NivaraColumn<T>` is not the outlier by accident: 6 of the 7 `NivaraColumn` factories
  (`Create`, `CreateForReferenceType`, `CreateFromSpans`, `CreateFromOwnedArray(es)`) do sit on the
  generic form, and `NivaraFrame.Create` sits on the non-generic form because `NivaraFrame` is
  inherently heterogeneous. The relocation was a considered change, not drift.

## Measurements taken on this branch

| Measure | GETTING-STARTED.md | EXAMPLES.md |
|---|---|---|
| blocks | 63 | 15 |
| carry their own `using` directives | 18 | 9 |
| mix a type declaration with statements | 8 | 7 |
| use `await` | 1 | 0 |

Two consequences:

1. **27 blocks need a compiler capability that does not exist.** A `using` emitted after the
   generated prologue but before the body is legal; a `using` *inside* the body is CS1529. Those 27
   blocks would fail on a rule unrelated to whether the documented API is real.
2. **15 blocks (19% of the corpus) are genuinely uncompilable.** The pattern is deliberate
   pedagogy — build the frame, *then* declare the row type, then query it:

   ```csharp
   var frame = NivaraFrame.Create(...);   // data first
   public sealed class Person { ... }     // then the row type - CS8803
   ```

   No wrapper can compile this: a type declaration cannot follow top-level statements. The reader
   cannot paste it into any file. Same defect class as `README.md:58`.

## Proposed changes

### 1. Fix the 15 stale call sites

`GETTING-STARTED.md` (9) and `EXAMPLES.md` (5): `NivaraColumn<int>.CreateFromNullable(x)` becomes
`NivaraColumn.CreateFromNullable(x)`. Inference resolves `T` from the `int?[]` / `float?[]`
argument at every site — verified individually, including the site inside
`NivaraFrame.Create((string, IColumn)[])` where the tuple target type does not affect inference.

`AGENTS.md:153` is a **signature, not a call**: `NivaraColumn<T>.CreateFromNullable(T?[])` becomes
`NivaraColumn.CreateFromNullable<T>(T?[])`. Highest-value single edit — it is the helper list an
agent reads before writing the call, which is how 15 sites drifted in the first place.

Already correct, left alone: `docs/AGENT-CODE-EXAMPLES.md`, `docs/AUTODIFF.md`,
`docs/BFLOAT16.md`, and `CHANGELOG.md:757`.

### 2. Add leading-`using` hoisting to `DocSnippetCompiler`

Split the block body's leading run of `using` directives (tolerating interleaved comments and blank
lines) out of the statement region and emit them into the generated prologue, deduplicated against
`BaseUsings`. **Document text is never edited** — this is an arrangement of the generated unit, not
a rewrite of the snippet.

Line mapping must account for the hoist, or diagnostics will be off by the number of hoisted lines.
That is the whole risk in this step and it gets a dedicated test.

### 3. Split the 15 mixed blocks

Each becomes two fenced blocks — the row type, then the statements — with prose between them
carrying the original comment. Splitting rather than reordering is deliberate: it preserves the
data-then-row-type teaching order *and* makes both halves compile. Reordering is a smaller diff but
teaches the row type before the reader knows why they need it.

This changes the repository's snippet count: 15 splits is +15 blocks, so the universe goes 233 → 248.

### 4. Gate both documents

- Move `GETTING-STARTED.md` and `EXAMPLES.md` from `UngatedDocuments` to `GatedDocuments`.
- One preamble per section for `GETTING-STARTED.md`'s 13 `##` headings rather than one per block:
  the document is cumulative (block #3 consumes `column` from block #2), and a section-scoped
  context is 13 preambles instead of 63. Where a section's context is wrong for one block, the
  compiler names that block.
- Extend `KnownPreambles`; the bidirectional assertion already fails on drift.
- Re-pin the coverage numbers. Projected: universe **248**, ungated **141**, compiled **105**,
  excluded **2** (the existing ASP.NET Core pair). These are projections; the real figures get
  pinned from what the gate reports.
- Add gate comments to all 78 blocks.

The gate goes red at this point. That is the intended sequence — it cannot be enabled until the docs
are green, and the docs cannot be fixed until it runs. CI only ever sees the pushed state.

### 5. Fix what the gate finds, one class of defect per commit

### 6. `CHANGELOG.md`

One entry under the existing `[Unreleased] → Fixed`, house style: cite #520, state that the shape
was stale since #222, and that the API was deliberately relocated so only the docs needed changing.

## Blast radius

- **Markdown only** plus one compiler change. No `src/` change, no public API change, no behaviour
  change. `NivaraColumn.CreateFromNullable` and `QueryFrame.AsStream` are untouched.
- `DocSnippetCompiler` gains the hoisting path, which every gated block goes through — including
  the 14 already gated on `docs/STREAMING.md` / `docs/AGENT-CODE-EXAMPLES.md`. A hoisting bug would
  corrupt line mapping for the whole gate, so its tests pin the mapping explicitly.
- **Reader-visible**: `GETTING-STARTED.md` is the primary onboarding document. 15 blocks change
  shape (one becomes two) and 14 call shapes change. Content is preserved; only structure and the
  factory form change.
- `AGENTS.md` edit changes what future agents are told to write.
- The citation gate (`DocCitations`, #518) also scans markdown, so a split that moves a
  `File.cs:NN` citation out of a block could orphan it. Both gates are run before each commit.
- CI needs no change. The gate carries no `[Category("Performance")]`, so
  `--filter "Category!=Performance"` keeps it in the run.
- No downstream callers: both gates are tests.

## Verification

1. `dotnet build Nivara.slnx -c Release` — preambles are compiled C#, so a broken preamble fails
   the build itself.
2. `dotnet test -c Release --filter "FullyQualifiedName~DocumentationSnippetTests"` — green, and
   reporting the real pinned numbers.
3. The citation gate still passes — `FullyQualifiedName~DocCitation`.
4. The negative control still fails: `Fixtures/Broken.md` reports its own line.
5. Every one of the 15 fixed call sites compiles — proven by the gate for the 14 in the two
   documents, and by inspection for `AGENTS.md:153`, which is prose rather than a fenced block.
6. Re-run the bogus-member injection check to confirm line mapping survived the hoist.
7. Full suite `dotnet test -c Release --filter "Category!=Performance"` — **ask first**; the
   previous branch declined it, and the same gap would recur here.

## Planned commits

1. `docs: plan #520 in TODO.md`
2. `docs: correct the stale NivaraColumn<T>.CreateFromNullable call shape`
3. `test: hoist leading using directives out of gated snippet bodies`
4. `docs: split the blocks that mix a row type with statements`
5. `test: gate GETTING-STARTED.md and EXAMPLES.md snippets`
6. `fix: <one commit per class of defect the gate finds>` (additive, count unknown)
7. `docs: remove TODO.md - plan executed`

## Risks and open items

- **Unknown defect yield.** One stale shape already spans 15 sites in documents nothing has ever
  compiled. I expect more, and the count is not knowable until the gate runs. If it is large this
  becomes several commits, and the documentation fixes are the deliverable rather than the gate.
- **Section-scoped preambles may be too coarse.** `GETTING-STARTED.md`'s "Automatic Differentiation"
  section alone has 16 blocks. Where a context is wrong for a specific block the gate names it, and
  the fix is a per-block `locals:` override — the mechanism already supports that, but it means
  preambles may not be as tidy as "one per section".
- **The hoist changes line mapping for all gated blocks.** Highest-risk step in the plan. Mitigated
  by dedicated tests and by re-running the injection check.
- **Stage 2 may need its own counted exclusions** — illustrative fragments that are not meant to
  compile. Each would be asserted by value, like the existing ASP.NET pair.
- `DocWrapMode.File` has no consumer among the gated blocks. Stage 2's type-only blocks will use it,
  which resolves the speculative-generality concern.

## GitHub issues log

- [x] #517 — `docs/ACCELERATION.md:22` cites a nonexistent `src/Nivara.Gpu`
- [x] #518 — `docs/LINQ.md` calls `QueryFrame` internal; it is `public sealed class`
- [x] #520 — the subject of this plan. Premise correction is recorded above and in the issue.
- [ ] `README.md:58` — mixes a row-type declaration with top-level statements, the same defect as
      the 15 blocks in step 3. Not in the stage-2 allowlist, so the gate will not cover it; to be
      confirmed and filed once step 3's exact shape is known.

Reminder: as each task executes, if you find deferred work or a concern (a known limitation, a
follow-up, a refactor) that is outside the current plan, create a tracked issue immediately via
`gh issue create --repo khurram-uworx/Nivara` and record its number in the log above. Do not rely on
memory or wait until the plan finishes — compaction during execution can lose important items.