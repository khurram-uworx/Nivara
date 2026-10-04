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
| carry leading `using` **directives** | 14 | 9 |
| carry `using` **statements** (`using var`, `using (…)`) | 8 | 2 |
| type declaration **before** statements (CS8803) | 7 | 7 |
| type declaration **after** statements (legal) | 1 | 0 |
| statements only / type-declaration only | 52 / 1 | 8 / 0 |

Two consequences:

1. **23 blocks need a compiler capability that does not exist.** A `using` emitted in the generated
   prologue is legal; a `using` *directive* inside the body is CS1529 ("A using clause must precede
   all other elements defined in the namespace except extern alias declarations"). Those 23 blocks
   would fail on a rule unrelated to whether the documented API is real.

   The other 10 blocks contain `using` **statements** — `using var csvQuery = Csv.ScanQuery<Employee>(…)`,
   `using (GradientUtils.Grad())` — which are ordinary statements, legal where they stand, and must
   never be hoisted. Conflating the two forms is the trap here; see step 2.
2. **14 blocks (18% of the corpus) are genuinely uncompilable.** The pattern is deliberate
   pedagogy — declare the row type, then build the frame, then query it:

   ```csharp
   public sealed class Person { ... }     // the row type first
   var frame = NivaraFrame.Create(...);   // then the data - CS8803
   ```

   CS8803 is "Top-level statements must precede namespace and type declarations", so no wrapper can
   compile this: the reader cannot paste it into any file. Same defect class as `README.md:58`.

   Order decides, and one block is on the right side of it. `GETTING-STARTED.md:210` is
   statements-then-type, which is the order CS8803 requires, so it compiles as written and needs no
   fix — it was a false positive of a shape-only count.

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

Parse the block body with `CSharpSyntaxTree.ParseText` and lift the leading run of
`CompilationUnitSyntax.Usings` out of the statement region, emitting them into the generated
prologue deduplicated against `BaseUsings`. **Document text is never edited** — this arranges the
generated unit, it does not rewrite the snippet.

Use the syntax tree rather than a regex, and this is not a style preference. Roslyn already puts
`using Nivara.Linq;` in `Usings` and `using var w = ReverseGradTensor<float>.FromArray(…);` in
`Members`, so the directive/statement distinction is structural. A line-based `^\s*using\s` hoist —
the obvious implementation — would have hoisted the `using var` at `EXAMPLES.md:457` and the
`using (GradientUtils.Grad())` blocks at `GETTING-STARTED.md:1085` and `:1274`, turning three
working snippets into compile failures and corrupting their line mapping. The 23/10 split in the
table above is the reason to reach for the parser.

Line mapping needs one added term: `Map` computes `block.StartLine + (generatedLine - bodyStart)`,
which under-counts by the number of hoisted lines, so it becomes
`block.StartLine + hoistedCount + (generatedLine - bodyStart)`. Diagnostics in the prologue keep
reporting `block.StartLine`. A test pins the mapping on a hoisted block, because silent off-by-N
would misdirect every future reader to the wrong line.

### 3. Split the 14 blocks that declare a type before using it

`GETTING-STARTED.md` lines 265, 332, 467, 774, 839, 899, 926 and `EXAMPLES.md` lines 253, 295, 328,
392, 501, 563, 669.

Each becomes two fenced blocks — the row type, then the statements — with prose between them
carrying the original comment. Splitting rather than reordering is deliberate: it preserves the
teaching order *and* makes both halves compile. Reordering is a smaller diff but moves the row type
ahead of the data it exists to describe.

This changes the repository's snippet count: 14 splits is +14 blocks, so the universe goes 233 → 247.

### 4. Gate both documents

- Move `GETTING-STARTED.md` and `EXAMPLES.md` from `UngatedDocuments` to `GatedDocuments`.
- Add gate comments to all 78 blocks. Almost all stay `TopLevel` (the default), which is already
  correct for statements and for the legal statements-then-type block at line 210. The one
  type-declaration-only block needs `mode: File`, because `File` is the only mode that emits a
  genuine library output kind — CS8805 rejects top-level statements in a `DynamicallyLinkedLibrary`,
  so `File` cannot hold statements and `TopLevel` must not hold a bare type.
- Progressive context, one preamble per section for `GETTING-STARTED.md`'s 13 `##` headings rather
  than one per block: the document is cumulative (block #3 consumes `column` from block #2), and a
  section-scoped context is 13 preambles instead of 63. Where a section's context is wrong for one
  block, the compiler names that block and the fix is a per-block `locals:` override.
- Extend `KnownPreambles`; the bidirectional assertion already fails on drift.
- Re-pin the coverage numbers, and update the prose in the assertion's failure message so it still
  describes the real split. Projected: universe **247**, ungated **134**, compiled **111**,
  excluded **2** (the existing ASP.NET Core pair). Projections; the real figures get pinned from
  what the gate reports.
- `GatedStreamixBlocks_OnlyDependOnTheStreamixPackageTheRepoReferences` filters on
  `DocumentPath == "docs/STREAMING.md"`, so it stays correct untouched.

The gate goes red at this point. That is the intended sequence — it cannot be enabled until the docs
are green, and the docs cannot be fixed until it runs. CI only ever sees the pushed state.

### 5. Fix what the gate finds, one class of defect per commit

### 6. `CHANGELOG.md`

One entry under the existing `[Unreleased] → Fixed`, house style: cite #520, state that the shape
was stale since #222, and that the API was deliberately relocated so only the docs needed changing.

## Grounding (G1) — done, and it changed the plan

Two numbers in the first draft of this plan were wrong. Both were caught by grounding rather than by
a failing build, which is the point of running the gate before implementing.

**CS1529 is real and is an error.** *"A using clause must precede all other elements defined in the
namespace except extern alias declarations"*, listed under the using-directive errors. So the 23
hoistable blocks would indeed fail today for a reason unrelated to the API.

**CS8803 makes order, not mixture, the defect.** *"Top-level statements must precede namespace and
type declarations"*, and the compiler messages page states a file with top-level statements can also
contain type definitions *"but they must come after the top-level statements"*. That makes the
mixed-block count 14, not 15: `GETTING-STARTED.md:210` is statements-then-type, which is the order
the error message requires, so it compiles as written. I had counted shape and called it a defect.

Also grounded against the source rather than assumed:

- `DocSnippetCompiler.Map` needs exactly one added term (`+ hoistedCount`), not a redesign.
- `File` and `TopLevel` share one generated-text branch and differ only in `OutputKind`, so
  CS8805 is what decides which mode a block can use — `File` for a bare type, `TopLevel` for
  statements and for the legal mixed block.
- `GatedStreamixBlocks_OnlyDependOnTheStreamixPackageTheRepoReferences` filters on
  `DocumentPath`, so adding documents cannot perturb it.
- 14 pinned numbers (12/2/219/233) each name their own split in prose; all four get re-pinned with
  their prose, not just the values.
- Duplicate directives are CS0105, a **warning**, so the error-only filter already tolerates them;
  deduping in step 2 is hygiene rather than a correctness requirement.

The blast radius below is unchanged by grounding. No decision here needs the human; the two the plan
already records (split over reorder, section-scoped preambles) were taken when the plan was first
put to them, and grounding refined the counts underneath them rather than the direction.

## Blast radius

- **Markdown only** plus one compiler change. No `src/` change, no public API change, no behaviour
  change. `NivaraColumn.CreateFromNullable` and `QueryFrame.AsStream` are untouched.
- `DocSnippetCompiler` gains the hoisting path, which every gated block goes through — including
  the 14 already gated on `docs/STREAMING.md` / `docs/AGENT-CODE-EXAMPLES.md`. A hoisting bug would
  corrupt line mapping for the whole gate, so its tests pin the mapping explicitly.
- **Reader-visible**: `GETTING-STARTED.md` is the primary onboarding document. 14 blocks change
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
- `DocWrapMode.File` previously had no consumer among the gated blocks, which was a speculative
  generality. Stage 2's one type-declaration-only block needs it, so the concern resolves itself.
- **The defect yield is still unknown, and grounding does not touch that.** Everything above is
  structural. Whether the 78 blocks reference Nivara members that do not exist is a separate
  question that only a run can answer, and it is the open risk on this branch.

## GitHub issues log

- [x] #517 — `docs/ACCELERATION.md:22` cites a nonexistent `src/Nivara.Gpu`
- [x] #518 — `docs/LINQ.md` calls `QueryFrame` internal; it is `public sealed class`
- [x] #520 — the subject of this plan. Premise correction is recorded above and in the issue.
- [ ] `README.md:58` — declares a row type before top-level statements, the same CS8803 defect as
      the 14 blocks in step 3. Not in the stage-2 allowlist, so the gate will not cover it; to be
      confirmed and filed once step 3's exact shape is known.

Reminder: as each task executes, if you find deferred work or a concern (a known limitation, a
follow-up, a refactor) that is outside the current plan, create a tracked issue immediately via
`gh issue create --repo khurram-uworx/Nivara` and record its number in the log above. Do not rely on
memory or wait until the plan finishes — compaction during execution can lose important items.