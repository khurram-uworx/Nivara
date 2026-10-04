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

- The #222 removal entry in `CHANGELOG.md` records **"Removed `NivaraColumn<T>.CreateFromNullable(Array)` (breaking,
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

- It counted 16 sites. The #222 removal entry in `CHANGELOG.md` is the *before* side of a migration
  instruction and is correct in context — **15** real sites.
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
`docs/BFLOAT16.md`, and the #222 removal entry in `CHANGELOG.md`.

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

**Correction, and the gate is what caught it.** An interim Roslyn diagnostic claimed zero CS8803
blocks remained after the 14 splits. That was wrong: it compared only the *first* type declaration
against the *first* top-level statement, so a `statements -> type -> statements` block read as legal
because its first statement preceded its type. The rule is that *no* top-level statement may follow
*any* type declaration, so the check must be "exists a type that starts before some statement". The
gate reports two survivors — `GETTING-STARTED.md` fences 210 and 378 — both of that shape, needing
2 more splits for **16** total and a universe of 249. Prefer the gate's own diagnostics to a
hand-rolled classifier; this is the second time a heuristic has been wrong here, the first being the
`using var` confusion.

### The first gate run was wrong about the cause of 5 of its own errors

The first run reported **140** errors, and five of the defects it named were not defects at all.
`DocSnippetCompiler.BaseUsings` listed 17 of the 32 namespaces the public surface occupies, so the
gate could not resolve names that genuinely exist:

| Reported as missing | Actually in | Errors withdrawn |
|---|---|---|
| `NivaraFrame.ToTensor` | `Nivara.Tensors` | 2 |
| `NivaraFrame.ToReverseGradTensors` | `Nivara.AutoDiff` | 1 |
| `ColumnDisambiguationStrategy` | `Nivara.Operations` | 2 |
| `ReverseGradOperations` | `Nivara.AutoDiff.Operations` | 5 |
| `FileStream` / `FileMode` | `System.IO` | — |
| `JoinException` | `Nivara.Exceptions` | — |

Widening `BaseUsings` to every public namespace dropped the count to **129**. Widening it cannot mask
a defect — it affects name resolution only, never whether a member is present — so this is strictly a
harness fix, and it is what makes the remaining diagnostics trustworthy. Two rounds of classification
were discarded because of this, and the CS0104 pair it introduced is a real API finding rather than a
doc defect (#532).

**Verified final partition — the parts sum to the whole:**

| Class | Count | Issue |
|---|---|---|
| Missing context (70 × CS0103, 30 × CS0246) | 100 | #524 |
| Column/series members that do not exist (CS0019, CS1061 ×2) | 3 | #525 |
| AutoDiff call sites (CS1061 ×2, CS1739 ×3, CS8130 ×1) | 6 | #526 |
| IO / Arrow / tokenizer call sites (CS1061 ×2, CS0117 ×5) | 7 | #527 |
| `Tensor<T>.AsSpan()` does not exist (CS1929 ×2) | 2 | #528 |
| `int[]` where `double[]` is required (CS1503 ×5) | 5 | #529 |
| Block-internal name collisions (CS0128, CS0136) | 2 | #530 |
| Blocks still needing a split (CS8803 ×2) | 2 | #531 |
| `SchemaValidationException` declared twice (CS0104 ×2) | 2 | #532 |
| **Total** | **129** | |

`frame` alone accounts for 36 of the missing-context errors and appears under several incompatible
shapes, so no single preamble can serve it; that work is per-block `locals:` authoring (#524).

Each of the 29 genuine-defect diagnostics needs the real API looked up before the doc is corrected,
which is the slow part and the actual deliverable — so it becomes follow-up work rather than more
commits on this branch (see *Split of work* below).

- Move `GETTING-STARTED.md` and `EXAMPLES.md` from `UngatedDocuments` to `GatedDocuments`.
- 14 blocks become TYPE-ONLY and need `mode: File` (7 in each document, all inserted); the other 92
  of the 106 gated blocks stay `TopLevel`. Two more still need splitting — see #531.
- Progressive context: the 100 context errors (#524) show a per-section preamble is too coarse, because
  `frame` has several different shapes. Expect mostly per-block `locals:` with a small number of
  named preambles for the row types the splits created.
- Extend `KnownPreambles`; the bidirectional assertion already fails on drift. **Deferred to #524** —
  no preamble is referenced yet, because every block that needs one is still missing context.
- Re-pin the coverage numbers from what the gate reports rather than from projection. The first
  projection (universe 247, ungated 141, compiled 104) was right on universe and ungated but the
  2 extra splits move the universe to 249.
- `GatedStreamixBlocks_OnlyDependOnTheStreamixPackageTheRepoReferences` filters on
  `DocumentPath == "docs/STREAMING.md"`, so it stays correct untouched.

The gate goes red at this point. That is the intended sequence — it cannot be enabled until the docs
are green, and the docs cannot be fixed until it runs. CI only ever sees the pushed state.

### 5. Split of work: gate here, defects as follow-up issues

The human's call was to split this: land the gate red, and file the defects it found rather than
fixing them here. The reasoning that matters is not scheduling but **what the branch is for**. Once
the gate is enabled, any fix lands *after* the check that catches it, so the fixes are ordinary
follow-up work with an issue number each — while the branch stays one reviewable unit whose only
purpose is to make the check exist.

Two alternatives were put to the human and rejected: pushing all 129 fixes through this branch (many
hours of mechanical `locals:` authoring mixed in with a harness change, so the harness change could
not be reviewed on its own), and narrowing the allowlist to blocks that already pass. The second was
rejected on principle — an allowlist that only admits passing blocks cannot fail, and a gate that
cannot fail is indistinguishable from no gate at all.

Nine issues filed, one per fix rather than one per diagnostic, each stating its site count so the
partition in section 4 can be audited against the gate output. Nothing is annotated in the documents:
the gate comment records a block's declared context, and defect commentary in prose is exactly the
kind of comment that goes stale.

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
2. `dotnet test -c Release --filter "FullyQualifiedName~DocumentationSnippetTests"` — **red by
   design**: `EveryGatedBlock_CompilesWithoutErrors` reports the 129 in section 4. The other ten pass,
   including the coverage assertion that pins 104/2/141/247, so a red here is the known count and
   not a regression. A change in that number means the partition above is stale.
3. The citation gate still passes — `FullyQualifiedName~DocCitation` (8 tests, verified green before
   each of the two gate commits).
4. The negative control still fails: `Fixtures/Broken.md` reports its own line.
5. The 15 fixed call sites are pinned by review, not by the gate. The gate cannot certify them while
   it is red — most of those blocks still fail for missing context — so the claim rests on the diff
   plus the fact that `NivaraColumn.CreateFromNullable<T>(T?[])` is the only such factory in
   `src/Nivara/NivaraColumn.Factory.cs:19`. #525 onward is where this stops being true.
6. Re-run the bogus-member injection check to confirm line mapping survived the hoist.
7. Full suite `dotnet test -c Release --filter "Category!=Performance"` — **ask first**; the
   previous branch declined it, and the same gap would recur here.

## Planned commits

1. `docs: plan #520 in TODO.md`
2. `docs: correct the stale NivaraColumn<T>.CreateFromNullable call shape`
3. `test: hoist leading using directives out of gated snippet bodies`
4. `docs: split the blocks that mix a row type with statements`
5. `test: widen snippet BaseUsings to every public namespace`
6. `test: gate GETTING-STARTED.md and EXAMPLES.md, red on 129 known errors`
7. `docs: remove TODO.md - plan executed`

Steps 5 and 6 are recorded separately because step 5 is green on its own merits and step 6 is the
red state. Commit 5 alone is red on one assertion — the coverage count the step-4 splits invalidated
and step 6 re-pins — so the branch is red from step 4 onward, by design.

**Step 7 is deliberately not taken.** The skill's normal ending is to delete the plan once its gates
clear, but these cannot clear: the branch ends on a failing assertion by the human's decision, and
steps 5 and 6 of the plan are now #524–#532. This file is therefore not a live plan but the record
of *why* the branch is red, which a red CI run will otherwise look like an unexplained regression.
It is deleted when the follow-ups land, not before.

## Risks and open items

- **The branch is red and cannot merge until the follow-ups land.** `EveryGatedBlock_CompilesWithoutErrors`
  reports 129 errors and the human chose to let CI show that rather than hide it behind a narrowed
  allowlist. The consequence to keep visible: this branch is a mechanism, not a mergeable change, and
  #524 (100 of the 129) is the gate for the gate — until that lands, merging it would break main.
- **Commit 5 in isolation is red on the coverage assertion.** Verified, not inferred: stashing steps
  6's changes and running `DocumentationSnippetTests|DocSnippetCompilerTests` gives 26 passed / 1
  failed, and the failure is the repository-wide count that step 4 moved from 233 to 247. Stated here
  because a bisect landing on that commit will see a red that no single commit caused.
- **Defect yield turned out to be 29, not the 21 first projected** — and 5 of the original 21 were
  never defects. The projection was made before the gate ran and was wrong twice over; the verified
  partition in section 4 supersedes it.
- **Section-scoped preambles are too coarse**, as the plan suspected but could not confirm until the
  gate ran. `GETTING-STARTED.md`'s "Automatic Differentiation" section alone has 16 blocks, and
  `frame` appears in 36 blocks under incompatible shapes. The fix is per-block `locals:` plus a small
  number of named preambles for the row types the splits created (#524).
- **The hoist changes line mapping for all gated blocks.** Highest-risk step in the plan. Mitigated
  by dedicated tests and by re-running the injection check.
- **Stage 2 may need its own counted exclusions** — illustrative fragments that are not meant to
  compile. Each would be asserted by value, like the existing ASP.NET pair. None were needed at 129
  errors: every one of them names a real member or a real missing local, so no block has to be
  excused yet. If #524 finds a fragment that genuinely cannot compile, it must be counted by value
  rather than added to the allowlist.
- `DocWrapMode.File` previously had no consumer among the gated blocks, which was a speculative
  generality. Stage 2's 14 type-declaration-only blocks need it, so the concern resolves itself.

## GitHub issues log

Fixed here:

- [x] #517 — `docs/ACCELERATION.md:22` cites a nonexistent `src/Nivara.Gpu`
- [x] #518 — `docs/LINQ.md` calls `QueryFrame` internal; it is `public sealed class`
- [x] #520 — the subject of this plan. Premise correction is recorded above; the issue body still
      carries the original framing and should be corrected when it is closed.

Filed by this branch, one per fix, partition in section 4:

- [ ] #524 — 100 gated blocks fail on missing context (70 CS0103 + 30 CS0246). **The gate for the
      gate**: until this lands, this branch cannot merge.
- [ ] #525 — `ValidCount` (×2) and `operator>` on `NivaraColumn<T>` (×1) do not exist
- [ ] #526 — five AutoDiff call sites: `EncodeDecode`, `ZeroGrad`, `vocabSize:`, `batchSize:`, `lr:`
- [ ] #527 — seven IO / Arrow / tokenizer call sites, including the removed zero-copy Arrow API
- [ ] #528 — `Tensor<T>.AsSpan()` does not exist; the property is `.Span`
- [ ] #529 — five snippets pass `int[]` where `double[]` is required
- [ ] #530 — two snippets declare the same variable name twice (`recon`, `result`)
- [ ] #531 — two blocks still need a third split (CS8803); re-pins the universe 247 → 249
- [ ] #532 — **API**, not documentation: `SchemaValidationException` is declared in both
      `Nivara.Exceptions` and `Nivara.IO`, so any user importing both gets CS0104

Still to file:

- [ ] `README.md:58` — declares a row type before top-level statements, the same CS8803 defect as
      the 14 blocks split in step 4. `README.md` is not in the stage-2 allowlist, so the gate will
      not cover it. #531 fixes the same shape in `GETTING-STARTED.md`; this should be folded into
      that issue or filed once #531's exact shape is known.

Reminder: as each task executes, if you find deferred work or a concern (a known limitation, a
follow-up, a refactor) that is outside the current plan, create a tracked issue immediately via
`gh issue create --repo khurram-uworx/Nivara` and record its number in the log above. Do not rely on
memory or wait until the plan finishes — compaction during execution can lose important items.
---

# Plan — #528: `Tensor<T>` row-span extraction in EXAMPLES.md, plus the .NET 11 correction

Branch: `khurram/528`, branched off `main` at `8be35dea`.

This section is appended to the #520 record above rather than replacing it. That file is
load-bearing: it is the only explanation of why `EveryGatedBlock_CompilesWithoutErrors` is red
in CI, and the earlier plan was explicit that it is deleted only when the follow-ups land
(#524–#532), not before. #528 is one of those follow-ups.

## Problem

`EXAMPLES.md:222` and `EXAMPLES.md:444` call `.AsSpan()` on a `Tensor<float>`:

```csharp
scores[i] = TensorPrimitives.CosineSimilarity(docVectors.AsSpan().Slice(i * dims, dims), query);
```

The gate reports both as `CS1929`.

## Premise correction: #528 as filed is also wrong about the fix

The issue says *"the property is `.Span`"*. **`Tensor<T>` has no `Span` property.** Its
entire public surface, read from the .NET runtime source at the pinned commit
`3551975be08744f0418857c5bed8ab1545c5dd47` (`System.Numerics.Tensors` 11.0.0-rc.1), is:

| Member | Kind |
|---|---|
| `FlattenedLength`, `IsDense`, `IsEmpty`, `IsPinned`, `HasAnyDenseDimensions` | property |
| `Lengths`, `Rank`, `Strides` | property |
| `GetSpan` / `TryGetSpan` / `FlattenTo` / `TryFlattenTo` / `Slice` / `CopyTo` / `ToArray` / `AsTensorSpan` / `AsReadOnlyTensorSpan` | method |

There is no `Span`, no `Memory`, and no `AsSpan()`. Applying the issue's prescribed
substitution produces `CS1061: 'Tensor<float>' does not contain a definition for 'Span'`
(verified — candidate D below). **Implementing #528 literally trades one compile error for
another.** This is the second time a filed issue's premise has been wrong in this area
(#520 claimed the `CreateFromNullable` API was the outlier; it was the docs).

The diagnostic shape in the issue is explained: the compiler *did* resolve `AsSpan`, to
`MemoryExtensions.AsSpan(string?)`, which is why this reads as CS1929 rather than CS1061.

## Measurements — four candidate forms, compile **and** runtime

Compile-only evidence is not sufficient here, and the reason is the finding in row 2.

| # | Candidate | Compiles | Runs |
|---|---|---|---|
| D | `.Span.Slice(i * dims, dims)` (the issue's advice) | no — `CS1061` | — |
| A | `GetSpan([i], dims)` | yes | **no — throws `ArgumentOutOfRangeException`** |
| B | `GetSpan(new nint[] { i }, dims)` | yes | no — same throw |
| C | `GetSpan([i, 0], dims)` | yes | **yes** |

Two things this table settles that the issue could not:

1. **`GetSpan` requires an index for every dimension.** `startIndexes` must have `Rank`
   elements; `[i]` alone on a rank-2 tensor passes the compiler and throws on the first
   loop iteration. A compile-only gate reports this snippet as correct. This is the
   non-obvious part of the fix and the reason the probe in step 2 exists.
2. **`nint` vs `NIndex` is genuinely ambiguous** — `GetSpan` is overloaded on
   `ReadOnlySpan<nint>` and `ReadOnlySpan<NIndex>`, and `int` converts implicitly to both.
   A collection expression `[i, 0]` resolves (standard `int`→`nint` beats user-defined
   `int`→`NIndex`); `new nint[] { i, 0 }` also resolves, but `[]` and `Slice([i], ..)`
   do not (`CS0121`, `CS9174`).

Correctness of candidate C was checked against the snippet's own Python reference, not
merely asserted to compile:

| | doc-101 | doc-102 | doc-103 |
|---|---|---|---|
| `GetSpan([i,0],dims)` | 0.985318 | 0.410305 | 0.974284 |
| Python `v·q / (‖v‖‖q‖)` | 0.985318 | 0.410305 | 0.974284 |

and the descending ranking `doc-101, doc-103` matches the top-2 the document claims.

## Proposed changes

### 1. Preserve the probe as a `tensor-api` mode in `tests/Nivara.SimdProbe`

The probe used to establish the table above was a temp scratch project. #524–#532 are
nine more "the doc names the wrong API" issues of exactly this kind, so a throwaway is the
wrong home: the next session needs to check an API shape against the real assembly rather
than guess, and the failure mode (compiles but throws) is invisible to the snippet gate.

New `TensorApiProbe.cs`, `tensor-api` mode, following the existing `Correctness`/`Benchmark`
pattern and the `TransposeKernelProbe` precedent of a self-contained `--mode`:

- print the resolved `Tensor<T>` public surface by reflection, so the next agent reads the
  API instead of inferring it;
- for each candidate row-extraction form, record **compiles / runs / row matches the flat
  row-major slice**, catching row 2 above as a first-class result rather than a surprise;
- exit non-zero if the recommended form stops compiling or stops matching, so the probe is a
  gate on the API, not a printout;
- keep the `GetSpan` rank trap and the `nint`/`NIndex` ambiguity in its output and README,
  because both are invisible to a reader who has not been bitten.

Self-contained like the rest of the project (only `System.Numerics.Tensors`), so it stays
decoupled from Nivara internals and builds fast.

### 2. Fix the two EXAMPLES.md call sites

- `:222` — `docVectors.AsSpan().Slice(i * dims, dims)` → `docVectors.GetSpan([i, 0], dims)`
- `:444` — `embeddings.AsSpan().Slice(i * 4, 4)` → `embeddings.GetSpan([i, 0], 4)`

No prose changes. Checked per the issue's own instruction: `:186` ("lays them out row-major
as a 2D tensor") and `:422` ("Scored against the stored embeddings") describe the *layout*,
not the method, so both remain accurate.

### 3. Correct AGENTS.md — this is the root cause, not a footnote

AGENTS.md is the file an agent reads before writing a call, which is how these defects get
written in the first place (the #520 plan made the same argument about
`NivaraColumn<T>.CreateFromNullable`). It is stale on the target framework:

| AGENTS.md says | The build says |
|---|---|
| "Target framework: .NET 10.0 with System.Numerics.Tensors 10.0.10" (×2) | `net11.0`, `System.Numerics.Tensors 11.0.0-rc.1.26425.128` |
| "Key BCL .NET 10 tensor patterns" | .NET 11 |
| "Microsoft.ML 5.0.0" (×2) | `6.0.0-preview.26457.2` |
| "Parquet.Net 6.0.3" (×2) | `6.1.1-pre.1` |
| — (absent) | `Streamix 1.2.3`, `Microsoft.Extensions.AI.Abstractions 10.10.0` |

Every `.csproj` in the repo is already `net11.0`; `docs/TENSORS.md` already says
`11.0.0-preview.7`. Only AGENTS.md was left behind, and it is the highest-traffic doc for
exactly the fact that was wrong.

Also add the row-span rule and a pointer to the probe, so the next session does not
re-derive it: `Tensor<T>` has no `Span`/`AsSpan()`; use `GetSpan`; supply every rank index.

Grounding note: AGENTS.md edits must not introduce new `File.cs:NN` citations — the citation
gate (#518) scans it and this plan adds none, so no new citation risk is taken.

## Blast radius

- **`EXAMPLES.md`** — two lines in two fenced blocks. Both blocks currently fail the gate for
  *other* reasons too (`CS0246` on the row types at `:277` and `:449`, owned by #524), so
  neither block goes green here. The claim is precisely "these two diagnostics are gone",
  not "EXAMPLES.md compiles".
- **The gate stays red**, by the human's decision: `GETTING-STARTED.md` still reports its
  #524/#525/#527/#529/#530/#531 errors. The EXAMPLES.md error count must move 13 → 11 and no
  diagnostic may be added or removed.
- **`tests/Nivara.SimdProbe`** — new file + one `Program.cs` switch arm + README section.
  Standalone, not referenced by `Nivara.slnx` CI paths, no `[Category]`, not in the NUnit
  suite. Cannot affect the gate.
- **`AGENTS.md`** — prose only. Changes what future agents are told to write; that is the
  intent.
- No `src/` change, no public API change, no behaviour change.

## Verification

1. `dotnet run -c Release --project tests/Nivara.SimdProbe -- tensor-api` — prints the API
   surface, all four candidate forms with their compile/run results, and exits 0.
2. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --nologo --filter
   "FullyQualifiedName~DocumentationSnippetTests"` — **red by design**;
   `EveryGatedBlock_CompilesWithoutErrors` must drop from 13 to 11 EXAMPLES.md diagnostics,
   with `:222` and `:444` absent and every other diagnostic byte-identical. The other 10
   tests in the fixture must pass, including the coverage pins and the `Broken.md` negative
   control.
3. `dotnet test ... --filter "FullyQualifiedName~DocCitation"` — must stay green, since
   AGENTS.md and EXAMPLES.md are both in its scope.
4. Full suite — **ask first**; out of scope for a two-line docs fix.
5. Never report a check as passing that has not been seen fail or pass. Step 2's baseline
   count is captured from a real run, not from the partition table in the #520 section above.

## Planned commits

1. `docs: plan #528 in TODO.md`
2. `test: add a tensor-api probe for row-span extraction`
3. `docs: fix the two EXAMPLES.md row-extraction call sites`
4. `docs: correct the .NET 11 target and tensor API notes in AGENTS.md`

Step 3 alone cannot be verified by the gate, because both blocks still fail on #524's
missing row types. The pinned claim is the diagnostic diff, and step 2's probe is what makes
the replacement independently checkable.

## GitHub issues log

- [x] #528 — the subject of this section. **Its premise is corrected here**; the issue body
      still says the property is `.Span`, which does not exist, so it must be corrected on
      the way out or the next reader will implement it literally and fail.
- [ ] #524 — 100 missing-context errors; the gate for the gate. Untouched here.
- [ ] #525 / #526 / #527 / #529 / #530 / #531 — the other filed defects. Untouched here.
- [ ] #532 — `SchemaValidationException` is an API defect, not a doc defect. Untouched here.

Reminder: as each task executes, if you find deferred work or a concern outside this plan,
create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and
record its number above. Do not rely on memory or wait until the plan finishes.
