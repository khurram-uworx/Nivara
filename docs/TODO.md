# Plan — #518: gate markdown `File.cs:NN` citations on resolving

## Problem

No test reads the markdown documentation, so a doc can cite a file, line, or member that no
longer exists and nothing notices. #516 is the worked example: `AGENTS.md` and
`docs/STREAMING.md` both stated the default streaming memory budget as 256 MB while an
executable test asserted 1 GB, and the worked example could not detect it because the clamp
gave both inputs the same answer.

The same rot is **live today** in the citation layer. Across the 80 tracked markdown files,
116 `File.cs:NN` citation instances resolve as follows:

| Status | Count | Where |
|---|---|---|
| resolve | 101 | — |
| cite a file that no longer exists | 11 | `docs/plan/ARROW-REVIEW.md`, `docs/plan/POLARS-REVIEW.md` |
| cite a line past end-of-file | 4 | `docs/blog/1-*`, `docs/blog/4-*`, `docs/blog/README.md` |

The 11 missing-file citations point at `MemoryStorage.cs` / `TensorStorage.cs`, deleted
2026-08-03 by the storage consolidation (`630e9a70`, `0670f414`), and
`ExpressionEvaluator.cs`, replaced by `FusedExpressionEvaluator.cs`. They have been dangling
for ~2 months. The 4 past-EOF citations include `NivaraTensorExtensions.cs:1366-1388`, a
range in a file now 187 lines long.

`docs/STREAMING.md` also cited `StreamingExecutionStrategy.calculateChunkSize`, a name that
ceased to exist when the method was renamed to `CalculateChunkSize` during #516 — the same
failure class, caught only because that edit was made by hand in the same change.

## Scope decisions (agreed with the human)

1. **Gate form** — assert every `File.cs:NN` citation resolves, rather than #518's proposed
   "`256` must be accompanied by `StreamingBufferManager`". The proposed rule only catches
   re-introduction of that one number pair; a doc claiming 512 MB would pass it.
2. **Not a symbol-existence gate.** A blanket check over the 507 backticked `Type.Member`
   citations false-positives on legitimate design docs — `BudgetEnforcement` has 0 symbols
   but is a *proposed* API in `docs/ACCELERATION.md`.
3. **Scan scope** — reuse `DocSnippetExtractor.MarkdownFiles()` (repo root `*.md` plus
   `docs/**`). All 15 existing failures are under `docs/`, so nothing broken is out of scope,
   and the deliberate node_modules exclusion is preserved.
4. **Repo root** — reuse `DocSnippetExtractor.RepoRoot`, which walks up to `Nivara.slnx` and
   **throws** if not found. A gate that silently skips when it cannot find the docs is a gate
   that can silently stop gating.
5. **`ARROW-REVIEW.md` — repair citations, keep the banner.** Line 11 already carries a
   deliberate `Superseded 2026-08-03` banner, dated the same day as the deletions, naming
   which conclusions survive. Re-auditing would overwrite the author's own supersession note.
6. **`POLARS-REVIEW.md` — full re-audit of §4.** No such banner exists and its central claim
   is now false (see below).

## Proposed changes

### 1. The gate — `tests/Nivara.Tests/Docs/DocCitations.cs` + `DocCitationTests.cs`

A sibling to the existing `DocumentationSnippetTests`, reusing `RepoRoot` and
`MarkdownFiles()` rather than introducing a second root-discovery mechanism.

Extract citations with a regex that captures an optional path qualifier, the file basename,
and an optional line **range** (the range matters — `1366-1388` slips past a start-only
check). For each:

- **Path-qualified** (`src/Nivara/Storage/ColumnStorage.cs:167`) — must resolve exactly.
- **Bare basename unique in the repo** — must resolve to that one file.
- **Bare basename ambiguous** — **fail**, asking the author to qualify the path. Ambiguity is
  real: `Program.cs` exists in 14 files, `AttentionKernels.cs` in 2. Six such citations need
  qualifying; most are already path-qualified elsewhere in the same document.

Two independent verifiers, no prose parsing:
- the file exists;
- the **last** line of the range is `<=` the file's line count.

Rule to enforce: a citation resolves to exactly one file and a line inside it.

### 2. Repairs — the 15 broken citations

**`docs/blog/*` (4, past-EOF).** Re-find the current locations. Published prose that is
simply wrong.

**`docs/plan/ARROW-REVIEW.md` (8, missing-file).** Repoint at `ColumnStorage.cs`, which the
banner already names as the post-consolidation owner. Verified current facts:

| Old citation | Claim | Current truth | New target |
|---|---|---|---|
| `MemoryStorage.cs:12-13` | strings are reference arrays | holds — `ColumnStorage<T>` stores `readonly T[]` | `ColumnStorage.cs:18` |
| `TensorStorage.cs:14`, `MemoryStorage.cs:13` | nulls are 1 byte/element, not bit-packed | **holds** — `readonly bool[]? nullMask` | `ColumnStorage.cs:21` |
| `TensorStorage.cs:207-210`, `:33` | view access allocates a copy | **no longer true** — `AsSpan()` / `TryGetSpan` | `ColumnStorage.cs:238`, `:245` |
| `TensorStorage.cs:145`, `:145-156` | `Slice` copies | **no longer true** — `Slice` passes the same array with an offset | `ColumnStorage.cs:167` |
| `MemoryStorage.cs:130-139` | zero-copy works in the storage path | **now true more broadly** | `ColumnStorage.cs:167` |

Pillar 2's validity-bitmap gap and Pillar 4's chunked-column gap are unchanged, as the banner
states — `bool[]` is 1 byte per element.

**`docs/plan/POLARS-REVIEW.md` (3, missing-file) — re-audit §4.** The section is titled "The
one fatal flaw: the boxed expression evaluator" and is presented as the highest-leverage gap
in the project. It is now false. Verified via code-memory, all with **0 symbols**:

- `ApplyBinaryOperation` — gone
- `AddValues` — gone
- `RowExpressionBuilder` — gone
- `NivaraLinqExtensions` — gone (now `NivaraQuery`)

Replaced by `FusedExpressionEvaluator` (`src/Nivara/Expressions/FusedExpressionEvaluator.cs:34`)
and `FusedKernel` (`src/Nivara/Expressions/FusedKernel.cs:17`), shipped in v1.4.0. Per
`AGENTS.md` the primary path now compiles the `ColumnExpression` AST to cached delegates
(SIMD auto-vectorized), with a `FusedKernel` fallback and **no boxed fallback** —
non-fusible expressions throw.

Rewrite must re-derive, not merely re-point: whether the `NivaraColumn<object?>` result
degradation still occurs, and whether `OrderBy` still rejects computed keys
(`SortByExpressionOperation` now exists and takes a `FusedExpressionEvaluator`, suggesting it
does not). **If any claim cannot be settled from the code, flag it rather than invent a
conclusion.**

### 3. Non-vacuity

Prove the gate catches breakage two ways, both established patterns in this repo:

- a **negative-control fixture** under `tests/Nivara.Tests/Docs/Fixtures/` citing a
  nonexistent file and an over-long line, asserted to fail — mirroring `Broken.md`;
- a temporary one-line break of a real citation, observed red, then restored, so the gate is
  shown to fail on the documents it actually covers.

No red commit is left on the branch.

## Verification steps

1. `dotnet build Nivara.slnx` — 0 warnings, 0 errors.
2. Targeted: `dotnet test -c Release --filter "FullyQualifiedName~Doc"` — gate green over the
   real corpus; negative-control fixture red as designed.
3. Full suite: `dotnet test -c Release --filter "Category!=Performance"`, capturing the exit
   status of `dotnet test` itself (not of a pipeline stage).
4. Re-derive every citation this branch adds or moves, and confirm each resolves.

## Planned commits

1. `docs: plan #518 in TODO.md`
2. `docs: repair four stale blog citations` (the 4 past-EOF)
3. `docs: repoint ARROW-REVIEW storage citations at ColumnStorage` (8)
4. `docs: re-audit POLARS-REVIEW section 4 against the fused expression engine` (3)
5. `test: gate markdown File.cs:NN citations on resolving` (gate + fixture)
6. `docs: record the documentation citation gate in the changelog`
7. `docs: remove TODO.md - plan executed`

Order puts repairs before the gate so no commit is red; the gate's teeth are then proven by
the fixture and by the temporary-break experiment rather than by a red intermediate commit.

## Blast radius

**Files touched**

- `tests/Nivara.Tests/Docs/DocCitations.cs` (new), `DocCitationTests.cs` (new),
  `Fixtures/` (new negative control)
- `docs/blog/1-the-workhorses-from-words-to-numbers.md`, `docs/blog/4-*.md`,
  `docs/blog/README.md`
- `docs/plan/ARROW-REVIEW.md`, `docs/plan/POLARS-REVIEW.md`
- `CHANGELOG.md`, `docs/TODO.md` (this file, removed at G2)

**Dependencies** — none outside `Nivara.Tests`. The gate reads files only; it adds no
project reference, no package, and touches no `src/` symbol. It reuses
`DocSnippetExtractor.RepoRoot` and `MarkdownFiles()`, both `internal static` in the same
assembly, so there is no new visibility surface and no InternalsVisibleTo change.

**Not touched** — no `src/` file. `StreamingExecutionStrategy`, `ColumnStorage`,
`ArrowInterop` and `FusedExpressionEvaluator` are read-only inputs to the re-audit; the
citations move to point at them, nothing about them changes.

**Test-suite impact** — adds one fixture to `Nivara.Tests`. CI runs
`--filter "Category!=Performance"`, and these tests are not timing tests, so no
`[Category("Performance")]` is needed. The new tests are the only new coverage; no existing
test is modified.

**Carried-over work — must never be staged.** `src/Nivara/Execution/NivaraExecutionContext.cs`
carries an uncommitted change belonging to **#514** (`WithChunkSize`). It rode onto this
branch with the checkout and is not part of #518. Every commit stages explicit paths.

**Stacking** — branched off `khurram/515` (`815663db`), which is itself stacked on the #516
commits and has open PR #521. #518 therefore targets a PR base of `khurram/515`, not `main`.

## GitHub issues log

- [ ] #518 — the gate itself (this work)
- [ ] #514 — no public knob for a streaming memory budget; referenced, not duplicated.
      Uncommitted `WithChunkSize` work on this branch is #514's, not #518's.
- [ ] #516 — the originating drift; fixed in PR #519.
- [ ] #521 — #515's doc-snippet gate; this work extends its infrastructure rather than
      duplicating it. **Sequence #518 after #521 merges** to avoid two branches editing
      `tests/Nivara.Tests/Docs/` and the shared document lists.
