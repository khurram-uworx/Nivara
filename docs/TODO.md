# #547 — make the `docs/TODO.md` scan exclusion verifiable instead of decorative

Branch: `khurram/547` · Issue: [#547](https://github.com/khurram-uworx/Nivara/issues/547)

## Problem

`DocSnippetExtractor.ScanExcludedDocuments` lists `docs/TODO.md`. That file does not exist on
`main` — it was deleted in `da3f3c66`, and its git history is a long create/delete cycle
(`docs: plan #528 in TODO.md` … `docs: remove TODO.md — plan executed`).

The exclusion is applied as an exact-ordinal path match inside `MarkdownFiles()`
(`tests/Nivara.Tests/Docs/DocSnippet.cs:124`). With the file absent the match removes nothing, so
today the entry costs nothing and proves nothing.

### The correction that shapes this plan

The issue's stated failure mode — *"if `docs/TODO.md` is ever recreated it will silently re-enter
the gate's scope"* — **does not happen.** `MarkdownFiles()` filters by exact ordinal match on the
repo-relative path, so when the file exists it is dropped correctly. And it *is* recreated:
`iterative-work` **commits** the plan (`SKILL.md:83`) and only deletes it at G2 (`SKILL.md:86`), so
the exclusion is live for most of any multi-step workflow's life.

The real defect is narrower and sharper:

> **No test distinguishes "the exclusion works" from "the exclusion was deleted."**

With the file absent, those two worlds are byte-identical. Delete the `.Where(...)` at
`DocSnippet.cs:124` and the whole suite still passes, because both existing assertions are
unfalsifiable:

- `DocumentationSnippetTests.ScanExclusions_AreTheSingleTransientPlanDocument` asserts the **list
  constant** equals `["docs/TODO.md"]` — it never consults the filter.
- `DocCitationTests.ScanScope_ExcludesTheTransientPlanDocument` asserts `MarkdownFiles()` does not
  contain a path that is not there.

That is the same evidence defect this repo already litigates elsewhere
(`NegativeControl_TheGateStillRejectsABrokenSnippet`, `NegativeControl_StillAcceptsTheResolvableCitation`):
an assertion that cannot fail is not a gate, it is a comment that costs a build.

### Option B was rejected

Dropping the name and documenting a new plan location is *worse* here, not better. The plan file is
committed mid-workflow, so removing the exclusion exposes five pinned counts to a scratch file. And
`docs/plan/` is not an alternative — it holds durable named deliverables (`AISTACK-REVIEW.md`,
`POLARS-ROADMAP.md`, …), not transient workflow state.

## Proposed changes

### 1. One authoritative decision site

Extract the filter into a testable seam on `DocSnippetExtractor`, and let `MarkdownFiles()` consult
it. Single decision site, no duplicated predicate (AGENTS.md rule 8).

```csharp
/// <summary>Whether a repo-relative markdown path is inside the gate's scan scope.</summary>
internal static bool IsScanned(string repoRelativePath) => IsScanned(repoRelativePath, ScanExcludedDocuments);

internal static bool IsScanned(string repoRelativePath, IReadOnlyList<string> exclusions) =>
    !exclusions.Contains(repoRelativePath, StringComparer.Ordinal);
```

### 2. The counterfactual that makes the entry causal

The single assertion `IsScanned("docs/TODO.md") is false` is still satisfiable by a rule that
excludes everything. Pair it with the counterfactual: **remove the entry and the same path must be
scanned.** The pair cannot be satisfied by over-broad or empty rules.

Plus near-miss controls (`docs/TODO.mdx`, `docs/blog/TODO.md`, `TODO.md`, `docs/todo.md` must all
still be scanned) proving exact-ordinal named exclusion rather than prefix/substring/case folding.

### 3. The end-to-end proof — the only test that kills the filter-removal mutation

No predicate unit test can prove `MarkdownFiles()` *consults* the predicate. That needs the file to
be present. Add a shared helper that materialises the plan document, asserts, and cleans up:

- **If `docs/TODO.md` already exists** (the normal case *during* a workflow) assert against the real
  committed plan and **never delete it**.
- **If it does not** (a clean `main` checkout) synthesise it, assert, delete in a `finally`.

This is the part that matters for this branch specifically: while this plan is in flight the helper
takes the first branch, so the gate is proven against a genuine plan file.

The synthesised body is engineered to move **every** pin it could if it leaked — a fenced
` ```csharp ` block *and* a resolvable `File.cs:NN` citation. The citation is resolvable on purpose:
if it were unresolvable, an incidental fault could mask the leak, and the count assertion would stop
being the thing that catches it.

Assert, with the file present, that none of these move:

| Pin | Consumer |
|---|---|
| `MarkdownFiles()` omits `docs/TODO.md` | the exclusion itself |
| document count (47) | `DocCitationTests.ExpectedDocumentCount` |
| repo-wide snippet total (249) | `DocumentationSnippetTests` coverage |
| `CitationCount()` (106) | `DocCitationTests.ExpectedCitationCount` |
| `CitedDocumentCount()` (16) | `DocCitationTests.ExpectedCitedDocumentCount` |

Applied in **both** gate fixtures — `DocumentationSnippetTests` and `DocCitationTests` — because both
consume the scope and both would otherwise keep a tautology in place, which is precisely the trap
#547 is about.

Precondition guard: if the file already exists the helper uses it; it never deletes a file it did
not create.

### 4. Make the doc-count pin's arithmetic explainable

`ExpectedDocumentCount = 47` is `9 root *.md + 38 under docs/`. State that, and state that a live
plan file does **not** make it 48 — that exclusion is the whole point and is what keeps the number
stable across a workflow that commits one.

Refresh the `ScanExcludedDocuments` doc comment: it currently reads "deleted once its plan
executes", which implies inertness. It is committed for the duration of a workflow and removed at G2.

### 5. Repoint six dangling `docs/TODO.md` references (same branch, separate commit)

The mirror image of #547: #547 is a name that outlived its file in config; these are readers still
pointed at a file that was deleted. All six are in `tests/**`, outside the gate's scan scope, so
none moves a pin.

| Site | Claim made | Replacement |
|---|---|---|
| `tests/Nivara.GpuProbe/LevelZero/L0Run.cs:830` | pivots to compiler-produced SPIR-V via SYCL | `tests/Nivara.GpuProbe/SYCL.md` |
| `tests/Nivara.GpuProbe/Kernels/CpuLeg.cs:13` | every GPU leg gates against CpuLeg | `docs/ILGPU.md` (gate-vs-CpuLeg table + backend series) |
| `tests/Nivara.PerformanceTests/GpuAllocProbe.cs:8` | Laya memory leg | `docs/LAYA.md` §"Memory: the binary gate passes" |
| `tests/Nivara.PerformanceTests/CpuGemmProbe.cs:8` | Laya CPU GEMM leg | `docs/LAYA.md` |
| `tests/Nivara.PerformanceTests/CpuGemmProbe.cs:16` | grounding corrections | `docs/LAYA.md` |
| `tests/Nivara.PerformanceTests/Program.cs:734` | Qwen prefill/decode split | `docs/QWEN.md` — **verify the anchor exists first** |
| `tests/Nivara.SimdProbe/README.md:359` | do not trust a filed `Tensor<T>` prescription | `AGENTS.md` §"Common gotchas" (the `tensor-api` probe instruction) |

**Deliberately untouched** — these are correct historical narrative about a file that *was* deleted
on purpose, not dangling pointers:

- `CHANGELOG.md:11`
- `docs/ROADMAP-SUGGESTION.md:5` ("was deleted at G2")
- `docs/adr/005-snippet-gate-authoring-contract.md:16` ("the completing PR deletes on purpose")

## Verification steps

1. `dotnet build Nivara.slnx -c Release` — green.
2. Run the two gate fixtures. Expected: green **with this plan file present**, which is the proof
   that the exclusion holds against a real plan.
3. Mutation check (the point of the whole change): temporarily delete the `.Where(...)` from
   `MarkdownFiles()` and confirm the new tests **fail**. Restore. A green run against the mutant
   means the tests are still decorative and this change has achieved nothing.
4. Full suite: `dotnet test -c Release --filter "Category!=Performance"`.

## Planned commits

1. `docs: plan #547 in TODO.md`
2. `test: prove the docs/TODO.md scan exclusion instead of asserting its name`
3. `Repoint six dangling docs/TODO.md references at the docs that replaced them`

## Blast radius

**Changed, and nothing else:**

- `tests/Nivara.Tests/Docs/DocSnippet.cs` — extract `IsScanned`; refresh the
  `ScanExcludedDocuments` comment; add the plan-document helper.
- `tests/Nivara.Tests/Docs/DocumentationSnippetTests.cs` — counterfactual + near-miss tests; the
  end-to-end exists-and-is-excluded test.
- `tests/Nivara.Tests/Docs/DocCitationTests.cs` — promote the tautology to an end-to-end test;
  expand the doc-count comment.
- Six comment/doc sites under `tests/**` (part 5) — comments and prose only, no behaviour.

**Not touched:** the exclusion's behaviour is unchanged. `ScanExcludedDocuments` keeps its single
entry. No production code — this is entirely inside `tests/Nivara.Tests`. No public API, so no
CHANGELOG entry is warranted.

**Dependents:** `DocSnippetExtractor.MarkdownFiles()` has three callers outside its own fixture —
`DocCitationExtractor.AllFaults/CitationCount/CitedDocumentCount` (`DocCitations.cs:114,122,126`).
All three are covered by the pins asserted in step 3, so a regression in the extraction surfaces as
a citation-gate failure rather than silently.

**Risk:** the new end-to-end tests write `docs/TODO.md` into the working tree when it is absent.
Mitigations: `finally` cleanup; the helper never deletes a file it did not create; NUnit runs
sequentially here (no `[Parallelizable]` anywhere in `tests/Nivara.Tests`), so no interleaving with
the coverage fixtures.

**Residual risk accepted:** a hard process kill mid-test could leave a synthesised
`docs/TODO.md` behind. The helper's existence-first branch then makes the next run assert against
it harmlessly rather than fail confusingly, and it stays invisible to the gate by design.

## GitHub issues log

- [ ] #547 — this work (the issue being implemented)
